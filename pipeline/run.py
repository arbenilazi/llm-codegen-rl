import argparse
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import DatasetDict, load_from_disk

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline.annotate import annotate_candidates, execute_wrapped_code
from pipeline.analyzer.analyze import run_analysis
from pipeline.online_grpo import train_online_grpo
from predictor import EVAL_PROGRESS_JSON_ENV, EVAL_PROGRESS_LOG_ENV
from sanitize import normalize_entry_point

import torch.multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# Subprocess correctness evaluator 
def evaluate_pass_masks_in_memory(cfg, split_dir, candidates_path):
    """
    Compute correctness masks by executing wrapped code using subprocesses.
    Does NOT write any annotated file. Returns:
    - per_task_n: number of candidates per task
    - per_task_m: number of passing candidates per task
    """
    dataset_cfg = cfg["dataset"]
    from dataset import get_dataset

    dataset_path = (REPO_ROOT / dataset_cfg["output_dir"]).resolve()
    ds = get_dataset(dataset_cfg["name"], str(dataset_path)).dataset["test"]

    id_order = list(ds["id"])
    id2entry = {ex_id: entry for ex_id, entry in zip(ds["id"], ds["entry_point"])}
    raw_tests = ds["test"]

    # Clean test fields (same fix as annotate.py)
    def sanitize_test(t):
        if isinstance(t, dict):
            t = t["test"]
        t = (t or "").rstrip()
        while t.endswith("_"):
            t = t[:-1].rstrip()
        return t + "\n"

    tests = {tid: sanitize_test(t) for tid, t in zip(ds["id"], raw_tests)}

    # Load candidate codes
    from collections import defaultdict

    codes_by_task = defaultdict(list)

    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            tid = obj["id"]
            code = obj.get("code", "")
            entry = id2entry[tid]
            clean_code = normalize_entry_point(code, entry)
            wrapped = f"{tests[tid]}\n\n{clean_code}\n\ncheck({entry})"
            codes_by_task[tid].append(wrapped)

    per_task_n = {}
    per_task_m = {}

    for tid in id_order:
        wrapped_list = codes_by_task.get(tid, [])
        n = len(wrapped_list)
        m = 0
        for wrapped in wrapped_list:
            ok = execute_wrapped_code(wrapped, timeout=2.0)
            if ok:
                m += 1
        per_task_n[tid] = n
        per_task_m[tid] = m

    return per_task_n, per_task_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the MBPP fine-tuning pipeline.")
    parser.add_argument("--config", required=True, help="Path to a JSON configuration file.")
    parser.add_argument(
        "--only",
        default="baseline",
        help="Comma-separated stages to run: baseline,test_after,train_gen,rl_build,train_all,online_grpo,analyze",
    )
    return parser.parse_args()


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {k: v for k, v in (base or {}).items()}
    if override is None:
        return merged
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    extends = data.pop("extends", None)
    if not extends:
        return data

    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list):
        raise ValueError(f"'extends' must be a string or list of strings in {path}")

    merged: Dict[str, Any] = {}
    for parent in extends:
        if not isinstance(parent, str):
            raise ValueError(f"'extends' entries must be strings in {path}")
        parent_path = (REPO_ROOT / parent).resolve()
        parent_cfg = load_config(parent_path)
        merged = _deep_merge(merged, parent_cfg)

    merged = _deep_merge(merged, data)
    print(f"[cfg] loaded {path} (with extends)")
    return merged


def validate_config(cfg: Dict[str, Any]) -> None:
    required_root = ["experiment_name", "dataset", "generation", "training", "rl_build", "reports"]
    for key in required_root:
        if key not in cfg:
            raise ValueError(f"Configuration missing required key '{key}'.")

    dataset = cfg["dataset"]
    if not isinstance(dataset.get("name"), str):
        raise ValueError("dataset.name must be a string.")
    if not isinstance(dataset.get("output_dir"), str):
        raise ValueError("dataset.output_dir must be a string path.")

    generation = cfg["generation"]
    for field in ("model_name", "num_candidates"):
        if field not in generation:
            raise ValueError(f"generation.{field} is required.")
    if not isinstance(generation["num_candidates"], int) or generation["num_candidates"] <= 0:
        raise ValueError("generation.num_candidates must be a positive integer.")

    training = cfg["training"]
    if "output_dir" in training and not isinstance(training["output_dir"], str):
        raise ValueError("training.output_dir must be a string path if provided.")

    rl_build = cfg["rl_build"]
    for field in ("out_dir", "n_best", "max_pairs", "pairs_per_winner"):
        if field not in rl_build:
            raise ValueError(f"rl_build.{field} is required.")

    reports = cfg["reports"]
    if not isinstance(reports.get("dir"), str):
        raise ValueError("reports.dir must be a string path.")

def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dataset(dataset_cfg: Dict[str, Any]) -> Path:
    from dataset import prepare_dataset

    dataset_root = (REPO_ROOT / dataset_cfg["output_dir"]).resolve()
    if dataset_root.exists():
        return dataset_root

    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    hf_id = dataset_cfg["hf_id"]
    split_ratio = dataset_cfg["split_ratio"]
    split_seed = dataset_cfg.get("split_seed")

    print(f"[i] Dataset not found at {dataset_root}. Preparing from {hf_id}…")
    try:
        prepare_dataset(hf_id, split_ratio, str(dataset_root), seed=split_seed)
    except TypeError:
        prepare_dataset(hf_id, split_ratio, str(dataset_root))
    return dataset_root


def ensure_required_splits(dataset_root: Path, *, train_share: float, seed: Optional[int]) -> None:
    """
    Guarantee that on-disk dataset has both 'train' and 'test' *directories*.
    If only one exists, create the missing one as a subset of the present split
    WITHOUT altering or rewriting the present split.

    Policy:
      - Desired shares: train_share (e.g., 0.9) / test_share = 1 - train_share.
      - If only 'train' exists: create 'test' as test_share subset of that train.
      - If only 'test' exists: create 'train' as train_share subset of that test.
      - If both exist: no-op.
    """
    dataset_root = dataset_root.resolve()
    train_dir = dataset_root / "train"
    test_dir = dataset_root / "test"

    if train_dir.exists() and test_dir.exists():
        return

    base = load_from_disk(str(dataset_root))
    try:
        present_splits = list(base.keys())
    except AttributeError:
        present_splits = []

    if not present_splits:
        return

    rng_seed = seed if seed is not None else 42
    train_share = float(train_share)
    test_share = max(0.0, min(1.0, 1.0 - train_share))

    def _materialize(split_name: str, ds):
        DatasetDict({split_name: ds}).save_to_disk(str(dataset_root / split_name))

    if "train" in present_splits and not test_dir.exists():
        src = base["train"]
        if test_share > 0.0 and len(src) > 0:
            subset = src.train_test_split(test_size=test_share, seed=rng_seed)["test"]
        else:
            subset = src.select([])
        _materialize("test", subset)

    if "test" in present_splits and not train_dir.exists():
        src = base["test"]
        if train_share > 0.0 and len(src) > 0:
            subset = src.train_test_split(train_size=train_share, seed=rng_seed)["train"]
        else:
            subset = src.select([])
        _materialize("train", subset)


def ensure_split_views(dataset_root: Path, splits: Dict[str, Path]) -> None:
    base = load_from_disk(str(dataset_root))
    for split_name, split_path in splits.items():
        split_path = split_path.resolve()
        if split_path.exists():
            continue
        if split_name not in base:
            raise ValueError(f"Split '{split_name}' missing in dataset at {dataset_root}")
        print(f"[i] Materializing split '{split_name}' at {split_path}")
        DatasetDict({split_name: base[split_name]}).save_to_disk(str(split_path))


def update_predictor_samples(predictor_module: Any, n_samples: int):
    original_n = predictor_module.NUMBER_OF_SOLUTIONS
    original_prompt = predictor_module.SYSTEM_PROMPT

    if original_n == n_samples:
        def restore() -> None:
            pass

        return restore

    pattern = re.compile(rf"\b{original_n}\b")
    predictor_module.NUMBER_OF_SOLUTIONS = n_samples
    predictor_module.SYSTEM_PROMPT = pattern.sub(str(n_samples), original_prompt)

    def restore() -> None:
        predictor_module.NUMBER_OF_SOLUTIONS = original_n
        predictor_module.SYSTEM_PROMPT = original_prompt

    return restore


@contextmanager
def override_mbpp_default_split(target_split: Optional[str]):
    if target_split is None:
        yield
        return

    import dataset as dataset_module

    if not hasattr(dataset_module, "MBPPDataset"):
        yield
        return

    original = dataset_module.MBPPDataset._default_split

    def _patched(self) -> str:  # type: ignore[override]
        return target_split

    dataset_module.MBPPDataset._default_split = _patched
    try:
        yield
    finally:
        dataset_module.MBPPDataset._default_split = original


def run_predictor_for_split(
    dataset_name: str,
    dataset_root: Path,
    model_name: str,
    split: Optional[str],
    n_samples: int,
    batch_size: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    adapter_path: Optional[Path] = None,
    tag: Optional[str] = None,
) -> Dict[str, Path]:
    import predictor as predictor_module

    restore = update_predictor_samples(predictor_module, n_samples)
    dataset_root = dataset_root.resolve()
    if not split:
        raise ValueError("Split must be provided when running predictor.")

    try:
        with override_mbpp_default_split(split):
            predictor_module.run_predictions(
                model_name=model_name,
                dataset_name=dataset_name,
                dataset_path=str(dataset_root),
                split=split,
                num_samples=n_samples,
                batch_size=batch_size,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                adapter_path=str(adapter_path) if adapter_path else None,
                tag=tag,
            )
    finally:
        restore()

    split_dir = dataset_root / split
    suffix = f"_{tag}" if tag else ""
    outputs = {
        "predictions": split_dir / f"predictions{suffix}.json",
        "candidates": split_dir / f"candidates{suffix}.jsonl",
        "evaluations": split_dir / f"evaluations{suffix}.json",
    }
    return outputs


def baseline_eval_on_test(cfg: Dict[str, Any]) -> Dict[str, float]:
    dataset_cfg = cfg["dataset"]
    generation_cfg = cfg["generation"]
    eval_settings = _resolve_eval_settings(cfg)
    gen_batch_size = int(generation_cfg.get("batch_size", 32))
    ks = eval_settings["ks"]
    num_procs = eval_settings["num_procs"]
    batch_size_eval = eval_settings["batch_size"]
    if cfg.get("_config_path"):
        print(f"[eval] baseline config: {cfg['_config_path']}")
    print(f"[eval] planned ks: {ks}")
    dataset_root = (REPO_ROOT / dataset_cfg["output_dir"]).resolve()
    split_dir = dataset_root / "test"
    eval_path = split_dir / "evaluations_baseline.json"
    if eval_path.exists():
        try:
            with eval_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            stored_ks = existing.get("ks")
            if stored_ks is not None and [int(x) for x in stored_ks] != ks:
                print(f"[eval] ks changed from {stored_ks} to {ks}; recomputing evaluations.")
        except Exception:
            pass
    env_overrides = _evaluation_env_overrides(cfg)
    reports_dir = (REPO_ROOT / cfg["reports"]["dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_log = reports_dir / f"{cfg['experiment_name']}_eval_progress.log"
    progress_json = reports_dir / f"{cfg['experiment_name']}_eval_progress.json"
    _write_resolved_eval_snapshot(cfg, ks, num_procs, batch_size_eval)
    env_overrides.update(
        {
            EVAL_PROGRESS_LOG_ENV: str(progress_log),
            EVAL_PROGRESS_JSON_ENV: str(progress_json),
            "EVAL_KS": json.dumps(ks),
            "EVAL_NUM_PROCS": str(num_procs),
            "EVAL_BATCH_SIZE": str(batch_size_eval),
            "EVAL_N_TEST": str(generation_cfg["num_candidates"]),
        }
    )
    with _temporary_env(env_overrides):
        outputs = run_predictor_for_split(
            dataset_name=dataset_cfg["name"],
            dataset_root=dataset_root,
            model_name=generation_cfg["model_name"],
            split="test",
            n_samples=generation_cfg["num_candidates"],
            batch_size=gen_batch_size,
            temperature=generation_cfg.get("temperature"),
            top_p=generation_cfg.get("top_p"),
            max_tokens=generation_cfg.get("max_tokens"),
            stop=generation_cfg.get("stop"),
            tag="baseline",
        )
    eval_path = split_dir / "evaluations_baseline.json"
    if outputs["evaluations"] != eval_path:
        copy_if_exists(outputs["evaluations"], eval_path)

    #pass@k using subprocess correctness
    per_task_n, per_task_m = evaluate_pass_masks_in_memory(cfg, split_dir, outputs["candidates"])

    ks_list = ks 
    pass_at_k = {}

    for k in ks_list:
        vals = []
        for tid in per_task_n:
            n = per_task_n[tid]
            m = per_task_m[tid]
            v = _compute_task_pass_at_k(n, m, k)
            if v is not None:
                vals.append(v)
        pass_at_k[str(k)] = (sum(vals) / len(vals)) if vals else 0.0

    # Load predictor's diversity
    with open(eval_path, "r", encoding="utf-8") as f:
        predictor_eval = json.load(f)

    div_ratio = predictor_eval.get("diversity_unique_ratio", 0.0)
    n_test = predictor_eval.get("n_test", len(per_task_n))

    final_payload = {
        "pass@k": pass_at_k,
        "ks": ks_list,
        "n_test": n_test,
        "diversity_unique_ratio": div_ratio,
    }

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    return pass_at_k


def post_training_eval_on_test(cfg: Dict[str, Any]) -> Dict[str, float]:
    dataset_cfg = cfg["dataset"]
    generation_cfg = cfg["generation"]
    eval_settings = _resolve_eval_settings(cfg)
    gen_batch_size = int(generation_cfg.get("batch_size", 32))
    ks = eval_settings["ks"]
    num_procs = eval_settings["num_procs"]
    batch_size_eval = eval_settings["batch_size"]
    if cfg.get("_config_path"):
        print(f"[eval] baseline config: {cfg['_config_path']}")
    print(f"[eval] planned ks: {ks}")
    dataset_root = (REPO_ROOT / dataset_cfg["output_dir"]).resolve()
    split_dir = dataset_root / "test"
    eval_path = split_dir / "evaluations_after.json"
    if eval_path.exists():
        try:
            with eval_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            stored_ks = existing.get("ks")
            if stored_ks is not None and [int(x) for x in stored_ks] != ks:
                print(f"[eval] ks changed from {stored_ks} to {ks}; recomputing evaluations.")
        except Exception:
            pass
    env_overrides = _evaluation_env_overrides(cfg)
    reports_dir = (REPO_ROOT / cfg["reports"]["dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    progress_log = reports_dir / f"{cfg['experiment_name']}_eval_progress.log"
    progress_json = reports_dir / f"{cfg['experiment_name']}_eval_progress.json"
    _write_resolved_eval_snapshot(cfg, ks, num_procs, batch_size_eval)
    env_overrides.update(
        {
            EVAL_PROGRESS_LOG_ENV: str(progress_log),
            EVAL_PROGRESS_JSON_ENV: str(progress_json),
            "EVAL_KS": json.dumps(ks),
            "EVAL_NUM_PROCS": str(num_procs),
            "EVAL_BATCH_SIZE": str(batch_size_eval),
            "EVAL_N_TEST": str(generation_cfg["num_candidates"]),
        }
    )
    with _temporary_env(env_overrides):
        outputs = run_predictor_for_split(
            dataset_name=dataset_cfg["name"],
            dataset_root=dataset_root,
            model_name=generation_cfg["model_name"],
            split="test",
            n_samples=generation_cfg["num_candidates"],
            batch_size=gen_batch_size,
            temperature=generation_cfg.get("temperature"),
            top_p=generation_cfg.get("top_p"),
            max_tokens=generation_cfg.get("max_tokens"),
            stop=generation_cfg.get("stop"),
            tag="after",
        )
    eval_path = split_dir / "evaluations_after.json"
    if outputs["evaluations"] != eval_path:
        copy_if_exists(outputs["evaluations"], eval_path)

    #pass@k using subprocess correctness
    per_task_n, per_task_m = evaluate_pass_masks_in_memory(cfg, split_dir, outputs["candidates"])

    ks_list = ks 
    pass_at_k = {}

    for k in ks_list:
        vals = []
        for tid in per_task_n:
            n = per_task_n[tid]
            m = per_task_m[tid]
            v = _compute_task_pass_at_k(n, m, k)
            if v is not None:
                vals.append(v)
        pass_at_k[str(k)] = (sum(vals) / len(vals)) if vals else 0.0

    # Load predictor's diversity
    with open(eval_path, "r", encoding="utf-8") as f:
        predictor_eval = json.load(f)

    div_ratio = predictor_eval.get("diversity_unique_ratio", 0.0)
    n_test = predictor_eval.get("n_test", len(per_task_n))

    final_payload = {
        "pass@k": pass_at_k,
        "ks": ks_list,
        "n_test": n_test,
        "diversity_unique_ratio": div_ratio,
    }

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)

    return pass_at_k


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    if not overrides:
        yield
        return
    original = {}
    try:
        for key, value in overrides.items():
            original[key] = os.environ.get(key)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in overrides.items():
            original_value = original.get(key)
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def _evaluation_env_overrides(cfg: Dict[str, Any]) -> Dict[str, Optional[str]]:
    eval_cfg = cfg.get("evaluation", {})
    overrides: Dict[str, Optional[str]] = {}
    if eval_cfg.get("num_procs") is not None:
        overrides["EVAL_NUM_PROCS"] = str(eval_cfg["num_procs"])
    if eval_cfg.get("batch_size") is not None:
        overrides["EVAL_BATCH_SIZE"] = str(eval_cfg["batch_size"])
    return overrides


def _resolve_eval_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})

    ks = eval_cfg.get("ks")
    if ks is None:
        env_ks = os.getenv("EVAL_KS")
        if env_ks:
            try:
                parsed = json.loads(env_ks)
                if isinstance(parsed, (list, tuple)):
                    ks = [int(x) for x in parsed]
                else:
                    ks = [int(x.strip()) for x in env_ks.split(",") if x.strip()]
            except Exception:
                ks = [1, 5, 10]
        else:
            ks = [1, 5, 10]

    num_procs = eval_cfg.get("num_procs")
    if num_procs is None:
        env_np = os.getenv("EVAL_NUM_PROCS")
        if env_np:
            try:
                num_procs = max(1, int(env_np))
            except ValueError:
                num_procs = max(1, os.cpu_count() or 1)
        else:
            num_procs = max(1, os.cpu_count() or 1)

    batch_size = eval_cfg.get("batch_size")
    if batch_size is None:
        env_bs = os.getenv("EVAL_BATCH_SIZE")
        if env_bs:
            try:
                batch_size = max(1, int(env_bs))
            except ValueError:
                batch_size = 32
        else:
            batch_size = 32

    return {
        "ks": [int(x) for x in ks],
        "num_procs": int(num_procs),
        "batch_size": int(batch_size),
    }


def _write_resolved_eval_snapshot(
    cfg: Dict[str, Any], ks: list[int], num_procs: int, batch_size: int
) -> Path:
    reports_dir = (REPO_ROOT / cfg["reports"]["dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = reports_dir / f"{cfg['experiment_name']}_resolved_eval.json"
    payload = {
        "ks": ks,
        "num_procs": num_procs,
        "batch_size": batch_size,
        "n_test": int(cfg["generation"]["num_candidates"]),
    }
    with snapshot_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return snapshot_path


def _resolve_manifest_ks(cfg: Dict[str, Any]) -> list[int]:
    reports_dir = (REPO_ROOT / cfg["reports"]["dir"]).resolve()
    resolved_path = reports_dir / f"{cfg['experiment_name']}_resolved_eval.json"
    if resolved_path.exists():
        try:
            with resolved_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            ks = data.get("ks")
            if ks:
                return [int(x) for x in ks]
        except Exception:
            pass

    eval_cfg = cfg.get("evaluation", {})
    if eval_cfg.get("ks"):
        return [int(x) for x in eval_cfg["ks"]]

    return [1, 3, 5, 10, 20, 50, 100]


def _compute_task_pass_at_k(n: int, m: int, k: int) -> Optional[float]:
    if n <= 0:
        return None
    if m <= 0:
        return 0.0 if n >= k else None
    if k >= n:
        return 1.0
    if n < k:
        return None
    try:
        denom = math.comb(n, k)
    except ValueError:
        return 0.0
    if denom == 0:
        return 0.0
    try:
        fail_comb = math.comb(n - m, k) if (n - m) >= k else 0
    except ValueError:
        fail_comb = 0
    value = 1.0 - fail_comb / denom
    return max(0.0, min(1.0, value))


def _update_manifest_with_annotations(
    cfg: Dict[str, Any],
    rl_dir: Path,
    annotated_path: Path,
) -> None:
    manifest_path = rl_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[rl] warning: manifest not found at {manifest_path}; skipping analytic pass@k update.")
        return

    annotated_path = Path(annotated_path)
    if not annotated_path.exists():
        print(f"[rl] warning: annotated candidates missing at {annotated_path}; leaving manifest untouched.")
        return

    ks = sorted({int(x) for x in _resolve_manifest_ks(cfg)})
    strict_used = False
    per_task: Dict[Any, Dict[str, Any]] = {}

    with annotated_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = data.get("id")
            if task_id is None:
                continue

            entry = per_task.setdefault(task_id, {"seen": set(), "n": 0, "m": 0, "pass_at_k": {}})
            seen = entry["seen"]

            candidate_idx = data.get("candidate_idx")
            if candidate_idx is not None:
                key = ("idx", candidate_idx)
            else:
                metrics = data.get("metrics") or {}
                ast_hash = metrics.get("ast_hash")
                if ast_hash:
                    key = ("hash", ast_hash)
                else:
                    code = data.get("code")
                    key = ("code", hash(code) if code is not None else len(seen))

            if key in seen:
                continue
            seen.add(key)

            passed_value = data.get("passed_strict")
            if passed_value is not None:
                strict_used = True
            else:
                passed_value = data.get("passed")

            passed = bool(passed_value)
            entry["n"] += 1
            if passed:
                entry["m"] += 1

    if not per_task:
        print(f"[rl] warning: no annotated candidates in {annotated_path}; manifest unchanged.")
        return

    for stats in per_task.values():
        stats.pop("seen", None)

    pass_at_k_global: Dict[str, float] = {}
    contributing_counts: Dict[int, int] = {}
    total_candidates = 0
    total_pass = 0
    tasks_with_pass = 0
    m_values = []

    for stats in per_task.values():
        total_candidates += stats["n"]
        total_pass += stats["m"]
        m_values.append(stats["m"])
        if stats["m"] > 0:
            tasks_with_pass += 1

    for k in ks:
        contributions = []
        for task_id, stats in per_task.items():
            value = _compute_task_pass_at_k(stats["n"], stats["m"], k)
            stats["pass_at_k"][str(k)] = value
            if value is None:
                continue
            contributions.append(value)
        contributing_counts[k] = len(contributions)
        if contributions:
            pass_at_k_global[str(k)] = round(sum(contributions) / len(contributions), 8)
        else:
            pass_at_k_global[str(k)] = 0.0

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if "items_with_pass" in manifest:
        manifest.setdefault("kept_items_with_pass", manifest["items_with_pass"])
    if "total_passing_candidates" in manifest:
        manifest.setdefault("kept_total_passing_candidates", manifest["total_passing_candidates"])

    manifest["items_with_pass"] = tasks_with_pass
    manifest["total_passing_candidates"] = total_pass
    manifest["total_candidates"] = total_candidates
    manifest["eval_pass@k"] = pass_at_k_global
    manifest["eval_source"] = "analytic_from_pass_masks"
    manifest["ks"] = ks
    manifest["strict_per_candidate_used"] = bool(strict_used)

    write_per_task = cfg.get("rl_build", {}).get("write_per_task_passk")
    if write_per_task is None:
        write_per_task = os.getenv("RL_WRITE_PER_TASK_PASSK") == "1"

    if write_per_task:
        per_task_path = rl_dir / "per_task_passk.jsonl"
        with per_task_path.open("w", encoding="utf-8") as handle:
            for task_id, stats in per_task.items():
                record = {
                    "id": task_id,
                    "n": stats["n"],
                    "m": stats["m"],
                    "pass_at_k": {str(k): stats["pass_at_k"].get(str(k)) for k in ks},
                }
                handle.write(json.dumps(record) + "\n")

    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    tasks_total = len(per_task)
    tasks_with_candidates = sum(1 for stats in per_task.values() if stats["n"] > 0)
    mean_m = statistics.mean(m_values) if m_values else 0.0
    median_m = statistics.median(m_values) if m_values else 0.0
    print(
        f"[rl] analytic pass@k: tasks_total={tasks_total}, tasks_with_candidates={tasks_with_candidates}, "
        f"mean_m={mean_m:.2f}, median_m={median_m:.2f}"
    )
    print(
        "[rl] contributing per k: "
        + json.dumps({str(k): contributing_counts[k] for k in ks})
    )
    print(f"[rl] updated manifest eval_pass@k from annotations: {pass_at_k_global}")


def read_pass_at_k(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    passk = data.get("pass@k", {})
    out: Dict[str, float] = {}
    for key, value in passk.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def build_rl_dataset(
    cmd_cfg: Dict[str, Any],
    dataset_path: Path,
    rl_out_dir: Path,
    candidates_path: Optional[Path] = None,
) -> None:
    rl_out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "build_rl_dataset.py"),
        "--dataset",
        cmd_cfg["dataset"],
        "--dataset_path",
        str(dataset_path),
        "--out_dir",
        str(rl_out_dir),
        "--n_best",
        str(cmd_cfg["n_best"]),
        "--max_pairs",
        str(cmd_cfg["max_pairs"]),
        "--pairs_per_winner",
        str(cmd_cfg["pairs_per_winner"]),
        "--within_pass_pairs_per_winner",
        str(cmd_cfg["within_pass_pairs_per_winner"]),
        "--fails_per_passer",
        str(cmd_cfg["fails_per_passer"]),
    ]
    if candidates_path is not None:
        cmd.extend(["--candidates_path", str(candidates_path)])
    print(f"[i] Building RL dataset via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def train_dpo(train_cfg: Dict[str, Any], dataset_cfg: Dict[str, Any], rl_dir: Path, model_name: str) -> Path:
    output_dir = (REPO_ROOT / train_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train_dpo.py"),
        "--model_name",
        model_name,
        "--dataset",
        dataset_cfg["name"],
        "--dataset_path",
        str(REPO_ROOT / dataset_cfg["output_dir"]),
        "--rl_dir",
        str(rl_dir),
        "--output_dir",
        str(output_dir),
        "--epochs",
        str(train_cfg["epochs"]),
        "--batch_size",
        str(train_cfg["batch_size"]),
        "--lr",
        str(train_cfg["lr"]),
        "--logging_steps",
        str(train_cfg["logging_steps"]),
        "--save_steps",
        str(train_cfg["save_steps"]),
        "--eval_steps",
        str(train_cfg["eval_steps"]),
    ]

    if "gradient_accumulation_steps" in train_cfg:
        cmd.extend(["--gradient_accumulation_steps", str(train_cfg["gradient_accumulation_steps"])])

    if "max_pairs_per_task" in train_cfg:
        cmd.extend(["--max_pairs_per_task", str(train_cfg["max_pairs_per_task"])])

    if train_cfg.get("use_lora"):
        cmd.append("--use_lora")
        cmd.extend(["--lora_r", str(train_cfg["lora_r"])])
        cmd.extend(["--lora_alpha", str(train_cfg["lora_alpha"])])
        cmd.extend(["--lora_dropout", str(train_cfg["lora_dropout"])])
        target_modules = train_cfg.get("target_modules")
        if target_modules:
            cmd.extend(["--target_modules", ",".join(target_modules)])

    print(f"[i] Launching DPO training via: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return find_latest_checkpoint(output_dir)


def find_latest_checkpoint(output_dir: Path) -> Path:
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return output_dir

    def _key(path: Path):
        match = re.search(r"checkpoint-(\d+)", path.name)
        step = int(match.group(1)) if match else -1
        return (step, path.stat().st_mtime)

    return max(checkpoints, key=_key)


def write_report(
    cfg: Dict[str, Any],
    baseline_pass: Dict[str, float],
    after_pass: Dict[str, float],
    checkpoint_path: Path,
) -> Dict[str, Path]:
    reports_cfg = cfg["reports"]
    experiment_name = cfg["experiment_name"]
    dataset_cfg = cfg["dataset"]
    rl_cfg = cfg["rl_build"]

    reports_dir = (REPO_ROOT / reports_cfg["dir"]).resolve()
    reports_dir.mkdir(parents=True, exist_ok=True)

    ks = ["1", "3", "5", "10"]
    delta_pass = {f"pass@{k}": after_pass.get(k, 0.0) - baseline_pass.get(k, 0.0) for k in ks}

    def _format_pass(pass_dict: Dict[str, float]) -> Dict[str, float]:
        return {f"pass@{k}": pass_dict.get(k, 0.0) for k in ks}

    manifest_path = REPO_ROOT / rl_cfg["out_dir"] / "manifest.json"
    json_payload = {
        "experiment": experiment_name,
        "seed": cfg.get("seed"),
        "train_ratio": dataset_cfg.get("split_ratio"),
        "baseline_test": _format_pass(baseline_pass),
        "after_test": _format_pass(after_pass),
        "delta": delta_pass,
        "rl_manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
    }

    outputs: Dict[str, Path] = {}
    json_path = reports_dir / f"{experiment_name}_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, indent=2)
    outputs["json"] = json_path

    if reports_cfg.get("write_markdown"):
        md_path = reports_dir / f"{experiment_name}_summary.md"
        rows = []
        for k in ks:
            base = baseline_pass.get(k, 0.0)
            aft = after_pass.get(k, 0.0)
            delta = aft - base
            rows.append(f"| pass@{k} | {base:.3f} | {aft:.3f} | {delta:+.3f} |")

        md_lines = [
            f"# {experiment_name} Summary",
            "",
            f"- Seed: `{cfg.get('seed')}`",
            f"- Train/Test split ratio: `{dataset_cfg.get('split_ratio')}`",
            f"- RL manifest: `{json_payload['rl_manifest']}`",
            f"- Checkpoint: `{json_payload['checkpoint']}`",
            "",
            "| Metric | Baseline | After | Delta |",
            "| --- | --- | --- | --- |",
            *rows,
        ]
        with md_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(md_lines) + "\n")
        outputs["markdown"] = md_path

    return outputs


def resolve_training_output_dir(
    training_cfg: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> Path:
    # Explicit always wins
    if training_cfg.get("output_dir"):
        return Path(training_cfg["output_dir"])

    rl_method = training_cfg.get("rl_method", "dpo").lower()
    dataset = dataset_cfg["name"]
    suffix = "_lora" if training_cfg.get("use_lora") else ""

    return Path("results") / f"{rl_method}_{dataset}{suffix}"


def main() -> None:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    config = load_config(config_path)
    validate_config(config)
    gen = config.get("generation", {})
    if "n_train" in gen or "n_test" in gen:
        gen["num_candidates"] = gen.get("n_train") or gen.get("n_test") or gen.get("num_candidates")
        gen.pop("n_train", None)
        gen.pop("n_test", None)
    config_json = json.dumps(config, indent=2)
    config["_config_path"] = str(config_path)

    seed = config.get("seed")
    if seed is not None:
        print(f"[i] Setting global seed to {seed}")
        set_global_seeds(seed)

    selected = {token.strip().lower() for token in args.only.split(",") if token.strip()}

    valid_stages = {
        "analyze",
        "baseline",
        "train_gen",
        "rl_build",
        "train_all",
        "test_after",
        "online_grpo",
    }

    unknown = {s for s in selected if s not in valid_stages}
    if unknown:
        raise ValueError(
            f"Unknown stage(s): {sorted(unknown)}. Valid: {sorted(valid_stages)}"
        )

    def stage_enabled(name: str) -> bool:
        return name.lower() in selected

    print(f"[i] Running stages: {', '.join(sorted(selected))}")

    dataset_cfg = config["dataset"]
    generation_cfg = config["generation"]
    batch_size = int(generation_cfg.get("batch_size", 32))
    rl_cfg = config["rl_build"]
    training_cfg = config["training"]

    experiment_name = config["experiment_name"]
    rl_config_path = REPO_ROOT / config["rl_build"]["out_dir"] / f"{experiment_name}.config.json"
    reports_config_path = REPO_ROOT / config["reports"]["dir"] / f"{experiment_name}.config.json"
    for dest in (rl_config_path, reports_config_path):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(config_json, encoding="utf-8")

    dataset_root = ensure_dataset(dataset_cfg)

    ensure_required_splits(
        dataset_root,
        train_share=float(dataset_cfg.get("split_ratio", 0.9)),
        seed=dataset_cfg.get("split_seed"),
    )
    split_paths = {
        "train": dataset_root / "train",
        "test": dataset_root / "test",
    }
    ensure_split_views(dataset_root, split_paths)

    train_split_path = split_paths["train"]
    test_split_path = split_paths["test"]

    # Online GRPO training (single-stage online RL)
    if stage_enabled("online_grpo"):
        print("[stage] Online GRPO training (train split)")
        checkpoint_path = train_online_grpo(config)
        print(f"[i] Online GRPO output dir: {checkpoint_path}")

    # Baseline evaluation
    baseline_eval_path = test_split_path / "evaluations_baseline.json"
    if stage_enabled("baseline"):
        print("[stage] Baseline evaluation (test split)")
        baseline_pass = baseline_eval_on_test(config)
    else:
        print("[skip] Baseline evaluation (test split)")
        baseline_pass = read_pass_at_k(baseline_eval_path)
        if not baseline_pass:
            print(f"[warn] Baseline results not found at {baseline_eval_path}")
    print(f"[i] Baseline pass@k: {baseline_pass}")

    if stage_enabled("analyze"):
        print("[stage] Baseline analysis (offline)")
        run_analysis(config)
        print("[ok] Analysis complete.")
        return
    else:
        print("[skip] Baseline analysis")

    # Training split generation
    train_candidates = train_split_path / "candidates.jsonl"
    annotated_candidates = train_split_path / "candidates_annotated.jsonl"

    if stage_enabled("train_gen"):
        print("[stage] Train split generation")
        train_outputs = run_predictor_for_split(
            dataset_name=dataset_cfg["name"],
            dataset_root=dataset_root,
            model_name=generation_cfg["model_name"],
            split="train",
            n_samples=generation_cfg["num_candidates"],
            batch_size=batch_size,
            temperature=generation_cfg.get("temperature"),
            top_p=generation_cfg.get("top_p"),
            max_tokens=generation_cfg.get("max_tokens"),
            stop=generation_cfg.get("stop"),
        )

        print("[stage] Candidate annotation (train split)")
        ann_path = Path(
            annotate_candidates(
                config,
                split="train",
                in_path=str(train_candidates),
                out_path=str(annotated_candidates),
            )
        )
        print(f"[i] Annotated candidates written to {ann_path}")
    else:
        ann_path = annotated_candidates
        print("[skip] Train split generation; reusing existing artifacts")

    # Build RL dataset
    rl_dir = (REPO_ROOT / rl_cfg["out_dir"]).resolve()
    if stage_enabled("rl_build"):
        print("[stage] RL dataset construction")
        rl_cmd_cfg = {
            "dataset": dataset_cfg["name"],
            "n_best": rl_cfg["n_best"],
            "max_pairs": rl_cfg["max_pairs"],
            "pairs_per_winner": rl_cfg["pairs_per_winner"],
            "within_pass_pairs_per_winner": rl_cfg.get("within_pass_pairs_per_winner", 1),
            "fails_per_passer": rl_cfg.get("fails_per_passer", 2),
        }

        build_rl_dataset(rl_cmd_cfg, dataset_root, rl_dir, candidates_path=ann_path)
    else:
        print("[skip] RL dataset construction; expecting existing RL artifacts")

    _update_manifest_with_annotations(config, rl_dir, ann_path)

    # Resolve training output directory (explicit or derived)
    resolved_output_dir = resolve_training_output_dir(training_cfg, dataset_cfg)
    training_cfg["output_dir"] = str(resolved_output_dir)
    print(f"[i] Training output dir resolved to: {resolved_output_dir}")

    # RL training (config-driven method selection)
    checkpoint_root = (REPO_ROOT / resolved_output_dir).resolve()
    if stage_enabled("train_all"):
        rl_method = training_cfg.get("rl_method", "dpo").lower()
        print(f"[i] RL method selected: {rl_method}")
        print(f"[stage] RL training ({rl_method.upper()})")
        
        if rl_method == "dpo":
            checkpoint_path = train_dpo(training_cfg, dataset_cfg, rl_dir, generation_cfg["model_name"])
        else:
            raise ValueError(f"Unknown rl_method: {rl_method}")
    else:
        print("[skip] RL training; reusing latest checkpoint")
        checkpoint_path = find_latest_checkpoint(checkpoint_root)
    print(f"[i] Latest training checkpoint (for reporting): {checkpoint_path}")

    # Post-training evaluation
    after_eval_path = test_split_path / "evaluations_after.json"
    if stage_enabled("test_after"):
        print("[stage] Post-training evaluation (test split)")
        after_pass = post_training_eval_on_test(config)
    else:
        print("[skip] Post-training evaluation (test split)")
        after_pass = read_pass_at_k(after_eval_path)
        if not after_pass:
            print(f"[warn] Post-training results not found at {after_eval_path}")
    print(f"[i] Post-training pass@k: {after_pass}")

    all_keys = sorted(set(baseline_pass.keys()) | set(after_pass.keys()))
    delta_preview = {k: after_pass.get(k, 0.0) - baseline_pass.get(k, 0.0) for k in all_keys}
    print(f"[i] Delta pass@k: {delta_preview}")

    report_paths = write_report(config, baseline_pass, after_pass, checkpoint_path)
    json_report = report_paths.get("json")
    print(f"[ok] Pipeline complete. Summary written to {json_report}")


if __name__ == "__main__":
    main()
