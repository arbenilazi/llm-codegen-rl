import multiprocessing as mp
import os
import json
import sys
from concurrent.futures import ProcessPoolExecutor
import tempfile
import subprocess
import uuid
import textwrap
try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:
    class BrokenProcessPool(Exception):
        pass

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import get_dataset  # noqa: E402
from metrics import ast_key, measure_execution_time, novelty_score, score_simple  # noqa: E402
from sanitize import normalize_entry_point  # noqa: E402


def execute_wrapped_code(wrapped_code: str, timeout: float = 2.0) -> bool:
    """
    Execute wrapped candidate code (test + solution + check(entry))
    in an isolated subprocess.
    Returns True if exit code == 0, else False.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / f"test_{uuid.uuid4().hex}.py"
            temp_path.write_text(wrapped_code, encoding="utf-8")

            # Ensure subprocess can import local `src/` modules by extending PYTHONPATH
            env = os.environ.copy()
            existing_py = env.get("PYTHONPATH", "")
            repo_src = str(SRC_DIR)
            env["PYTHONPATH"] = repo_src + (":" + existing_py if existing_py else "")

            proc = subprocess.run(
                ["python", str(temp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
                env=env,
            )

            return proc.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def _extract_pass1_value(result: Dict) -> float:
    for key in (1, "1", "pass@1", "pass_at_1"):
        if key in result:
            try:
                return float(result[key])
            except Exception:
                continue
    for key, value in result.items():
        if "1" in str(key):
            try:
                return float(value)
            except Exception:
                continue
    return 0.0


def _compute_pass_mask_strict(_, test: str, candidates: List[str]) -> List[bool]:
    """
    Real execution-based correctness:
    For each candidate, build the wrapped code and run in a subprocess.
    """
    results = []
    # Print only once per task
    printed_preview = False

    for candidate in candidates:
        wrapped_code = candidate

        if not printed_preview:
            print("\n[debug] Annotate task preview")
            print("[debug] Test snippet")
            print(test[:180])
            print("[debug] Wrapped code snippet")
            print(wrapped_code[:220])
            printed_preview = True

        # Run oracle
        passed = execute_wrapped_code(wrapped_code)

        if printed_preview:  # print result of first candidate only
            print(f"[debug] First candidate passed={passed}")
            print("[debug] End preview\n")
            printed_preview = False

        results.append(bool(passed))
    return results



def _normalize_eval_output(result):
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    if isinstance(result, np.ndarray):
        if result.size == 1:
            return result.item()
        return result
    return result


def _metric_payload(code: str) -> Dict[str, float]:
    loc, tokens, cc, depth = score_simple(code)
    return {
        "loc": loc,
        "tokens": tokens,
        "cyclomatic_complexity": cc,
        "nesting_depth": depth,
        "ast_hash": ast_key(code),
    }


def _annotate_worker(args):
    tests, preds, start_index = args
    results = []
    for offset, (test, candidates) in enumerate(zip(tests, preds)):
        # preds already contain fully wrapped code strings
        mask = _compute_pass_mask_strict(None, test, candidates)
        results.append((start_index + offset, mask))
    return results, True


def _compute_pass_masks_parallel(
    references: Sequence[str],
    predictions: Sequence[Sequence[str]],
    num_procs: int,
) -> Tuple[Dict[int, List[bool]], bool]:
    passed_mask: Dict[int, List[bool]] = defaultdict(list)
    strict_used = True

    if num_procs <= 1:
        for idx, (test, preds) in enumerate(zip(references, predictions)):
            passed_mask[idx] = _compute_pass_mask_strict(None, test, preds)
        return passed_mask, strict_used

    chunk_size = max(1, len(references) // num_procs)
    batches = []
    for start in range(0, len(references), chunk_size):
        end = min(start + chunk_size, len(references))
        batch_refs = references[start:end]
        batch_preds = [predictions[i] for i in range(start, end)]
        batches.append((batch_refs, batch_preds, start))

    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(max_workers=num_procs, mp_context=ctx) as executor:
            for batch_result, _ in executor.map(_annotate_worker, batches):
                for idx, mask in batch_result:
                    passed_mask[idx] = [_normalize_eval_output(v) for v in mask]
    except (BrokenProcessPool, AttributeError, RuntimeError) as exc:
        print(f"[warn] Parallel annotation failed ({exc}); falling back to serial evaluation.")
        return _compute_pass_masks_parallel(references, predictions, 1)

    return passed_mask, strict_used


def annotate_candidates(cfg: Dict, split: str, in_path: str, out_path: str) -> str:
    dataset_cfg = cfg["dataset"]
    dataset_name = dataset_cfg["name"]
    dataset_path = (REPO_ROOT / dataset_cfg["output_dir"]).resolve()

    ds = get_dataset(dataset_name, str(dataset_path))
    if split not in ds.dataset:
        raise ValueError(f"Split '{split}' not found in dataset at {dataset_path}")

    hf_split = ds.dataset[split]
    id_order = list(hf_split["id"])
    id2entry = {ex_id: entry for ex_id, entry in zip(hf_split["id"], hf_split["entry_point"])}
    raw_tests = hf_split["test"]
    references = [
        t["test"] if isinstance(t, dict) else t
        for t in raw_tests
    ]
    task_id_to_test = {tid: test for tid, test in zip(id_order, references)}

    records_by_id: Dict[int, List[Dict]] = defaultdict(list)
    eval_codes_by_id: Dict[int, List[str]] = defaultdict(list)

    in_file = Path(in_path)
    if not in_file.exists():
        raise FileNotFoundError(f"Missing candidates file: {in_file}")

    with in_file.open("r", encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            task_id = data.get("id")
            if task_id not in id2entry:
                continue

            code = data.get("code") or ""
            metrics = _metric_payload(code)

            records_by_id[task_id].append(
                {
                    "candidate_idx": data.get("candidate_idx"),
                    "code": code,
                    "metrics": metrics,
                }
            )
            clean_code = normalize_entry_point(code, id2entry[task_id])

            # Wrap into an executable program
            entry = id2entry[task_id]
            raw_test = task_id_to_test[task_id]

            # Normalize trailing artifacts in test code before execution.

            # Strip trailing whitespace.
            t = (raw_test or "").rstrip()

            # Remove dangling underscore suffixes.
            while t.endswith("_"):
                t = t[:-1].rstrip()

            # Ensure trailing newline.
            test = t + "\n"

            # Now build the wrapped executable program
            wrapped = f"{test}\n\n{clean_code}\n\ncheck({entry})"

            eval_codes_by_id[task_id].append(wrapped)


    env_procs = os.getenv("ANNOTATE_NUM_PROCS")
    num_procs = int(env_procs) if env_procs else max(1, os.cpu_count() or 1)

    predictions = [
        list(eval_codes_by_id.get(task_id, []))
        for task_id in id_order
    ]
    passed_mask_raw, strict_used = _compute_pass_masks_parallel(references, predictions, num_procs)

    passed_mask: Dict[int, List[bool]] = defaultdict(list)
    for idx, task_id in enumerate(id_order):
        mask = passed_mask_raw.get(idx, [])
        mask = [_normalize_eval_output(m) for m in mask]
        if any(not isinstance(m, bool) for m in mask):
            mask = [bool(m) for m in mask]
        passed_mask[task_id] = mask

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"[debug] Annotating {len(id_order)} tasks")
    for _task_id in id_order[:5]:
        print(f"[debug] sample task {_task_id}: {len(eval_codes_by_id.get(_task_id, []))} candidates")

    # Compute execution time for each candidate
    for task_id in id_order:
        records = records_by_id.get(task_id, [])
        wrapped_codes = eval_codes_by_id.get(task_id, [])

        for i, rec in enumerate(records):
            wrapped = wrapped_codes[i] if i < len(wrapped_codes) else None
            if wrapped is None:
                rec["metrics"]["execution_time"] = 10.0
            else:
                exec_time = measure_execution_time(wrapped, timeout=1.0)
                rec["metrics"]["execution_time"] = float(exec_time)

    # Compute novelty per task
    for task_id in id_order:
        records = records_by_id.get(task_id, [])

        # Compute novelty for each candidate within this task
        codes = [rec["code"] for rec in records]
        for idx, rec in enumerate(records):
            other_codes = [c for j, c in enumerate(codes) if j != idx]
            rec["metrics"]["novelty"] = float(novelty_score(rec["code"], other_codes))

    # Ensure all required metrics are present before writing
    required = ("loc", "tokens", "cyclomatic_complexity", "nesting_depth")
    for task_id in id_order:
        records = records_by_id.get(task_id, [])
        wrapped_codes = eval_codes_by_id.get(task_id, [])
        codes = [rec["code"] for rec in records]

        for idx, rec in enumerate(records):
            metrics = rec.setdefault("metrics", {})
            loc, tokens, cc, depth = score_simple(rec["code"])
            for key, val in zip(required, (loc, tokens, cc, depth)):
                metrics.setdefault(key, val)
            metrics.setdefault("ast_hash", ast_key(rec["code"]))

            if "execution_time" not in metrics:
                wrapped = wrapped_codes[idx] if idx < len(wrapped_codes) else None
                metrics["execution_time"] = float(
                    measure_execution_time(wrapped, timeout=1.0) if wrapped is not None else 10.0
                )

            if "novelty" not in metrics:
                other = [c for j, c in enumerate(codes) if j != idx]
                metrics["novelty"] = float(novelty_score(rec["code"], other))

    with out_file.open("w", encoding="utf-8") as handle:
        for task_id in id_order:
            records = records_by_id.get(task_id, [])
            mask = passed_mask.get(task_id, [])
            entry = id2entry.get(task_id)
            for idx, record in enumerate(records):
                candidate_idx = record["candidate_idx"]
                if candidate_idx is None:
                    candidate_idx = idx

                passed = bool(mask[idx]) if idx < len(mask) else False
                payload = {
                    "id": task_id,
                    "candidate_idx": candidate_idx,
                    "code": record["code"],
                    "metrics": record["metrics"],
                    "passed": passed,
                    "entry_point": entry,
                }
                handle.write(json.dumps(payload) + "\n")

    return str(out_file)


__all__ = ["annotate_candidates"]
