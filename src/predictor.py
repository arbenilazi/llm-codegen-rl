import argparse
import json
import multiprocessing as mp
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
try:
    from concurrent.futures.process import BrokenProcessPool
except Exception:
    class BrokenProcessPool(Exception): 
        pass
from math import comb
from pathlib import Path
from textwrap import dedent
from types import MethodType
from typing import Dict, List, Optional, Tuple

import datetime
import random
import logging
import sys

import torch.multiprocessing as _mp

try:
    _mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import pandas as pd

from datasets import Dataset
from tqdm import tqdm

from dataset import get_dataset, prepare_dataset
from model import Model
from prompt_utils import (
    canonical_signature,
    synthesize_problem_description,
    build_difficulty_prompt,
    build_strategy_prompt,
    VARIANTS,
)
from sanitize import normalize_entry_point, _syntactic_ok
from pipeline.annotate import execute_wrapped_code

EVAL_PROGRESS_LOG_ENV = "EVAL_PROGRESS_LOG"
EVAL_PROGRESS_JSON_ENV = "EVAL_PROGRESS_JSON"
PROMPT_MODE_ENV = "PROMPT_MODE"
PROMPT_SHOTS_ENV = "PROMPT_SHOTS"
PROMPT_MAX_ASSERTS_ENV = "PROMPT_MAX_ASSERTS"

NUMBER_OF_SOLUTIONS = 10
SYSTEM_PROMPT_TEMPLATE = """
You are an expert Python programmer.

Your objective is to produce a fully correct Python solution that passes all unit tests.

Principles:
- Correctness is always the top priority.
- Prefer clear, direct, and reliable solutions.
- Use only the Python standard library.
- One Python code block containing one function.
- No comments, no prints, no main guard, no explanation.

About diversity:
- If multiple natural algorithms exist, choose one standard algorithm 
  that fits the requested variant (e.g., hashing, sorting, recursion, DP).
- If the problem has only ONE reasonable algorithm, vary only structure.
- Never create artificial diversity through renaming, reformatting, or
  style changes that keep the same algorithm without justification.
- All diversity must come from **meaningful algorithmic or structural choices**.

About structural or algorithmic styles (variants):
- If a variant is provided, follow it only when it naturally fits a correct solution.
- Never force a variant if it reduces correctness or makes the solution unnatural.
- If multiple correct algorithms exist, choose one that aligns with the variant.
- Do not force techniques that harm clarity or correctness.

Focus on: correctness -> clarity -> robustness -> meaningful diversity.
""".strip()


SYSTEM_PROMPT = SYSTEM_PROMPT_TEMPLATE


def _candidate_config_paths(dataset_name: str) -> List[Path]:
    candidates: List[Path] = []
    cfg_env = os.getenv("PREDICTOR_CONFIG_PATH")
    if cfg_env:
        candidates.append(Path(cfg_env))

    repo_root = Path(__file__).resolve().parents[1]
    configs_dir = repo_root / "configs"

    name_variants = set()
    if dataset_name:
        raw = str(dataset_name)
        name_variants.add(raw)
        name_variants.add(raw.lower())
        sanitized = raw.replace("/", "_").replace("\\", "_")
        name_variants.add(sanitized)
        name_variants.add(sanitized.lower())
    for variant in name_variants:
        candidates.append(configs_dir / f"{variant}.json")
    return candidates


_CONFIG_CACHE: Dict[str, Optional[Dict[str, object]]] = {}


def _load_dataset_config(dataset_name: str, dataset_path: str) -> Optional[Dict[str, object]]:
    cache_key = (dataset_name or "").lower()
    if cache_key in _CONFIG_CACHE:
        return _CONFIG_CACHE[cache_key]

    candidates = _candidate_config_paths(dataset_name)
    seen: set = set()
    for path in candidates:
        if path is None:
            continue
        try:
            resolved = path.resolve()
        except Exception:
            continue
        if resolved in seen or not resolved.exists():
            continue
        seen.add(resolved)
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                cfg = json.load(handle)
        except Exception as exc:
            print(f"[warn] Failed to load config from {resolved}: {exc}")
            continue
        _CONFIG_CACHE[cache_key] = cfg
        return cfg

    _CONFIG_CACHE[cache_key] = None
    return None


def _load_generation_config(dataset_name: str, dataset_path: str) -> Optional[Dict[str, object]]:
    cfg = _load_dataset_config(dataset_name, dataset_path)
    if isinstance(cfg, dict):
        generation = cfg.get("generation")
        if isinstance(generation, dict):
            return generation
    return None

def detect_strategy(code: str, entry_point: Optional[str] = None) -> str:
    """Derives a short, human-readable hint describing the algorithmic idea."""
    snippet = (code or "").strip()
    if not snippet:
        return "direct iterative computation"
    lowered = snippet.lower()

    def _has(pattern: str) -> bool:
        return re.search(pattern, snippet, re.MULTILINE | re.DOTALL) is not None

    def _contains_all(*tokens: str) -> bool:
        return all(tok in lowered for tok in tokens)

    if "lru_cache" in lowered or "@cache" in lowered or "@lru_cache" in lowered or "memo" in lowered:
        return "memoized recursion"

    target = entry_point
    if not target:
        match = re.search(r"\bdef\s+([a-zA-Z_]\w*)\s*\(", snippet)
        if match:
            target = match.group(1)
    if target:
        match = re.search(rf"\bdef\s+{re.escape(target)}\s*\(", snippet)
        if match:
            body = snippet[match.end():]
            if re.search(rf"\b{re.escape(target)}\s*\(", body):
                return "recursive decomposition"

    if "heapq" in lowered:
        return "heap-based selection"

    if "collections.deque" in lowered or "deque(" in lowered or "popleft" in lowered:
        return "deque-driven bfs/queue"

    if "counter" in lowered:
        return "frequency counting with Counter"

    if "set(" in lowered or ".add(" in lowered and "set" in lowered:
        return "hash-set membership pruning"

    if "dict" in lowered and "update" in lowered:
        return "dictionary aggregation"

    if "sorted(" in lowered or ".sort(" in lowered:
        if _contains_all("left", "right") or _has(r"while\s+[a-z_]+\s*<\s*[a-z_]+"):
            return "sorting + two-pointer sweep"
        return "sorting + linear scan"

    if (
        ("while" in lowered and "mid" in lowered and "// 2" in lowered)
        or _contains_all("left", "right", "mid")
    ):
        return "binary search"

    if _contains_all("left", "right") and _has(r"(while|for).*\b(left|l)\s*<\s*(right|r)"):
        return "two-pointer sweep"

    if "dp" in lowered and re.search(r"\bdp\s*=\s*\[", lowered):
        return "tabular dynamic programming"

    if "dp" in lowered and ("cache" in lowered or "memo" in lowered):
        return "memoized dynamic programming"

    if "for" in lowered and "range" in lowered and "stack" in lowered:
        return "stack-based iteration"

    if (
        "math." in lowered
        or "pow(" in lowered
        or "**" in snippet
        or "sqrt" in lowered
        or "log" in lowered
    ):
        return "mathematical derivation"

    if "while" in lowered:
        return "iterative while-loop refinement"

    return "direct iterative computation"

def _extract_code_for_entry(text: str, entry: str) -> str:
    if not text:
        return ""
    source = str(text).strip()

    blocks = re.findall(r"```(?:[a-zA-Z]+\n)?(.*?)```", source, re.DOTALL)
    blocks = [b.strip() for b in blocks]

    if not blocks:
        m = re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", source)
        if m:
            return source[m.start():].replace("```", "").strip()
        return source.replace("```", "").strip()

    def_has_def = [("def " in b) for b in blocks]

    idx_entry = None
    for i, block in enumerate(blocks):
        if re.search(rf"\bdef\s+{re.escape(entry)}\s*\(", block):
            idx_entry = i
            break

    if idx_entry is not None:
        helpers = [blocks[i] for i in range(idx_entry) if def_has_def[i]]
        merged = ("\n\n".join(helpers + [blocks[idx_entry]])).strip()
        return merged

    candidates = [(i, b) for i, b in enumerate(blocks) if "def " in b]
    if candidates:
        i_best, best = max(candidates, key=lambda t: len(t[1]))
        helpers = [blocks[i] for i in range(i_best) if def_has_def[i]]
        merged = ("\n\n".join(helpers + [best])).strip()
        return merged

    return ("\n\n".join(blocks)).strip()


def _concat_all_code_blocks(text: str) -> str:
    if not text:
        return ""
    source = str(text).strip()
    blocks = re.findall(r"```(?:[a-zA-Z]+\n)?(.*?)```", source, re.DOTALL)
    blocks = [b.strip() for b in blocks if b.strip()]
    if blocks:
        return ("\n\n".join(blocks)).strip()
    return source.replace("```", "").strip()


def _tests_for_prompt(test_code: str, entry: str) -> str:
    """
    Convert our wrapped `check(candidate)` into plain asserts using the real entry name,
    and cap how many assert lines we expose to keep prompts compact.
    """
    try:
        max_asserts = int(os.getenv(PROMPT_MAX_ASSERTS_ENV, "6"))
    except ValueError:
        max_asserts = 6
    if max_asserts < 0:
        max_asserts = 0
    if max_asserts == 0:
        return ""
    out: List[str] = []
    count = 0
    for line in (test_code or "").splitlines():
        line = line.strip()
        if not line or line.startswith("def check("):
            continue
        m = re.match(r"assert\s+candidate\((.*)\)\s*==\s*(.*)$", line)
        if m:
            out.append(f"assert {entry}({m.group(1)}) == {m.group(2)}")
        else:
            line = re.sub(r"\bcandidate\s*\(", f"{entry}(", line)
            if line.startswith("assert "):
                out.append(line)
        if out and out[-1].startswith("assert "):
            count += 1
        if count >= max_asserts:
            break
    return "\n".join(out)


def _exemplar_block(example: dict) -> str:
    entry = example.get("entry_point", "solution")
    tests_for_prompt = _tests_for_prompt(example.get("test", ""), entry)
    return dedent(
        f"""
        ### Example (format)
        {example['input']}

        Requirements:
        - Implement a single function `{entry}`.
        - No top-level code (no prints, no input(), no main guard).
        - Put the solution in one Python code block (```python ... ```).

        Tests to satisfy:
        ```python
        {tests_for_prompt}
        ```
        """
    ).strip()


def set_number_of_solutions(n: int) -> None:
    global NUMBER_OF_SOLUTIONS
    if NUMBER_OF_SOLUTIONS != n:
        NUMBER_OF_SOLUTIONS = n


def _pass1_value(result: dict) -> float:
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


def _sanitize_test_for_exec(t) -> str:
    if isinstance(t, dict):
        t = t.get("test") or t.get("text") or ""
    t = (t or "").rstrip()
    while t.endswith("_"):
        t = t[:-1].rstrip()
    return t + "\n"


def _compute_pass_mask_exec(test: str, entry: str, candidates: List[str]) -> List[bool]:
    mask: List[bool] = []
    cleaned_test = _sanitize_test_for_exec(test)
    entry_point = entry or "solution"
    for candidate in candidates:
        wrapped = f"{cleaned_test}\n\n{candidate or ''}\n\ncheck({entry_point})"
        mask.append(bool(execute_wrapped_code(wrapped, timeout=2.0)))
    return mask


def _estimate_pass_at_k(total: int, correct: int, k: int) -> float:
    if total < k or total == 0:
        return 0.0
    remaining = total - correct
    if remaining < k:
        return 1.0
    return 1.0 - comb(remaining, k) / comb(total, k)


def _aggregate_pass_at_k(passed_masks: List[List[bool]], ks: List[int]) -> dict:
    results = {}
    totals = [len(mask) for mask in passed_masks]
    corrects = [sum(mask) for mask in passed_masks]
    for k in ks:
        acc = 0.0
        count = 0
        for total, correct in zip(totals, corrects):
            if total < k or total == 0:
                continue
            acc += _estimate_pass_at_k(total, correct, k)
            count += 1
        results[int(k)] = float(acc / count) if count else 0.0
    return results


def _normalize_pass_dict(values: dict) -> dict:
    normalized = {}
    for key, val in values.items():
        try:
            normalized[int(key)] = float(val)
        except Exception:
            normalized[key] = float(val)
    return normalized


def _compute_pass_at_k_serial(tests, predictions, entries, ks) -> dict:
    passed_masks: List[List[bool]] = []
    for test, preds, entry in zip(tests, predictions, entries):
        passed_masks.append(_compute_pass_mask_exec(test, entry, preds))
    res = _aggregate_pass_at_k(passed_masks, ks)
    return _normalize_pass_dict(res)


def _format_eta(seconds: float) -> str:
    if seconds is None or seconds <= 0:
        return "0s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return "".join(parts)


class _EvalProgressMonitor:
    def __init__(
        self,
        *,
        stage: str,
        mode: str,
        ks: List[int],
        total_tasks: int,
        total_batches: int,
        num_procs: int,
        batch_size: int,
        n_test: int,
        text_path: Optional[str],
        json_path: Optional[str],
    ) -> None:
        self.stage = stage
        self.mode = mode
        self.ks = ks
        self.total_tasks = total_tasks
        self.total_batches = total_batches
        self.num_procs = num_procs
        self.batch_size = batch_size
        self.n_test = n_test
        self.completed_tasks = 0
        self.batch_count = 0
        self.batch_time_sum = 0.0
        self.start_time = time.monotonic()
        self.last_time = self.start_time
        self.logger: Optional[logging.Logger] = None
        self.handler: Optional[logging.Handler] = None
        self.json_file = None
        self.text_path = None
        self.json_path = None
        if text_path:
            try:
                resolved = Path(text_path).resolve()
            except Exception:
                resolved = Path(text_path)
            self.text_path = str(resolved)
            logger = logging.Logger(f"eval_progress_{id(self)}")
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = logging.FileHandler(resolved, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            self.logger = logger
            self.handler = handler
        if json_path:
            try:
                resolved_json = Path(json_path).resolve()
            except Exception:
                resolved_json = Path(json_path)
            resolved_json.parent.mkdir(parents=True, exist_ok=True)
            self.json_path = str(resolved_json)
            self.json_file = resolved_json.open("a", encoding="utf-8", buffering=1)
        self.fsync = os.getenv("EVAL_PROGRESS_FSYNC") == "1"

    def _timestamp(self) -> str:
        return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()

    def _log(self, message: str, payload: Optional[dict]) -> None:
        if self.logger:
            self.logger.info(message)
            if self.handler:
                self.handler.flush()
        print(message)
        sys.stdout.flush()
        if payload is not None and self.json_file:
            record = {
                "timestamp": self._timestamp(),
                "stage": self.stage,
                "mode": payload.get("mode", self.mode),
            }
            record.update(payload)
            self.json_file.write(json.dumps(record) + "\n")
            self.json_file.flush()
            if self.fsync:
                try:
                    os.fsync(self.json_file.fileno())
                except OSError:
                    pass

    def log_start(self) -> None:
        message = (
            f"[eval][test] start mode={self.mode} ks={self.ks} n_test={self.n_test} "
            f"num_procs={self.num_procs} batch_size={self.batch_size} "
            f"log={self.text_path or '-'} json={self.json_path or '-'}"
        )
        payload = {
            "event": "start",
            "mode": self.mode,
            "total_batches": self.total_batches,
            "total_tasks": self.total_tasks,
            "ks": self.ks,
            "num_procs": self.num_procs,
            "batch_size": self.batch_size,
            "n_test": self.n_test,
            "log_path": self.text_path,
            "json_path": self.json_path,
        }
        self._log(message, payload)

    def log_batch(self, batch_idx: int, tasks_increment: int, dt_override: Optional[float] = None) -> None:
        self.completed_tasks = min(self.total_tasks, self.completed_tasks + tasks_increment)
        now = time.monotonic()
        if dt_override is not None:
            dt = dt_override
            self.last_time += dt_override
        else:
            dt = now - self.last_time
            self.last_time = now
        self.batch_count += 1
        self.batch_time_sum += dt
        avg_dt = self.batch_time_sum / self.batch_count if self.batch_count else 0.0
        remaining_batches = max(0, self.total_batches - batch_idx)
        eta = avg_dt * remaining_batches
        message = (
            f"[eval][test] batch {batch_idx}/{self.total_batches} "
            f"completed={self.completed_tasks}/{self.total_tasks} ks={self.ks} "
            f"dt={dt:.1f}s avg={avg_dt:.1f}s ETA={_format_eta(eta)}"
        )
        payload = {
            "event": "batch",
            "mode": self.mode,
            "batch_idx": batch_idx,
            "total_batches": self.total_batches,
            "completed_tasks": self.completed_tasks,
            "total_tasks": self.total_tasks,
            "ks": self.ks,
            "dt_sec": dt,
            "avg_dt_sec": avg_dt,
            "eta_sec": max(0.0, eta),
        }
        self._log(message, payload)

    def log_fallback(self, reason: str) -> None:
        message = f"[eval][test] fallback to serial ({reason})"
        payload = {"event": "fallback", "mode": self.mode, "reason": str(reason)}
        self._log(message, payload)

    def log_done(
        self,
        total_time: float,
        compute_time: float,
        aggregate_time: float,
        pass_at_k: dict,
        mode_override: Optional[str] = None,
    ) -> None:
        mode = mode_override or self.mode
        message = (
            f"[eval][test] done in {total_time:.2f}s (compute={compute_time:.2f}s, "
            f"aggregate={aggregate_time:.2f}s) pass@k={pass_at_k}"
        )
        payload = {
            "event": "done",
            "mode": mode,
            "total_sec": total_time,
            "compute_sec": compute_time,
            "aggregate_sec": aggregate_time,
            "pass_at_k": pass_at_k,
        }
        self._log(message, payload)

    def close(self) -> None:
        if self.handler and self.logger:
            self.logger.removeHandler(self.handler)
            self.handler.close()
        if self.json_file:
            self.json_file.close()

def _eval_worker(args):
    batch_tests, batch_preds, batch_entries, _ = args
    passed = []
    for test, preds, entry in zip(batch_tests, batch_preds, batch_entries):
        passed.append(_compute_pass_mask_exec(test, entry, preds))
    return passed


def _compute_pass_at_k_parallel(
    tests,
    predictions,
    entries,
    ks,
    num_procs: int,
    batch_size: int,
    progress_paths: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> dict:
    tasks = len(tests)
    log_path, json_path = progress_paths or (None, None)
    batches = []
    for start in range(0, tasks, batch_size):
        end = min(start + batch_size, tasks)
        batch_tests = list(tests[start:end])
        batch_preds = [list(predictions[idx]) for idx in range(start, end)]
        batch_entries = list(entries[start:end])
        batches.append((batch_tests, batch_preds, batch_entries, ks))

    total_batches = max(1, len(batches))
    monitor = _EvalProgressMonitor(
        stage="eval_test",
        mode="parallel",
        ks=list(ks),
        total_tasks=tasks,
        total_batches=total_batches,
        num_procs=num_procs,
        batch_size=batch_size,
        n_test=tasks,
        text_path=log_path,
        json_path=json_path,
    )

    try:
        monitor.log_start()
        passed_masks: List[List[bool]] = []
        try:
            ctx = mp.get_context("spawn")
            exec_start = time.monotonic()
            with ProcessPoolExecutor(max_workers=num_procs, mp_context=ctx) as executor:
                for idx, batch_passed in enumerate(executor.map(_eval_worker, batches), start=1):
                    passed_masks.extend(batch_passed)
                    batch_tasks = len(batch_passed)
                    monitor.log_batch(idx, batch_tasks)
            exec_time = time.monotonic() - exec_start
        except (BrokenProcessPool, AttributeError, RuntimeError) as exc:
            monitor.log_fallback(str(exc))
            serial_start = time.monotonic()
            serial_res = _compute_pass_at_k_serial(tests, predictions, entries, ks)
            serial_time = time.monotonic() - serial_start
            total_time = time.monotonic() - monitor.start_time
            monitor.completed_tasks = monitor.total_tasks
            monitor.log_done(total_time, serial_time, 0.0, serial_res, mode_override="serial")
            return serial_res

        agg_start = time.monotonic()
        result = _aggregate_pass_at_k(passed_masks, ks)
        agg_time = time.monotonic() - agg_start
        total_time = time.monotonic() - monitor.start_time
        monitor.log_done(total_time, exec_time, agg_time, result)
    finally:
        monitor.close()

    if os.getenv("EVAL_DEBUG") == "1":
        serial = _compute_pass_at_k_serial(tests, predictions, entries, ks)
        for key in result:
            if abs(result[key] - serial.get(key, result[key])) > 1e-9:
                print(
                    f"[warn] parallel vs serial mismatch for k={key}: parallel={result[key]:.6f}, serial={serial.get(key, 0.0):.6f}"
                )
    return result


def _resolve_eval_num_procs() -> int:
    cfg_val = os.getenv("EVAL_NUM_PROCS")
    if cfg_val is not None:
        try:
            return max(1, int(cfg_val))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def _resolve_eval_batch_size() -> int:
    cfg_val = os.getenv("EVAL_BATCH_SIZE")
    if cfg_val is not None:
        try:
            size = int(cfg_val)
            return size if size > 0 else 32
        except ValueError:
            pass
    return 32


def _resolve_eval_ks() -> List[int]:
    env_val = os.getenv("EVAL_KS")
    if env_val:
        try:
            parsed = json.loads(env_val)
            if isinstance(parsed, (list, tuple)):
                return [int(x) for x in parsed]
        except Exception:
            try:
                return [int(x.strip()) for x in env_val.split(",") if x.strip()]
            except Exception:
                pass
    return [1, 5, 10]



def run_predictions(
    model_name: str,
    dataset_name: str,
    dataset_path: str,
    split: Optional[str] = None,
    num_samples: Optional[int] = None,
    batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[str] = None,
    adapter_path: Optional[str] = None,
    tag: Optional[str] = None,
):
    print(f" Loading dataset: {dataset_name} from {dataset_path}")
    # NOTE: Splits are materialized on disk ahead of time by run.py's ensure_required_splits().
    # predictor.py never re-splits; it just reads the requested split directory ('train' or 'test').
    dataset = get_dataset(dataset_name, dataset_path)
    if dataset is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if split:
        dataset._default_split = MethodType(lambda self: split, dataset)  # type: ignore[assignment]
        target_split = split
    else:
        target_split = dataset._default_split()

    hf_split = dataset.dataset[target_split]

    generation_cfg = _load_generation_config(dataset_name, dataset_path) or {}

    generation_dtype = None
    temperature_cfg = None
    top_p_cfg = None
    num_candidates_cfg = None
    if isinstance(generation_cfg, dict):
        dtype_val = generation_cfg.get("dtype")
        if dtype_val is not None:
            generation_dtype = str(dtype_val)
        temperature_cfg = generation_cfg.get("temperature")
        top_p_cfg = generation_cfg.get("top_p")
        num_candidates_cfg = generation_cfg.get("num_candidates")

    if num_candidates_cfg is None:
        num_candidates_cfg = NUMBER_OF_SOLUTIONS

    try:
        num_candidates_cfg = int(num_candidates_cfg)
    except (TypeError, ValueError):
        print(f"[warn] Invalid generation.num_candidates={num_candidates_cfg}; defaulting to {NUMBER_OF_SOLUTIONS}")
        num_candidates_cfg = NUMBER_OF_SOLUTIONS

    samples = num_samples if num_samples is not None else max(1, num_candidates_cfg)

    set_number_of_solutions(samples)

    # Prompting mode (0/1/few-shot) from env (default: zero).
    prompt_mode = os.getenv(PROMPT_MODE_ENV, "zero").strip().lower()
    try:
        shots_k = int(os.getenv(PROMPT_SHOTS_ENV, "0"))
    except ValueError:
        shots_k = 0
    if prompt_mode == "one":
        shots_k = max(1, shots_k or 1)
    elif prompt_mode == "few":
        shots_k = max(1, shots_k)
    else:
        shots_k = 0

    rng = random.Random(42)
    indices = list(range(len(hf_split)))

    batch_size = int(batch_size or 32)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    output_root = Path(dataset_path).resolve()
    split_dir = output_root / target_split
    split_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""

    temp_override = os.getenv("GEN_TEMPERATURE")
    top_p_override = os.getenv("GEN_TOP_P")
    max_tokens_override = os.getenv("GEN_MAX_TOKENS")
    stop_override = os.getenv("GEN_STOP")

    resolved_temperature = (
        float(temp_override)
        if temp_override is not None
        else float(temperature_cfg) if temperature_cfg is not None
        else float(temperature) if temperature is not None
        else 0.9
    )
    resolved_top_p = (
        float(top_p_override)
        if top_p_override is not None
        else float(top_p_cfg) if top_p_cfg is not None
        else float(top_p) if top_p is not None
        else 0.95
    )
    resolved_max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(max_tokens) if max_tokens is not None
        else 512
    )
    resolved_stop = (
        stop_override
        if stop_override is not None
        else stop if stop is not None
        else "<END>"
    )

    print(f" Initializing model: {model_name}")
    model = Model(
        model_name=model_name,
        max_tokens=resolved_max_tokens,
        temperature=resolved_temperature,
        top_p=resolved_top_p,
        stop=resolved_stop,
        n=samples,
        dtype=generation_dtype,
        adapter_path=adapter_path,
    )

    effective_batch_size = batch_size
    if getattr(model, "backend", None) == "hf":
        try:
            fallback_bs = int(os.getenv("PREDICTOR_FALLBACK_BATCH_SIZE", "1"))
        except ValueError:
            fallback_bs = 1
            print("[predictor] Invalid PREDICTOR_FALLBACK_BATCH_SIZE provided; defaulting to 1")
        fallback_bs = max(1, fallback_bs)
        if fallback_bs != batch_size:
            print(f"[predictor] HF fallback active: overriding batch_size -> {fallback_bs}")
        effective_batch_size = fallback_bs

    print(f" Generating predictions on split '{target_split}' with n={samples} (batch={effective_batch_size})...")
    results = []

    progress = tqdm(total=len(hf_split), desc=f"Predicting[{target_split}]")

    for example in hf_split:
        entry = example.get("entry_point", "solution") or "solution"
        tests_for_prompt = _tests_for_prompt(example.get("test", ""), entry)
        description = synthesize_problem_description(example)
        full_problem_text = f"{example.get('input', '')}\n\n{example.get('test', '')}"
        difficulty = "unknown"
        try:
            diff_prompt = build_difficulty_prompt(full_problem_text)  
            diff_result = model.generate(diff_prompt, n_override=1)
            if diff_result:
                val = (diff_result[0] or "").strip().lower().replace(".", "")
                if val in ("easy", "medium", "hard"):
                    difficulty = val
        except Exception as e:
            print(f"[warn] Difficulty classification failed ({e}); defaulting to unknown.")

        # Optional exemplar preface
        exemplar_preface = ""
        if shots_k > 0:
            cur_id = example.get("id")
            pool = [i for i in indices if hf_split[i].get("id") != cur_id]
            rng.shuffle(pool)
            chosen = pool[:shots_k]
            exemplars = [_exemplar_block(hf_split[i]) for i in chosen]
            exemplar_preface = (
                "You will see a few format examples.\n"
                + "\n\n".join(exemplars)
                + "\n\n---\n\n"
            )
        # Try to generate structural variants from the model.
        variants = []
        try:
            strat_prompt_text = build_strategy_prompt(full_problem_text)
            strat_prompt = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": strat_prompt_text},
            ]
            strat_outputs = []
            if strat_prompt_text and strat_prompt_text.strip():
                try:
                    strat_outputs = model.generate(strat_prompt, n_override=1)
                except Exception as e:
                    print("[warn] Strategy generation failed:", e)
            else:
                variants = VARIANTS[:]

            if strat_outputs:
                for line in strat_outputs[0].split("\n"):
                    cleaned = line.strip().lstrip("-• ").strip()

                    # Reject empty, symbols, code fences, headings, noise
                    if not cleaned:
                        continue
                    if cleaned in {"```", "``", "`"}:
                        continue
                    if cleaned.startswith("```") or cleaned.endswith("```"):
                        continue
                    if cleaned.lower().startswith("example"):
                        continue
                    if cleaned.lower().startswith("python"):
                        continue
                    if re.fullmatch(r"[#*>\-_=]+", cleaned):
                        continue
                    if len(cleaned) < 3:  # avoid noise like ";", "{}"
                        continue

                    variants.append(cleaned)

        except Exception as e:
            print("[error] Strategy generation failed:", repr(e))
            raise


        if not variants:
            variants = VARIANTS[:]

        planned_hints: List[str] = []
        planned_origins: List[str] = []
        raw_predictions: List[str] = []
        candidate_batch: List[List[dict]] = []

        def _flush_candidate_batch():
            nonlocal candidate_batch
            if not candidate_batch:
                return

            # Filter empty prompts before batch generation.
            filtered = []
            for p in candidate_batch:
                if (
                    isinstance(p, list)
                    and any(m.get("content", "").strip() for m in p)
                ):
                    filtered.append(p)

            candidate_batch = []
            if not filtered:
                return

            def _generate(msgs: List[List[dict]]) -> List[List[str]]:
                msgs = [
                    m for m in msgs
                    if any(x.get("content", "").strip() for x in m)
                ]
                if not msgs:
                    return []
                try:
                    return model.generate_batch(msgs, n_override=1)
                except AssertionError:
                    if len(msgs) == 1:
                        raise
                    mid = len(msgs) // 2
                    left = _generate(msgs[:mid])
                    right = _generate(msgs[mid:])
                    return left + right

            batch_outputs = _generate(filtered)
            for outs in batch_outputs:
                text = outs[0] if outs else ""
                raw_predictions.append(text)

        # Deterministic assignment: cycle through structural variants.
        rng.shuffle(variants)
        assigned = [variants[i % len(variants)] for i in range(samples)]

        for cand_idx in range(samples):
            variant = assigned[cand_idx]
            planned_hints.append(variant)
            planned_origins.append("variant")

            desc_text = example.get("input")
            if not desc_text:
                raise ValueError("Dataset example is missing the 'input' field.")
            base_prompt = dedent(f"""
                Task Description:
                {desc_text}

                You must implement a correct Python function named `{entry}`.

                Use the following structural or algorithmic style (optional):
                - {variant}

                Guidelines:
                - Use the variant only if it fits naturally with a correct solution.
                - Prioritize correctness above all.
                - Use only the Python standard library.
                - Keep the solution clear and straightforward.
                - Exactly one function, inside one Python code block.
                - No comments, no prints, no main guards.
            """).strip()


            sections: List[str] = [base_prompt]
            if exemplar_preface:
                sections.append(exemplar_preface.strip())
            if tests_for_prompt:
                sections.append(
                    dedent(f"""\
                    Unit tests to satisfy:
                    ```python
                    {tests_for_prompt}
                    ```
                    """).strip()
                )
            user_msg = "\n\n".join(part for part in sections if part)

            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]
            candidate_batch.append(prompt)
            if len(candidate_batch) >= effective_batch_size:
                _flush_candidate_batch()

        _flush_candidate_batch()

        sanitized_codes: List[str] = []
        target_entry = entry or "solution"
        for raw_txt in raw_predictions:
            code = _extract_code_for_entry(raw_txt, target_entry)
            if code and not _syntactic_ok(code):
                fallback_code = _concat_all_code_blocks(raw_txt)
                if fallback_code and _syntactic_ok(fallback_code):
                    code = fallback_code
            sanitized_codes.append(code)

        strategy_hints = list(planned_hints[:len(raw_predictions)])
        strategy_origins = list(planned_origins[:len(raw_predictions)])
        if len(strategy_hints) < len(raw_predictions):
            diff = len(raw_predictions) - len(strategy_hints)
            strategy_hints.extend([""] * diff)
            strategy_origins.extend([""] * diff)

        for idx, (code, raw_txt) in enumerate(zip(sanitized_codes, raw_predictions)):
            if idx >= len(strategy_hints) or not strategy_hints[idx]:
                detected = detect_strategy(code or raw_txt, target_entry)
                if idx >= len(strategy_hints):
                    strategy_hints.append(detected)
                    strategy_origins.append("inferred")
                else:
                    strategy_hints[idx] = detected
                    strategy_origins[idx] = "inferred"

        results.append({
            "id": example.get("id"),
            "problem_description": description,
            "difficulty": difficulty,
            "input": example["input"],
            "entry_point": entry,
            "prediction": raw_predictions,
            "codes": sanitized_codes,
            "strategy_hints": strategy_hints,
            "strategy_origins": strategy_origins,
        })
        progress.update(1)

    progress.close()

    # Convert list of dicts → Hugging Face Dataset
    pred_dataset = Dataset.from_pandas(pd.DataFrame(results))

    # Save predictions as JSON (Hugging Face format)
    output_path = split_dir / f"predictions{suffix}.json"
    print(f" Saving predictions to {output_path}")
    pred_dataset.to_json(str(output_path), orient="records", lines=True)

    # Also save a flat list of candidates (one line per candidate) for filtering/testing
    cands_path = split_dir / f"candidates{suffix}.jsonl"
    print(f" Saving flattened candidates to {cands_path}")
    with open(cands_path, "w") as f:
        for rec in results:
            _id = rec.get("id")
            _inp = rec.get("input")
            entry = rec.get("entry_point")
            # Use the short synthesized description (no redundancy)
            desc = rec.get("problem_description")
            diff = rec.get("difficulty", "unknown")
            hints = rec.get("strategy_hints") or []
            origins = rec.get("strategy_origins") or []
            codes = rec.get("codes") or []
            target_entry = entry or "solution"
            for idx, txt in enumerate(rec["prediction"]):
                hint = hints[idx] if idx < len(hints) else None
                origin = origins[idx] if idx < len(origins) else ""
                code = codes[idx] if idx < len(codes) else _extract_code_for_entry(txt, target_entry)
                if code and not _syntactic_ok(code):
                    fallback_code = _concat_all_code_blocks(txt)
                    if fallback_code and _syntactic_ok(fallback_code):
                        code = fallback_code
                if not hint:
                    hint = detect_strategy(code or txt, target_entry)
                    origin = "inferred"
                out = {
                    "id": _id,
                    "problem_description": desc,
                    "difficulty": diff,
                    "input": _inp,
                    "entry_point": target_entry,
                    "candidate_idx": idx,
                    "strategy_hint": hint,
                    "strategy_origin": origin,
                    "text": txt,
                    "code": code,
                    "chars": len(code or txt or ""),
                }
                f.write(json.dumps(out) + "\n")

    should_eval = (target_split.lower() == "test") or os.getenv("FORCE_EVAL_TRAIN") == "1"
    if should_eval:
        ks = _resolve_eval_ks()
        print(f"[eval] effective ks: {ks}")
        # Legacy single-sample evaluation is disabled by default.
        # Set LEGACY_EVAL_LOG=1 to re-enable this debugging-only path.
        if os.getenv("LEGACY_EVAL_LOG") == "1":
            print(" Evaluating predictions (legacy single-sample path)...")
            try:
                pred_df = pred_dataset.to_pandas()
                legacy_report = dataset.evaluate(pred_df, ks=ks, split="")
            except (AttributeError, NotImplementedError):
                print("[legacy] skipping (adapter has no evaluate())")
            except Exception as exc:
                print(f"[legacy] skipping (error: {exc})")
            else:
                print(" Legacy evaluation results:\n", legacy_report)

        # Multi-candidate evaluation.
        print(f" Re-evaluating split '{target_split}' from candidates.jsonl for true pass@k...")
        ds_eval = dataset
        refs = [ex["test"] for ex in ds_eval.dataset[target_split]]
        entries_list = [ex["entry_point"] for ex in ds_eval.dataset[target_split]]
        id2entry = {ex["id"]: ex["entry_point"] for ex in ds_eval.dataset[target_split]}

        by_id = defaultdict(list)
        with open(cands_path) as f:
            for line in f:
                j = json.loads(line)
                ex_id = j.get("id")
                if ex_id in id2entry:
                    code = j.get("code") or ""
                    if code:
                        by_id[ex_id].append(
                            normalize_entry_point(code, id2entry[ex_id])
                        )

        preds = [by_id[ex["id"]] for ex in ds_eval.dataset[target_split]]

        diversity_scores: List[float] = []
        for codes in preds:
            if not codes:
                diversity_scores.append(0.0)
                continue
            unique_codes = {canonical_signature(code or "") for code in codes if code is not None}
            diversity_scores.append(len(unique_codes) / max(1, len(codes)))
        diversity_unique_ratio = (
            sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
        )

        if target_split.lower() == "test":
            num_procs = _resolve_eval_num_procs()
            batch_size_eval = _resolve_eval_batch_size()
            progress_paths = (
                os.getenv(EVAL_PROGRESS_LOG_ENV),
                os.getenv(EVAL_PROGRESS_JSON_ENV),
            )
            if progress_paths == (None, None):
                progress_paths = None
            if num_procs > 1:
                true_res = _compute_pass_at_k_parallel(refs, preds, entries_list, ks, num_procs, batch_size_eval, progress_paths)
            else:
                log_path, json_path = progress_paths or (None, None)
                monitor = _EvalProgressMonitor(
                    stage="eval_test",
                    mode="serial",
                    ks=list(ks),
                    total_tasks=len(refs),
                    total_batches=1,
                    num_procs=1,
                    batch_size=batch_size_eval,
                    n_test=len(refs),
                    text_path=log_path,
                    json_path=json_path,
                )
                try:
                    monitor.log_start()
                    start_serial = time.monotonic()
                    true_res = _compute_pass_at_k_serial(refs, preds, entries_list, ks)
                    serial_time = time.monotonic() - start_serial
                    monitor.log_batch(1, len(refs), dt_override=serial_time)
                    monitor.log_done(serial_time, serial_time, 0.0, true_res, mode_override="serial")
                finally:
                    monitor.close()
        else:
            true_res = _compute_pass_at_k_serial(refs, preds, entries_list, ks)
        true_res = _normalize_pass_dict(true_res)
        print(" True pass@k:", true_res)

        # Overwrite evaluations.json with the true metrics
        eval_path = split_dir / f"evaluations{suffix}.json"
        with open(eval_path, "w") as f:
            payload = {
                "pass@k": true_res,
                "ks": ks,
                "n_test": len(refs),
                "diversity_unique_ratio": diversity_unique_ratio,
            }
            json.dump(payload, f, indent=2)
    else:
        print("[skip] pass@k evaluation on training split (annotation provides per-candidate pass/fail; test-only metrics preserved)")

    print(" Done! Results saved to:")
    print(f"   - Predictions: {output_path}")
    print(f"   - Candidates:  {cands_path}")
    if should_eval:
        print(f"   - Evaluation:  {eval_path}")


def main():
    parser = argparse.ArgumentParser(description="Run model predictions and evaluation on a dataset.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., mbpp).")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Qwen/... or meta-llama/...).")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder.")
    parser.add_argument("--split", type=str, choices=["train", "test"], default="test", help="Dataset split to process.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate per prompt.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of prompts to process per batch.")

    args = parser.parse_args()

    # Prepare dataset if not present
    if not os.path.exists(args.dataset_path):
        print(f"Dataset not found at {args.dataset_path}. Preparing dataset...")
        prepare_dataset(args.dataset, split_ratio=0.8, output_dir=args.dataset_path)

    run_predictions(
        model_name=args.model,
        dataset_name=args.dataset,
        dataset_path=args.dataset_path,
        split=args.split,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
