import ast
import json
import os
import re
from math import comb
from typing import Any, Iterator, List, Optional

import pandas as pd
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk

from dataset import Dataset, register_dataset
from prompt_utils import synthesize_problem_description
from sanitize import normalize_entry_point


def _sanitize_test_for_exec(t: str) -> str:
    t = (t or "").rstrip()
    while t.endswith("_"):
        t = t[:-1].rstrip()
    return t + "\n"


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


def _safe_parse_io(payload) -> Optional[dict]:
    """
    Robustly parse the `input_output` field into a dict.

    Accepts:
      - dict (already parsed)
      - JSON string
      - Python literal string (ast.literal_eval)
    """
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload

    text = str(payload).strip()
    if not text:
        return None

    # Try JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Try Python literal
    try:
        literal = ast.literal_eval(text)
        if isinstance(literal, dict):
            return literal
    except Exception:
        pass

    return None


def _is_function_item(example: dict) -> bool:
    """
    Keep only function-style items:
      input_output = { fn_name: str, inputs: [...], outputs: [...] }
    """
    io = _safe_parse_io(example.get("input_output"))
    return bool(
        io
        and isinstance(io.get("fn_name"), str)
        and "inputs" in io
        and "outputs" in io
    )


def _py_literal(value: Any) -> str:
    """
    Safe Python literal representation (used to embed args/expected in tests).
    """
    return repr(value)


def _build_io_asserts(io: dict, entry: str) -> str:
    """
    Build raw assert lines from TACO `input_output`, e.g.:

        assert fn_name(1, 2) == 3

    These will later be wrapped into a `check(candidate)` function by
    `_wrap_tests_as_check`, which takes care of normalization and _equivalent.
    """
    inputs = io.get("inputs", [])
    outputs = io.get("outputs", [])
    lines: List[str] = []

    count = min(len(inputs), len(outputs))
    for idx in range(count):
        args = inputs[idx]
        expected = outputs[idx]

        if isinstance(args, (list, tuple)):
            call = f"{entry}(*{_py_literal(list(args))})"
        else:
            call = f"{entry}({_py_literal(args)})"

        expected_code = _py_literal(expected)
        lines.append(f"assert {call} == {expected_code}")

    return "\n".join(lines).strip()


# TACO-Verified Dataset Adapter

class TacoVerifiedDataset(Dataset):
    """
    Adapter for likaixin/TACO-verified saved via datasets.save_to_disk.

    Keeps only function-style rows (via input_output) and produces the
    unified schema:
      { id, problem_description, input, solution, entry_point, test }

    Evaluation is done via subprocess + execute_wrapped_code 
    """

    DEFAULT_HF_ID = "likaixin/TACO-verified"

    def __init__(self, datapath: str):
        super().__init__()
        self._root_path = datapath

        raw = load_from_disk(datapath)

        # Normalize to DatasetDict
        if isinstance(raw, HFDataset):
            raw = DatasetDict({"train": raw})
        elif isinstance(raw, DatasetDict):
            # Keep only known splits if present
            raw = DatasetDict({k: v for k, v in raw.items() if k in {"train", "test"}})

        def _format_split(ds: HFDataset) -> HFDataset:
            # Filter to function-style items
            filtered = ds.filter(_is_function_item, desc="TACO-verified: keep function-mode items")
            columns = filtered.column_names
            return filtered.map(
                self._format_example,
                remove_columns=columns,
                load_from_cache_file=False,
                desc="TACO-verified: formatting schema + injecting check(candidate)",
            )

        if isinstance(raw, DatasetDict):
            self.dataset = DatasetDict({split: _format_split(ds) for split, ds in raw.items()})
        else:
            self.dataset = _format_split(raw)

        # id -> entry_point mapping
        self._by_id = {}
        if isinstance(self.dataset, DatasetDict):
            for _, ds in self.dataset.items():
                for ex in ds:
                    self._by_id[ex["id"]] = ex["entry_point"]
        else:
            for ex in self.dataset:
                self._by_id[ex["id"]] = ex["entry_point"]

    # Entry point inference
    def _guess_entry_point_from_tests(self, test_code: str) -> Optional[str]:
        m = re.search(r"assert\s+([a-zA-Z_]\w*)\s*\(", test_code or "")
        return m.group(1) if m else None

    def _guess_entry_point_from_prompt(self, prompt: str) -> Optional[str]:
        m = re.search(r"\bdef\s+([a-zA-Z_]\w*)\s*\(", prompt or "")
        return m.group(1) if m else None

    # Robust code extraction
    def _extract_code(self, text: str) -> str:
        """
        Robust extraction:
        - Prefer fenced blocks; pick one containing 'def ' if possible, else longest.
        - If none, strip a leading 'python\n'.
        - If still none, slice from the first 'def '.
        - Strip leftover backticks/END markers/whitespace.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        s = text.strip()

        blocks = re.findall(r"```(?:[a-zA-Z]+\n)?(.*?)```", s, re.DOTALL)
        if blocks:
            best = None
            best_len = -1
            for b in blocks:
                cand = b.strip()
                score = (1 if "def " in cand else 0, len(cand))
                if score > ((1 if (best and "def " in best) else 0), best_len):
                    best = cand
                    best_len = len(cand)
            return (best or max(blocks, key=len)).strip()

        if s.lower().startswith("python\n"):
            s = s.split("\n", 1)[1].lstrip()

        m = re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", s)
        if m:
            s = s[m.start():]

        s = s.replace("```", "").strip()
        s = re.sub(r"\b<?END>?\b\s*$", "", s, flags=re.IGNORECASE)
        return s

    def postprocess(self, prediction) -> List[str]:
        items = prediction if isinstance(prediction, list) else [prediction]
        out: List[str] = []
        for it in items:
            code = self._extract_code(it)
            if code:
                out.append(code)
        return out

    # Wrap tests into check(candidate)
    def _wrap_tests_as_check(self, test_code: str, entry: str) -> str:
        """
        Wrap raw assert lines into a check(candidate) function that uses
        normalize_code_output + equivalent, similar to the MBPP adapter.
        """
        body = [
            "from prompt_utils import normalize_code_output as _normalize_code_output, equivalent as _equivalent",
            "import math",
            "",
            "def check(candidate):",
        ]

        call_pat = re.compile(rf"\b{re.escape(entry)}\s*\(")

        for raw_line in (test_code or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue

            converted = call_pat.sub("candidate(", line)

            # Equality: use _equivalent
            if converted.startswith("assert ") and "==" in converted:
                expr = converted[len("assert "):]
                lhs, rhs = expr.split("==", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                body.append(f"    _lhs = _normalize_code_output({lhs})")
                body.append(f"    _rhs = _normalize_code_output({rhs})")
                body.append("    assert _equivalent(_lhs, _rhs)")
                continue

            # Inequality !=
            if converted.startswith("assert ") and "!=" in converted:
                expr = converted[len("assert "):]
                lhs, rhs = expr.split("!=", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                body.append(f"    _lhs = _normalize_code_output({lhs})")
                body.append(f"    _rhs = _normalize_code_output({rhs})")
                body.append("    assert not _equivalent(_lhs, _rhs)")
                continue

            # For all other asserts, keep literal evaluation
            if converted.startswith("assert "):
                body.append(f"    {converted}")
                continue

            body.append("    " + converted)

        return "\n".join(body) + "\n"

    # Format example (core schema logic)
    def _format_example(self, example: dict) -> dict:
        """
        Convert a raw TACO-verified example into the unified schema.

        - Parse input_output to get fn_name, inputs, outputs
        - Build tests as assert fn_name(...) == expected
        - Wrap tests into check(candidate)
        - Build a slightly enriched prompt
        """
        prompt = (example.get("question") or "").strip()

        io = _safe_parse_io(example.get("input_output")) or {}
        fn = io.get("fn_name")
        entry_hint = fn if isinstance(fn, str) and fn else "solution"

        if prompt and not prompt.endswith("."):
            prompt_instruction = "\nImplement the function as described."
        else:
            prompt_instruction = "\nImplement the function."
        full_prompt = (prompt + prompt_instruction).strip()

        raw_test_code = _build_io_asserts(io, entry_hint)

        entry = (
            self._guess_entry_point_from_tests(raw_test_code)
            or self._guess_entry_point_from_prompt(prompt)
            or entry_hint
        )

        test_wrapped = self._wrap_tests_as_check(raw_test_code, entry)

        ref_solution = ""
        sols = example.get("solutions")
        if isinstance(sols, list) and sols:
            ref_solution = sols[0]

        formatted = {
            "id": example.get("task_id") or example.get("id"),
            "problem_description": prompt,
            "input": full_prompt,
            "solution": ref_solution,
            "entry_point": entry,
            "test": test_wrapped,
        }

        formatted["problem_description"] = synthesize_problem_description(formatted)
        return formatted

    def _default_split(self) -> str:
        if isinstance(self.dataset, DatasetDict):
            for s in ("validation", "test", "train"):
                if s in self.dataset:
                    return s
        return "train"
        
    # Subprocess-based evaluation
    def check(self, solutions, split: str, ks: List[int]) -> pd.DataFrame:
        """
        Run subprocess-based correctness oracle over wrapped test+candidate.
        """
        from pipeline.annotate import execute_wrapped_code  # late import to avoid circularity

        split = split or self._default_split()
        refs = self.dataset[split]["test"]
        entries = self.dataset[split]["entry_point"]

        print(f"[eval] split={split} refs={len(refs)} preds={len(solutions)} ks={ks}")

        passed_masks: List[List[bool]] = []
        for test, preds, entry in zip(refs, solutions, entries):
            cleaned = _sanitize_test_for_exec(test)
            entry_point = entry or "solution"

            mask: List[bool] = []
            for cand in preds:
                wrapped = f"{cleaned}\n\n{cand or ''}\n\ncheck({entry_point})"
                ok = bool(execute_wrapped_code(wrapped, timeout=2.0))
                mask.append(ok)

            passed_masks.append(mask)

        stats = _aggregate_pass_at_k(passed_masks, ks)
        vals = [stats.get(k, 0.0) for k in ks]

        return pd.DataFrame({"k": ks, "pass@k": vals}).set_index("k")

    # Iterator and evaluate()
    def __iter__(self) -> Iterator[dict]:
        split = self._default_split()
        for item in self.dataset[split]:
            yield item

    def evaluate(self, predictions: pd.DataFrame, ks: List[int], split: str):
        """
        Prefer evaluating from candidates.jsonl (robust; always many per id).
        Fallback to predictions DataFrame if the file is missing.
        """
        import collections
        import json as _json
        import numpy as _np
        import os as _os

        split = split or self._default_split()
        ref_split = self.dataset[split]
        ref_ids = list(ref_split["id"])
        entry_map = {ex_id: ep for ex_id, ep in zip(ref_split["id"], ref_split["entry_point"])}

        cand_path = _os.path.join(self._root_path, "candidates.jsonl")
        use_file = _os.path.exists(cand_path)

        solutions_by_id = collections.defaultdict(list)

        if use_file:
            want = set(ref_ids)
            with open(cand_path) as f:
                for line in f:
                    j = _json.loads(line)
                    _id = j.get("id")
                    if _id not in want:
                        continue

                    raw = j.get("code") or ""
                    cleaned = self.postprocess(raw)
                    if cleaned:
                        # keep first cleaned snippet per line
                        solutions_by_id[_id].append(cleaned[0])
        else:
            for _, row in predictions.iterrows():
                _id = row.get("id")
                raw = row.get("prediction")

                if isinstance(raw, list):
                    items = raw
                elif isinstance(raw, _np.ndarray):
                    items = raw.tolist()
                elif isinstance(raw, str):
                    try:
                        maybe = _json.loads(raw)
                        items = maybe if isinstance(maybe, list) else [raw]
                    except Exception:
                        items = [raw]
                else:
                    items = [str(raw)]

                cleaned = self.postprocess(items)
                if cleaned:
                    solutions_by_id[_id].extend(cleaned)

        aligned: List[List[str]] = []
        for _id in ref_ids:
            entry = entry_map.get(_id, "solution")
            codes = solutions_by_id.get(_id, [])
            normalized = [normalize_entry_point(c or "", entry) for c in codes]
            aligned.append(normalized)

        return self.check(aligned, split, ks)


register_dataset("taco_verified", TacoVerifiedDataset, aliases=["likaixin/TACO-verified"])
TacoVerifiedDataset.DEFAULT_HF_ID = "likaixin/TACO-verified"
