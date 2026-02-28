import json
import os
import re
from math import comb
from typing import Iterator, List, Optional

import pandas as pd
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


class MBPPDataset(Dataset):
    """
    Adapter for MBPP saved with datasets.save_to_disk.
    Produces the unified schema used by the pipeline.
    """

    def __init__(self, datapath: str):
        super().__init__()
        self._root_path = datapath  # for debug
        raw = load_from_disk(datapath)  # DatasetDict({'train','validation','test',...})
        # keep only known splits (ignore 'prompt')
        if isinstance(raw, DatasetDict):
            raw = DatasetDict({k: v for k, v in raw.items() if k in {"train", "validation", "test"}})
        self.dataset = raw.map(
            self._format_example,
            load_from_cache_file=False,
            desc="MBPP: formatting examples + injecting check(candidate)",
        )

        # Build id -> entry_point map across splits
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

    def _guess_entry_point_from_prompt(self, text: str) -> Optional[str]:
        m = re.search(r"\bdef\s+([a-zA-Z_]\w*)\s*\(", text or "")
        return m.group(1) if m else None

    def _format_example(self, example: dict) -> dict:
        prompt = example.get("text") or example.get("prompt") or ""

        tests = example.get("test_list") or []
        test_code = "\n".join(map(str, tests)) if isinstance(tests, list) else str(tests)

        # Prefer entry point inferred from tests; fall back to prompt; then 'solution'
        entry = (
            self._guess_entry_point_from_tests(test_code)
            or self._guess_entry_point_from_prompt(prompt)
            or "solution"
        )

        # Include the runner so code_eval executes asserts.
        test_wrapped = self._wrap_tests_as_check(test_code, entry)

        formatted = {
            "id": example.get("task_id") or example.get("problem_id") or example.get("id"),
            "problem_description": prompt,
            "input": prompt,
            "solution": example.get("code", ""),
            "entry_point": entry,
            "test": test_wrapped,  # <-- use the one with check(...)
        }
        formatted["problem_description"] = synthesize_problem_description(formatted)
        return formatted

    def _default_split(self) -> str:
        if isinstance(self.dataset, DatasetDict):
            for s in ("validation", "test", "train"):
                if s in self.dataset:
                    return s
        return "train"

    # Robust code extraction

    def _extract_code(self, text: str) -> str:
        """
        Robust extraction:
        - Prefer fenced blocks; pick one containing 'def ' if possible, else longest.
        - If none, strip a leading 'python\n'.
        - If still none, slice from the first 'def '.
        - Strip leftover backticks/whitespace.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        s = text.strip()

        # fenced blocks: ```python ...``` or ``` ... ```
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

        # no fences: drop stray 'python\n'
        if s.lower().startswith("python\n"):
            s = s.split("\n", 1)[1].lstrip()

        # slice from first 'def '
        m = re.search(r"\bdef\s+[A-Za-z_]\w*\s*\(", s)
        if m:
            s = s[m.start():]

        # strip leftover backticks and any trailing END markers
        s = s.replace("```", "").strip()
        s = re.sub(r"\b<?END>?\b\s*$", "", s, flags=re.IGNORECASE)
        return s

    def postprocess(self, prediction) -> List[str]:
        """
        Accept a single string OR a list of strings; return a list of cleaned code candidates.
        (Entry-point normalization happens in evaluate where we know the target name.)
        """
        items = prediction if isinstance(prediction, list) else [prediction]
        out: List[str] = []
        for it in items:
            code = self._extract_code(it)
            if code:
                out.append(code)
        return out

    def _wrap_tests_as_check(self, test_code: str, entry: str) -> str:
        """
        Convert raw asserts like `assert find_Min_Sum(12) == 7` into:
        def check(candidate):
            assert candidate(12) == 7
            ...
        """
        body = [
            "from prompt_utils import normalize_code_output as _normalize_code_output, equivalent as _equivalent",
            "",
            "def check(candidate):",
        ]
        pat = re.compile(rf"\b{re.escape(entry)}\s*\(")
        for line in (test_code or "").splitlines():
            line = line.strip()
            if not line:
                continue
            converted = pat.sub("candidate(", line)
            if converted.startswith("assert ") and "==" in converted:
                expr = converted[len("assert "):]
                lhs, rhs = expr.split("==", 1)
                lhs = lhs.strip()
                rhs = rhs.strip()
                body.append(f"    _lhs = _normalize_code_output({lhs})")
                body.append(f"    _rhs = _normalize_code_output({rhs})")
                body.append("    assert _equivalent(_lhs, _rhs)")
            else:
                body.append("    " + converted)
        return "\n".join(body) + "\n"

    # Metric bridge
    def check(self, solutions, split: str, ks: List[int]) -> pd.DataFrame:
        """
        Run subprocess-based correctness oracle over wrapped test+candidate.
        """
        split = split or self._default_split()
        ref_split = self.dataset[split]
        references = ref_split["test"]
        entries = ref_split["entry_point"]

        # Minimal debug snapshot
        try:
            dbg = {
                "ref_example": references[0][:2000],
                "num_preds_for_first": len(solutions[0]) if solutions and isinstance(solutions[0], list) else None,
                "pred_first_snippet": (solutions[0][0] or "")[:300]
                if solutions
                and isinstance(solutions[0], list)
                and solutions[0]
                else None,
            }
            with open(os.path.join(self._root_path, "eval_debug_min.json"), "w") as f:
                json.dump(dbg, f)
        except Exception:
            pass

        print(f"[eval] split={split}  refs={len(references)}  preds={len(solutions)}  ks={ks}")
        from pipeline.annotate import execute_wrapped_code

        passed_masks: List[List[bool]] = []
        for test, preds, entry in zip(references, solutions, entries):
            cleaned_test = _sanitize_test_for_exec(test)
            entry_point = entry or "solution"
            mask: List[bool] = []
            for candidate in preds:
                wrapped = f"{cleaned_test}\n\n{candidate or ''}\n\ncheck({entry_point})"
                mask.append(bool(execute_wrapped_code(wrapped, timeout=2.0)))
            passed_masks.append(mask)

        pass_at_k = _aggregate_pass_at_k(passed_masks, ks)

        vals = []
        for k in ks:
            if k in pass_at_k:
                vals.append(pass_at_k[k])
            elif str(k) in pass_at_k:
                vals.append(pass_at_k[str(k)])
            else:
                vals.append(0.0)

        return pd.DataFrame({"k": ks, "pass@k": vals}).set_index("k")

    def __iter__(self) -> Iterator[dict]:
        split = self._default_split()
        for item in self.dataset[split]:
            yield item

    def evaluate(self, predictions: pd.DataFrame, ks: List[int], split: str) -> pd.DataFrame:
        """
        Prefer evaluating from candidates.jsonl (preserves generated candidates per id).
        Fallback to predictions DataFrame if the file is missing.
        """
        import collections
        import json as _json
        import numpy as _np
        import os as _os

        split = split or self._default_split()
        ref_split = self.dataset[split]
        ref_ids = list(ref_split["id"])
        id_to_entry = {ex_id: ep for ex_id, ep in zip(ref_split["id"], ref_split["entry_point"])}

        cand_path = _os.path.join(self._root_path, "candidates.jsonl")
        use_candidates_file = _os.path.exists(cand_path)

        solutions_by_id = collections.defaultdict(list)

        if use_candidates_file:
            # Option B: read directly from candidates.jsonl.
            want = set(ref_ids)
            with open(cand_path) as f:
                for line in f:
                    j = _json.loads(line)
                    _id = j.get("id")
                    if _id in want:
                        code = j.get("code") or ""
                        if code:
                            solutions_by_id[_id].append(code)
        else:
            # Fallback: use predictions DataFrame.
            for _, row in predictions.iterrows():
                _id = row.get("id")
                raw = row.get("prediction")

                # Normalize to a Python list of strings
                if isinstance(raw, list):
                    items = raw
                elif isinstance(raw, _np.ndarray):
                    items = raw.tolist()
                elif isinstance(raw, str):
                    # Try parse as JSON array first
                    try:
                        maybe = _json.loads(raw)
                        items = maybe if isinstance(maybe, list) else [raw]
                    except Exception:
                        items = [raw]
                else:
                    items = [str(raw)]

                # Use postprocess to extract code blocks (may be many per string)
                codes = self.postprocess(items)
                if codes:
                    solutions_by_id[_id].extend(codes)

        # Normalize entry points and align to reference order.
        aligned_solutions = []
        for _id in ref_ids:
            entry = id_to_entry.get(_id, "solution")
            codes = solutions_by_id.get(_id, [])
            normed = [normalize_entry_point(c or "", entry) for c in codes]
            aligned_solutions.append(normed)

        # Debug breadcrumb
        try:
            first_count = len(aligned_solutions[0]) if aligned_solutions else 0
            first_pred_snip = ""
            if aligned_solutions and aligned_solutions[0]:
                first_pred_snip = aligned_solutions[0][0][:180]
            debug = {
                "source": "candidates.jsonl" if use_candidates_file else "predictions_df",
                "ref_example": ref_split[0]["test"] if len(ref_split) else "",
                "num_preds_for_first": first_count,
                "pred_first_snippet": first_pred_snip,
            }
            with open(_os.path.join(self._root_path, "eval_debug_min.json"), "w") as f:
                _json.dump(debug, f)
        except Exception:
            pass

        return self.check(aligned_solutions, split, ks)


register_dataset("mbpp", MBPPDataset, aliases=["google-research-datasets/mbpp"])
MBPPDataset.DEFAULT_HF_ID = "google-research-datasets/mbpp"
