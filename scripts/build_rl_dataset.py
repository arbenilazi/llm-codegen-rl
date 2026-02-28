#!/usr/bin/env python3
"""
Build an offline RL dataset of (prompt → ranked correct solutions) from model samples.

Pipeline:
1) Load dataset and generated candidates.jsonl.
2) Normalize entry points for each candidate.
3) Correctness is taken from candidates_annotated.jsonl (the annotation stage).
   No runtime evaluation is performed here.
4) Keep only passing candidates, deduplicate by AST, rank by simplicity (LOC, tokens, CC, nesting).
5) Export:
   - nbest.jsonl   → ranked correct solutions per task
   - pairs.jsonl   → preference pairs (indices) for DPO/IPO training
   - manifest.json → run config + summary stats (+ pass@k from evaluations.json if present)
"""

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np

# allow code execution for the evaluate 'code_eval' metric
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

from evaluate import load

# local imports
import sys
sys.path.append("src")
from sanitize import normalize_entry_point
from metrics import ast_key, cyclomatic_complexity, nesting_depth, score_simple

# Helpers

def _extract_pass1_value(res: Dict) -> float:
    """
    Robustly extract pass@1 from 'evaluate' result dicts that may look like:
      {1: 1.0}, {"1": 1.0}, {"pass@1": 1.0}, {"pass_at_1": 1.0}, or similar.
    """
    for key in (1, "1", "pass@1", "pass_at_1"):
        if key in res:
            try:
                return float(res[key])
            except Exception:
                pass
    # last resort: find any key that contains "1"
    for k, v in res.items():
        if "1" in str(k):
            try:
                return float(v)
            except Exception:
                continue
    return 0.0


def _normalize_eval_output(result):
    # Normalize singleton outputs like [True] → True for consistent evaluation.
    if isinstance(result, list) and len(result) == 1:
        return result[0]
    if isinstance(result, np.ndarray):
        if result.size == 1:
            return result.item()
        return result
    return result


def compute_pass_mask_strict(metric, reference: str, candidates: List[str]) -> List[bool]:
    """
    Strict fallback: evaluate each candidate alone with code_eval; pass if pass@1 == 1.0.
    Conservative: any exception counts as fail.
    """
    mask: List[bool] = []
    for c in candidates:
        try:
            res, _det = metric.compute(references=[reference], predictions=[[c]], k=[1])
            val = _extract_pass1_value(res)
            mask.append(val >= 1.0 - 1e-12)
        except Exception:
            mask.append(False)
    return mask


# Main build

def build(args):
    # 1) load dataset wrapper to get split ids/entry points and raw tests
    sys.path.append("src")
    from dataset import get_dataset
    ds = get_dataset(args.dataset, args.dataset_path)
    split = "train" if hasattr(ds.dataset, "__contains__") and "train" in ds.dataset else ds._default_split()
    ref_split = ds.dataset[split]           # HF dataset split
    id_order = list(ref_split["id"])
    id2entry = {ex["id"]: ex["entry_point"] for ex in ref_split}
    references = [ex["test"] for ex in ref_split]  # per-id test harness (wrapped check(candidate))

    # Optionally show cached evaluation summary (pass@1/5/10).
    eval_path = os.path.join(args.dataset_path, "evaluations.json")
    eval_data = {}
    if os.path.exists(eval_path):
        try:
            with open(eval_path) as f:
                eval_data = json.load(f)
            passk = eval_data.get("pass@k", {})
            if passk:
                msg = ", ".join(f"{k}={float(v):.3f}" for k, v in passk.items())
                print(f"[summary] Cached evaluation: {msg}")
            else:
                print("[warn] evaluations.json found but no pass@k inside.")
        except Exception:
            print("[warn] Failed to parse evaluations.json; continuing without cached summary.")

    # 2) read all candidates, keep only those for ids in split
    # Use candidates_annotated.jsonl (not raw candidates.jsonl) to get correct passed flags.
    cand_path = args.candidates_path or os.path.join(args.dataset_path, "candidates_annotated.jsonl")
    assert os.path.exists(cand_path), f"Missing {cand_path}. Make sure to use candidates_annotated.jsonl (not raw candidates.jsonl)"

    candidates_by_id: Dict[int, List[Dict]] = defaultdict(list)

    with open(cand_path) as f:
        for line in f:
            j = json.loads(line)
            _id = j.get("id")
            if _id not in id2entry:
                continue
            code = (j.get("code") or "").strip()
            if not code:
                continue

            entry = id2entry[_id]
            normalized = normalize_entry_point(code, entry)

            metrics = j.get("metrics") or {}
            has_metrics = all(
                key in metrics for key in ("loc", "tokens", "cyclomatic_complexity", "nesting_depth")
            )
            if not has_metrics:
                loc, toks, cc, depth = score_simple(normalized)
                metrics = {
                    "loc": loc,
                    "tokens": toks,
                    "cyclomatic_complexity": cc,
                    "nesting_depth": depth,
                }
            ast_hash = metrics.get("ast_hash") or ast_key(normalized)
            metrics["ast_hash"] = ast_hash

            passed = j.get("passed")
            if passed is None:
                # annotation stage is the only source of truth
                passed = False
            passed = bool(passed)

            candidates_by_id[_id].append({
                "code": normalized,
                "metrics": metrics,
                "passed": passed,
            })

    # 4) per id: separate passers and fails, dedup by ast_key, rank passers, sample fails
    nbest_records = []
    pair_records = []
    kept_total = 0
    num_items = 0
    num_pass_total = 0

    for _id in id_order:
        entry = id2entry[_id]
        records = candidates_by_id.get(_id, [])

        pass_recs = [rec for rec in records if rec.get("passed")]
        fail_recs = [rec for rec in records if not rec.get("passed")]

        if not pass_recs:
            continue  # no positives => can't form correctness pairs

        num_items += 1
        num_pass_total += len(pass_recs)

        # Deduplicate passers by AST.
        seen = set()
        pass_uniq = []
        for rec in pass_recs:
            h = rec["metrics"].get("ast_hash")
            if h not in seen:
                seen.add(h)
                pass_uniq.append(rec)

        # Deduplicate fails by AST.
        seen_f = set()
        fail_uniq = []
        for rec in fail_recs:
            h = rec["metrics"].get("ast_hash")
            if h not in seen_f:
                seen_f.add(h)
                fail_uniq.append(rec)

        # Rank passers by style metrics.
        # Do not use failure-sentinel execution_time in ranking.
        scored_pass = []
        for rec in pass_uniq:
            m = rec["metrics"]
            exec_time = m.get("execution_time", 10.0)
            nov = m.get("novelty", 0.0)
            score = (
                m["loc"],
                m["tokens"],
                m["cyclomatic_complexity"],
                m["nesting_depth"],
                exec_time,
                -nov,
            )
            scored_pass.append((rec["code"], m, score, exec_time, nov, True))
        scored_pass.sort(key=lambda x: x[2])

        # Keep top passers (your n_best)
        pass_keep = scored_pass[:args.n_best]

        # Sample a subset of failing candidates.
        # If you want, you can prefer "near-miss" fails later, but keep it simple for now.
        max_fails = max(0, int(args.max_fails_per_task))
        fail_keep = fail_uniq[:max_fails]

        # Build pool: passers first, then fails
        pool = []
        for code, m, score, exec_time, nov, _is_pass in pass_keep:
            pool.append({
                "code": code,
                "passed": True,
                "score": {
                    "loc": score[0],
                    "tokens": score[1],
                    "cc": score[2],
                    "nest": score[3],
                    "execution_time": float(exec_time),
                    "novelty": float(nov),
                }
            })

        for rec in fail_keep:
            m = rec["metrics"]
            pool.append({
                "code": rec["code"],
                "passed": False,
                "score": {
                    "loc": m.get("loc"),
                    "tokens": m.get("tokens"),
                    "cc": m.get("cyclomatic_complexity"),
                    "nest": m.get("nesting_depth"),
                    "execution_time": float(m.get("execution_time", 10.0)),
                    "novelty": float(m.get("novelty", 0.0)),
                }
            })

        # Build pairs:
        # 1) correctness pairs: kept passers beat kept failures
        # 2) optional within-pass pairs for style ranking signal
        pairs = []

        num_pass_in_pool = len(pass_keep)
        num_fail_in_pool = len(fail_keep)

        # (1) pass > fail (capped to avoid pair explosion)
        fails_per_passer = max(0, int(args.fails_per_passer))
        for i in range(num_pass_in_pool):
            for j in range(num_pass_in_pool, min(num_pass_in_pool + num_fail_in_pool, num_pass_in_pool + fails_per_passer)):
                pairs.append([i, j])

        # (2) within-pass style pairs (optional, small)
        wppw = max(0, int(args.within_pass_pairs_per_winner))
        for i in range(min(num_pass_in_pool, args.max_pairs)):
            for j in range(i + 1, min(num_pass_in_pool, i + 1 + wppw)):
                pairs.append([i, j])

        nbest_records.append({
            "id": _id,
            "entry_point": entry,
            "best_list": pool,               # now includes passed flags
            "num_pass": len(pass_recs),
            "num_uniq": len(pass_uniq),
            "num_fail": len(fail_recs),
            "fail_uniq": len(fail_uniq),
        })
        pair_records.append({
            "id": _id,
            "pairs": pairs
        })
        
        kept_total += len(pool)

    # 6) write outputs
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "nbest.jsonl"), "w") as f:
        for r in nbest_records:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "pairs.jsonl"), "w") as f:
        for r in pair_records:
            f.write(json.dumps(r) + "\n")

    # 7) manifest with full pass@k summary if available
    manifest = {
        "dataset": args.dataset,
        "dataset_path": args.dataset_path,
        "n_best": args.n_best,
        "max_pairs": args.max_pairs,
        "pairs_per_winner": args.pairs_per_winner,
        "max_fails_per_task": args.max_fails_per_task,
        "within_pass_pairs_per_winner": args.within_pass_pairs_per_winner,
        "fails_per_passer": args.fails_per_passer,
        "kept_total": kept_total,
        "items_with_pass": num_items,
        "total_passing_candidates": num_pass_total,
        "strict_per_candidate_used": False,
        "eval_pass@k": eval_data.get("pass@k", {}),  # pass@1, pass@5, pass@10 when present
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[ok] wrote {len(nbest_records)} items → {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="mbpp or leetcode")
    p.add_argument("--dataset_path", required=True, help="path to saved HF dataset dir (where candidates.jsonl lives)")
    p.add_argument("--out_dir", default="assets/rl/mbpp", help="where to write nbest/pairs")
    p.add_argument("--n_best", type=int, default=10)
    p.add_argument("--max_pairs", type=int, default=20)
    p.add_argument("--pairs_per_winner", type=int, default=3)
    p.add_argument("--candidates_path", type=str, help="Optional override for candidates.jsonl path")
    p.add_argument("--max_fails_per_task", type=int, default=5)
    p.add_argument("--within_pass_pairs_per_winner", type=int, default=1)
    p.add_argument("--fails_per_passer", type=int, default=2, help="Max number of fails to pair with each passer (caps pass>fail pairs)")
    args = p.parse_args()
    build(args)
