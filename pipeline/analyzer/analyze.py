from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

from .diff_utils import diff_reference_vs_others, extremal_pairs
from .embeddings import compute_embeddings, pairwise_distances
from .hash_utils import hash_similarity_per_task
from .loader import load_candidates
from .plots import plot_global_length_histogram, plot_length_histogram, plot_length_scatter
from .stats import compute_length_stats

REPO_ROOT = Path(__file__).resolve().parents[2]


def _distance_summary(matrix: np.ndarray) -> Dict[str, Any]:
    if matrix.size == 0 or matrix.shape[0] < 2:
        return {"min": None, "max": None, "mean": None, "median": None}
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    values = matrix[mask]
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
    }


def _vector_distance_summary(distances: Iterable[float]) -> Dict[str, Any]:
    values = list(distances)
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None, "count": 0}
    arr = np.array(values, dtype=float)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "count": int(arr.size),
    }


def _flatten_distances(matrix: np.ndarray) -> List[float]:
    if matrix.size == 0 or matrix.shape[0] < 2:
        return []
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask].ravel().tolist()


def _global_length_stats(lengths: List[int]) -> Dict[str, Any]:
    if not lengths:
        return {"lengths": [], "min": 0, "max": 0, "mean": 0.0, "median": 0.0}
    arr = np.array(lengths, dtype=float)
    return {
        "lengths": lengths,
        "min": int(arr.min()),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
    }


def _task_output_dir(cfg: Dict[str, Any]) -> Path:
    reports_dir = cfg.get("reports", {}).get("dir", "results/reports")
    return (REPO_ROOT / reports_dir).resolve()


def run_analysis(cfg: Dict[str, Any]) -> None:
    dataset_cfg = cfg["dataset"]
    experiment_name = cfg["experiment_name"]
    dataset_root = (REPO_ROOT / dataset_cfg["output_dir"]).resolve()
    test_dir = dataset_root / "test"
    candidates_path = test_dir / "candidates_baseline.jsonl"

    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Baseline candidates not found at {candidates_path}. "
            "Run the baseline stage before analysis."
        )

    candidates_by_task = load_candidates(candidates_path)

    analysis_root = _task_output_dir(cfg) / f"{experiment_name}_analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    per_task_summary: Dict[str, Any] = {}
    all_lengths: List[int] = []
    all_task_ids_for_lengths: List[int] = []
    all_pairwise_distances: List[float] = []

    for task_id in sorted(candidates_by_task.keys()):
        task_candidates = candidates_by_task[task_id]
        codes = [c.code for c in task_candidates]

        embeddings = compute_embeddings(codes)
        distance_matrix = pairwise_distances(embeddings)
        distance_stats = _distance_summary(distance_matrix)
        all_pairwise_distances.extend(_flatten_distances(distance_matrix))

        min_pair, max_pair = extremal_pairs(embeddings)

        diff_text = diff_reference_vs_others(task_candidates)
        diff_path = analysis_root / f"{task_id}_diff.txt"
        diff_path.write_text(diff_text, encoding="utf-8")

        length_stats = compute_length_stats(task_candidates)
        all_lengths.extend(length_stats["lengths"])
        all_task_ids_for_lengths.extend([task_id] * len(task_candidates))
        length_plot_path = None

        hash_stats = hash_similarity_per_task(task_candidates)

        per_task_summary[str(task_id)] = {
            "num_candidates": len(task_candidates),
            "length_stats": {
                "lengths": length_stats["lengths"],
                "min": length_stats["min"],
                "max": length_stats["max"],
                "mean": length_stats["mean"],
                "median": length_stats["median"],
            },
            "embedding_distances": distance_stats,
            "extremal_pairs": {
                "min_pair": list(min_pair) if min_pair else None,
                "max_pair": list(max_pair) if max_pair else None,
            },
            "hash_similarity": hash_stats,
            "diff_file": diff_path.name,
            "length_histogram":None,
        }

    plot_global_length_histogram(all_lengths, analysis_root)
    scatter_path = plot_length_scatter(all_task_ids_for_lengths, all_lengths, analysis_root)

    summary = {
        "experiment": experiment_name,
        "dataset": dataset_cfg.get("name"),
        "candidates_path": str(candidates_path),
        "output_dir": str(analysis_root),
        "tasks_total": len(candidates_by_task),
        "total_candidates": sum(len(v) for v in candidates_by_task.values()),
        "global": {
            "length_stats": _global_length_stats(all_lengths),
            "embedding_distances": _vector_distance_summary(all_pairwise_distances),
            "length_scatter_plot": scatter_path.name,
        },
        "per_task": per_task_summary,
    }

    summary_path = analysis_root / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
