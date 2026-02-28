from __future__ import annotations

from statistics import mean, median
from typing import Any, Dict, List

from .loader import Candidate


def compute_length_stats(task_candidates: List[Candidate]) -> Dict[str, Any]:
    lengths = [
        c.chars if c.chars is not None else len(c.code or "")
        for c in task_candidates
    ]
    if not lengths:
        return {
            "lengths": [],
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }

    return {
        "lengths": lengths,
        "min": min(lengths),
        "max": max(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
    }

