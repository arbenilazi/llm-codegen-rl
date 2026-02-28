from __future__ import annotations

import difflib
from typing import List, Optional, Tuple

import numpy as np

from .embeddings import pairwise_distances
from .loader import Candidate


def diff_reference_vs_others(task_candidates: List[Candidate]) -> str:
    """
    Unified diffs comparing the reference candidate (idx 0) to all others.
    """
    if not task_candidates:
        return ""

    reference = next((c for c in task_candidates if c.candidate_idx == 0), None)
    if reference is None:
        return "# Reference candidate (idx 0) not found."

    ref_lines = reference.code.splitlines(keepends=True)
    chunks = []
    for candidate in task_candidates:
        if candidate is reference:
            continue
        other_lines = candidate.code.splitlines(keepends=True)
        diff_lines = difflib.unified_diff(
            ref_lines,
            other_lines,
            fromfile=f"task_{reference.id}_ref",
            tofile=f"task_{reference.id}_cand_{candidate.candidate_idx}",
            lineterm="",
        )
        result = "\n".join(diff_lines)
        if result:
            chunks.append(result)
        else:
            chunks.append(
                f"# Candidate {candidate.candidate_idx} identical to reference for task {reference.id}."
            )
    if not chunks:
        return f"# No comparison candidates for task {reference.id}."
    return "\n\n".join(chunks)


def extremal_pairs(embeddings: np.ndarray) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """
    Return (min_pair, max_pair) indices for smallest/largest distances.
    """
    if embeddings.shape[0] < 2:
        return None, None

    distances = pairwise_distances(embeddings)
    np.fill_diagonal(distances, np.nan)

    min_idx = int(np.nanargmin(distances))
    max_idx = int(np.nanargmax(distances))
    n = distances.shape[0]

    min_pair = divmod(min_idx, n)
    max_pair = divmod(max_idx, n)
    return (int(min_pair[0]), int(min_pair[1])), (int(max_pair[0]), int(max_pair[1]))
