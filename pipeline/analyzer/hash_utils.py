from __future__ import annotations

import ast
import hashlib
from typing import Any, Dict, List

from .loader import Candidate


def ast_hash(code: str) -> str:
    """
    Compute a stable hash of the AST; return 'invalid' on parse failures.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "invalid"

    dumped = ast.dump(tree, include_attributes=False)
    digest = hashlib.sha256(dumped.encode("utf-8")).hexdigest()
    return digest


def hash_similarity_per_task(task_candidates: List[Candidate]) -> Dict[str, Any]:
    hashes = [ast_hash(c.code) for c in task_candidates]
    total = len(hashes)
    unique = len(set(hashes))

    hash_counts: Dict[str, int] = {}
    for h in hashes:
        hash_counts[h] = hash_counts.get(h, 0) + 1

    cluster_sizes = sorted(hash_counts.values(), reverse=True)
    ratio = (unique / total) if total else 0.0

    return {
        "total_candidates": total,
        "unique_hashes": unique,
        "unique_ratio": ratio,
        "cluster_sizes": cluster_sizes,
        "hash_counts": hash_counts,
    }

