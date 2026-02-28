import ast
import difflib
import hashlib
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def ast_key(code: str) -> str:
    """
    Stable hash for near-duplicate detection.
    Parses to AST, strips docstrings, dumps structure; falls back to whitespace-normalized text.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(
                node,
                (
                    ast.Module,
                    ast.ClassDef,
                    ast.FunctionDef,
                    getattr(ast, "AsyncFunctionDef", type("X", (), {})),
                ),
            ):
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(getattr(node.body[0], "value", None), ast.Str)
                ):
                    node.body = node.body[1:]
        norm = ast.dump(tree, include_attributes=False)
    except Exception:
        norm = re.sub(r"\s+", " ", code or "").strip()
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def ast_dump_normalized(code: str) -> str:
    """
    Produce a normalized AST dump for similarity comparison.
    """
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            # Remove docstrings
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
                if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Str):
                    node.body = node.body[1:]
        return ast.dump(tree, include_attributes=False)
    except Exception:
        # Fallback: normalized code text (rare)
        return re.sub(r"\s+", " ", code or "").strip()


def ast_similarity(code_a: str, code_b: str) -> float:
    """
    Returns similarity in [0,1], where 1 = identical AST.
    Based on SequenceMatcher over AST dumps.
    """
    dump_a = ast_dump_normalized(code_a)
    dump_b = ast_dump_normalized(code_b)
    return difflib.SequenceMatcher(None, dump_a, dump_b).ratio()


def novelty_score(code: str, others: list) -> float:
    """
    Novelty = 1 - average similarity to all other candidates.
    If alone, novelty = 1.0.
    """
    if not others:
        return 1.0

    sims = []
    for other in others:
        sim = ast_similarity(code, other)
        sims.append(sim)

    avg_sim = sum(sims) / len(sims)
    return 1.0 - avg_sim


def measure_execution_time(wrapped_code: str, timeout: float = 1.0) -> float:
    """
    Execute candidate solution and measure runtime.
    Timeout ⇒ return large penalty value.
    Returns time in seconds.
    """
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / f"time_{uuid.uuid4().hex}.py"
            temp_path.write_text(wrapped_code, encoding="utf-8")

            env = os.environ.copy()
            existing_py = env.get("PYTHONPATH", "")
            repo_src = str(SRC_DIR)
            env["PYTHONPATH"] = repo_src + (":" + existing_py if existing_py else "")

            start = time.perf_counter()

            proc = subprocess.run(
                ["python", str(temp_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
                env=env,
            )

            end = time.perf_counter()

            if proc.returncode != 0:
                # Failed execution gets high penalty
                return 10.0

            return round(end - start, 6)

    except subprocess.TimeoutExpired:
        return 10.0  # max penalty for timeout
    except Exception:
        return 10.0  # safety fallback


def cyclomatic_complexity(code: str) -> int:
    """
    Lightweight CC estimate: counts branch-like AST nodes present in this Python version.
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return 10**9
    names = ["If", "For", "While", "Try", "ExceptHandler", "BoolOp", "With", "Match"]
    types = tuple(getattr(ast, name) for name in names if hasattr(ast, name))
    return sum(isinstance(node, types) for node in ast.walk(tree))


def nesting_depth(code: str) -> int:
    try:
        tree = ast.parse(code)
    except Exception:
        return 10**9

    max_depth = 0

    def walk(node, depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, depth)
        for child in ast.iter_child_nodes(node):
            child_depth = depth + (
                1 if isinstance(child, (ast.FunctionDef, ast.For, ast.While, ast.If, ast.Try)) else 0
            )
            walk(child, child_depth)

    walk(tree, 0)
    return max_depth


def score_simple(code: str) -> Tuple[int, int, int, int]:
    """
    Primary = shorter is better (lines).
    Tie-breakers = tokens, cyclomatic complexity, max nesting.
    """
    loc = len([line for line in (code or "").splitlines() if line.strip()])
    tokens = len(re.findall(r"\w+|\S", code or ""))
    cc = cyclomatic_complexity(code)
    depth = nesting_depth(code)
    return loc, tokens, cc, depth
