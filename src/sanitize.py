import ast
import re
from typing import Optional

__all__ = ["normalize_entry_point", "_syntactic_ok"]


def _syntactic_ok(s: str) -> bool:
    try:
        ast.parse(s)
        return True
    except Exception:
        return False


def _extract_first_def_name(code: str) -> Optional[str]:
    m = re.search(r"^\s*def\s+([A-Za-z]\w*)\s*\(", code or "", flags=re.M)
    return m.group(1) if m else None


def normalize_entry_point(code: str, target: str) -> str:
    """
    Make the first function definition use the target entry-point name and
    fix recursive/self-calls. Idempotent and conservative.

    - Strips stray code fences/backticks.
    - Keeps everything else unchanged.
    """
    if not code:
        return code

    s = str(code).replace("```", "").strip()
    if s.lower().startswith("python\n"):
        s = s.split("\n", 1)[1].lstrip()

    current = _extract_first_def_name(s)
    if not current or current == target:
        return s

    s = re.sub(
        rf"^(\s*def\s+){re.escape(current)}(\s*\()",
        rf"\1{target}\2",
        s,
        count=1,
        flags=re.M,
    )

    s = re.sub(
        rf"(?<![A-Za-z0-9_]){re.escape(current)}\s*\(",
        f"{target}(",
        s,
    )
    return s
