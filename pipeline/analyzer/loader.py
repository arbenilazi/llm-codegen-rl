from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Candidate:
    id: int
    candidate_idx: int
    code: str
    chars: int


def load_candidates(path: Path | str) -> Dict[int, List[Candidate]]:
    """
    Load baseline candidates, keeping only a few stable fields.
    Returns a mapping of task id -> list of candidates.
    """
    path = Path(path)
    tasks: Dict[int, List[Candidate]] = {}
    if not path.exists():
        raise FileNotFoundError(f"Candidates file not found at {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue

            task_id = raw.get("id")
            candidate_idx = raw.get("candidate_idx")
            code = raw.get("code", "")
            chars = raw.get("chars")

            if task_id is None or candidate_idx is None:
                continue

            if chars is None:
                try:
                    chars = len(code)
                except Exception:
                    chars = 0

            candidate = Candidate(
                id=int(task_id),
                candidate_idx=int(candidate_idx),
                code=str(code),
                chars=int(chars),
            )
            tasks.setdefault(candidate.id, []).append(candidate)

    return tasks

