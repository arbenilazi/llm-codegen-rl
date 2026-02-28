from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save_histogram(lengths: List[int], title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.hist(lengths, bins=20, color="steelblue", edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("Code length (chars)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_length_histogram(task_id: int, lengths: List[int], output_dir: Path) -> Path:
    path = Path(output_dir) / f"{task_id}_length_hist.png"
    _save_histogram(lengths, f"Task {task_id} length distribution", path)
    return path


def plot_global_length_histogram(lengths: Iterable[int], output_dir: Path) -> Path:
    lengths_list = list(lengths)
    path = Path(output_dir) / "length_hist_global.png"
    _save_histogram(lengths_list, "Global length distribution", path)
    return path

def plot_length_scatter(task_ids: List[int], lengths: List[int], output_dir: Path) -> Path:
    """
    Creates a scatter plot where each point is one candidate,
    x = task_id, y = code length.
    """
    path = Path(output_dir) / "length_scatter_global.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(task_ids, lengths, alpha=0.4, s=12)
    plt.xlabel("Task ID")
    plt.ylabel("Code length (chars)")
    plt.title("Global scatter of code lengths across tasks")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return path
