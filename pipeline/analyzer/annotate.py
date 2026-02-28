import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Data loading and setup

def load_data(path: str) -> pd.DataFrame:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            obj = json.loads(line)
            m = obj.get("metrics") or {}
            records.append(
                {
                    "id": obj.get("id"),
                    "candidate_idx": obj.get("candidate_idx"),
                    "passed": bool(obj.get("passed", False)),
                    "loc": m.get("loc"),
                    "tokens": m.get("tokens"),
                    "cyclomatic_complexity": m.get("cyclomatic_complexity"),
                    "nesting_depth": m.get("nesting_depth"),
                    "execution_time": m.get("execution_time"),
                    "novelty": m.get("novelty"),
                    "ast_hash": m.get("ast_hash"),
                }
            )
    df = pd.DataFrame(records)
    num_cols = ["loc", "tokens", "cyclomatic_complexity", "nesting_depth", "execution_time", "novelty"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df["passed_int"] = df["passed"].astype(int)
    df = add_simplicity_score(df)
    return df


def add_simplicity_score(df: pd.DataFrame) -> pd.DataFrame:
    norm_cols = ["loc", "tokens", "cyclomatic_complexity", "nesting_depth"]
    for col in norm_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        if pd.isna(col_min) or pd.isna(col_max) or col_max == col_min:
            df[f"{col}_norm"] = 0.0
        else:
            df[f"{col}_norm"] = (df[col] - col_min) / (col_max - col_min)
    df["simplicity_score"] = df[[f"{c}_norm" for c in norm_cols]].sum(axis=1)
    return df


def ensure_out_dir(out_dir: str) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Statistics

def compute_global_stats(df: pd.DataFrame) -> Dict:
    pass_per_task = df.groupby("id")["passed_int"].sum()
    cand_per_task = df.groupby("id")["passed_int"].count()
    pass_rate_per_task = pass_per_task / cand_per_task.replace(0, np.nan)

    novelty_vals = df["novelty"].dropna()
    exec_vals = df["execution_time"].dropna()
    timeouts = (df["execution_time"] > 1.0).sum()
    exec_no_timeout = df.loc[df["execution_time"] < 10.0, "execution_time"]

    stats = {
        "num_tasks": int(df["id"].nunique()),
        "total_candidates": int(len(df)),
        "global_pass_rate": float(df["passed_int"].mean()),
        "total_passing": int(df["passed_int"].sum()),
        "avg_passing_per_task": float(pass_per_task.mean()),
        "pass_rate_mean": float(pass_rate_per_task.mean()),
        "pass_rate_median": float(pass_rate_per_task.median()),
        "pass_rate_std": float(pass_rate_per_task.std()),
        "novelty_mean": float(novelty_vals.mean()) if not novelty_vals.empty else float("nan"),
        "novelty_median": float(novelty_vals.median()) if not novelty_vals.empty else float("nan"),
        "novelty_p1": float(novelty_vals.quantile(0.01)) if not novelty_vals.empty else float("nan"),
        "novelty_p99": float(novelty_vals.quantile(0.99)) if not novelty_vals.empty else float("nan"),
        "execution_mean": float(exec_no_timeout.mean()) if not exec_no_timeout.empty else float("nan"),
        "execution_median": float(exec_no_timeout.median()) if not exec_no_timeout.empty else float("nan"),
        "execution_std": float(exec_no_timeout.std()) if not exec_no_timeout.empty else float("nan"),
        "timeouts_gt1s": int(timeouts),
    }

    uniq_hashes = df.groupby("id")["ast_hash"].nunique()
    diversity_ratio = uniq_hashes / cand_per_task.replace(0, np.nan)
    stats["diversity_mean"] = float(diversity_ratio.mean())
    stats["diversity_median"] = float(diversity_ratio.median())
    stats["diversity_std"] = float(diversity_ratio.std())

    corr_cols = ["passed_int", "novelty", "execution_time", "loc", "tokens", "cyclomatic_complexity", "nesting_depth", "simplicity_score"]
    corr_df = df[corr_cols].dropna()
    pearson = corr_df.corr(method="pearson") if not corr_df.empty else pd.DataFrame()
    spearman = corr_df.corr(method="spearman") if not corr_df.empty else pd.DataFrame()
    stats["correlation_matrix"] = pearson
    stats["spearman_matrix"] = spearman
    stats["pearson_novelty_pass"] = float(pearson.loc["novelty", "passed_int"]) if "novelty" in pearson and "passed_int" in pearson else float("nan")
    stats["spearman_novelty_pass"] = float(spearman.loc["novelty", "passed_int"]) if "novelty" in spearman and "passed_int" in spearman else float("nan")

    return stats


def compute_task_level(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("id")
    timeout_rate = grouped.apply(lambda g: (g["execution_time"] > 1.0).mean())
    summary = pd.DataFrame(
        {
            "best_loc": grouped["loc"].min(),
            "best_novelty": grouped["novelty"].max(),
            "novelty_range": grouped["novelty"].max() - grouped["novelty"].min(),
            "num_pass": grouped["passed_int"].sum(),
            "total_candidates": grouped["passed_int"].count(),
            "timeout_rate": timeout_rate,
        }
    )
    summary["pass_rate"] = summary["num_pass"] / summary["total_candidates"].replace(0, np.nan)
    return summary.reset_index()


def task_level_stats_summary(task_df: pd.DataFrame) -> Dict:
    metrics = ["best_loc", "best_novelty", "novelty_range", "num_pass", "timeout_rate", "pass_rate"]
    summary: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        series = task_df[m].dropna()
        summary[m] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
        }
    return summary


def select_case_studies(task_df: pd.DataFrame) -> pd.DataFrame:
    picks: List[int] = []

    # 2 tasks where all candidates fail
    fail_tasks = task_df[task_df["num_pass"] == 0].sort_values("id").head(2)["id"].tolist()
    picks.extend(fail_tasks)

    # 2 tasks with high pass rates
    high_pass = task_df.sort_values("pass_rate", ascending=False).head(4)["id"].tolist()
    picks.extend([i for i in high_pass if i not in picks][:2])

    # 2 tasks with highest novelty range
    high_novelty_span = task_df.sort_values("novelty_range", ascending=False).head(4)["id"].tolist()
    picks.extend([i for i in high_novelty_span if i not in picks][:2])

    # 2 tasks with most execution timeouts
    high_timeouts = task_df.sort_values("timeout_rate", ascending=False).head(4)["id"].tolist()
    picks.extend([i for i in high_timeouts if i not in picks][:2])

    picks = list(dict.fromkeys(picks))[:10]
    return task_df[task_df["id"].isin(picks)]


# Plotting helpers

PALETTE = {
    "primary": "#4c72b0",
    "secondary": "#55a868",
    "tertiary": "#c44e52",
    "quaternary": "#8172b3",
    "quinary": "#64b5cd",
    "senary": "#dd8452",
}


def _kde(values: np.ndarray, grid_size: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    if values.size == 0:
        return np.array([]), np.array([])
    vmin, vmax = values.min(), values.max()
    if vmin == vmax:
        return np.array([vmin, vmax]), np.array([1.0, 1.0])
    grid = np.linspace(vmin, vmax, grid_size)
    bw = np.std(values) * (len(values) ** (-1 / 5)) if len(values) > 1 else 0.1
    bw = max(float(bw), 1e-3)
    diff = (grid[:, None] - values[None, :]) / bw
    density = np.exp(-0.5 * diff**2).sum(axis=1)
    density = density / (len(values) * bw * np.sqrt(2 * np.pi))
    return grid, density


def plot_global(df: pd.DataFrame, task_df: pd.DataFrame, out_dir: Path) -> None:
    plt.style.use("seaborn-v0_8")
    out_dir.mkdir(parents=True, exist_ok=True)

    pass_per_task = task_df["num_pass"]
    pass_rate_per_task = task_df["pass_rate"].fillna(0)

    # Correctness
    plt.figure(figsize=(8, 5))
    plt.hist(pass_per_task, bins=20, color=PALETTE["primary"], edgecolor="black")
    plt.title("Passing Candidates per Task")
    plt.xlabel("Number of passing candidates")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(out_dir / "correctness_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(pass_rate_per_task, bins=20, color=PALETTE["secondary"], edgecolor="black", range=(0, 1))
    plt.title("Pass Rate per Task")
    plt.xlabel("Pass rate")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(out_dir / "passrate_histogram.png", dpi=200)
    plt.close()

    # Novelty
    nov = df["novelty"].dropna().values
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(nov, bins=25, density=True, color=PALETTE["quaternary"], edgecolor="black", alpha=0.7)
    grid, dens = _kde(nov)
    if grid.size:
        ax[0].plot(grid, dens, color="black", linewidth=1.5, label="KDE")
        ax[0].legend()
    ax[0].set_title("Novelty Distribution")
    ax[0].set_xlabel("Novelty")
    ax[0].set_ylabel("Density")

    scatter = ax[1].scatter(df["loc"], df["novelty"], c=df["passed_int"], cmap="coolwarm", alpha=0.6)
    ax[1].set_title("Novelty vs LOC")
    ax[1].set_xlabel("LOC")
    ax[1].set_ylabel("Novelty")
    cbar = fig.colorbar(scatter, ax=ax[1], label="Passed")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["False", "True"])
    plt.tight_layout()
    plt.savefig(out_dir / "novelty_distribution.png", dpi=200)
    plt.close(fig)

    # Execution time
    exec_vals = df["execution_time"].dropna().values
    plt.figure(figsize=(8, 5))
    plt.hist(exec_vals, bins=30, color=PALETTE["quinary"], edgecolor="black", alpha=0.8)
    plt.title("Execution Time Distribution")
    plt.xlabel("Execution time (s)")
    plt.ylabel("Candidates")
    plt.axvline(1.0, color="red", linestyle="--", label="Timeout bucket (>1s)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exec_time_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    passed_times = df.loc[df["passed"], "execution_time"].dropna()
    failed_times = df.loc[~df["passed"], "execution_time"].dropna()
    bins = np.linspace(0, max(exec_vals.max() if exec_vals.size else 1.0, 1.0), 30)
    plt.hist(passed_times, bins=bins, alpha=0.6, label="Passed", color=PALETTE["secondary"], edgecolor="black")
    plt.hist(failed_times, bins=bins, alpha=0.6, label="Failed", color=PALETTE["tertiary"], edgecolor="black")
    plt.axvline(1.0, color="red", linestyle="--", linewidth=1.0)
    plt.title("Execution Time: Passed vs Failed")
    plt.xlabel("Execution time (s)")
    plt.ylabel("Candidates")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "exec_time_passfail.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df["loc"], df["execution_time"], c=df["passed_int"], cmap="coolwarm", alpha=0.6)
    plt.xlabel("LOC")
    plt.ylabel("Execution time (s)")
    plt.title("Execution Time vs LOC")
    cbar = plt.colorbar(scatter, label="Passed")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["False", "True"])
    plt.tight_layout()
    plt.savefig(out_dir / "exec_time_vs_loc.png", dpi=200)
    plt.close()

    # Simplicity metrics
    plt.figure(figsize=(10, 5))
    plt.boxplot(
        [df["loc"], df["tokens"], df["cyclomatic_complexity"], df["nesting_depth"]],
        labels=["LOC", "Tokens", "CC", "Depth"],
        showfliers=False,
    )
    plt.title("Simplicity Metrics (All Candidates)")
    plt.tight_layout()
    plt.savefig(out_dir / "simplicity_boxplot.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.boxplot(
        [
            df.loc[df["passed"], "loc"],
            df.loc[df["passed"], "tokens"],
            df.loc[df["passed"], "cyclomatic_complexity"],
            df.loc[df["passed"], "nesting_depth"],
        ],
        labels=["LOC", "Tokens", "CC", "Depth"],
        showfliers=False,
    )
    plt.title("Simplicity Metrics (Passing Candidates)")
    plt.tight_layout()
    plt.savefig(out_dir / "simplicity_boxplot_passing.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(df["simplicity_score"].dropna(), bins=25, color=PALETTE["senary"], edgecolor="black", alpha=0.8)
    plt.title("Simplicity Score Distribution")
    plt.xlabel("Simplicity score (lower = simpler)")
    plt.ylabel("Candidates")
    plt.tight_layout()
    plt.savefig(out_dir / "simplicity_score_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df["simplicity_score"], df["novelty"], c=df["passed_int"], cmap="coolwarm", alpha=0.6)
    plt.xlabel("Simplicity score")
    plt.ylabel("Novelty")
    plt.title("Novelty vs Simplicity")
    cbar = plt.colorbar(scatter, label="Passed")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["False", "True"])
    plt.tight_layout()
    plt.savefig(out_dir / "novelty_vs_simplicity.png", dpi=200)
    plt.close()

    # Diversity
    diversity_ratio = df.groupby("id")["ast_hash"].nunique().values / task_df["total_candidates"].replace(0, np.nan).values
    plt.figure(figsize=(8, 5))
    plt.hist(pd.Series(diversity_ratio).dropna().values, bins=20, color=PALETTE["quinary"], edgecolor="black")
    plt.title("Diversity Ratio per Task")
    plt.xlabel("Diversity ratio (#unique / #total)")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(out_dir / "diversity_distribution.png", dpi=200)
    plt.close()


def plot_task_level(task_df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["best_loc", "best_novelty", "novelty_range", "num_pass", "timeout_rate"]
    titles = [
        "Best LOC per Task",
        "Best Novelty per Task",
        "Novelty Range per Task",
        "Passing Candidates per Task",
        "Timeout Rate per Task",
    ]
    for metric, title in zip(metrics, titles):
        plt.figure(figsize=(8, 5))
        plt.hist(task_df[metric].dropna(), bins=25, color=PALETTE["primary"], edgecolor="black")
        plt.title(title)
        plt.xlabel(metric.replace("_", " ").title())
        plt.ylabel("Tasks")
        plt.tight_layout()
        plt.savefig(out_dir / f"task_{metric}_hist.png", dpi=200)
        plt.close()


def plot_ranking_analysis(df: pd.DataFrame, out_dir: Path) -> None:
    grouped = df.groupby("id")
    top1_records = []
    rank_frames: List[pd.DataFrame] = []

    for task_id, group in grouped:
        grp = group.copy()
        grp["simplicity_rank"] = grp["simplicity_score"].rank(method="dense")
        grp["loc_rank"] = grp["loc"].rank(method="dense")
        grp["novelty_rank"] = grp["novelty"].rank(method="dense", ascending=False)
        grp["exec_time_rank"] = grp["execution_time"].rank(method="dense")

        top = grp.sort_values(["simplicity_rank", "novelty_rank"]).iloc[0]
        top1_records.append(
            {
                "id": task_id,
                "simplicity_score": top["simplicity_score"],
                "novelty": top["novelty"],
                "execution_time": top["execution_time"],
            }
        )
        rank_frames.append(grp[["loc_rank", "novelty_rank", "exec_time_rank", "simplicity_rank"]])

    top1_df = pd.DataFrame(top1_records)
    plt.figure(figsize=(8, 5))
    plt.hist(top1_df["simplicity_score"].dropna(), bins=20, color=PALETTE["secondary"], edgecolor="black")
    plt.title("Simplicity of Top-1 (per Task)")
    plt.xlabel("Simplicity score")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_simplicity_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(top1_df["novelty"].dropna(), bins=20, color=PALETTE["quaternary"], edgecolor="black")
    plt.title("Novelty of Top-1 (per Task)")
    plt.xlabel("Novelty")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_novelty_hist.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(top1_df["execution_time"].dropna(), bins=20, color=PALETTE["quinary"], edgecolor="black")
    plt.title("Execution Time of Top-1 (per Task)")
    plt.xlabel("Execution time (s)")
    plt.ylabel("Tasks")
    plt.tight_layout()
    plt.savefig(out_dir / "top1_exec_time_hist.png", dpi=200)
    plt.close()

    rank_df = pd.concat(rank_frames, axis=0, ignore_index=True).dropna()
    if not rank_df.empty:
        corr = rank_df.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.index)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.index)
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman")
        ax.set_title("Metric Overlap Heatmap (Ranks)")
        fig.tight_layout()
        fig.savefig(out_dir / "ranking_overlap_heatmap.png", dpi=200)
        plt.close(fig)


# Reporting

def print_summary(stats: Dict) -> None:
    print("=== Annotated MBPP Candidate Analysis ===")
    print(f"Tasks: {stats['num_tasks']}")
    print(f"Total candidates: {stats['total_candidates']}")
    print(f"Total passing: {stats['total_passing']} (global pass rate: {stats['global_pass_rate']:.3f})")
    print(f"Avg passing candidates per task: {stats['avg_passing_per_task']:.3f}")
    print(
        f"Pass rate per task — mean: {stats['pass_rate_mean']:.3f}, median: {stats['pass_rate_median']:.3f}, std: {stats['pass_rate_std']:.3f}"
    )
    print(
        f"Novelty — mean: {stats['novelty_mean']:.3f}, median: {stats['novelty_median']:.3f}, p1: {stats['novelty_p1']:.3f}, p99: {stats['novelty_p99']:.3f}"
    )
    print(
        f"Execution time (<10s) — mean: {stats['execution_mean']:.3f}s, median: {stats['execution_median']:.3f}s, std: {stats['execution_std']:.3f}s; timeouts (>1s): {stats['timeouts_gt1s']}"
    )
    print(f"Diversity ratio — mean: {stats['diversity_mean']:.3f}, median: {stats['diversity_median']:.3f}")
    print(
        f"Novelty vs correctness — Pearson: {stats['pearson_novelty_pass']:.3f}, Spearman: {stats['spearman_novelty_pass']:.3f}"
    )
    corr = stats.get("correlation_matrix")
    if isinstance(corr, pd.DataFrame) and not corr.empty:
        print("Correlation matrix (Pearson):")
        print(corr.round(3))
    else:
        print("Correlation matrix unavailable (insufficient data).")


def print_task_level_summary(task_stats: Dict) -> None:
    print("=== Task-Level Metric Summary ===")
    for metric, vals in task_stats.items():
        print(
            f"{metric}: mean={vals['mean']:.3f}, median={vals['median']:.3f}, std={vals['std']:.3f}"
        )


def save_summary_json(stats: Dict, out_dir: Path) -> None:
    serializable = {
        k: (
            v.to_dict() if isinstance(v, pd.DataFrame) else v
        )
        for k, v in stats.items()
    }
    with open(out_dir / "annotate_summary.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)


def save_case_studies(task_df: pd.DataFrame, out_dir: Path) -> None:
    case_df = select_case_studies(task_df)
    case_df.to_csv(out_dir / "task_case_studies.csv", index=False)


# Main entry point

def analyze_candidates(path: str, out_dir: str = "results/annotate") -> None:
    out_path = ensure_out_dir(out_dir)
    df = load_data(path)
    if df.empty:
        print("[warn] No candidates loaded; exiting.")
        return

    task_df = compute_task_level(df)
    stats = compute_global_stats(df)
    stats["task_level_summary"] = task_level_stats_summary(task_df)

    print_summary(stats)
    print_task_level_summary(stats["task_level_summary"])
    save_summary_json(stats, out_path)
    save_case_studies(task_df, out_path)
    plot_global(df, task_df, out_path)
    plot_task_level(task_df, out_path)
    plot_ranking_analysis(df, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="Path to train/candidates_annotated.jsonl")
    parser.add_argument("--out_dir", default="results/annotate", help="Directory to store analysis outputs")
    args = parser.parse_args()
    analyze_candidates(args.path, args.out_dir)
