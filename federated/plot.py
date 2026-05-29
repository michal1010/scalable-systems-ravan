"""Visualization for Ravan federated fine-tuning experiments.

Called automatically at the end of every training run via:
    plot_single_run(run_name, history)   -- learning curve for this run
    plot_all()                           -- comparison figures from all runs

Outputs written to results/:

  <run_name>_curve.png
      Per-run accuracy vs. round.  Always generated.

  comparison_iid.png / comparison_noniid.png
      All three methods (FedIT, Ravan-GS, Ravan-SVD) on the same axes for
      one split.  Mean across seeds, shaded ±1 std when >1 seed exists.
      Directly answers RQ1 (FedIT vs Ravan) and RQ2 (GS vs SVD).

  iid_vs_noniid.png
      2-panel figure: left = I.I.D., right = Non-I.I.D., same methods on each.
      The key figure showing whether Ravan's advantage widens under heterogeneity.

  final_accuracy.png
      Grouped bar chart: x = method, two bars per method (IID vs non-IID),
      error bars = std across seeds.  Report-ready summary.

All comparison figures regenerate from scratch each time so they accumulate
results as more runs finish.  Missing method/split combinations are skipped
gracefully — single-run and partial results always produce valid output.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")   # works headless on cluster nodes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent / "results"

# ── visual identity ────────────────────────────────────────────────────────────

METHOD_ORDER = ["fedit", "ravan_gram_schmidt", "ravan_svd"]

METHOD_LABELS = {
    "fedit":             "FedIT",
    "ravan_gram_schmidt": "Ravan-GS",
    "ravan_svd":          "Ravan-SVD",
}
METHOD_COLORS = {
    "fedit":             "#e74c3c",   # red
    "ravan_gram_schmidt": "#27ae60",  # green
    "ravan_svd":          "#2980b9",  # blue
}
SPLIT_LABELS  = {"iid": "I.I.D.", "noniid": "Non-I.I.D."}
SPLIT_HATCHES = {"iid": "",        "noniid": "///"}
SPLIT_COLORS  = {"iid": "#5dade2", "noniid": "#e59866"}

plt.rcParams.update({
    "figure.dpi":     150,
    "font.size":      11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "lines.linewidth": 2.0,
    "axes.grid":      True,
    "grid.alpha":     0.3,
    "grid.linestyle": "--",
})


# ── internal data helpers ──────────────────────────────────────────────────────

def _load_summaries(results_dir: Path) -> pd.DataFrame:
    """Load every summary.json from per-run subdirectories into a DataFrame.

    Each run lives in results/<run_name>/summary.json.
    When multiple runs share the same (method, split, seed) — e.g. re-runs —
    only the most recent (alphabetically last run_name) is kept.
    """
    records = []
    for f in sorted(results_dir.glob("*/summary.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
            d["_run_dir"]    = str(f.parent)
            d["_rounds_path"] = str(f.parent / "rounds.csv")
            records.append(d)
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    keep = {"method", "split", "seed"}
    if keep.issubset(df.columns):
        df = (df.sort_values("run_name")
                .drop_duplicates(subset=list(keep), keep="last")
                .reset_index(drop=True))
    return df


def _load_histories(summaries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {run_name: rounds_df} for every run that has a rounds CSV."""
    out: dict[str, pd.DataFrame] = {}
    for _, row in summaries.iterrows():
        p = Path(row["_rounds_path"])
        if p.exists():
            out[row["run_name"]] = pd.read_csv(p)
    return out


def _curves_for(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    method: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (rounds, mean_acc, std_acc) for a (method, split) group, or None."""
    mask = (summaries["method"] == method) & (summaries["split"] == split)
    seed_series: list[np.ndarray] = []
    for _, row in summaries[mask].iterrows():
        if row["run_name"] in histories:
            seed_series.append(histories[row["run_name"]]["test_acc"].values)

    if not seed_series:
        return None

    min_len = min(len(s) for s in seed_series)
    arr     = np.array([s[:min_len] for s in seed_series])   # [n_seeds, n_rounds]
    return np.arange(1, min_len + 1), arr.mean(0), arr.std(0)


def _draw_method_curves(
    ax: plt.Axes,
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    split: str,
    methods: list[str] | None = None,
) -> bool:
    """Draw one line per method on ax.  Returns True if anything was drawn."""
    drawn = False
    for method in (methods or METHOD_ORDER):
        result = _curves_for(summaries, histories, method, split)
        if result is None:
            continue
        rounds, mean, std = result
        color  = METHOD_COLORS[method]
        label  = METHOD_LABELS[method]
        n_seeds = int(((summaries["method"] == method) & (summaries["split"] == split)).sum())

        ax.plot(rounds, mean, label=label, color=color)
        if n_seeds > 1:
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)
        drawn = True
    return drawn


# ── public API ─────────────────────────────────────────────────────────────────

def plot_single_run(
    run_name: str,
    history: list[dict],
    run_dir: Path | None = None,
) -> None:
    """Per-run learning curve — saved into the run's own subdirectory."""
    if run_dir is None:
        run_dir = RESULTS_DIR / run_name

    rounds = [r["round"]    for r in history]
    accs   = [r["test_acc"] for r in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rounds, accs, color="#2980b9", marker="o" if len(rounds) <= 10 else None)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(run_name.replace("_", " "))
    ax.set_xlim(left=max(0, rounds[0] - 0.5), right=rounds[-1] + 0.5)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylim(bottom=0, top=min(1.0, max(accs) * 1.15 + 0.05))

    path = run_dir / "curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Curve    → {path}")


def plot_all(results_dir: Path = RESULTS_DIR) -> None:
    """Regenerate all comparison figures from every completed run."""
    summaries = _load_summaries(results_dir)
    if summaries.empty or "method" not in summaries.columns:
        return

    histories = _load_histories(summaries)

    # Per-split learning curve comparison (answers RQ1 + RQ2)
    for split in ["iid", "noniid"]:
        _plot_learning_curves(summaries, histories, split, results_dir)

    # Side-by-side IID vs non-IID (the key heterogeneity figure)
    _plot_iid_vs_noniid(summaries, histories, results_dir)

    # Final accuracy bar chart (report summary)
    _plot_final_accuracy_bar(summaries, results_dir)


# ── individual figure generators ──────────────────────────────────────────────

def _plot_learning_curves(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    split: str,
    results_dir: Path,
) -> None:
    """All methods on one split — answers RQ1 and RQ2 directly."""
    fig, ax = plt.subplots(figsize=(9, 5))

    if not _draw_method_curves(ax, summaries, histories, split):
        plt.close(fig)
        return

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Method Comparison — {SPLIT_LABELS.get(split, split)} split")
    ax.legend(loc="lower right")

    # Note seed count
    n_seeds = summaries[summaries["split"] == split]["seed"].nunique()
    if n_seeds > 1:
        ax.annotate(
            f"mean ± std  ({n_seeds} seeds)",
            xy=(0.02, 0.97), xycoords="axes fraction",
            fontsize=9, color="gray", va="top",
        )

    path = results_dir / f"comparison_{split}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison ({split}) → {path}")


def _plot_iid_vs_noniid(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    results_dir: Path,
) -> None:
    """2-panel: IID left, Non-IID right, all methods on each panel.

    This is the key heterogeneity figure: it shows whether Ravan's advantage
    over FedIT widens as clients become more data-heterogeneous (RQ1), and
    whether SVD init helps more under non-IID data (RQ2).
    """
    available_splits = [s for s in ["iid", "noniid"] if s in summaries["split"].values]
    if len(available_splits) < 2:
        return   # only generate once both splits have at least one run

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, split in zip(axes, ["iid", "noniid"]):
        drawn = _draw_method_curves(ax, summaries, histories, split)
        ax.set_xlabel("Communication Round")
        ax.set_title(SPLIT_LABELS[split])
        if drawn:
            ax.legend(loc="lower right")

    axes[0].set_ylabel("Test Accuracy")
    fig.suptitle("FedIT vs Ravan: I.I.D. and Non-I.I.D. Comparison", fontsize=13)
    fig.tight_layout()

    path = results_dir / "iid_vs_noniid.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"IID vs non-IID → {path}")


def _plot_final_accuracy_bar(
    summaries: pd.DataFrame,
    results_dir: Path,
) -> None:
    """Grouped bar chart: x = method, bars = IID / Non-IID, error = std over seeds.

    Provides a report-ready single-figure summary of all results.
    Each bar is annotated with its mean accuracy value.
    """
    if "final_test_acc" not in summaries.columns:
        return

    present_methods = [m for m in METHOD_ORDER if m in summaries["method"].values]
    if len(present_methods) < 2:
        return

    present_splits = [s for s in ["iid", "noniid"] if s in summaries["split"].values]
    n_methods = len(present_methods)
    n_splits  = len(present_splits)

    x       = np.arange(n_methods)
    width   = 0.35
    offsets = np.linspace(-(n_splits - 1) * width / 2,
                           (n_splits - 1) * width / 2,
                           n_splits)

    fig, ax = plt.subplots(figsize=(9, 5))

    for offset, split in zip(offsets, present_splits):
        means, stds = [], []
        for method in present_methods:
            vals = summaries.loc[
                (summaries["method"] == method) & (summaries["split"] == split),
                "final_test_acc",
            ].values
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            stds.append( float(vals.std())  if len(vals) > 1 else 0.0)

        bars = ax.bar(
            x + offset, means, width,
            yerr=stds if any(e > 0 for e in stds) else None,
            capsize=4,
            label=SPLIT_LABELS.get(split, split),
            color=SPLIT_COLORS.get(split, "#aaa"),
            hatch=SPLIT_HATCHES.get(split, ""),
            alpha=0.85,
            error_kw={"elinewidth": 1.5},
        )

        # Annotate bar tops with accuracy value
        for bar, mean, std in zip(bars, means, stds):
            top = mean + std + 0.004 if std > 0 else mean + 0.004
            if mean > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    top,
                    f"{mean:.3f}",
                    ha="center", va="bottom", fontsize=8.5,
                )

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in present_methods])
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Final Test Accuracy by Method and Data Split\n"
                 "(error bars = std across seeds; see iid_vs_noniid.png for learning curves)")
    ax.legend(title="Split")
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    path = results_dir / "final_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Bar chart → {path}")
