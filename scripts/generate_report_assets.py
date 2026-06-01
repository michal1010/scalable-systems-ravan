"""Generate all paper-ready tables and figures from completed experiment runs.

Usage:
    python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets

This script reads:
  - results/all_results.csv   (one row per run, written by training scripts)
  - results/*/summary.json    (per-run summaries, as fallback)
  - results/*/rounds.csv      (per-round accuracy curves)
  - results/*/singular_values.csv  (optional, for SVD spectral plots)

It regenerates all tables and figures from scratch, so it can be re-run any
time as more runs finish.  Missing method/split combinations are gracefully
skipped.

Outputs written to paper_assets/:
  tables/
    main_results.csv / .tex
    init_costs.csv   / .tex
    setup_summary.csv
    parameter_budget.csv
    gap_analysis.csv
  figures/
    learning_curves_iid.png
    learning_curves_noniid.png
    final_accuracy_by_method_split.png
    iid_vs_noniid_drop.png
    gap_analysis.png
    init_cost_vs_accuracy.png
    singular_values.png  (only if singular value data exists)
  report_assets_summary.md
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# ── visual identity ────────────────────────────────────────────────────────────

METHOD_KEY_ORDER = ["fedit", "ravan_gram_schmidt", "ravan_svd"]
METHOD_DISPLAY = {
    "fedit":              "FedIT",
    "ravan_gram_schmidt": "Ravan-GS",
    "ravan_svd":          "Ravan-SVD",
}
METHOD_COLORS = {
    "fedit":              "#e74c3c",
    "ravan_gram_schmidt": "#27ae60",
    "ravan_svd":          "#2980b9",
}
SPLIT_DISPLAY  = {"iid": "I.I.D.", "noniid": "Non-I.I.D."}

plt.rcParams.update({
    "figure.dpi":      300,
    "font.size":       10,
    "axes.titlesize":  11,
    "axes.labelsize":  10,
    "legend.fontsize": 9,
    "lines.linewidth": 1.8,
    "axes.grid":       True,
    "grid.alpha":      0.3,
    "grid.linestyle":  "--",
})

# ── data loading ───────────────────────────────────────────────────────────────

def load_summaries(results_dir: Path) -> pd.DataFrame:
    """Load summaries from all_results.csv (preferred) plus any summary.json files."""
    records: list[dict] = []

    master = results_dir / "all_results.csv"
    if master.exists():
        try:
            df = pd.read_csv(master)
            records.extend(df.to_dict("records"))
        except Exception as e:
            print(f"Warning: could not read {master}: {e}")

    # Also scan per-run summary.json (fills gaps if master CSV is incomplete)
    seen_runs = {r.get("run_name") for r in records if r.get("run_name")}
    for f in sorted(results_dir.glob("*/summary.json")):
        try:
            with open(f) as fp:
                d = json.load(fp)
            if d.get("run_name") not in seen_runs:
                records.append(d)
                seen_runs.add(d.get("run_name"))
        except Exception:
            pass

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Normalise method names (old runs may have used different conventions)
    if "method" in df.columns:
        df["method"] = df["method"].str.replace("ravan_gs", "ravan_gram_schmidt", regex=False)

    # Keep only most-recent run per (method, split, seed)
    key_cols = ["method", "split", "seed"]
    if all(c in df.columns for c in key_cols):
        df = (df.sort_values("run_name", na_position="first")
                .drop_duplicates(subset=key_cols, keep="last")
                .reset_index(drop=True))

    # Unify final accuracy column name
    if "final_acc" not in df.columns and "final_test_acc" in df.columns:
        df["final_acc"] = df["final_test_acc"]

    return df


def load_histories(results_dir: Path, summaries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return {run_name: rounds_df} for every run that has a rounds.csv."""
    out: dict[str, pd.DataFrame] = {}
    for _, row in summaries.iterrows():
        run_name = row.get("run_name")
        if not run_name:
            continue
        p = results_dir / str(run_name) / "rounds.csv"
        if p.exists():
            try:
                out[str(run_name)] = pd.read_csv(p)
            except Exception:
                pass
    return out


def load_singular_values(results_dir: Path, summaries: pd.DataFrame) -> pd.DataFrame:
    """Collect all singular_values.csv from SVD warm-up runs."""
    frames = []
    for _, row in summaries.iterrows():
        if str(row.get("method", "")) != "ravan_svd":
            continue
        run_name = row.get("run_name")
        if not run_name:
            continue
        p = results_dir / str(run_name) / "singular_values.csv"
        if p.exists():
            try:
                df = pd.read_csv(p)
                df["run_name"] = run_name
                df["split"]    = row.get("split", "?")
                df["seed"]     = row.get("seed", -1)
                frames.append(df)
            except Exception:
                pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── helpers ────────────────────────────────────────────────────────────────────

def _method_split_stats(df: pd.DataFrame) -> dict[tuple, dict]:
    """Return {(method, split): {mean, std, n}} for final_acc."""
    out = {}
    for method in METHOD_KEY_ORDER:
        for split in ["iid", "noniid"]:
            vals = df.loc[(df["method"] == method) & (df["split"] == split), "final_acc"].dropna()
            if len(vals) > 0:
                out[(method, split)] = {
                    "mean": float(vals.mean()),
                    "std":  float(vals.std()) if len(vals) > 1 else 0.0,
                    "n":    len(vals),
                }
    return out


def _curves_for(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    method: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    mask = (summaries["method"] == method) & (summaries["split"] == split)
    series = []
    for _, row in summaries[mask].iterrows():
        run_name = str(row.get("run_name", ""))
        if run_name in histories:
            series.append(histories[run_name]["test_acc"].dropna().values)
    if not series:
        return None
    min_len = min(len(s) for s in series)
    arr = np.array([s[:min_len] for s in series])
    return np.arange(1, min_len + 1), arr.mean(0), arr.std(0)


def _fmt(v, decimals: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _mean_std_str(mean: float, std: float, n: int, decimals: int = 4) -> str:
    if n == 1:
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


# ── table generators ──────────────────────────────────────────────────────────

def make_main_results_table(df: pd.DataFrame, out_tables: Path) -> None:
    stats = _method_split_stats(df)
    if not stats:
        print("  Skipping main_results: no data")
        return

    fedit_noniid_mean = stats.get(("fedit", "noniid"), {}).get("mean")

    rows = []
    for method in METHOD_KEY_ORDER:
        row = df[df["method"] == method].iloc[0] if (df["method"] == method).any() else None
        if row is None:
            continue

        s_iid    = stats.get((method, "iid"),    {})
        s_noniid = stats.get((method, "noniid"), {})

        iid_mean_std    = _mean_std_str(s_iid["mean"],    s_iid["std"],    s_iid["n"])    if s_iid    else "—"
        noniid_mean_std = _mean_std_str(s_noniid["mean"], s_noniid["std"], s_noniid["n"]) if s_noniid else "—"

        gap_vs_fedit = "—"
        if s_noniid and fedit_noniid_mean is not None and method != "fedit":
            gap = s_noniid["mean"] - fedit_noniid_mean
            gap_vs_fedit = f"{gap:+.4f}"

        iid_to_noniid_drop = "—"
        if s_iid and s_noniid:
            drop = s_iid["mean"] - s_noniid["mean"]
            iid_to_noniid_drop = f"{drop:.4f}"

        comm = row.get("communicated_params_per_round", "?")
        adapter_p = row.get("trainable_adapter_params", "?")
        head_p    = row.get("trainable_head_params", "?")
        total_p   = row.get("total_trainable_params", "?")

        init_str = str(row.get("init", "—"))
        if init_str == "lora":
            init_str = "LoRA (FedIT)"
        elif init_str == "gram_schmidt":
            init_str = "GS"
        elif init_str == "svd":
            init_str = "SVD warm-up"

        rank_str = str(row.get("rank", "?"))
        if method != "fedit" and row.get("heads"):
            rank_str = f"h={row['heads']}, r={row['rank']}"

        rows.append({
            "Method":           METHOD_DISPLAY[method],
            "Init":             init_str,
            "Config":           rank_str,
            "Trainable params": _fmt(total_p, 0),
            "Comm. params/round": _fmt(comm, 0),
            "IID Acc. mean±std":     iid_mean_std,
            "Non-IID Acc. mean±std": noniid_mean_std,
            "Non-IID gap vs FedIT":  gap_vs_fedit,
            "IID-to-Non-IID drop":   iid_to_noniid_drop,
        })

    if not rows:
        print("  Skipping main_results: no rows")
        return

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_tables / "main_results.csv", index=False)
    print(f"  → {out_tables / 'main_results.csv'}")

    _df_to_latex(
        result_df,
        out_tables / "main_results.tex",
        caption="Main results: final test accuracy for each method and split, averaged across seeds.",
        label="tab:main_results",
    )
    print(f"  → {out_tables / 'main_results.tex'}")


def make_gap_analysis_table(df: pd.DataFrame, out_tables: Path) -> None:
    stats = _method_split_stats(df)
    if not stats:
        return

    fedit_iid_mean    = stats.get(("fedit", "iid"),    {}).get("mean")
    fedit_noniid_mean = stats.get(("fedit", "noniid"), {}).get("mean")
    gs_iid_mean       = stats.get(("ravan_gram_schmidt", "iid"),    {}).get("mean")
    gs_noniid_mean    = stats.get(("ravan_gram_schmidt", "noniid"), {}).get("mean")

    rows = []
    for method in METHOD_KEY_ORDER:
        s_iid    = stats.get((method, "iid"),    {})
        s_noniid = stats.get((method, "noniid"), {})
        if not s_iid and not s_noniid:
            continue

        def _gap(a, b):
            return (a - b) if a is not None and b is not None else None

        gap_fedit_iid    = _gap(s_iid.get("mean"),    fedit_iid_mean)    if method != "fedit" else None
        gap_fedit_noniid = _gap(s_noniid.get("mean"), fedit_noniid_mean) if method != "fedit" else None
        svd_vs_gs_iid    = _gap(s_iid.get("mean"),    gs_iid_mean)    if method == "ravan_svd" else None
        svd_vs_gs_noniid = _gap(s_noniid.get("mean"), gs_noniid_mean) if method == "ravan_svd" else None

        drop = _gap(s_iid.get("mean"), s_noniid.get("mean")) if s_iid and s_noniid else None

        rows.append({
            "method":           method,
            "display":          METHOD_DISPLAY[method],
            "iid_mean":         s_iid.get("mean"),
            "iid_std":          s_iid.get("std"),
            "noniid_mean":      s_noniid.get("mean"),
            "noniid_std":       s_noniid.get("std"),
            "gap_vs_fedit_iid":    gap_fedit_iid,
            "gap_vs_fedit_noniid": gap_fedit_noniid,
            "iid_to_noniid_drop":  drop,
            "svd_vs_gs_iid":    svd_vs_gs_iid,
            "svd_vs_gs_noniid": svd_vs_gs_noniid,
        })

    if rows:
        gap_df = pd.DataFrame(rows)
        gap_df.to_csv(out_tables / "gap_analysis.csv", index=False)
        print(f"  → {out_tables / 'gap_analysis.csv'}")


def make_init_costs_table(df: pd.DataFrame, out_tables: Path) -> None:
    stats = _method_split_stats(df)

    rows = []
    for method in METHOD_KEY_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        row = sub.iloc[0]
        s_noniid = stats.get((method, "noniid"), {})
        noniid_str = _mean_std_str(s_noniid["mean"], s_noniid["std"], s_noniid["n"]) if s_noniid else "—"

        comm_round = _fmt(row.get("communicated_params_per_round"), 0)

        if method == "fedit":
            rows.append({
                "Method":            METHOD_DISPLAY[method],
                "Initialization":    "LoRA (Kaiming/zeros)",
                "Warm-up steps":     "—",
                "Warm-up clients":   "—",
                "Temporary rank":    "—",
                "Extra comm. (params)": "0",
                "Warm-up runtime (s)":  "—",
                "SVD runtime (s)":      "—",
                "Total init runtime (s)": "—",
                "Main comm./round":     comm_round,
                "Final Non-IID Acc.":   noniid_str,
            })
        elif method == "ravan_gram_schmidt":
            rows.append({
                "Method":            METHOD_DISPLAY[method],
                "Initialization":    "Gram-Schmidt QR",
                "Warm-up steps":     "—",
                "Warm-up clients":   "—",
                "Temporary rank":    "—",
                "Extra comm. (params)": "0",
                "Warm-up runtime (s)":  "—",
                "SVD runtime (s)":      "—",
                "Total init runtime (s)": "—",
                "Main comm./round":     comm_round,
                "Final Non-IID Acc.":   noniid_str,
            })
        else:  # ravan_svd
            rows.append({
                "Method":            METHOD_DISPLAY[method],
                "Initialization":    "Federated SVD warm-up",
                "Warm-up steps":     _fmt(row.get("warmup_steps"), 0),
                "Warm-up clients":   _fmt(row.get("warmup_clients"), 0),
                "Temporary rank":    _fmt(row.get("warmup_rank"), 0),
                "Extra comm. (params)": _fmt(row.get("warmup_communicated_params"), 0),
                "Warm-up runtime (s)":  _fmt(row.get("warmup_train_runtime_s"), 1),
                "SVD runtime (s)":      _fmt(row.get("warmup_svd_runtime_s"), 3),
                "Total init runtime (s)": _fmt(row.get("warmup_total_runtime_s"), 1),
                "Main comm./round":     comm_round,
                "Final Non-IID Acc.":   noniid_str,
            })

    if rows:
        cost_df = pd.DataFrame(rows)
        cost_df.to_csv(out_tables / "init_costs.csv", index=False)
        print(f"  → {out_tables / 'init_costs.csv'}")
        _df_to_latex(
            cost_df,
            out_tables / "init_costs.tex",
            caption="Initialization cost comparison across methods.",
            label="tab:init_costs",
        )
        print(f"  → {out_tables / 'init_costs.tex'}")


def make_setup_summary_table(df: pd.DataFrame, out_tables: Path) -> None:
    if df.empty:
        return
    row = df.iloc[0]
    params = {
        "Model":           "DistilBERT (distilbert-base-uncased)",
        "Dataset":         "20 Newsgroups (20 classes)",
        "Clients":         str(row.get("clients", 20)),
        "Clients/round":   str(row.get("clients_per_round", 3)),
        "Local steps":     str(row.get("local_steps", 50)),
        "Rounds":          str(row.get("rounds", 50)),
        "Non-IID alpha":   str(row.get("dirichlet_alpha", 0.3)),
        "Seeds":           "0, 1, 2",
        "Adapted layers":  "12 (q_lin + v_lin in all 6 transformer layers)",
    }
    setup_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
    setup_df.to_csv(out_tables / "setup_summary.csv", index=False)
    print(f"  → {out_tables / 'setup_summary.csv'}")


def make_parameter_budget_table(df: pd.DataFrame, out_tables: Path) -> None:
    rows = []
    for method in METHOD_KEY_ORDER:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        row = sub.iloc[0]
        rows.append({
            "Method":          METHOD_DISPLAY[method],
            "Rank config":     f"r={row.get('rank', '?')}" if method == "fedit"
                               else f"h={row.get('heads','?')}, r={row.get('rank','?')}",
            "Adapter params":  _fmt(row.get("trainable_adapter_params"), 0),
            "Head params":     _fmt(row.get("trainable_head_params"), 0),
            "Total trainable": _fmt(row.get("total_trainable_params"), 0),
            "Comm./round":     _fmt(row.get("communicated_params_per_round"), 0),
        })
    if rows:
        budget_df = pd.DataFrame(rows)
        budget_df.to_csv(out_tables / "parameter_budget.csv", index=False)
        print(f"  → {out_tables / 'parameter_budget.csv'}")


def _df_to_latex(df: pd.DataFrame, path: Path, caption: str = "", label: str = "") -> None:
    """Write a booktabs-style LaTeX table."""
    cols = list(df.columns)
    col_spec = "l" + "r" * (len(cols) - 1)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        line = " & ".join(str(v) for v in row.values) + r" \\"
        lines.append(line)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── figure generators ──────────────────────────────────────────────────────────

def _draw_learning_curve(
    ax: plt.Axes,
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    split: str,
) -> bool:
    drawn = False
    for method in METHOD_KEY_ORDER:
        result = _curves_for(summaries, histories, method, split)
        if result is None:
            continue
        rounds, mean, std = result
        color  = METHOD_COLORS[method]
        label  = METHOD_DISPLAY[method]
        n_seeds = int(((summaries["method"] == method) & (summaries["split"] == split)).sum())
        ax.plot(rounds, mean, label=label, color=color)
        if n_seeds > 1:
            ax.fill_between(rounds, mean - std, mean + std, alpha=0.15, color=color)
        drawn = True
    return drawn


def fig_learning_curves(
    summaries: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
    out_figures: Path,
) -> None:
    for split in ["iid", "noniid"]:
        fig, ax = plt.subplots(figsize=(7, 4))
        if not _draw_learning_curve(ax, summaries, histories, split):
            plt.close(fig)
            continue

        n_seeds = summaries[summaries["split"] == split]["seed"].nunique()
        ax.set_xlabel("Communication Round")
        ax.set_ylabel("Test Accuracy")
        ax.set_title(f"Learning Curves — {SPLIT_DISPLAY.get(split, split)}")
        ax.legend(loc="lower right")
        if n_seeds > 1:
            ax.annotate(f"mean ± std  ({n_seeds} seeds)", xy=(0.02, 0.97),
                        xycoords="axes fraction", fontsize=8, color="gray", va="top")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        path = out_figures / f"learning_curves_{split}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {path}")


def fig_final_accuracy_bar(
    summaries: pd.DataFrame,
    out_figures: Path,
) -> None:
    if "final_acc" not in summaries.columns:
        return

    present_methods = [m for m in METHOD_KEY_ORDER if m in summaries["method"].values]
    present_splits  = [s for s in ["iid", "noniid"] if s in summaries["split"].values]
    if len(present_methods) < 1:
        return

    x       = np.arange(len(present_methods))
    width   = 0.35
    n_spl   = len(present_splits)
    offsets = np.linspace(-(n_spl - 1) * width / 2, (n_spl - 1) * width / 2, n_spl)
    split_colors = {"iid": "#5dade2", "noniid": "#e59866"}

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, split in zip(offsets, present_splits):
        means, stds = [], []
        for method in present_methods:
            vals = summaries.loc[
                (summaries["method"] == method) & (summaries["split"] == split), "final_acc"
            ].dropna().values
            means.append(float(vals.mean()) if len(vals) > 0 else 0.0)
            stds.append( float(vals.std())  if len(vals) > 1 else 0.0)

        bars = ax.bar(
            x + offset, means, width,
            yerr=stds if any(e > 0 for e in stds) else None,
            capsize=4, label=SPLIT_DISPLAY.get(split, split),
            color=split_colors.get(split, "#aaa"), alpha=0.85,
            error_kw={"elinewidth": 1.5},
        )
        for bar, mean, std in zip(bars, means, stds):
            top = mean + std + 0.004 if std > 0 else mean + 0.004
            if mean > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, top,
                        f"{mean:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in present_methods])
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Final Test Accuracy by Method and Data Split")
    ax.legend(title="Split")
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    path = out_figures / "final_accuracy_by_method_split.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_iid_vs_noniid_drop(summaries: pd.DataFrame, out_figures: Path) -> None:
    stats = _method_split_stats(summaries)
    present = [m for m in METHOD_KEY_ORDER
               if (m, "iid") in stats and (m, "noniid") in stats]
    if not present:
        return

    drops = []
    stds  = []
    for method in present:
        s_iid    = stats[(method, "iid")]
        s_noniid = stats[(method, "noniid")]
        drop = s_iid["mean"] - s_noniid["mean"]
        combined_std = math.sqrt(s_iid["std"]**2 + s_noniid["std"]**2) / math.sqrt(2) \
            if (s_iid["n"] > 1 or s_noniid["n"] > 1) else 0.0
        drops.append(drop)
        stds.append(combined_std)

    x = np.arange(len(present))
    colors = [METHOD_COLORS[m] for m in present]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(x, drops, color=colors, alpha=0.85,
                  yerr=stds if any(s > 0 for s in stds) else None,
                  capsize=4, error_kw={"elinewidth": 1.5})
    for bar, v in zip(bars, drops):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in present])
    ax.set_ylabel("IID Accuracy − Non-IID Accuracy (lower is better)")
    ax.set_title("IID-to-Non-IID Accuracy Drop by Method")
    fig.tight_layout()

    path = out_figures / "iid_vs_noniid_drop.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_gap_analysis(summaries: pd.DataFrame, out_figures: Path) -> None:
    stats = _method_split_stats(summaries)
    fedit_iid    = stats.get(("fedit", "iid"),    {}).get("mean")
    fedit_noniid = stats.get(("fedit", "noniid"), {}).get("mean")
    if fedit_iid is None and fedit_noniid is None:
        return

    methods_to_plot = [m for m in ["ravan_gram_schmidt", "ravan_svd"]
                       if (m, "iid") in stats or (m, "noniid") in stats]
    if not methods_to_plot:
        return

    splits = ["iid", "noniid"]
    fedit_baselines = {"iid": fedit_iid, "noniid": fedit_noniid}
    x = np.arange(len(methods_to_plot))
    width = 0.35
    offsets = [-width/2, width/2]
    split_colors = {"iid": "#5dade2", "noniid": "#e59866"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for offset, split in zip(offsets, splits):
        base = fedit_baselines[split]
        if base is None:
            continue
        gaps, errs = [], []
        for method in methods_to_plot:
            s = stats.get((method, split), {})
            gaps.append((s["mean"] - base) if s else 0.0)
            errs.append(s.get("std", 0.0))

        bars = ax.bar(x + offset, gaps, width,
                      label=SPLIT_DISPLAY.get(split, split),
                      color=split_colors.get(split, "#aaa"), alpha=0.85,
                      yerr=errs if any(e > 0 for e in errs) else None,
                      capsize=4, error_kw={"elinewidth": 1.5})
        for bar, v in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.002 if v >= 0 else -0.006),
                    f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_DISPLAY[m] for m in methods_to_plot])
    ax.set_ylabel("Accuracy gap vs. FedIT (positive = better than FedIT)")
    ax.set_title("Ravan Accuracy Gap Relative to FedIT Baseline")
    ax.legend(title="Split")
    fig.tight_layout()

    path = out_figures / "gap_analysis.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_init_cost_vs_accuracy(summaries: pd.DataFrame, out_figures: Path) -> None:
    stats = _method_split_stats(summaries)
    points = []
    for method in METHOD_KEY_ORDER:
        s = stats.get((method, "noniid"), {})
        if not s:
            continue
        sub = summaries[summaries["method"] == method]
        if sub.empty:
            continue
        row = sub.iloc[0]
        wc = float(row.get("warmup_communicated_params", 0) or 0)
        points.append({
            "method": method,
            "warmup_comm": wc,
            "noniid_mean": s["mean"],
            "noniid_std":  s["std"],
        })

    if len(points) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    for p in points:
        color = METHOD_COLORS[p["method"]]
        label = METHOD_DISPLAY[p["method"]]
        ax.errorbar(
            p["warmup_comm"], p["noniid_mean"],
            yerr=p["noniid_std"] if p["noniid_std"] > 0 else None,
            fmt="o", color=color, label=label, markersize=8, capsize=4,
        )
        ax.annotate(label, (p["warmup_comm"], p["noniid_mean"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Extra warm-up communication (parameters)")
    ax.set_ylabel("Final Non-IID Test Accuracy")
    ax.set_title("Initialization Cost vs. Final Non-IID Accuracy")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = out_figures / "init_cost_vs_accuracy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def fig_singular_values(sv_df: pd.DataFrame, out_figures: Path) -> None:
    if sv_df.empty:
        return

    splits = sv_df["split"].unique()
    n_panels = len(splits)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, split in zip(axes, splits):
        sub = sv_df[sv_df["split"] == split]
        for proj in ["q", "v"]:
            proj_sub = sub[sub["projection"] == proj]
            if proj_sub.empty:
                continue
            # Average over seeds and layers
            mean_sv = (
                proj_sub.groupby("sv_rank")["singular_value"]
                .mean()
                .reset_index()
                .sort_values("sv_rank")
            )
            ax.semilogy(mean_sv["sv_rank"].values, mean_sv["singular_value"].values,
                        label=f"{proj}_lin", marker=".", markersize=4)

        ax.set_xlabel("Singular value rank")
        ax.set_title(f"{SPLIT_DISPLAY.get(split, split)}")
        ax.legend()

    axes[0].set_ylabel("Singular value (log scale)")
    fig.suptitle("Warm-up ΔW Singular Value Spectra (Ravan-SVD)", fontsize=11)
    fig.tight_layout()

    path = out_figures / "singular_values.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


# ── report summary markdown ───────────────────────────────────────────────────

def write_summary_md(
    summaries: pd.DataFrame,
    out_dir: Path,
    tables_generated: list[str],
    figures_generated: list[str],
) -> None:
    stats = _method_split_stats(summaries)
    lines = [
        "# Report Assets Summary",
        "",
        f"Generated from {len(summaries)} completed runs.",
        "",
        "## Results snapshot",
        "",
        "| Method | IID Acc | Non-IID Acc |",
        "|--------|---------|-------------|",
    ]
    for method in METHOD_KEY_ORDER:
        s_iid    = stats.get((method, "iid"),    {})
        s_noniid = stats.get((method, "noniid"), {})
        iid_str    = _mean_std_str(s_iid["mean"],    s_iid["std"],    s_iid["n"])    if s_iid    else "—"
        noniid_str = _mean_std_str(s_noniid["mean"], s_noniid["std"], s_noniid["n"]) if s_noniid else "—"
        lines.append(f"| {METHOD_DISPLAY[method]} | {iid_str} | {noniid_str} |")

    lines += [
        "",
        "## Tables generated",
        "",
    ] + [f"- `{t}`" for t in tables_generated] + [
        "",
        "## Figures generated",
        "",
    ] + [f"- `{f}`" for f in figures_generated]

    path = out_dir / "report_assets_summary.md"
    with open(path, "w") as fp:
        fp.write("\n".join(lines) + "\n")
    print(f"  → {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate paper-ready tables and figures")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory containing experiment results")
    parser.add_argument("--out_dir",     type=str, default="paper_assets",
                        help="Output directory for tables and figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_tables  = out_dir / "tables"
    out_figures = out_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figures.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir} ...")
    summaries = load_summaries(results_dir)
    if summaries.empty:
        print("No results found — nothing to generate.")
        return

    print(f"Found {len(summaries)} runs: "
          f"{sorted(summaries['method'].unique()) if 'method' in summaries.columns else '?'}")

    histories = load_histories(results_dir, summaries)
    sv_df     = load_singular_values(results_dir, summaries)
    print(f"Loaded {len(histories)} round-history files, "
          f"{len(sv_df)} singular value rows.\n")

    # ── tables ────────────────────────────────────────────────────────────────
    print("Generating tables...")
    make_main_results_table(summaries, out_tables)
    make_gap_analysis_table(summaries, out_tables)
    make_init_costs_table(summaries, out_tables)
    make_setup_summary_table(summaries, out_tables)
    make_parameter_budget_table(summaries, out_tables)

    # ── figures ───────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    fig_learning_curves(summaries, histories, out_figures)
    fig_final_accuracy_bar(summaries, out_figures)
    fig_iid_vs_noniid_drop(summaries, out_figures)
    fig_gap_analysis(summaries, out_figures)
    fig_init_cost_vs_accuracy(summaries, out_figures)
    if not sv_df.empty:
        fig_singular_values(sv_df, out_figures)

    # ── summary md ────────────────────────────────────────────────────────────
    tables_generated  = [str(p.relative_to(out_dir)) for p in sorted(out_tables.iterdir())]
    figures_generated = [str(p.relative_to(out_dir)) for p in sorted(out_figures.iterdir())]
    write_summary_md(summaries, out_dir, tables_generated, figures_generated)

    print(f"\nDone. Paper assets in: {out_dir}/")


if __name__ == "__main__":
    main()
