"""Plot test accuracy from all rounds_live.csv files under a sweep parts dir.

Usage:
    python -m scripts.plot_parts_rounds_live
    python -m scripts.plot_parts_rounds_live --parts-dir results_100_t5/big_aggregation/parts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_PARTS_DIR = Path("results_100_t5/big_aggregation/parts")

METHOD_LABELS = {
    "t5_no_lora": "T5-No-LoRA",
    "t5_fedit": "T5-FedIT",
    "t5_ravan_gram_schmidt": "T5-Ravan-GS",
    "t5_ravan_svd": "T5-Ravan-SVD",
}
LABEL_COLORS = {
    "FedIT Non-IID": "#c0392b",
    "No-LoRA Non-IID": "#566573",
    "Ravan-GS Non-IID 12": "#1e8449",
    "Ravan-GS Non-IID 4": "#52be80",
    "Ravan-SVD 12": "#1a5276",
    "Ravan-SVD 4": "#5dade2",
}
SPLIT_LABELS = {"iid": "IID", "noniid": "Non-IID"}


def _read_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _read_rounds(path: Path) -> tuple[list[int], list[float]]:
    rounds: list[int] = []
    accs: list[float] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            acc = row.get("test_acc")
            if acc in (None, "", "None"):
                continue
            rounds.append(int(row["round"]))
            accs.append(float(acc))
    return rounds, accs


def _pretty_parts_name(name: str) -> str:
    name = name.replace(" copy", "").removeprefix("t5_")
    tokens = [token for token in name.split("_") if not token.startswith("seed")]

    if tokens[:2] == ["no", "lora"]:
        label = ["No-LoRA"]
        rest = tokens[2:]
    elif tokens[:2] == ["ravan", "gs"]:
        label = ["Ravan-GS"]
        rest = tokens[2:]
    elif tokens[:2] == ["ravan", "svd"]:
        label = ["Ravan-SVD"]
        rest = tokens[2:]
    elif tokens[:1] == ["fedit"]:
        label = ["FedIT"]
        rest = tokens[1:]
    else:
        label = []
        rest = tokens

    for token in rest:
        label.append(SPLIT_LABELS.get(token, token.upper() if token.isdigit() else token.title()))
    return " ".join(label)


def _label(run_dir: Path) -> str:
    return _pretty_parts_name(run_dir.parent.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all rounds_live.csv accuracy curves from a parts directory")
    parser.add_argument("--parts-dir", type=Path, default=DEFAULT_PARTS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    parts_dir = args.parts_dir
    out_path = args.out or parts_dir.parent / "rounds_live_accuracy.png"

    runs = []
    for rounds_path in sorted(parts_dir.glob("*/*/rounds_live.csv")):
        run_dir = rounds_path.parent
        meta = _read_summary(run_dir)
        rounds, accs = _read_rounds(rounds_path)
        if not rounds:
            continue
        runs.append((meta, run_dir, rounds, accs))

    if not runs:
        raise SystemExit(f"No non-empty rounds_live.csv files found under {parts_dir}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for meta, run_dir, rounds, accs in runs:
        label = _label(run_dir)
        ax.plot(rounds, accs, label=label, color=LABEL_COLORS.get(label), linewidth=2)

    ax.set_title("T5 Sweep Accuracy From rounds_live.csv")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=160)
    plt.close(fig)

    print(f"Plotted {len(runs)} runs")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
