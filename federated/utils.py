"""Result logging utilities: JSON configs, per-round CSV, master CSV."""

import csv
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_run_name(method: str, split: str, seed: int, extra: str = "") -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{method}_{split}_seed{seed}"
    if extra:
        name += f"_{extra}"
    name += f"_{ts}"
    return name


def save_config(cfg: dict, run_name: str):
    path = RESULTS_DIR / f"{run_name}_config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config  → {path}")


def save_results(metrics: dict, history: list[dict], run_name: str):
    """Save summary JSON + per-round CSV + append to master CSV."""
    # --- summary JSON ---
    summary_path = RESULTS_DIR / f"{run_name}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary → {summary_path}")

    # --- per-round CSV ---
    if history:
        csv_path = RESULTS_DIR / f"{run_name}_rounds.csv"
        fieldnames = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"Rounds  → {csv_path}")

    # --- master CSV (one row per experiment) ---
    master = RESULTS_DIR / "all_results.csv"
    write_header = not master.exists()
    with open(master, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Master  → {master}")
