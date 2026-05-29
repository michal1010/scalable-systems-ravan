"""Result logging utilities: per-run directories, JSON configs, CSVs, master index."""

import csv
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def make_run_name(method: str, split: str, seed: int, extra: str = "") -> str:
    ts   = time.strftime("%Y%m%d_%H%M%S")
    name = f"{method}_{split}_seed{seed}"
    if extra:
        name += f"_{extra}"
    name += f"_{ts}"
    return name


def make_run_dir(run_name: str) -> Path:
    """Create and return results/<run_name>/ — unique folder for this run."""
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(cfg: dict, run_dir: Path) -> None:
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config  → {path}")


def save_results(metrics: dict, history: list[dict], run_dir: Path) -> None:
    """Save summary JSON + per-round CSV inside run_dir, append to master CSV."""
    # --- summary JSON (inside run folder) ---
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary → {summary_path}")

    # --- per-round CSV (inside run folder) ---
    if history:
        csv_path = run_dir / "rounds.csv"
        fieldnames = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"Rounds  → {csv_path}")

    # --- master CSV at top-level results/ (one row per experiment) ---
    master      = RESULTS_DIR / "all_results.csv"
    write_header = not master.exists()
    with open(master, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(metrics)
    print(f"Master  → {master}")
