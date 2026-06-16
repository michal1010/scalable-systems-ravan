#!/usr/bin/env python
"""Collect parallel VM sweep outputs into one final results directory."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from plot import plot_all


def copy_run_dirs(parts_dir: Path, final_dir: Path) -> list[Path]:
    final_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for summary_path in sorted(parts_dir.glob("*/*/summary.json")):
        run_dir = summary_path.parent
        dest = final_dir / run_dir.name
        shutil.copytree(run_dir, dest, dirs_exist_ok=True)
        copied.append(dest)

    return copied


def write_master_csv(final_dir: Path) -> Path:
    rows: list[dict] = []
    fields: list[str] = []

    for summary_path in sorted(final_dir.glob("*/summary.json")):
        with open(summary_path) as f:
            row = json.load(f)
        rows.append(row)
        for field in row:
            if field not in fields:
                fields.append(field)

    master = final_dir / "all_results.csv"
    with open(master, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})

    return master


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate A100 VM sweep results")
    parser.add_argument("--parts-dir", required=True, type=Path)
    parser.add_argument("--final-dir", required=True, type=Path)
    args = parser.parse_args()

    copied = copy_run_dirs(args.parts_dir, args.final_dir)
    master = write_master_csv(args.final_dir)

    try:
        # from federated.plot import plot_all

        plot_all(args.final_dir)
    except Exception as exc:
        print(f"Plot regeneration skipped: {exc}")

    print(f"Copied {len(copied)} runs into {args.final_dir}")
    print(f"Master CSV: {master}")


if __name__ == "__main__":
    main()
