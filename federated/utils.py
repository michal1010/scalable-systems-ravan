"""Result logging utilities: per-run directories, JSON configs, CSVs, master index."""

import csv
import json
import subprocess
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def get_git_hash() -> str:
    """Return the short HEAD commit hash, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def make_run_name(method: str, split: str, seed: int) -> str:
    """Build a unique, sortable run identifier.

    Format: <method>_<split>_seed<seed>_<YYYYMMDD_HHMMSS>
    The method string should already encode init mode (e.g. "ravan_svd").
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"{method}_{split}_seed{seed}_{ts}"


def make_run_dir(run_name: str, output_dir: str | None = None) -> Path:
    """Create and return <output_dir>/<run_name>/ — unique folder for this run."""
    base = Path(output_dir) if output_dir else RESULTS_DIR
    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_config(cfg: dict, run_dir: Path) -> None:
    path = run_dir / "config.json"
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config  → {path}")


def save_profile_report(profile_data: dict, run_dir: Path) -> None:
    """Save profiling metrics to run_dir/profile_report.json and print summary."""
    path = run_dir / "profile_report.json"
    with open(path, "w") as f:
        json.dump(profile_data, f, indent=2)

    print("\n=== PROFILE REPORT ===")
    if profile_data.get("peak_gpu_memory_mb") is not None:
        print(f"  Peak GPU memory    : {profile_data['peak_gpu_memory_mb']:.0f} MB")
    print(f"  Avg time/client    : {profile_data['avg_client_time_s']:.2f} s")
    print(f"  Time for 2 rounds  : {profile_data['total_2round_time_s']:.1f} s")
    est_s = profile_data.get("estimated_total_s", 0)
    print(f"  Estimated total    : {est_s:.0f} s  ({est_s / 3600:.2f} h)  "
          f"[{profile_data.get('total_rounds_requested', '?')} rounds]")
    print(f"  Profile report  → {path}")
    print("=== END PROFILE ===\n")


def save_results(
    summary: dict,
    history: list[dict],
    run_dir: Path,
    results_dir: Path | None = None,
) -> None:
    """Save summary JSON + per-round CSV inside run_dir, append to master CSV.

    Args:
        summary     : flat dict with all summary fields (becomes summary.json)
        history     : list of per-round dicts (becomes rounds.csv)
        run_dir     : destination directory for this run's files
        results_dir : top-level results directory for all_results.csv
                      (defaults to RESULTS_DIR)
    """
    if results_dir is None:
        results_dir = RESULTS_DIR

    # --- summary JSON ---
    summary_path = run_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary → {summary_path}")

    # --- per-round CSV ---
    if history:
        csv_path = run_dir / "rounds.csv"
        fieldnames = list(history[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)
        print(f"Rounds  → {csv_path}")

    # --- master CSV at top-level results/ (one row per experiment) ---
    master = results_dir / "all_results.csv"

    existing_rows: list[dict] = []
    existing_fields: list[str] = []
    if master.exists():
        try:
            with open(master, "r", newline="") as f:
                reader = csv.DictReader(f)
                existing_fields = list(reader.fieldnames or [])
                existing_rows   = list(reader)
        except Exception:
            pass

    new_fields = list(summary.keys())
    all_fields = existing_fields + [f for f in new_fields if f not in existing_fields]

    with open(master, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k) for k in all_fields})
        writer.writerow({k: summary.get(k) for k in all_fields})
    print(f"Master  → {master}")
