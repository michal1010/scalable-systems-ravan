#!/usr/bin/env bash
# Preflight check: run tests, validate contract, smoke grid, and generate partial assets.
# Stop on first error.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=============================="
echo " STEP 1: pytest"
echo "=============================="
python -m pytest tests/ -v

echo ""
echo "=============================="
echo " STEP 2: validate_experiment_contract"
echo "=============================="
python -m scripts.validate_experiment_contract

echo ""
echo "=============================="
echo " STEP 3: smoke grid (LIMIT=500)"
echo "=============================="
LIMIT=500 bash scripts/run_smoke_grid.sh

echo ""
echo "=============================="
echo " STEP 4: generate report assets"
echo "=============================="
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets

echo ""
echo "=============================="
echo " PREFLIGHT COMPLETE"
echo "=============================="
