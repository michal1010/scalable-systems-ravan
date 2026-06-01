#!/bin/bash
# Local smoke-test grid: all 3 methods × 2 splits × 1 seed = 6 tiny runs.
# Uses --rounds 2 --local_steps 5 so each run completes in minutes on CPU.
# Does NOT change any default research hyper-parameters.
#
# Usage (from project root):
#   bash scripts/run_smoke_grid.sh
#
# To also limit training data for faster tokenization:
#   LIMIT=500 bash scripts/run_smoke_grid.sh
#
set -e

ROUNDS=2
STEPS=5
SEED=0

# On CPU, limit examples to keep smoke tests fast (ignored if GPU available).
# Set LIMIT=0 to disable limiting (use full data).
LIMIT=${LIMIT:-500}
LIMIT_TEST=${LIMIT_TEST:-200}

LIMIT_ARG=""
if [ "${LIMIT}" != "0" ] && [ -n "${LIMIT}" ]; then
    LIMIT_ARG="--limit_train_examples ${LIMIT} --limit_test_examples ${LIMIT_TEST}"
fi

echo "=== Smoke grid: rounds=$ROUNDS, local_steps=$STEPS, seed=$SEED ==="
echo ""

for SPLIT in iid noniid; do
    echo "--- FedIT ($SPLIT) ---"
    python -m federated.train_fedit \
        --split $SPLIT --seed $SEED \
        --rounds $ROUNDS --local_steps $STEPS \
        --rank 8 --lr 1e-3 \
        $LIMIT_ARG

    echo "--- Ravan-GS ($SPLIT) ---"
    python -m federated.train_ravan \
        --init gram_schmidt \
        --split $SPLIT --seed $SEED \
        --rounds $ROUNDS --local_steps $STEPS \
        --heads 4 --rank 55 --lr 5e-4 \
        $LIMIT_ARG

    echo "--- Ravan-SVD ($SPLIT) ---"
    python -m federated.train_ravan \
        --init svd \
        --split $SPLIT --seed $SEED \
        --rounds $ROUNDS --local_steps $STEPS \
        --heads 4 --rank 55 --lr 5e-4 \
        --warmup_clients 2 --warmup_steps 5 \
        $LIMIT_ARG
done

echo ""
echo "=== Smoke grid complete. Generating report assets... ==="
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
echo "=== Done. ==="
