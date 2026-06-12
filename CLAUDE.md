# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project (TU Delft) comparing three federated fine-tuning strategies for 20 Newsgroups text classification (20 classes) using **DistilBERT** (`distilbert-base-uncased`):

| Method | File | Description |
|---|---|---|
| **FedIT** | `federated/train_fedit.py` | Federated LoRA — server averages A and B factors separately (inexact aggregation baseline) |
| **Ravan-GS** | `federated/train_ravan.py --init gram_schmidt` | Ravan with Gram-Schmidt QR frozen bases — exact aggregation, data-agnostic init |
| **Ravan-SVD** | `federated/train_ravan.py --init svd` | Ravan with federated LoRA warm-up SVD frozen bases — exact aggregation, data-aware init |

## Federated Module Structure (`federated/`)

| File | Role |
|---|---|
| `data.py` | Load 20 Newsgroups; IID and non-IID (Dirichlet α=0.3) client splits |
| `model.py` | DistilBERT factory (`make_distilbert`), adapter injection (`inject_lora`, `inject_ravan`), param counting |
| `lora.py` | `LoRALinear` — frozen base + trainable A, B (FedIT baseline) |
| `ravan.py` | `RavanLinear` — frozen bases B_i, A_i + trainable H_i, scales; GS and SVD inits |
| `client.py` | `local_train()` and `evaluate()` — client-side gradient steps |
| `server.py` | `fedit_*` and `ravan_*` — state extraction, aggregation, loading |
| `warmup.py` | `federated_svd_init()` — federated LoRA warm-up → SVD → frozen bases for Ravan |
| `utils.py` | Result logging (JSON config, per-round CSV, master CSV) |
| `plot.py` | `plot_single_run` and `plot_all` — per-run curves and multi-run comparison figures |
| `model_t5.py` | T5-encoder classification wrapper, adapter injection, param counting (T5 mode) |
| `train_fedit.py` | Main script for FedIT experiment (supports `--model_type distilbert\|t5`) |
| `train_ravan.py` | Main script for Ravan experiment (supports `--model_type distilbert\|t5`) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments Locally

Full default hyperparameters:

```bash
# FedIT (DistilBERT)
python -m federated.train_fedit \
    --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --rank 8 --lr 1e-3

# Ravan (Gram-Schmidt, DistilBERT)
python -m federated.train_ravan \
    --init gram_schmidt --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4

# Ravan (SVD warm-up, DistilBERT)
python -m federated.train_ravan \
    --init svd --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4 \
    --warmup_clients 5 --warmup_steps 50
```

## T5 Mode (T5-Encoder Classification Approximation)

Uses `T5EncoderModel` (encoder-only) + masked mean pooling + linear head. **Not** text-to-text generation. t5-base: d_model=768, 12 encoder layers, ~110M params.

Adapters are injected on `q` and `v` of all 12 encoder self-attention blocks (24 adapted layers).

### Profile before running full experiments:

```bash
# Always profile first to estimate memory and runtime
python -m federated.train_fedit --model_type t5 --profile \
    --split noniid --seed 0 --rounds 100 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --rank 32 --lr 1e-3 --max_length 256 \
    --use_amp --grad_accum_steps 2

python -m federated.train_ravan --model_type t5 --init gram_schmidt --profile \
    --split noniid --seed 0 --rounds 100 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 110 --lr 5e-4 --max_length 256 \
    --use_amp --grad_accum_steps 2
```

### Stage 1 — FedIT + Ravan-GS (seed 0):

```bash
# T5 FedIT
python -m federated.train_fedit \
    --model_type t5 --split noniid --seed 0 --rounds 100 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --rank 32 --lr 1e-3 --max_length 256 \
    --use_amp --grad_accum_steps 2

# T5 Ravan-GS
python -m federated.train_ravan \
    --model_type t5 --init gram_schmidt --split noniid --seed 0 --rounds 100 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 110 --lr 5e-4 --max_length 256 \
    --use_amp --grad_accum_steps 2
```

### Stage 2 (optional) — Ravan-SVD:

```bash
python -m federated.train_ravan \
    --model_type t5 --init svd --split noniid --seed 0 --rounds 100 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 110 --lr 5e-4 --max_length 256 \
    --use_amp --grad_accum_steps 2 \
    --warmup_clients 5 --warmup_steps 50
```

### T5 parameter budget (d=768, 24 adapted layers):

| Method | Rank | Adapter params | Comm/client/round |
|---|---|---|---|
| T5-FedIT | 32 | 24 × 49,152 = 1,179,648 | 1,179,648 |
| T5-Ravan-GS (h=4, r=110) | 110 | 24 × 48,404 = 1,161,696 | 1,161,600 (H only) |

Budget formula: r_Ravan ≈ √(2 × d × r_LoRA / h) = √(2 × 768 × 32 / 4) ≈ 110

### T5 new CLI flags (both train scripts):

| Flag | Default | Description |
|---|---|---|
| `--model_type` | `distilbert` | `distilbert` or `t5` |
| `--use_amp` | off | Mixed precision (float16, CUDA only) |
| `--grad_accum_steps` | 1 | Gradient accumulation (effective_bs = batch_size × steps) |
| `--grad_checkpoint` | off | Gradient checkpointing (reduce activation memory) |
| `--profile` | off | Run 2 rounds, report GPU memory + timing + estimated runtime |

## Tests

```bash
# All tests
python -m pytest tests/ -v

# Single test
python -m pytest tests/test_aggregation.py::test_ravan_exact_aggregation -v
```

12 tests in `tests/test_aggregation.py`: Ravan exact aggregation, FedIT mismatch, GS orthogonality, SVD orthogonality, zero initial Ravan output, zero initial LoRA output, parameter budget counting, report asset generation, warm-up factor communication count (20,275,200), SVD head isolation, non-empty client splits, exact parameter accounting.

## Smoke Tests and Validation

```bash
# 6 tiny runs (all methods × splits, 2 rounds / 5 steps, LIMIT=500 examples)
bash scripts/run_smoke_grid.sh

# Fast contract check (~15s): verifies protocol constants, optimizer, comm counts
python -m scripts.validate_experiment_contract

# Full preflight: pytest → contract → smoke grid → asset generation
bash scripts/preflight.sh
```

## Paper Asset Generation

```bash
# After experiments complete; safe to re-run with partial results
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
```

Outputs `paper_assets/tables/` (LaTeX + CSV: `main_results`, `init_costs`, `gap_analysis`, `parameter_budget`) and `paper_assets/figures/` (7 plots). `paper_assets/manifest.json` records which of the 18 `(method, split, seed)` configs are present.

## Running on DAIC Cluster (Slurm)

Partition `general`, QoS `short` (max 4 hours). Set `HF_HOME` to project storage in all three `jobs/submit_*.sh` files before submitting to avoid home quota issues.

```bash
# Full sweep (3 methods × 2 splits × 3 seeds = 18 jobs)
bash jobs/sweep.sh

# Single job
sbatch jobs/submit_fedit.sh
sbatch jobs/submit_ravan_gs.sh
sbatch jobs/submit_ravan_svd.sh
```

One-time environment setup on the login node:
```bash
module use /opt/insy/modulefiles && module load miniconda
conda create -n ravan python=3.10 -y && conda activate ravan
pip install -r requirements.txt
```

## Running on a GPU VM (no Slurm)

Scripts under `jobs/gpu_vm/` run the same 18-job grid directly on a VM. Parallel runs are controlled with `MAX_PARALLEL` (default 2; try 3 on A100, 1 if OOM). Logs go to `logs_a100_vm/`, results to `results_a100_vm/` with per-run subdirectories to avoid parallel writes; `aggregate_results.py` merges them after the sweep.

```bash
bash jobs/gpu_vm/sweep.sh
# or: MAX_PARALLEL=3 bash jobs/gpu_vm/sweep.sh
```

## Result Storage

Each run writes to `results/<method>_<split>_seed<N>_<timestamp>/`:
- `config.json` — all CLI arguments + param counts
- `summary.json` — final/best accuracy, all cost fields, git hash
- `rounds.csv` — per-round: `round`, `test_acc`, `elapsed_seconds`, `selected_clients`, `train_runtime_seconds`
- `curve.png` — per-run learning curve

`results/all_results.csv` is the master index appended after each run.

## Ravan Architecture

`RavanLinear` (in `federated/ravan.py`) wraps a **frozen** `nn.Linear` and adds trainable multi-head adapters:
- **Frozen buffers**: `B` [heads × d_out × rank] and `A` [heads × rank × d_in] — initialized once (GS or SVD)
- **Trainable params**: `H` [heads × rank × rank, zero-init] and `scales` [heads, ones-init]
- **Forward**: `output = frozen(x) + Σ_h scales[h] * B[h] @ H[h] @ A[h] @ x`
- `H=0` initialization means the adapter contributes zero at start (preserves pretrained weights)
- Adapters are inserted on `q_lin` and `v_lin` of every DistilBERT attention layer (6 layers × 2 = 12 adapted layers)

**Aggregation correctness**: clients upload `s[h]·H[h]` products (not H and scales separately). Because B[h] and A[h] are frozen and identical across all clients, `mean_c[s_c·H_c]` gives the exact average weight update. FedIT's `mean(B_c) @ mean(A_c) ≠ mean(B_c @ A_c)` is the intentional baseline mismatch.

## Parameter Budget

For DistilBERT (d=768), default settings achieve approximate budget matching:

| Method | Rank | Adapter params/layer | Total adapter params |
|---|---|---|---|
| FedIT | 8 | 2 × 768 × 8 = 12,288 | 147,456 |
| Ravan (h=4, r=55) | 55 | 4 × 55² + 4 ≈ 12,104 | 145,248 |

Budget formula: r_Ravan ≈ √(2 × d × r_LoRA / h) = √(2 × 768 × 8 / 4) ≈ 55

Head params (shared): `pre_classifier` + `classifier` = 605,972. Total trainable: FedIT 753,428 / Ravan 751,220.

## Federated SVD Warm-Up (Ravan-SVD)

1. Sample `warmup_clients` (default 5) clients using seed `seed + 9999`
2. Each trains a temporary LoRA of total rank R = heads × rank (= 220) for `warmup_steps`
3. Each client uploads factors B_c [d_out × R] and A_c [R × d_in] per layer
4. Server reconstructs ΔW_c = B_c @ A_c and averages: ΔW_warm = mean_c(ΔW_c)
5. Truncated SVD: U_R, Vh_R = top-R components of ΔW_warm (singular values **not** absorbed)
6. Frozen bases: B[h] = U_R[:, h·r:(h+1)·r], A[h] = Vh_R[h·r:(h+1)·r, :]
7. Main Ravan model constructed fresh from pretrained weights; warm-up model discarded
8. Warm-up extra comm cost: 5 clients × 12 layers × 220 × (768 + 768) = 20,275,200 params

The main FL loop re-applies the same random seeds after warm-up so model initialization is deterministic regardless of whether SVD warm-up ran.

## Legacy Scripts

`bert_20newsgroups.py`, `bert_20newsgroups_ravan.py`, and `federated_bert/` are earlier prototypes (different BERT variant, Flower framework). Not part of the current research pipeline.
