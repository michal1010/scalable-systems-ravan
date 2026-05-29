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
| `train_fedit.py` | Main script for FedIT experiment |
| `train_ravan.py` | Main script for Ravan experiment (both init modes) |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments Locally

```bash
# FedIT
python -m federated.train_fedit --split noniid --seed 0 --rounds 50

# Ravan (Gram-Schmidt)
python -m federated.train_ravan --init gram_schmidt --split noniid --seed 0

# Ravan (SVD warm-up)
python -m federated.train_ravan --init svd --split noniid --seed 0

# Fast smoke test (2 rounds, 5 steps)
python -m federated.train_fedit --rounds 2 --local_steps 5 --split iid
```

## Tests

```bash
python -m pytest tests/ -v
```

Tests in `tests/test_aggregation.py`:
- Ravan exact aggregation (mathematical correctness check)
- FedIT mismatch (verifies the mismatch exists — expected behavior)
- Gram-Schmidt orthogonality of B_i, A_i bases
- SVD orthogonality of B_i, A_i bases
- Zero initial adapter output (H=0 means adapter contributes nothing at init)
- Zero initial LoRA output (B=0 means LoRA contributes nothing at init)

## Running on DAIC Cluster

```bash
# Single job
sbatch jobs/submit_fedit.sh
sbatch jobs/submit_ravan_gs.sh
sbatch jobs/submit_ravan_svd.sh

# Full sweep (3 methods × 2 splits × 3 seeds = 18 jobs)
bash jobs/sweep.sh
```

The cluster uses **Slurm**, partition `general`, QoS `short` (max 4 hours).
Set up a conda environment named `ravan` with `pip install -r requirements.txt`.
Module load: `module use /opt/insy/modulefiles && module load miniconda`.

## Ravan Architecture

`RavanLinear` (in `federated/ravan.py`) wraps a **frozen** `nn.Linear` and adds trainable multi-head adapters:
- **Frozen buffers**: `B` [heads × d_out × rank] and `A` [heads × rank × d_in] — initialized once (GS or SVD)
- **Trainable params**: `H` [heads × rank × rank, zero-init] and `scales` [heads, ones-init]
- **Forward**: `output = frozen(x) + Σ_h scales[h] * (x @ A[h]ᵀ @ H[h]ᵀ @ B[h]ᵀ)`
- `H=0` initialization means the adapter contributes zero at start (preserves pretrained weights)
- Adapters are inserted on `q_lin` and `v_lin` of every DistilBERT attention layer (6 layers × 2 = 12 adapted layers)

## Parameter Budget

For DistilBERT (d=768), default settings achieve approximate budget matching:

| Method | Rank | Adapter params/layer | Total adapter params |
|---|---|---|---|
| FedIT | 8 | 2 × 768 × 8 = 12,288 | 147,456 |
| Ravan (h=4, r=55) | 55 | 4 × 55² + 4 ≈ 12,104 | 145,248 |

Budget formula: r_Ravan ≈ √(2 × d × r_LoRA / h) = √(2 × 768 × 8 / 4) ≈ 55

## Federated SVD Warm-Up (Ravan-SVD)

1. Sample `warmup_clients` (default 5) clients
2. Each trains a temporary LoRA model of total rank R = heads × rank (= 220)
3. Each client computes ΔW_c = B_c @ A_c per layer (products, not factors)
4. Server aggregates: ΔW_warm = mean_c(ΔW_c) — this avoids FedIT mismatch
5. Truncated SVD: ΔW_warm ≈ U_R Σ_R Vh_R
6. Frozen bases: B_i = U_R[:, slice_i], A_i = Vh_R[slice_i, :]
7. H_i = 0, scales_i = 1 (adapter starts at zero; main Ravan training begins)

## Legacy Scripts

The root-level scripts and `federated_bert/` directory contain earlier prototypes (tiny BERT, non-federated, Flower-based). These are kept for reference but are not part of the current research pipeline.
