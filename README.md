# Scalable Systems — RAVAN

Research project comparing three federated fine-tuning strategies for 20 Newsgroups text classification using **DistilBERT** (`distilbert-base-uncased`, 66M params).

**Paper:** "Reimplementing and Extending Ravan with Federated Data-Aware Initialization"
Anton & Jakomulski, Delft University of Technology, 2026.

## Methods

| Method | Adapter | Init | Aggregation | Communicated per round |
|---|---|---|---|---|
| **FedIT** | LoRA — trainable A, B | A: Kaiming, B: zeros | Average A, B separately *(inexact)* | A + B + head |
| **Ravan-GS** | Ravan — trainable H, scales; B/A frozen | Gram-Schmidt QR *(data-agnostic)* | Average s·H products *(exact)* | s·H + head |
| **Ravan-SVD** | Ravan — trainable H, scales; B/A frozen | Federated LoRA warm-up + SVD *(data-aware)* | Average s·H products *(exact)* | s·H + head |

All methods use the same frozen DistilBERT backbone and a shared trainable classification head (pre_classifier + classifier).

Adapters are injected into the **query** and **value** projections of all 6 transformer layers (12 adapted layers total).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running Experiments

### FedIT — federated LoRA baseline

```bash
# Non-IID split (Dirichlet α=0.3, recommended)
python -m federated.train_fedit \
    --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --rank 8 --lr 1e-3

# IID split
python -m federated.train_fedit --split iid --seed 0 --rounds 50
```

### Ravan — Gram-Schmidt initialization

```bash
python -m federated.train_ravan \
    --init gram_schmidt --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4
```

### Ravan — SVD warm-up initialization (data-aware)

```bash
# Runs a short federated LoRA warm-up before main Ravan training.
# Raw client data stays local; only ΔW products are communicated.
python -m federated.train_ravan \
    --init svd --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4 \
    --warmup_clients 5 --warmup_steps 50
```

### Quick smoke test

```bash
python -m federated.train_fedit  --rounds 2 --local_steps 5 --split iid
python -m federated.train_ravan  --init gram_schmidt --rounds 2 --local_steps 5 --split iid
python -m federated.train_ravan  --init svd          --rounds 2 --local_steps 5 --split iid \
    --warmup_steps 5 --warmup_clients 2
```

## Parameter Budget

The default rank choices achieve approximate parameter budget matching between FedIT and Ravan. For DistilBERT attention projections (d = 768):

$$r_\text{Ravan} \approx \sqrt{\frac{2 \cdot d \cdot r_\text{LoRA}}{h}} = \sqrt{\frac{2 \times 768 \times 8}{4}} \approx 55$$

| Method | Config | Adapter params (12 layers) | Head params | Total trainable |
|---|---|---|---|---|
| FedIT | rank=8 | 147,456 | ~606K | ~754K |
| Ravan | heads=4, rank=55 | 145,248 | ~606K | ~751K |

Adapter params per layer: FedIT = 2 × d × r = 12,288; Ravan = h × r² + h ≈ 12,104.

## Results

Results are written to `results/` after each run:

```
results/
  fedit_noniid_seed0_<timestamp>_config.json    # full run config
  fedit_noniid_seed0_<timestamp>_summary.json   # final + best accuracy
  fedit_noniid_seed0_<timestamp>_rounds.csv     # per-round test_acc, time
  all_results.csv                               # one row per experiment (accumulates)
```

## Correctness Tests

```bash
python -m pytest tests/ -v
```

| Test | Checks |
|---|---|
| `test_ravan_exact_aggregation` | `mean_c [Σ_i B_i (s_c,i H_c,i) A_i] == Σ_i B_i [mean_c (s_c,i H_c,i)] A_i` |
| `test_fedit_mismatch` | `mean(B_c @ A_c) ≠ mean(B_c) @ mean(A_c)` — verifies the known FedIT mismatch |
| `test_gram_schmidt_orthogonality` | `B_i` columns and `A_i` rows are orthonormal across all heads |
| `test_svd_init_orthogonality` | same check for SVD-initialized bases |
| `test_ravan_zero_init_output` | adapter contributes zero at init (H=0) |
| `test_lora_zero_init_output` | LoRA adapter contributes zero at init (B=0) |

## Running on DAIC (TU Delft Cluster)

The experiments are independent single-machine FL simulations — no multi-node distributed training needed. Submit one Slurm job per (method, split, seed) combination.

### One-off job

```bash
sbatch jobs/submit_fedit.sh
sbatch jobs/submit_ravan_gs.sh
sbatch jobs/submit_ravan_svd.sh

# Override defaults via extra args
sbatch jobs/submit_fedit.sh --split iid --seed 2
```

### Full sweep (18 jobs)

```bash
bash jobs/sweep.sh
```

### Environment setup on DAIC

```bash
# Run once on a login node
module use /opt/insy/modulefiles
module load miniconda
conda create -n ravan python=3.10
conda activate ravan
pip install -r requirements.txt
```

### Cluster specs

| GPU | Count | VRAM |
|---|---|---|
| A40 | 84 | 46 GB |
| L40 | 18 | 49 GB |
| V100 | 11 | 32 GB |
| RTX 2080 Ti | 24 | 11 GB |

Request a specific GPU: `#SBATCH --gres=gpu:a40:1`

### Monitoring

```bash
squeue -u $USER          # check job status
seff <jobID>             # efficiency report after job finishes
tail -f logs/fedit_*.out # live log output
```

## Ravan Architecture (Summary)

`RavanLinear` wraps a **frozen** `nn.Linear` and adds multi-head adapter:

```
output = W·x  +  Σ_h  scales[h] × B[h] @ H[h] @ A[h] @ x
                  \_________________________/
                    adapter contribution
                    (zero at init since H=0)
```

- `B[h]` ∈ ℝ^{d_out × r}, `A[h]` ∈ ℝ^{r × d_in} — **frozen** after initialization
- `H[h]` ∈ ℝ^{r × r} — **trainable**, zero-initialized
- `scales[h]` ∈ ℝ — **trainable**, initialized to 1

Clients upload `{scales[h] × H[h]}` products; the server averages them exactly.

## Federated SVD Warm-Up

Before main Ravan training, `--init svd` runs:

1. Sample `warmup_clients` (default 5) clients
2. Each trains a temporary LoRA model (rank R = heads × rank)
3. Each client computes `ΔW_c = B_c @ A_c` per layer *(products, not factors)*
4. Server averages: `ΔW_warm = mean_c(ΔW_c)` — avoids FedIT mismatch
5. Truncated SVD: `ΔW_warm ≈ U_R Σ_R Vh_R`
6. Frozen bases: `B_i = U_R[:, slice_i]`, `A_i = Vh_R[slice_i, :]`
7. Main Ravan training starts with `H_i = 0`, `scales_i = 1`

Raw client data never leaves the client. Extra initialization cost is reported separately in the results.

## Module Reference

```
federated/
  data.py          # 20 Newsgroups loading, IID and Dirichlet non-IID splits
  model.py         # DistilBERT factory, inject_lora, inject_ravan, param counting
  lora.py          # LoRALinear (FedIT adapter)
  ravan.py         # RavanLinear + gram_schmidt_init + svd_init
  client.py        # local_train(), evaluate()
  server.py        # fedit_* and ravan_* aggregation helpers
  warmup.py        # federated_svd_init()
  train_fedit.py   # FedIT training script (run with python -m federated.train_fedit)
  train_ravan.py   # Ravan training script (run with python -m federated.train_ravan)
  utils.py         # Result logging

jobs/
  submit_fedit.sh     # Slurm job script for FedIT
  submit_ravan_gs.sh  # Slurm job script for Ravan (Gram-Schmidt)
  submit_ravan_svd.sh # Slurm job script for Ravan (SVD)
  sweep.sh            # Submit full experiment grid

tests/
  test_aggregation.py  # Correctness tests (pytest)

results/              # Experiment outputs (auto-created)
logs/                 # Slurm stdout/stderr (auto-created by job scripts)
```

## Legacy Scripts

The root-level `bert_20newsgroups.py`, `bert_20newsgroups_ravan.py`, and `federated_bert/` directory contain earlier prototypes (tiny BERT, Flower-based FL). They are kept for reference but are **not** part of the current research pipeline.
