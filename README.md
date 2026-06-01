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

# Save top singular values for spectral analysis:
python -m federated.train_ravan --init svd ... --save_singular_values true
```

### Smoke tests (2 rounds, 5 steps)

```bash
python -m federated.train_fedit  --rounds 2 --local_steps 5 --split iid --seed 0
python -m federated.train_ravan  --init gram_schmidt --rounds 2 --local_steps 5 --split iid --seed 0
python -m federated.train_ravan  --init svd --rounds 2 --local_steps 5 --split iid --seed 0 \
    --warmup_steps 5 --warmup_clients 2

# Or run all 6 smoke configurations at once:
bash scripts/run_smoke_grid.sh
```

### Full 18-job grid (local)

```bash
bash scripts/run_smoke_grid.sh          # smoke (6 tiny runs)
# For full experiments, use the cluster sweep below
```

## Generating Paper Assets

Each training run writes raw metrics (JSON, CSV) and a per-run learning curve to `results/`.
After all runs finish, run the report-assets script to regenerate all paper-ready tables and figures:

```bash
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
```

The script can also be run after each partial batch of runs — it gracefully skips missing method/split combinations.

### Generated tables (`paper_assets/tables/`)

| File | Contents |
|---|---|
| `main_results.csv` / `.tex` | Method × split accuracy table, mean±std across seeds, gap vs FedIT |
| `gap_analysis.csv` | IID/Non-IID gaps relative to FedIT, SVD vs GS comparison |
| `init_costs.csv` / `.tex` | Warm-up cost breakdown per method |
| `setup_summary.csv` | Experiment hyper-parameters |
| `parameter_budget.csv` | Adapter, head, and total trainable params per method |

### Generated figures (`paper_assets/figures/`)

| File | Shows |
|---|---|
| `learning_curves_iid.png` | Test accuracy vs round, mean±std across seeds (IID) |
| `learning_curves_noniid.png` | Same for Non-IID split |
| `final_accuracy_by_method_split.png` | Grouped bar chart: method × split, error bars = std |
| `iid_vs_noniid_drop.png` | Per-method accuracy drop from IID to Non-IID |
| `gap_analysis.png` | Ravan advantage over FedIT for IID and Non-IID |
| `init_cost_vs_accuracy.png` | Initialization cost vs final Non-IID accuracy |
| `singular_values.png` | Warm-up ΔW singular value spectra (only if `--save_singular_values true`) |

## Parameter Budget

Default rank choices achieve approximate budget matching. For DistilBERT (d = 768):

```
r_Ravan ≈ sqrt(2 × d × r_LoRA / h) = sqrt(2 × 768 × 8 / 4) ≈ 55
```

| Method | Config | Adapter params (12 layers) | Head params | Total trainable |
|---|---|---|---|---|
| FedIT | rank=8 | 147,456 | ~591K | ~738K |
| Ravan | heads=4, rank=55 | 145,248 | ~591K | ~736K |

Per layer: FedIT = 2 × d × r = 12,288; Ravan = h × r² + h = 12,104.

## Correctness Tests

```bash
python -m pytest tests/ -v
```

| Test | Checks |
|---|---|
| `test_ravan_exact_aggregation` | `mean_c[Σ_i B_i(s H)_c,i A_i] == Σ_i B_i mean_c[(s H)_c,i] A_i` |
| `test_fedit_mismatch` | `mean(B_c@A_c) ≠ mean(B_c)@mean(A_c)` — verifies the known FedIT mismatch |
| `test_gram_schmidt_orthogonality` | B_i columns and A_i rows orthonormal across all heads |
| `test_svd_init_orthogonality` | same check for SVD-initialized bases |
| `test_ravan_zero_init_output` | adapter contributes zero at init (H=0) |
| `test_lora_zero_init_output` | LoRA adapter contributes zero at init (B=0) |
| `test_param_budget_counting` | FedIT rank=8 and Ravan h=4 r=55 adapter budgets agree within 5% |
| `test_result_asset_generation_with_dummy_data` | generate_report_assets produces all expected tables and figures |

## Running on DAIC (TU Delft Cluster)

### Environment setup (once)

```bash
# On a login node
module use /opt/insy/modulefiles
module load miniconda
conda create -n ravan python=3.10
conda activate ravan
pip install -r requirements.txt
```

### Submit individual jobs

```bash
sbatch jobs/submit_fedit.sh
sbatch jobs/submit_ravan_gs.sh
sbatch jobs/submit_ravan_svd.sh

# Override defaults
sbatch jobs/submit_fedit.sh --split iid --seed 2
sbatch jobs/submit_ravan_svd.sh --split noniid --seed 1
```

### Full sweep — 18 jobs (3 methods × 2 splits × 3 seeds)

```bash
bash jobs/sweep.sh
```

### Monitoring

```bash
squeue -u $USER                  # job status
seff <jobID>                     # efficiency after completion
tail -f logs/fedit_<jobID>.out   # live output
```

### Cluster specs (DAIC general partition)

| GPU | Count | VRAM |
|---|---|---|
| A40 | 84 | 46 GB |
| L40 | 18 | 49 GB |
| V100 | 11 | 32 GB |
| RTX 2080 Ti | 24 | 11 GB |

Partition `general`, QoS `short` (max 4 hours per job).
Request specific GPU: `#SBATCH --gres=gpu:a40:1`

### After sweep: generate paper assets on the cluster

```bash
# On a login node after all jobs complete:
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
# Then download paper_assets/ to your local machine
```

## Ravan Architecture

`RavanLinear` wraps a **frozen** `nn.Linear` and adds multi-head adapters:

```
output = W·x  +  Σ_h  scales[h] × B[h] @ H[h] @ A[h] @ x
```

- `B[h]` ∈ ℝ^{d_out × r}, `A[h]` ∈ ℝ^{r × d_in} — **frozen** after initialization
- `H[h]` ∈ ℝ^{r × r} — **trainable**, zero-initialized
- `scales[h]` ∈ ℝ — **trainable**, initialized to 1

Clients upload `{scales[h] × H[h]}` products; server averages exactly.

## Federated SVD Warm-Up (`--init svd`)

1. Sample `warmup_clients` (default 5) clients
2. Each trains a temporary LoRA model (rank R = heads × rank = 220)
3. Each client computes `ΔW_c = B_c @ A_c` per layer *(products, not factors)*
4. Server averages: `ΔW_warm = mean_c(ΔW_c)` — avoids FedIT mismatch
5. Truncated SVD: `ΔW_warm ≈ U_R Σ_R Vh_R`
6. Frozen bases: `B_i = U_R[:, slice_i]`, `A_i = Vh_R[slice_i, :]`
   (singular values not absorbed)
7. Main Ravan training starts with `H_i = 0`, `scales_i = 1`

Raw client data never leaves the client.  All warm-up costs are recorded in `summary.json`.

## Module Reference

```
federated/
  data.py          load_20newsgroups; IID and Dirichlet non-IID splits
  model.py         make_distilbert, inject_lora, inject_ravan, param counting
  lora.py          LoRALinear (FedIT adapter)
  ravan.py         RavanLinear + gram_schmidt_init + svd_init
  client.py        local_train(), evaluate()
  server.py        fedit_* and ravan_* aggregation helpers
  warmup.py        federated_svd_init() with timing and cost logging
  train_fedit.py   FedIT training script
  train_ravan.py   Ravan training script (both init modes)
  utils.py         Result logging, git hash, run naming
  plot.py          Per-run and comparison figures (called from training scripts)

scripts/
  generate_report_assets.py   Paper-ready tables + figures from all results
  run_smoke_grid.sh            Quick smoke test: all 3 methods, 2 splits

jobs/
  submit_fedit.sh              Slurm job: FedIT
  submit_ravan_gs.sh           Slurm job: Ravan-GS
  submit_ravan_svd.sh          Slurm job: Ravan-SVD
  sweep.sh                     Submit full 18-job grid

tests/
  test_aggregation.py          8 correctness tests (pytest)

results/                       Experiment outputs (auto-created)
paper_assets/                  Generated tables and figures (from script)
logs/                          Slurm stdout/stderr (auto-created by job scripts)
```

## Legacy Scripts

`bert_20newsgroups.py`, `bert_20newsgroups_ravan.py`, and `federated_bert/` are earlier
prototypes (tiny BERT, Flower-based FL). They are **not** part of the current research pipeline.
