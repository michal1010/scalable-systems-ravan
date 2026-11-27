# Scalable Systems — RAVAN

Research project comparing three federated fine-tuning strategies for 20-class text classification
using **DistilBERT** (`distilbert-base-uncased`, 66M parameters).

**Paper:** "Reimplementing and Extending Ravan with Federated Data-Aware Initialization"
Anton & Jakomulski, Delft University of Technology, 2026.

---

## Methods overview

| Method | Adapter | Frozen | Trainable | Aggregation |
|---|---|---|---|---|
| **FedIT** | LoRA | `W` | `A`, `B`, head | FedAvg on `A` and `B` separately (inexact) |
| **Ravan-GS** | Ravan | `W`, `B[h]`, `A[h]` | `H[h]`, `scales[h]`, head | FedAvg on `s·H` products (exact) |
| **Ravan-SVD** | Ravan | `W`, `B[h]`, `A[h]` | `H[h]`, `scales[h]`, head | FedAvg on `s·H` products (exact) |

Ravan-GS and Ravan-SVD are identical after initialization — the only difference is how the frozen
bases `B[h]` and `A[h]` are constructed.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running experiments

### Default hyperparameters

| Parameter | FedIT | Ravan-GS / Ravan-SVD |
|---|---|---|
| `--rounds` | 50 | 50 |
| `--clients` | 20 | 20 |
| `--clients_per_round` | 3 | 3 |
| `--local_steps` | 50 | 50 |
| `--lr` | 1e-3 | 5e-4 |
| `--batch_size` | 16 | 16 |
| `--rank` | 8 | 55 |
| `--heads` | — | 4 |
| `--dirichlet_alpha` | 0.3 | 0.3 |
| `--max_length` | 128 | 128 |
| `--warmup_clients` | — | 5 |
| `--warmup_steps` | — | 50 |

### FedIT

```bash
python -m federated.train_fedit \
    --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --rank 8 --lr 1e-3 --batch_size 16
```

### Ravan — Gram-Schmidt initialization

```bash
python -m federated.train_ravan \
    --init gram_schmidt --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4 --batch_size 16
```

### Ravan — SVD warm-up initialization

```bash
python -m federated.train_ravan \
    --init svd --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4 --batch_size 16 \
    --warmup_clients 5 --warmup_steps 50
```

To save singular value spectra for plotting (adds `singular_values.png` to paper assets):

```bash
python -m federated.train_ravan --init svd ... --save_singular_values true
```

### Additional CLI flags (all scripts)

| Flag | Default | Description |
|---|---|---|
| `--eval_every` | 1 | Evaluate on test set every N rounds |
| `--output_dir` | `results/` | Root directory for run outputs |
| `--device` | auto | e.g. `cuda` or `cpu` |
| `--cache_dir` | None | HuggingFace model/tokenizer cache |
| `--train_classifier_head` | `true` | Whether to train the shared classification head |
| `--save_checkpoints` | `false` | Save final model as `checkpoint_final.pt` |
| `--warmup_lr` | same as `--lr` | Learning rate for SVD warm-up phase |
| `--warmup_weighting` | `uniform` | `uniform` or `examples` (weight by dataset size) |

Flags `--limit_train_examples` and `--limit_test_examples` are for smoke tests only; do not
use in real experiments.

---

## Smoke tests and validation

```bash
# 6 tiny runs: all methods × splits, 2 rounds / 5 steps, 500 train examples, 200 test examples
bash scripts/run_smoke_grid.sh

# Set LIMIT=0 to disable example limiting (still uses 2 rounds / 5 steps)
LIMIT=0 bash scripts/run_smoke_grid.sh

# Fast contract check (~15s): verifies protocol constants, data preprocessing,
# optimizer settings, warm-up comm counts, parameter accounting, zero init
python -m scripts.validate_experiment_contract

# Full preflight: pytest → contract check → smoke grid → asset generation
bash scripts/preflight.sh
```

---

## Correctness tests

```bash
# All 12 tests
python -m pytest tests/ -v

# Single test
python -m pytest tests/test_aggregation.py::test_ravan_exact_aggregation -v
```

Tests in `tests/test_aggregation.py`:

| Test | What it checks |
|---|---|
| `test_ravan_exact_aggregation` | Averaging `s·H` products = exact ΔW aggregation |
| `test_fedit_mismatch` | `mean(B_c)@mean(A_c) ≠ mean(B_c@A_c)` — verifies the FedIT baseline mismatch exists |
| `test_gram_schmidt_orthogonality` | GS-initialized `B[h]` columns and `A[h]` rows are orthonormal |
| `test_svd_init_orthogonality` | Same for SVD-initialized bases |
| `test_ravan_zero_init_output` | With `H=0`, adapter output is zero |
| `test_lora_zero_init_output` | With `B=0`, LoRA output is zero |
| `test_param_budget_counting` | FedIT `r=8` and Ravan `h=4, r=55` have matching adapter param counts |
| `test_result_asset_generation_with_dummy_data` | Tables and figures are generated from synthetic results |
| `test_warmup_factor_communication_count` | Warm-up comm cost = 20,275,200 params |
| `test_ravan_svd_head_matches_pretrained` | SVD init preserves pretrained head weights |
| `test_nonempty_client_splits` | All clients receive at least one sample in both IID and non-IID splits |
| `test_parameter_accounting` | Exact trainable and communicated param counts for all methods |

---

## Paper asset generation

```bash
# Safe to re-run with partial results — missing method/split/seed combos are skipped
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
```

### Tables (`paper_assets/tables/`)

| File | Contents |
|---|---|
| `main_results.csv` / `.tex` | Method × split accuracy (mean±std), adapter params, comm params, gaps |
| `init_costs.csv` / `.tex` | Warm-up stages, clients, temp. rank, extra comm., SVD runtime, Non-IID acc. |
| `gap_analysis.csv` | Per-method accuracy gaps vs FedIT; SVD vs GS |
| `setup_summary.csv` | Shared experimental setup (model, optimizer, splits, seeds, hyperparameters) |
| `parameter_budget.csv` | Adapter, head, and total trainable/communicated params per method |

### Figures (`paper_assets/figures/`)

| File | Shows |
|---|---|
| `learning_curves_iid.png` | Test accuracy vs round, mean±std (IID) |
| `learning_curves_noniid.png` | Same for Non-IID |
| `final_accuracy_by_method_split.png` | Grouped bar chart, method × split, error bars = std |
| `iid_vs_noniid_drop.png` | Per-method IID − Non-IID accuracy drop |
| `gap_analysis.png` | Ravan accuracy gain relative to FedIT baseline |
| `init_cost_vs_accuracy.png` | Warm-up communication cost vs final Non-IID accuracy |
| `singular_values.png` | ΔW singular value spectra (only if `--save_singular_values true` was used) |

`paper_assets/manifest.json` records the ISO timestamp of last generation, which of the
18 expected `(method, split, seed)` configs were found, and which are still missing.

---

## Result structure

Each run writes to a timestamped directory `results/<run_name>/`:

```
results/
  all_results.csv                  ← master index, one row per run, appended on each run
  <method>_<split>_seed<N>_<ts>/
    config.json                    ← all CLI args + param counts + device info
    summary.json                   ← flat dict: final/best acc, all cost fields, git hash, timestamp
    rounds.csv                     ← per-round: round, test_acc, elapsed_seconds,
    │                                           selected_clients, train_runtime_seconds
    curve.png                      ← per-run learning curve
    checkpoint_final.pt            ← (only if --save_checkpoints true)
```

Run names follow the pattern:
- `fedit_iid_seed0_20260610_140000`
- `ravan_gram_schmidt_noniid_seed2_20260610_140000`
- `ravan_svd_iid_seed1_20260610_140000`

`summary.json` key fields:

| Key | Description |
|---|---|
| `final_acc` / `best_acc` | Test accuracy at last round / best round |
| `communicated_adapter_params_per_client` | Adapter params uploaded per client per round |
| `total_main_communication_params` | `comm_per_client × clients_per_round × rounds` |
| `warmup_communicated_params` | Extra params for SVD warm-up (0 for GS) |
| `warmup_train_runtime_s` / `warmup_svd_runtime_s` | Timing breakdown for SVD init |
| `total_runtime_s` | Wall-clock time for the full run |
| `git_commit` | Commit hash at run time |

---

## Results snapshot

> **Note:** The results below are from smoke-test runs (2 rounds, 5 steps, 500 training
> examples, seed 0 only). They are not representative of converged results. The full
> 18-job grid (3 methods × 2 splits × 3 seeds, 50 rounds, 50 steps) is pending.

| Method | Init | IID Acc | Non-IID Acc | Non-IID drop | Adapter comm./client |
|---|---|---|---|---|---|
| FedIT | LoRA | 17.5% | 14.0% | 3.5 pp | 147,456 |
| Ravan-GS | Gram-Schmidt | 15.5% | 5.5% | 10.0 pp | 145,200 |
| Ravan-SVD | SVD warm-up | 12.5% | 6.0% | 6.5 pp | 145,200 |

Parameter budget (shared across all runs):

| Method | Adapter params | Head params | Total trainable | Comm./round |
|---|---|---|---|---|
| FedIT | 147,456 | 605,972 | 753,428 | 2,260,284 |
| Ravan-GS | 145,248 | 605,972 | 751,220 | 2,253,516 |
| Ravan-SVD | 145,248 | 605,972 | 751,220 | 2,253,516 |

---

## Running on DAIC (TU Delft cluster)

The DAIC jobs use an **Apptainer container** (`jobs/container/ravan-experiments.sif`).
Build and deploy the container before submitting jobs.

### Step 0 — Build the container (once, on a machine with Apptainer)

```bash
cd jobs/container
./build.sh          # produces ravan-experiments.sif (~5 min)
```

### Step 1 — Deploy the container to DAIC

```bash
cd jobs/container
./deploy.sh <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<PROJECT>/apptainer/ravan-experiments.sif
```

### Step 2 — Upload the repo to project storage

```bash
# On campus / with eduVPN
rsync --progress -avz --no-perms \
    ./ <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<PROJECT>/scalable-systems-ravan/

# Off campus — route through bastion
rsync --progress -avz --no-perms \
    -e "ssh -J <NetID>@linux-bastion.tudelft.nl" \
    ./ <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<PROJECT>/scalable-systems-ravan/
```

### Step 3 — Point job scripts at the container image

In all three `jobs/submit_*.sh` files (or via the `APPTAINER_IMAGE` env var), set:

```bash
export APPTAINER_IMAGE=/tudelft.net/staff-umbrella/<PROJECT>/apptainer/ravan-experiments.sif
```

The `container_env.sh` sourced by all job scripts automatically sets
`HF_HOME` to `<project_root>/.cache/huggingface` inside the container to avoid home quota issues.

### Step 4 — Submit 18 jobs

```bash
cd /tudelft.net/staff-umbrella/<PROJECT>/scalable-systems-ravan
mkdir -p logs results                # Slurm creates the log file before the script runs
bash jobs/sweep.sh                   # 3 methods × 2 splits × 3 seeds = 18 jobs
```

To submit a single job or override split/seed:

```bash
sbatch jobs/submit_fedit.sh                        # default: noniid, seed 0
sbatch jobs/submit_fedit.sh --split iid --seed 2   # override via $@
```

Slurm settings (all jobs): `--partition=general`, `--qos=short`, `--time=4:00:00`,
1 GPU, 4 CPUs, 16 GB RAM. Logs go to `logs/<jobname>_<jobid>.out/.err`.

### Step 5 — Monitor

```bash
squeue -u $USER
tail -f logs/fedit_<jobID>.out
seff <jobID>                         # CPU/memory efficiency after completion
```

### Step 6 — Generate paper assets (login node, after all jobs finish)

```bash
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
```

### Step 7 — Download results

```bash
rsync --progress -avz --no-perms \
    <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<PROJECT>/scalable-systems-ravan/results/ \
    ./results/

rsync --progress -avz --no-perms \
    <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<PROJECT>/scalable-systems-ravan/paper_assets/ \
    ./paper_assets/
```

---

## Running on a GPU VM (no Slurm)

Scripts under `jobs/gpu_vm/` run the same 18-job grid directly on a VM with a GPU,
without Slurm or a container. Results go to `results_a100_vm/`, logs to `logs_a100_vm/`.

```bash
# Full 18-job sweep (default: 2 parallel jobs)
bash jobs/gpu_vm/sweep.sh

# Control parallelism
MAX_PARALLEL=3 bash jobs/gpu_vm/sweep.sh   # more parallel jobs (more GPU memory needed)
MAX_PARALLEL=1 bash jobs/gpu_vm/sweep.sh   # sequential (safest if OOM)
```

Single runs:

```bash
bash jobs/gpu_vm/run_fedit.sh --split noniid --seed 0
bash jobs/gpu_vm/run_ravan_gs.sh --split iid --seed 1
bash jobs/gpu_vm/run_ravan_svd.sh --split noniid --seed 2
```

Environment overrides:

```bash
CONDA_ENV=ravan MAX_PARALLEL=2 bash jobs/gpu_vm/sweep.sh
PYTHON_BIN=/path/to/python bash jobs/gpu_vm/run_fedit.sh
CUDA_VISIBLE_DEVICES=0 bash jobs/gpu_vm/sweep.sh
HF_HOME=/scratch/.cache/huggingface bash jobs/gpu_vm/sweep.sh
```

After the sweep, `jobs/gpu_vm/aggregate_results.py` is called automatically to merge
per-run subdirectories from `results_a100_vm/sweep_<ts>/parts/` into
`results_a100_vm/sweep_<ts>/final/` and rebuild `all_results.csv`.

---

## Component reference

### `federated/data.py` — dataset and client splits

Loads 20 Newsgroups via `sklearn.datasets.fetch_20newsgroups` with
`remove=("headers", "footers", "quotes")` applied to both train and test subsets.

Tokenization uses `DistilBertTokenizerFast` (`distilbert-base-uncased`) with
`padding="max_length"`, `truncation=True`, `max_length=128` (default). All client
DataLoaders and the central test loader contain `(input_ids, attention_mask, labels)` tuples.

**IID split** (`iid_split`): shuffles all training indices with `np.random.default_rng(seed)` then
splits evenly using `np.array_split`. Each client receives approximately the same number of
examples with a balanced class distribution.

**Non-IID split** (`dirichlet_split`): for each of the 20 classes, draws per-client proportions
from a `Dirichlet(α · 1_K)` distribution (`α=0.3` by default) and assigns indices accordingly.
Rounding discrepancies are resolved by distributing leftover samples round-robin. This creates
strong label skew at `α=0.3`.

Client DataLoaders are created with `shuffle=True`, `drop_last=False`. The central test
DataLoader uses no shuffle.

---

### `federated/lora.py` — LoRA adapter (FedIT)

`LoRALinear` wraps a frozen `nn.Linear` and adds a low-rank update:

```
y = W·x  +  scaling × (x @ A^T) @ B^T
```

- `A` ∈ ℝ^{r × d_in} — trainable, initialized with **Kaiming uniform** (same as `nn.Linear` default)
- `B` ∈ ℝ^{d_out × r} — trainable, initialized to **zero** (so the adapter output is zero at init)
- `scaling = 1.0` (fixed; not the `α/r` convention from the original LoRA paper)
- `W` is frozen in place; its `requires_grad` is set to `False`

Parameter counts per layer: `r·d_in + d_out·r`. For `d=768`, `r=8`: **12,288 params per layer**.

---

### `federated/ravan.py` — Ravan adapter

`RavanLinear` wraps a frozen `nn.Linear` and adds `H` multi-head adapters:

```
y = W·x  +  Σ_{h=1}^{H}  scales[h] × B[h] @ H[h] @ A[h] @ x
```

Equivalently in forward order: `x → A[h] → H[h] → B[h] → scaled add`.

Dimensions per head `h`:
- `B[h]` ∈ ℝ^{d_out × r} — **frozen buffer** (not a parameter; travels with `.to(device)`)
- `A[h]` ∈ ℝ^{r × d_in} — **frozen buffer**
- `H[h]` ∈ ℝ^{r × r} — **trainable**, zero-initialized → adapter output is zero at init
- `scales[h]` ∈ ℝ — **trainable**, initialized to `1.0`

Stored shapes: `B` ∈ ℝ^{H × d_out × r}, `A` ∈ ℝ^{H × r × d_in}, `H` ∈ ℝ^{H × r × r},
`scales` ∈ ℝ^H.

Parameter counts per layer: `H·r² + H` (trainable). For `H=4`, `r=55`: **12,104 params per layer**.

**Gram-Schmidt initialization** (`gram_schmidt_init`):
Constructs globally orthonormal bases using QR decomposition.

1. Sample `B_rand ∈ ℝ^{d_out × R}` where `R = H·r`, then `Q_B, _ = torch.linalg.qr(B_rand)`.
   Take the first `R` columns of `Q_B` → `B_orth ∈ ℝ^{d_out × R}` with orthonormal columns.
2. Sample `A_rand ∈ ℝ^{R × d_in}`, compute `Q_A, _ = torch.linalg.qr(A_rand.T)`.
   Take `A_orth = Q_A.T[:R, :]` → rows are orthonormal.
3. Slice into per-head blocks: `B[h] = B_orth[:, h·r:(h+1)·r]`, `A[h] = A_orth[h·r:(h+1)·r, :]`.

All `H·r` columns of `B` and all `H·r` rows of `A` form a globally orthonormal set.

**SVD initialization** (`svd_init`):
Takes pre-computed truncated SVD matrices `(U_R, Vh_R)` where `R = H·r`.

1. `B[h] = U_R[:, h·r:(h+1)·r]` — left singular vectors, sliced per head.
2. `A[h] = Vh_R[h·r:(h+1)·r, :]` — right singular vectors (V^H form), sliced per head.

Singular values are **not absorbed** into `B` or `A`. The bases span the principal subspace of
the warm-up `ΔW`, but `H` starts at zero so the adapter still contributes nothing at init.

---

### `federated/model.py` — model construction and adapter injection

`make_distilbert` loads `distilbert-base-uncased` (6 transformer layers, hidden size 768,
20 output classes). All backbone parameters are frozen (`requires_grad=False`). The
classification head (`pre_classifier: Linear(768, 768)` and `classifier: Linear(768, 20)`)
is unfrozen by default (`train_head=True`).

`inject_lora` replaces `q_lin` and `v_lin` in **all 6 transformer layers** with `LoRALinear`.
`inject_ravan` does the same with `RavanLinear`.

This gives **12 adapted layers** (6 blocks × 2 projections: query and value).

`count_communicated_per_round` counts total params uploaded per client per round (adapter + head).

`count_adapter_communicated` counts adapter-only params per client per round (head excluded):
- FedIT: `lora_A.numel() + lora_B.numel()` per layer = **147,456**
- Ravan: `H.numel()` per layer (scales absorbed into `s·H` before upload) = **145,200**

This value is saved as `communicated_adapter_params_per_client` in `summary.json` and used
in the Table 4 "Adapter comm." column of `main_results.tex`.

---

### `federated/client.py` — local training and evaluation

`local_train` runs `local_steps` gradient updates on the client's DataLoader using
**AdamW** (`weight_decay=0.01`) and **CrossEntropyLoss**. The step count is exact:
the DataLoader is looped repeatedly until `local_steps` is reached (not epoch-based).
Optimizer state is created fresh each time `local_train` is called (no persistence across rounds).

`evaluate` computes top-1 accuracy on a DataLoader under `torch.no_grad()`.

---

### `federated/server.py` — aggregation

**FedIT aggregation** (`fedit_aggregate`):
Receives a list of state dicts containing `{lora_A, lora_B, head_params}` from each client.
Aggregates by element-wise mean across clients for every tensor key.

This is **intentionally inexact**: `mean(B_c) @ mean(A_c) ≠ mean(B_c @ A_c)` in general,
because the average of outer products does not equal the outer product of averages.

**Ravan aggregation** (`ravan_aggregate`):
Clients upload `{s[h] · H[h]}` products per layer (keys prefixed `ravan_sH/`), plus head params
(keys prefixed `head/`). The server averages all tensors element-wise.

On `ravan_load_global`, the averaged `s·H` tensor is loaded directly into `module.H`, and
`module.scales` is reset to `1.0`. Because `B[h]` and `A[h]` are frozen and identical across all
clients, averaging the `s·H` products gives an **exact** aggregation of the true per-client
weight updates:

```
mean_c [ Σ_h  B[h] (s_{c,h} H_{c,h}) A[h] ]
  = Σ_h  B[h] ( mean_c [ s_{c,h} H_{c,h} ] ) A[h]
```

The equality holds because `B[h]` and `A[h]` are constant across clients and rounds.

Both methods share the same head FedAvg logic.

---

### `federated/warmup.py` — federated SVD warm-up (Ravan-SVD only)

`federated_svd_init` implements the data-aware initialization for Ravan-SVD:

1. **Client selection**: samples `warmup_clients` clients uniformly at random using
   `np.random.default_rng(seed + 9999)` (separate seed from the main FL loop).

2. **Warm-up training**: each selected client trains a temporary LoRA model with
   rank `R = heads × per_head_rank` (default: `4 × 55 = 220`) for `warmup_steps`
   gradient steps using the same `local_train` function as the main FL loop.

3. **Factor upload**: each client uploads the temporary LoRA factors `B_c ∈ ℝ^{d_out × R}`
   and `A_c ∈ ℝ^{R × d_in}` per adapted layer — **not** the full product `ΔW_c`.
   The server reconstructs `ΔW_c = B_c @ A_c` internally before aggregating products.
   This avoids the FedIT factor-averaging mismatch while keeping the upload smaller than `ΔW_c`.

4. **Server aggregation**: the server accumulates weighted `ΔW_c` matrices.
   - `warmup_weighting="uniform"` (default): each client has weight `1.0`; result is the mean.
   - `warmup_weighting="examples"`: weight proportional to `len(client_loaders[cid].dataset)`.

5. **Truncated SVD**: for each averaged `ΔW_warm ∈ ℝ^{d_out × d_in}`, compute
   `U, S, Vh = torch.linalg.svd(ΔW_warm, full_matrices=False)`, then keep the top-`R`
   components: `U_R = U[:, :R]`, `Vh_R = Vh[:R, :]`.

6. **Basis assignment**: sliced as in `svd_init`. Singular values `S` are **not absorbed**
   into `U_R` or `Vh_R`. The Ravan model is then initialized with `H=0`, `scales=1`.
   **The trained warm-up head and adapter state are discarded. The main Ravan model is
   constructed fresh from pretrained DistilBERT weights.**

7. **Cost recording**: all warm-up costs (clients, steps, rank, communicated params,
   train runtime, SVD runtime) are recorded in `summary.json` for reporting.

Raw client data never leaves the client. Each warm-up client uploads only the temporary LoRA
factors `B_c` and `A_c` — the warm-up head, backbone state, and LoRA adapter are fully
discarded after `federated_svd_init` returns.

**Warm-up extra communication cost** (reported in `warmup_communicated_params`):
Per layer per client: `R × (d_out + d_in) = 220 × (768 + 768) = 337,920`.
Total: `5 clients × 12 layers × 337,920 = 20,275,200` parameters.

The main FL loop re-applies the same random seeds after warm-up completes, so model
initialization is deterministic and independent of whether the SVD warm-up ran.

---

### `federated/train_fedit.py` and `federated/train_ravan.py` — training loop

Both scripts share the same FL loop structure:

1. Set random seeds: `random.seed(seed)`, `np.random.seed(seed)`, `torch.manual_seed(seed)`,
   `torch.cuda.manual_seed_all(seed)`.
2. Build federated data loaders (`data.build_federated_loaders`).
3. (Ravan-SVD only) Run warm-up (`warmup.federated_svd_init`) using seed `seed + 9999`.
4. Re-apply the same random seeds (step 1) before model construction, so model initialization
   is deterministic and independent of whether the SVD warm-up ran.
5. Construct DistilBERT and inject adapters.
6. **FL loop** for `rounds` rounds:
   - Sample `clients_per_round` clients without replacement using
     `np.random.default_rng(seed + 1000)`.
   - For each selected client: load global state → `local_train` → extract updated state.
   - Aggregate client states.
   - Evaluate on the central test set every `eval_every` rounds (default: every round).
7. Save `summary.json`, `rounds.csv`, `config.json`, per-run learning curve.
8. Regenerate all comparison figures from `results/` (calling `plot.plot_all`).

---

### `federated/utils.py` — result persistence

Each run writes to a unique directory `results/<method>_<split>_seed<N>_<YYYYMMDD_HHMMSS>/`:
- `config.json` — all CLI arguments plus parameter counts
- `summary.json` — flat dict with final/best accuracy, all cost fields, git hash, timestamp
- `rounds.csv` — one row per round: `round`, `test_acc`, `elapsed_seconds`, `selected_clients`, `train_runtime_seconds`
- `curve.png` — per-run learning curve

`results/all_results.csv` is a master index appended after each run (one row per run).

---

### `federated/plot.py` — live comparison figures (updated after each run)

`plot_single_run` saves the per-run accuracy curve to `results/<run_name>/curve.png`.

`plot_all` regenerates four comparison figures in `results/` from all completed runs:
- `comparison_iid.png` — all methods on the IID split; shaded ±1 std when multiple seeds exist
- `comparison_noniid.png` — same for the Non-IID split
- `iid_vs_noniid.png` — side-by-side IID/Non-IID panels (the main heterogeneity figure)
- `final_accuracy.png` — grouped bar chart: method × split, error bars = std over seeds

When multiple runs share the same `(method, split, seed)` key, the most recent run
(alphabetically last `run_name`) is kept.

---

## Parameter budget

Default rank choices are chosen so that FedIT and Ravan have approximately equal adapter
parameter counts.

For DistilBERT (`d = 768`, 12 adapted layers):
```
FedIT per layer:   A + B = r·d + d·r = 2·768·8 = 12,288
Ravan per layer:   H + scales = H·r² + H = 4·55² + 4 = 12,104

Total adapter params:
  FedIT:  12,288 × 12 = 147,456
  Ravan:  12,104 × 12 = 145,248

Budget-matching formula:  r_Ravan ≈ sqrt(2 · d · r_LoRA / H) = sqrt(2·768·8/4) ≈ 55
```

Head params (shared across methods): `pre_classifier` (768×768 + 768 = 590,592) +
`classifier` (768×20 + 20 = 15,380) = **605,972** (≈ 606K) head params.

Total trainable:
- FedIT: 147,456 + 605,972 = **753,428**
- Ravan: 145,248 + 605,972 = **751,220**

Ravan trains 48 additional scale parameters (one per head per layer) that are absorbed into
the `s·H` upload and not communicated separately. FedIT communicates 753,428 per client per
round; Ravan communicates 751,172.

---

## Data flow summary

```
fetch_20newsgroups()
        │
        ▼
  tokenize (max_length=128)
        │
        ├─── IID split ──────────────────────────────── 20 client DataLoaders
        └─── Dirichlet(α=0.3) non-IID split ─────────── 20 client DataLoaders

         Central test DataLoader (full sklearn test split)


Ravan-SVD warm-up (once, before main FL):

  select warmup_clients (seed = main_seed + 9999)
       │ local_train (LoRA, rank=220, warmup_steps)
       ▼
  client uploads B_c [768×220] and A_c [220×768] per layer  ← factors, not ΔW
       │ server reconstructs ΔW_c = B_c @ A_c, aggregates, runs SVD
       ▼
  frozen bases U_R, Vh_R → inject_ravan  (H=0, scales=1)
  warm-up model discarded; main seeds re-applied


Main FL loop (train_fedit.py / train_ravan.py):

  global_state (server)
       │
       ▼
  sample clients_per_round (seed = main_seed + 1000)
       │ load global_state
       ▼
  local_train (AdamW, local_steps gradient steps, CrossEntropyLoss)
       │ extract updated state
       ▼
  fedit_aggregate  ─── mean(A_c), mean(B_c) separately  (INEXACT)
  ravan_aggregate  ─── mean(s_c·H_c) products            (EXACT)
       │
       ▼
  evaluate on central test set → test_acc
       │
       ▼
  save rounds.csv / summary.json / update all_results.csv
  regenerate results/comparison_*.png
```

---

## Legacy scripts

`bert_20newsgroups.py`, `bert_20newsgroups_ravan.py`, and `federated_bert/` are earlier
prototypes using a different BERT variant and the Flower framework. They are **not** part of
the current research pipeline and should be ignored.
