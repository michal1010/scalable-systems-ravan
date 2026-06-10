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

### FedIT

```bash
python -m federated.train_fedit \
    --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --rank 8 --lr 1e-3
```

### Ravan — Gram-Schmidt initialization

```bash
python -m federated.train_ravan \
    --init gram_schmidt --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4
```

### Ravan — SVD warm-up initialization

```bash
python -m federated.train_ravan \
    --init svd --split noniid --seed 0 --rounds 50 \
    --clients 20 --clients_per_round 3 --local_steps 50 \
    --heads 4 --rank 55 --lr 5e-4 \
    --warmup_clients 5 --warmup_steps 50
```

### Smoke tests (fast local check, 2 rounds / 5 steps)

```bash
bash scripts/run_smoke_grid.sh
```

This runs all 3 methods × 2 splits, limiting train data to 500 examples by default. Pass `LIMIT=0`
to use the full dataset.

### Full 18-job grid (cluster)

```bash
bash jobs/sweep.sh          # 3 methods × 2 splits × 3 seeds on DAIC
```

### Generate paper assets

```bash
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
```

Can be run after each partial batch of results — missing method/split combinations are skipped.

---

## Correctness tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_aggregation.py::test_ravan_exact_aggregation -v  # single test
```

12 tests covering: Ravan exact aggregation, FedIT mismatch, GS orthogonality, SVD orthogonality,
zero initial adapter output (Ravan + LoRA), parameter budget counting, report asset generation,
warm-up factor communication count (20,275,200), SVD head isolation, non-empty client splits,
and exact parameter accounting for all methods.

---

## Validation

```bash
# Fast contract check (~15s): verifies all protocol constants against expected values
python -m scripts.validate_experiment_contract

# Full preflight (pytest → contract → smoke grid → asset generation)
bash scripts/preflight.sh
```

`validate_experiment_contract` checks 7 areas: static constants (`MODEL_NAME`, `NUM_LABELS`,
`MAX_LENGTH`), data preprocessing (`remove=headers,footers,quotes`), optimizer settings
(AdamW, `weight_decay=0.01`, CrossEntropyLoss), warm-up protocol (client uploads factors;
server reconstructs `ΔW_c = B_c @ A_c`; comm counts `B_c.numel() + A_c.numel()`),
exact adapter param counts for both methods, zero initial adapter output, and SVD singular
values not absorbed.

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

**Default hyperparameters:**

| Parameter | FedIT | Ravan-GS / Ravan-SVD |
|---|---|---|
| `--rounds` | 50 | 50 |
| `--clients` | 20 | 20 |
| `--clients_per_round` | 3 | 3 |
| `--local_steps` | 50 | 50 |
| `--lr` | 1e-3 | 5e-4 |
| `--batch_size` | 16 | 16 |
| `--dirichlet_alpha` | 0.3 | 0.3 |
| `--max_length` | 128 | 128 |
| `--rank` | 8 | 55 |
| `--heads` | — | 4 |
| `--warmup_clients` | — | 5 |
| `--warmup_steps` | — | 50 |

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

### `scripts/generate_report_assets.py` — paper tables and figures

Run after all experiments complete. Reads `results/all_results.csv` and per-run `rounds.csv`
files. Outputs to `paper_assets/`:

**Tables** (`paper_assets/tables/`):

| File | Contents |
|---|---|
| `main_results.csv` / `.tex` | Table 4: Method × split accuracy (mean±std), adapter trainable, adapter comm., gap vs FedIT, IID-to-Non-IID drop |
| `init_costs.csv` / `.tex` | Table 5: Ravan init comparison — warm-up stages, clients, temp. rank, extra comm., SVD runtime, main comm., Non-IID acc. |
| `gap_analysis.csv` | Per-method accuracy gaps: vs FedIT for both splits, SVD vs GS |
| `setup_summary.csv` | Shared experimental setup (model, optimizer, splits, seeds, all hyperparameters) |
| `parameter_budget.csv` | Adapter, head, and total trainable/communicated params per method |

`paper_assets/manifest.json` records which of the 18 expected `(method, split, seed)` configs
were found and which are still missing, with ISO timestamp of last generation.

**Figures** (`paper_assets/figures/`):

| File | Shows |
|---|---|
| `learning_curves_iid.png` | Test accuracy vs round, mean±std (IID) |
| `learning_curves_noniid.png` | Same for Non-IID |
| `final_accuracy_by_method_split.png` | Grouped bar chart, method × split, error bars = std |
| `iid_vs_noniid_drop.png` | Per-method IID − Non-IID accuracy drop |
| `gap_analysis.png` | Ravan accuracy gain relative to FedIT baseline |
| `init_cost_vs_accuracy.png` | Warm-up communication cost vs final Non-IID accuracy |
| `singular_values.png` | ΔW singular value spectra (only if `--save_singular_values true` was used) |

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

Ravan trains 48 additional scale parameters (one per head per layer) that are absorbed into the s·H upload and not communicated separately. FedIT communicates 753,428 per client per round; Ravan communicates 751,172.

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

  select warmup_clients
       │ local_train (LoRA, rank=220)
       ▼
  client uploads B_c [768×220] and A_c [220×768] per layer  ← factors, not ΔW
       │ server reconstructs ΔW_c = B_c @ A_c, aggregates, runs SVD
       ▼
  frozen bases U_R, Vh_R → inject_ravan  (H=0, scales=1)
  warm-up model discarded


Main FL loop (train_fedit.py / train_ravan.py):

  global_state (server)
       │
       ▼
  sample clients_per_round
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
```

---

## Running on DAIC (TU Delft cluster)

### Storage

Put the repo on project storage to avoid home quota issues:

```bash
/tudelft.net/staff-umbrella/<DAIC_PROJECT>/scalable-systems-ravan/
```

### Step 1 — Upload

```bash
# On campus / with eduVPN
rsync --progress -avz --no-perms \
    ./ <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<DAIC_PROJECT>/scalable-systems-ravan/

# Off campus — route through bastion
rsync --progress -avz --no-perms \
    -e "ssh -J <NetID>@linux-bastion.tudelft.nl" \
    ./ <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<DAIC_PROJECT>/scalable-systems-ravan/
```

### Step 2 — Environment (once, on login node)

```bash
module use /opt/insy/modulefiles
module load miniconda
conda create -n ravan python=3.10 -y
conda activate ravan
pip install -r requirements.txt
```

Set HuggingFace cache to project storage in all three `jobs/submit_*.sh` files:

```bash
export HF_HOME=/tudelft.net/staff-umbrella/<DAIC_PROJECT>/.cache/huggingface
```

### Step 3 — Submit 18 jobs

```bash
cd /tudelft.net/staff-umbrella/<DAIC_PROJECT>/scalable-systems-ravan
bash jobs/sweep.sh        # 3 methods × 2 splits × 3 seeds
```

Each job: `--partition=general`, `--qos=short`, `--time=4:00:00`, 1 GPU, 4 CPUs, 16 GB RAM.

### Step 4 — Monitor

```bash
squeue -u $USER
tail -f logs/fedit_<jobID>.out
seff <jobID>
```

### Step 5 — Generate paper assets (login node, after all jobs)

```bash
python -m scripts.generate_report_assets --results_dir results --out_dir paper_assets
```

### Step 6 — Download results

```bash
rsync --progress -avz --no-perms \
    <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<DAIC_PROJECT>/scalable-systems-ravan/paper_assets/ \
    ./paper_assets/

rsync --progress -avz --no-perms \
    <NetID>@login.daic.tudelft.nl:/tudelft.net/staff-umbrella/<DAIC_PROJECT>/scalable-systems-ravan/results/ \
    ./results/
```

---

## Legacy scripts

`bert_20newsgroups.py`, `bert_20newsgroups_ravan.py`, and `federated_bert/` are earlier
prototypes using a different BERT variant and the Flower framework. They are **not** part of
the current research pipeline and should be ignored.
