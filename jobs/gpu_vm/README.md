# A100 VM Jobs

These scripts run the same experiment settings as `jobs/submit_*.sh`, but directly
on a VM with an A100 GPU. They do not use Slurm or Apptainer.

Run from the project root:

```bash
bash jobs/gpu_vm/run_fedit.sh
bash jobs/gpu_vm/run_ravan_gs.sh
bash jobs/gpu_vm/run_ravan_svd.sh
```

Run the full grid:

```bash
bash jobs/gpu_vm/sweep.sh
```

The sweep runs 18 experiments:

- FedIT, Ravan-GS, Ravan-SVD
- `iid` and `noniid`
- seeds `0 1 2`

By default, the sweep runs two experiments at the same time on the A100:

```bash
MAX_PARALLEL=2 bash jobs/gpu_vm/sweep.sh
```

For an A100 with enough memory, you can try:

```bash
MAX_PARALLEL=3 bash jobs/gpu_vm/sweep.sh
```

If GPU memory becomes tight, use:

```bash
MAX_PARALLEL=1 bash jobs/gpu_vm/sweep.sh
```

Configure Ravan heads/rank from the sweep command:

```bash
bash jobs/gpu_vm/sweep.sh --heads 16 --rank 28
```

FedIT rank can be changed separately:

```bash
bash jobs/gpu_vm/sweep.sh --fedit-rank 8 --ravan-heads 16 --ravan-rank 28
```

Useful environment overrides:

```bash
CONDA_ENV=ravan MAX_PARALLEL=2 bash jobs/gpu_vm/sweep.sh
PYTHON_BIN=/path/to/python bash jobs/gpu_vm/run_fedit.sh
CUDA_VISIBLE_DEVICES=0 bash jobs/gpu_vm/sweep.sh
BATCH_SIZE=32 RAVAN_HEADS=16 RAVAN_RANK=28 bash jobs/gpu_vm/sweep.sh
```

Outputs:

- Logs: `logs_a100_vm/sweep_<timestamp>/`
- Per-run temporary outputs: `results_a100_vm/sweep_<timestamp>/parts/`
- Final aggregated outputs: `results_a100_vm/sweep_<timestamp>/final/`

The per-run temporary output directories avoid parallel writes to the same
`all_results.csv`. After all runs finish, `aggregate_results.py` collects the
run directories into `final/`, rebuilds `all_results.csv`, and regenerates the
comparison plots.
