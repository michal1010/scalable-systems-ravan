# GPU VM Jobs

These scripts run the same experiment settings as `jobs/submit_*.sh`, but directly
on a VM with a CUDA GPU. They do not use Slurm or Apptainer.

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

Extra training flags are forwarded to every run in the sweep:

```bash
bash jobs/gpu_vm/sweep.sh --use_amp --grad_checkpoint --profile
```

## T5-Base Minimal Reproduction

Run the one-seed T5-Base setup:

```bash
bash jobs/gpu_vm/sweep_t5_minimal.sh
```

This runs six experiments:

- T5-FedIT, T5-Ravan-GS, and T5-Ravan-SVD
- `iid` and `noniid`
- seed `0`

Default T5 settings:

- clients: `20`
- clients per round: `3`
- local steps: `50`
- rounds: `100`
- max length: `256`
- non-IID alpha: `0.3`
- FedIT rank: `32`
- Ravan heads/rank: `4 x 110`
- Ravan-SVD warm-up clients/steps: `5 x 50`
- mixed precision and gradient checkpointing: enabled
- batch: `16`, gradient accumulation: `2` for effective batch size `32`

On an RTX 6000 with enough memory, try true batch size 32:

```bash
T5_BATCH_SIZE=32 T5_GRAD_ACCUM_STEPS=1 bash jobs/gpu_vm/sweep_t5_minimal.sh
```

For a first memory/timing check, run only two rounds per experiment:

```bash
bash jobs/gpu_vm/sweep_t5_minimal.sh --profile
```

The T5 sweep defaults to one process at a time. If GPU memory is comfortable,
try two parallel processes:

```bash
MAX_PARALLEL=2 bash jobs/gpu_vm/sweep_t5_minimal.sh
```

The sweep runs 18 experiments:

- FedIT, Ravan-GS, Ravan-SVD
- `iid` and `noniid`
- seeds `0 1 2`

By default, the sweep runs two experiments at the same time on the GPU:

```bash
MAX_PARALLEL=2 bash jobs/gpu_vm/sweep.sh
```

For a large GPU with enough memory, you can try:

```bash
MAX_PARALLEL=3 bash jobs/gpu_vm/sweep.sh
```

If GPU memory becomes tight, use:

```bash
MAX_PARALLEL=1 bash jobs/gpu_vm/sweep.sh
```

Useful environment overrides:

```bash
CONDA_ENV=ravan MAX_PARALLEL=2 bash jobs/gpu_vm/sweep.sh
PYTHON_BIN=/path/to/python bash jobs/gpu_vm/run_fedit.sh
CUDA_VISIBLE_DEVICES=0 bash jobs/gpu_vm/sweep.sh
```

Outputs:

- Logs: `logs_gpu_vm/sweep_<timestamp>/`
- Per-run temporary outputs: `results_gpu_vm/sweep_<timestamp>/parts/`
- Final aggregated outputs: `results_gpu_vm/sweep_<timestamp>/final/`

The per-run temporary output directories avoid parallel writes to the same
`all_results.csv`. After all runs finish, `aggregate_results.py` collects the
run directories into `final/`, rebuilds `all_results.csv`, and regenerates the
comparison plots.
