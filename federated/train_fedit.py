"""FedIT: federated LoRA fine-tuning baseline.

FedIT adapts LoRA to federated learning by averaging the A and B factor
matrices separately after each round.  This is intentionally inexact:
mean(B_c) @ mean(A_c) != mean(B_c @ A_c) in general.

Supports --model_type distilbert (default) and --model_type t5.
T5 mode uses T5EncoderModel + mean-pooling classification head (encoder
approximation, not text-to-text).

Usage:
    # DistilBERT (original)
    python -m federated.train_fedit \\
        --split noniid --seed 0 --rounds 50 \\
        --clients 20 --clients_per_round 3 --local_steps 50 \\
        --rank 8 --lr 1e-3

    # T5-base (encoder approximation)
    python -m federated.train_fedit \\
        --model_type t5 \\
        --split noniid --seed 0 --rounds 100 \\
        --clients 20 --clients_per_round 3 --local_steps 50 \\
        --rank 32 --lr 1e-3 --max_length 256 \\
        --use_amp --grad_accum_steps 2

    # Profile mode (2 rounds, memory + timing report)
    python -m federated.train_fedit --model_type t5 --profile \\
        --split noniid --seed 0 --rounds 100 ...
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .client import local_train, evaluate
from .data import build_federated_loaders
from .plot import plot_all, plot_single_run
from .server import fedit_aggregate, fedit_get_state, fedit_load_state
from .utils import (
    get_git_hash, make_run_dir, make_run_name,
    save_config, save_profile_report, save_results,
)

# Tokenizer name per model type
_TOKENIZER = {"distilbert": "distilbert-base-uncased", "t5": "t5-base"}
# Adapted attention projection names per model type
_ADAPTER_TARGETS = {"distilbert": ["q_lin", "v_lin"], "t5": ["q", "v"]}
# Number of adapted linear layers per model type
_ADAPTED_MATRICES = {"distilbert": 12, "t5": 24}


def _load_model_components(model_type: str):
    """Return (make_model, inject_lora, count_params_detailed,
               count_communicated_per_round, count_adapter_communicated,
               print_param_summary) for the given model_type."""
    if model_type == "distilbert":
        from .model import (
            make_distilbert as make_model,
            inject_lora,
            count_params_detailed,
            count_communicated_per_round,
            count_adapter_communicated,
            print_param_summary,
        )
    else:
        from .model_t5 import (
            make_t5_encoder as make_model,
            inject_lora_t5 as inject_lora,
            count_params_detailed_t5 as count_params_detailed,
            count_communicated_per_round_t5 as count_communicated_per_round,
            count_adapter_communicated_t5 as count_adapter_communicated,
            print_param_summary_t5 as print_param_summary,
        )
    return (make_model, inject_lora, count_params_detailed,
            count_communicated_per_round, count_adapter_communicated,
            print_param_summary)


def run(args):
    # ── reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  model_type: {args.model_type}")

    # ── profile mode ─────────────────────────────────────────────────────────
    profile_total_rounds = args.rounds
    if args.profile:
        args.rounds = 2
        print(f"[PROFILE] Running 2 rounds (full run would be {profile_total_rounds})")

    t_run_start = time.time()

    # ── data ─────────────────────────────────────────────────────────────────
    client_loaders, test_loader, _ = build_federated_loaders(
        split_type=args.split,
        num_clients=args.clients,
        batch_size=args.batch_size,
        seed=args.seed,
        alpha=args.dirichlet_alpha,
        max_length=args.max_length,
        cache_dir=args.cache_dir or None,
        limit_examples=args.limit_train_examples or None,
        limit_test_examples=args.limit_test_examples or None,
        tokenizer_name=_TOKENIZER[args.model_type],
    )

    # ── model ─────────────────────────────────────────────────────────────────
    (make_model, inject_lora, count_params_detailed,
     count_communicated_per_round, count_adapter_communicated,
     print_param_summary) = _load_model_components(args.model_type)

    train_head = args.train_classifier_head.lower() not in ("false", "0", "no")
    model = make_model(train_head=train_head, cache_dir=args.cache_dir or None)
    inject_lora(model, rank=args.rank)
    model.to(device)

    method_label = "FedIT" if args.model_type == "distilbert" else f"T5-FedIT"
    print(f"\nParameter summary ({method_label}):")
    print_param_summary(model)

    param_detail        = count_params_detailed(model)
    comm_per_client     = count_communicated_per_round(model)
    comm_per_round      = comm_per_client * args.clients_per_round
    total_main_comm     = comm_per_round * args.rounds
    adapter_comm_per_client = count_adapter_communicated(model)

    print(f"  Communicated / round : {comm_per_round:,}")
    print(f"  NOTE: FedIT aggregation is INEXACT — averaging A and B separately\n")

    # Initial global adapter state
    global_state = fedit_get_state(model)

    # ── FL loop ───────────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed + 1000)
    history = []
    profile_client_times: list[float] = []
    profile_peak_gpu_mb:  list[float] = []

    prefix = "" if args.model_type == "distilbert" else f"{args.model_type}_"
    print(f"{method_label} — split={args.split}  seed={args.seed}  "
          f"rounds={args.rounds}  cpr={args.clients_per_round}  "
          f"steps={args.local_steps}  rank={args.rank}  lr={args.lr}\n")

    for rnd in range(1, args.rounds + 1):
        t_round_start = time.time()

        selected = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
        client_states = []

        t_train_start = time.time()
        for cid in selected:
            if args.profile and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t_client_start = time.time()

            fedit_load_state(model, global_state)
            local_train(
                model, client_loaders[cid], args.local_steps, args.lr, device,
                use_amp=args.use_amp,
                grad_accum_steps=args.grad_accum_steps,
                use_grad_checkpoint=args.grad_checkpoint,
            )
            client_states.append(fedit_get_state(model))

            if args.profile:
                profile_client_times.append(time.time() - t_client_start)
                if device.type == "cuda":
                    profile_peak_gpu_mb.append(
                        torch.cuda.max_memory_allocated(device) / 1e6
                    )

        train_runtime_s = time.time() - t_train_start

        global_state = fedit_aggregate(client_states)

        acc = None
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            fedit_load_state(model, global_state)
            acc = evaluate(model, test_loader, device)

        elapsed = time.time() - t_round_start

        row = {
            "round":                 rnd,
            "test_acc":              round(acc, 6) if acc is not None else None,
            "test_loss":             None,
            "elapsed_seconds":       round(elapsed, 2),
            "selected_clients":      str(selected),
            "train_runtime_seconds": round(train_runtime_s, 2),
        }
        history.append(row)

        if rnd == 1 or rnd % 5 == 0 or rnd == args.rounds:
            acc_str = f"{acc:.4f}" if acc is not None else "—"
            print(f"Round {rnd:3d}/{args.rounds}  "
                  f"clients={selected}  "
                  f"test_acc={acc_str}  "
                  f"time={elapsed:.1f}s")

    # ── save ──────────────────────────────────────────────────────────────────
    run_name = make_run_name(f"{prefix}fedit", args.split, args.seed)
    run_dir  = make_run_dir(run_name, output_dir=args.output_dir or None)

    accs = [r["test_acc"] for r in history if r["test_acc"] is not None]

    cfg = vars(args).copy()
    cfg.update({
        **param_detail,
        "communicated_params_per_round": comm_per_round,
        "device": str(device),
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "loss": "CrossEntropyLoss",
        "newsgroups_remove": "headers,footers,quotes",
        "lora_scaling": 1.0,
        "train_head": train_head,
        "adapter_targets": _ADAPTER_TARGETS[args.model_type],
        "adapted_matrices": _ADAPTED_MATRICES[args.model_type],
    })
    save_config(cfg, run_dir)

    summary = {
        "run_name":                    run_name,
        "method":                      f"{prefix}fedit",
        "init":                        "lora",
        "split":                       args.split,
        "seed":                        args.seed,
        "model_type":                  args.model_type,
        "rounds":                      args.rounds,
        "clients":                     args.clients,
        "clients_per_round":           args.clients_per_round,
        "local_steps":                 args.local_steps,
        "rank":                        args.rank,
        "heads":                       None,
        "lr":                          args.lr,
        "batch_size":                  args.batch_size,
        "grad_accum_steps":            args.grad_accum_steps,
        "effective_batch_size":        args.batch_size * args.grad_accum_steps,
        "use_amp":                     args.use_amp,
        "dirichlet_alpha":             args.dirichlet_alpha,
        "max_length":                  args.max_length,
        "final_acc":                   accs[-1] if accs else None,
        "best_acc":                    max(accs) if accs else None,
        "final_test_acc":              accs[-1] if accs else None,
        "best_test_acc":               max(accs) if accs else None,
        "final_loss":                  None,
        **param_detail,
        "communicated_params_per_round":           comm_per_round,
        "communicated_adapter_params_per_client":  adapter_comm_per_client,
        "total_main_communication_params":         total_main_comm,
        "warmup_clients":              None,
        "warmup_steps":                None,
        "warmup_rank":                 None,
        "warmup_weighting":            None,
        "warmup_trainable_params":     None,
        "warmup_communicated_params":  None,
        "warmup_train_runtime_s":      None,
        "warmup_svd_runtime_s":        None,
        "warmup_total_runtime_s":      None,
        "total_runtime_s":             round(time.time() - t_run_start, 1),
        "git_commit":                  get_git_hash(),
        "timestamp":                   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_dir = Path(args.output_dir) if args.output_dir else None
    save_results(summary, history, run_dir, results_dir=results_dir)

    # ── per-run plot ──────────────────────────────────────────────────────────
    valid_history = [r for r in history if r["test_acc"] is not None]
    if valid_history:
        plot_single_run(run_name, valid_history, run_dir)
    plot_all(results_dir)

    if args.save_checkpoints.lower() not in ("false", "0", "no"):
        ckpt_path = run_dir / "checkpoint_final.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint → {ckpt_path}")

    # ── profile report ────────────────────────────────────────────────────────
    if args.profile and profile_client_times:
        total_elapsed = sum(r["elapsed_seconds"] for r in history)
        profile_data = {
            "model_type":             args.model_type,
            "split":                  args.split,
            "seed":                   args.seed,
            "num_rounds_profiled":    2,
            "total_rounds_requested": profile_total_rounds,
            "avg_client_time_s":      round(sum(profile_client_times) / len(profile_client_times), 3),
            "total_2round_time_s":    round(total_elapsed, 2),
            "estimated_total_s":      round((total_elapsed / 2) * profile_total_rounds, 1),
            "peak_gpu_memory_mb":     round(max(profile_peak_gpu_mb), 1) if profile_peak_gpu_mb else None,
        }
        save_profile_report(profile_data, run_dir)


def main():
    parser = argparse.ArgumentParser(description="FedIT federated LoRA baseline")

    # Model selection
    parser.add_argument("--model_type", choices=["distilbert", "t5"], default="distilbert",
                        help="Backbone model: distilbert (default) or t5 (T5-encoder approximation)")

    # Federated setup
    parser.add_argument("--split",            choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--rounds",           type=int,   default=50)
    parser.add_argument("--clients",          type=int,   default=20)
    parser.add_argument("--clients_per_round",type=int,   default=3)
    parser.add_argument("--local_steps",      type=int,   default=50)
    parser.add_argument("--eval_every",       type=int,   default=1,
                        help="Evaluate on test set every N rounds")

    # LoRA
    parser.add_argument("--rank",             type=int,   default=8,
                        help="LoRA rank (default 8 for distilbert; 32 recommended for t5)")

    # Optimisation
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--grad_accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps (effective_bs = batch_size × grad_accum_steps)")
    parser.add_argument("--use_amp",          action="store_true",
                        help="Enable mixed-precision training (float16, CUDA only)")
    parser.add_argument("--grad_checkpoint",  action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory")
    parser.add_argument("--max_length",       type=int,   default=128,
                        help="Tokenizer max sequence length (use 256 for t5)")

    # Non-IID concentration
    parser.add_argument("--dirichlet_alpha",  type=float, default=0.3)
    parser.add_argument("--alpha",            type=float, default=None,
                        help="Alias for --dirichlet_alpha (deprecated)")

    # Infrastructure
    parser.add_argument("--output_dir",       type=str,   default=None)
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--cache_dir",        type=str,   default=None)
    parser.add_argument("--train_classifier_head", type=str, default="true")
    parser.add_argument("--save_checkpoints", type=str,   default="false")

    # Profiling
    parser.add_argument("--profile",          action="store_true",
                        help="Profile mode: run 2 rounds then report GPU memory, "
                             "seconds/client, and estimated total runtime")

    # Smoke-test helpers
    parser.add_argument("--limit_train_examples", type=int, default=None)
    parser.add_argument("--limit_test_examples",  type=int, default=None)

    args = parser.parse_args()

    if args.alpha is not None and args.dirichlet_alpha == 0.3:
        args.dirichlet_alpha = args.alpha

    run(args)


if __name__ == "__main__":
    main()
