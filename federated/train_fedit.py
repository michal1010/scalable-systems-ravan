"""FedIT: federated LoRA fine-tuning baseline.

FedIT adapts LoRA to federated learning by averaging the A and B factor
matrices separately after each round.  This is intentionally inexact:
mean(B_c) @ mean(A_c) != mean(B_c @ A_c) in general.

Usage:
    python -m federated.train_fedit \\
        --split noniid --seed 0 --rounds 50 \\
        --clients 20 --clients_per_round 3 --local_steps 50 \\
        --rank 8 --lr 1e-3

Cluster (DAIC):
    sbatch jobs/submit_fedit.sh
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
from .model import (
    count_params_detailed,
    count_communicated_per_round,
    inject_lora,
    make_distilbert,
    print_param_summary,
)
from .plot import plot_all, plot_single_run
from .server import fedit_aggregate, fedit_get_state, fedit_load_state
from .utils import get_git_hash, make_run_dir, make_run_name, save_config, save_results


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
    print(f"Device: {device}")

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
    )

    # ── model ─────────────────────────────────────────────────────────────────
    train_head = args.train_classifier_head.lower() not in ("false", "0", "no")
    model = make_distilbert(train_head=train_head, cache_dir=args.cache_dir or None)
    inject_lora(model, rank=args.rank)
    model.to(device)

    print("\nParameter summary (FedIT):")
    print_param_summary(model)

    param_detail = count_params_detailed(model)
    comm_per_round = count_communicated_per_round(model)
    total_main_comm = comm_per_round * args.rounds

    print(f"  Communicated / round : {comm_per_round:,}")
    print(f"  NOTE: FedIT aggregation is INEXACT — averaging A and B separately")
    print(f"        mean(B_c)@mean(A_c) != mean(B_c@A_c) in general.\n")

    # Initial global adapter state
    global_state = fedit_get_state(model)

    # ── FL loop ───────────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed + 1000)
    history = []

    print(f"FedIT — split={args.split}  seed={args.seed}  "
          f"rounds={args.rounds}  cpr={args.clients_per_round}  "
          f"steps={args.local_steps}  rank={args.rank}  lr={args.lr}\n")

    for rnd in range(1, args.rounds + 1):
        t_round_start = time.time()

        selected = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
        client_states = []

        t_train_start = time.time()
        for cid in selected:
            fedit_load_state(model, global_state)
            local_train(model, client_loaders[cid], args.local_steps, args.lr, device)
            client_states.append(fedit_get_state(model))
        train_runtime_s = time.time() - t_train_start

        global_state = fedit_aggregate(client_states)

        # Evaluate every eval_every rounds
        acc = None
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            fedit_load_state(model, global_state)
            acc = evaluate(model, test_loader, device)

        elapsed = time.time() - t_round_start

        row = {
            "round":                rnd,
            "test_acc":             round(acc, 6) if acc is not None else None,
            "test_loss":            None,
            "elapsed_seconds":      round(elapsed, 2),
            "selected_clients":     str(selected),
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
    run_name = make_run_name("fedit", args.split, args.seed)
    run_dir  = make_run_dir(run_name, output_dir=args.output_dir or None)

    accs = [r["test_acc"] for r in history if r["test_acc"] is not None]

    cfg = vars(args).copy()
    cfg.update({
        **param_detail,
        "communicated_params_per_round": comm_per_round,
        "device": str(device),
    })
    save_config(cfg, run_dir)

    summary = {
        "run_name":                    run_name,
        "method":                      "fedit",
        "init":                        "lora",
        "split":                       args.split,
        "seed":                        args.seed,
        "rounds":                      args.rounds,
        "clients":                     args.clients,
        "clients_per_round":           args.clients_per_round,
        "local_steps":                 args.local_steps,
        "rank":                        args.rank,
        "heads":                       None,
        "lr":                          args.lr,
        "batch_size":                  args.batch_size,
        "dirichlet_alpha":             args.dirichlet_alpha,
        "max_length":                  args.max_length,
        "final_acc":                   accs[-1] if accs else None,
        "best_acc":                    max(accs) if accs else None,
        "final_test_acc":              accs[-1] if accs else None,
        "best_test_acc":               max(accs) if accs else None,
        "final_loss":                  None,
        **param_detail,
        "communicated_params_per_round": comm_per_round,
        "total_main_communication_params": total_main_comm,
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
    plot_all(results_dir)  # uses RESULTS_DIR default when results_dir is None

    if args.save_checkpoints.lower() not in ("false", "0", "no"):
        ckpt_path = run_dir / "checkpoint_final.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"Checkpoint → {ckpt_path}")


def main():
    parser = argparse.ArgumentParser(description="FedIT federated LoRA baseline")

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
    parser.add_argument("--rank",             type=int,   default=8)

    # Optimisation
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--max_length",       type=int,   default=128,
                        help="Tokenizer max sequence length")

    # Non-IID concentration
    parser.add_argument("--dirichlet_alpha",  type=float, default=0.3,
                        help="Dirichlet concentration for noniid split (alpha=0.3)")
    # Legacy alias kept for backward compat
    parser.add_argument("--alpha",            type=float, default=None,
                        help="Alias for --dirichlet_alpha (deprecated; use --dirichlet_alpha)")

    # Infrastructure
    parser.add_argument("--output_dir",       type=str,   default=None,
                        help="Root directory for results (default: results/)")
    parser.add_argument("--device",           type=str,   default=None,
                        help="torch device string, e.g. 'cuda' or 'cpu'")
    parser.add_argument("--cache_dir",        type=str,   default=None,
                        help="HuggingFace model/tokenizer cache directory")
    parser.add_argument("--train_classifier_head", type=str, default="true",
                        help="Whether to train the shared classification head (default: true)")
    parser.add_argument("--save_checkpoints", type=str,   default="false",
                        help="Save final model checkpoint (default: false)")

    # Smoke-test helpers (do NOT use for real experiments)
    parser.add_argument("--limit_train_examples", type=int, default=None,
                        help="[smoke test only] Subsample training data to N examples")
    parser.add_argument("--limit_test_examples",  type=int, default=None,
                        help="[smoke test only] Subsample test data to N examples")

    args = parser.parse_args()

    # Resolve --alpha alias
    if args.alpha is not None and args.dirichlet_alpha == 0.3:
        args.dirichlet_alpha = args.alpha

    run(args)


if __name__ == "__main__":
    main()
