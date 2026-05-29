"""FedIT: federated LoRA fine-tuning baseline.

FedIT adapts LoRA to federated learning by averaging the A and B factor
matrices separately after each round.  This is simple and communication-
efficient, but the aggregation is inexact: mean(B_c) @ mean(A_c) != mean(B_c @ A_c).

Usage:
    python -m federated.train_fedit \\
        --split noniid --seed 0 --rounds 50 \\
        --clients 20 --clients_per_round 3 --local_steps 50 \\
        --rank 8 --lr 1e-3

Cluster (DAIC):
    sbatch jobs/submit_fedit.sh  (see jobs/submit_fedit.sh for SBATCH directives)
"""

import argparse
import random
import time

import numpy as np
import torch

from .client import local_train, evaluate
from .data import build_federated_loaders
from .model import count_params, inject_lora, make_distilbert, print_param_summary
from .server import fedit_aggregate, fedit_get_state, fedit_load_state
from .utils import make_run_name, save_config, save_results


def run(args):
    # ── reproducibility ──────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ─────────────────────────────────────────────────────────────────
    client_loaders, test_loader, _ = build_federated_loaders(
        split_type=args.split,
        num_clients=args.clients,
        batch_size=args.batch_size,
        seed=args.seed,
        alpha=args.alpha,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = make_distilbert()
    inject_lora(model, rank=args.rank)
    model.to(device)

    print("\nParameter summary (FedIT):")
    print_param_summary(model)
    trainable, total = count_params(model)

    # Initial global adapter state
    global_state = fedit_get_state(model)

    # ── FL loop ───────────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed + 1000)
    history = []

    print(f"\nFedIT — split={args.split}  seed={args.seed}  "
          f"rounds={args.rounds}  cpr={args.clients_per_round}  "
          f"steps={args.local_steps}  rank={args.rank}  lr={args.lr}\n")

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()

        selected = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
        client_states = []

        for cid in selected:
            fedit_load_state(model, global_state)
            local_train(model, client_loaders[cid], args.local_steps, args.lr, device)
            client_states.append(fedit_get_state(model))

        global_state = fedit_aggregate(client_states)

        # Evaluate with aggregated state
        fedit_load_state(model, global_state)
        acc = evaluate(model, test_loader, device)

        elapsed = time.time() - t0
        history.append({"round": rnd, "test_acc": round(acc, 6), "time_s": round(elapsed, 2)})

        if rnd == 1 or rnd % 5 == 0 or rnd == args.rounds:
            print(f"Round {rnd:3d}/{args.rounds}  "
                  f"clients={selected}  "
                  f"test_acc={acc:.4f}  "
                  f"time={elapsed:.1f}s")

    # ── save ──────────────────────────────────────────────────────────────────
    run_name = make_run_name("fedit", args.split, args.seed)

    cfg = vars(args).copy()
    cfg.update({"trainable_params": trainable, "total_params": total, "device": str(device)})

    final = {
        "run_name":        run_name,
        "method":          "fedit",
        "split":           args.split,
        "seed":            args.seed,
        "rank":            args.rank,
        "rounds":          args.rounds,
        "clients_per_round": args.clients_per_round,
        "local_steps":     args.local_steps,
        "lr":              args.lr,
        "batch_size":      args.batch_size,
        "trainable_params": trainable,
        "final_test_acc":  history[-1]["test_acc"],
        "best_test_acc":   max(r["test_acc"] for r in history),
    }

    save_config(cfg, run_name)
    save_results(final, history, run_name)


def main():
    parser = argparse.ArgumentParser(description="FedIT federated LoRA baseline")
    # Federated setup
    parser.add_argument("--split",            choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--rounds",           type=int,   default=50)
    parser.add_argument("--clients",          type=int,   default=20)
    parser.add_argument("--clients_per_round",type=int,   default=3)
    parser.add_argument("--local_steps",      type=int,   default=50)
    # LoRA
    parser.add_argument("--rank",             type=int,   default=8)
    # Optimisation
    parser.add_argument("--lr",               type=float, default=1e-3)
    parser.add_argument("--batch_size",       type=int,   default=16)
    # Non-IID concentration
    parser.add_argument("--alpha",            type=float, default=0.3,
                        help="Dirichlet concentration for noniid split")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
