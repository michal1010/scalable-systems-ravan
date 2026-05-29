"""Ravan federated fine-tuning: Gram-Schmidt and SVD warm-up initialization.

Ravan replaces each LoRA update BA with a sum of frozen-basis heads:
    ΔW = Σ_i  s_i * B_i * H_i * A_i
where B_i, A_i are frozen (shared across clients) and only H_i, s_i are
trained and communicated.  Clients upload  s_i * H_i  products; the server
averages them exactly, avoiding FedIT's factor-averaging mismatch.

Two initialization modes for the frozen bases B_i, A_i:
  gram_schmidt — QR orthonormalization of random matrices (data-agnostic)
  svd          — singular vectors of a federated LoRA warm-up (data-aware)

Usage:
    # Gram-Schmidt
    python -m federated.train_ravan \\
        --init gram_schmidt --split noniid --seed 0 \\
        --rounds 50 --clients 20 --clients_per_round 3 \\
        --local_steps 50 --heads 4 --rank 55

    # SVD warm-up (data-aware)
    python -m federated.train_ravan \\
        --init svd --split noniid --seed 0 \\
        --rounds 50 --clients 20 --clients_per_round 3 \\
        --local_steps 50 --heads 4 --rank 55 \\
        --warmup_rounds 2 --warmup_steps 50 --warmup_clients 5
"""

import argparse
import random
import time

import numpy as np
import torch

from .client import evaluate, local_train
from .data import build_federated_loaders
from .model import (
    count_params, inject_ravan, make_distilbert,
    print_param_summary,
)
from .plot import plot_all, plot_single_run
from .server import ravan_aggregate, ravan_get_upload, ravan_load_global
from .utils import make_run_dir, make_run_name, save_config, save_results
from .warmup import federated_svd_init


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

    # ── initialize frozen bases ───────────────────────────────────────────────
    svd_matrices_per_layer = None

    if args.init == "svd":
        print("\nRunning federated SVD warm-up...")
        total_rank = args.heads * args.rank
        svd_matrices_per_layer = federated_svd_init(
            client_loaders=client_loaders,
            warmup_clients=args.warmup_clients,
            total_rank=total_rank,
            warmup_steps=args.warmup_steps,
            lr=args.warmup_lr if args.warmup_lr is not None else args.lr,
            device=device,
            seed=args.seed + 9999,
        )

    # ── model ─────────────────────────────────────────────────────────────────
    model = make_distilbert()
    inject_ravan(
        model,
        heads=args.heads,
        rank=args.rank,
        init_method=args.init,
        svd_matrices_per_layer=svd_matrices_per_layer,
    )
    model.to(device)

    print(f"\nParameter summary (Ravan, init={args.init}):")
    print_param_summary(model)
    trainable, total = count_params(model)

    # Initial global state: H matrices are 0, scales are 1 (from RavanLinear init)
    global_upload = ravan_get_upload(model)

    # ── FL loop ───────────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed + 1000)
    history = []

    print(f"\nRavan — init={args.init}  split={args.split}  seed={args.seed}  "
          f"rounds={args.rounds}  cpr={args.clients_per_round}  "
          f"steps={args.local_steps}  heads={args.heads}  rank={args.rank}  lr={args.lr}\n")

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()

        # Reset scales to 1 and load H from global state before each round
        ravan_load_global(model, global_upload)

        selected = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
        client_uploads = []

        for cid in selected:
            # Each client starts from the same global H with scales=1
            ravan_load_global(model, global_upload)
            local_train(model, client_loaders[cid], args.local_steps, args.lr, device)
            client_uploads.append(ravan_get_upload(model))

        global_upload = ravan_aggregate(client_uploads)

        # Evaluate
        ravan_load_global(model, global_upload)
        acc = evaluate(model, test_loader, device)

        elapsed = time.time() - t0
        history.append({"round": rnd, "test_acc": round(acc, 6), "time_s": round(elapsed, 2)})

        if rnd == 1 or rnd % 5 == 0 or rnd == args.rounds:
            print(f"Round {rnd:3d}/{args.rounds}  "
                  f"clients={selected}  "
                  f"test_acc={acc:.4f}  "
                  f"time={elapsed:.1f}s")

    # ── save ──────────────────────────────────────────────────────────────────
    run_name = make_run_name(f"ravan_{args.init}", args.split, args.seed)
    run_dir  = make_run_dir(run_name)

    cfg = vars(args).copy()
    cfg.update({"trainable_params": trainable, "total_params": total, "device": str(device)})

    final = {
        "run_name":        run_name,
        "method":          f"ravan_{args.init}",
        "init":            args.init,
        "split":           args.split,
        "seed":            args.seed,
        "heads":           args.heads,
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

    if args.init == "svd":
        final["warmup_clients"] = args.warmup_clients
        final["warmup_steps"]   = args.warmup_steps

    save_config(cfg, run_dir)
    save_results(final, history, run_dir)

    # ── visualize ─────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_single_run(run_name, history, run_dir)
    plot_all()


def main():
    parser = argparse.ArgumentParser(description="Ravan federated fine-tuning")
    # Init
    parser.add_argument("--init", choices=["gram_schmidt", "svd"], default="gram_schmidt")
    # Federated setup
    parser.add_argument("--split",            choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--rounds",           type=int,   default=50)
    parser.add_argument("--clients",          type=int,   default=20)
    parser.add_argument("--clients_per_round",type=int,   default=3)
    parser.add_argument("--local_steps",      type=int,   default=50)
    # Ravan adapter
    parser.add_argument("--heads",            type=int,   default=4)
    parser.add_argument("--rank",             type=int,   default=55,
                        help="Per-head rank (default 55 matches FedIT rank=8 budget)")
    # Optimisation
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--alpha",            type=float, default=0.3,
                        help="Dirichlet concentration for noniid split")
    # SVD warm-up (only used when --init svd)
    parser.add_argument("--warmup_clients",   type=int,   default=5,
                        help="Number of clients to use in the SVD warm-up phase")
    parser.add_argument("--warmup_steps",     type=int,   default=50,
                        help="Local gradient steps per client during warm-up")
    parser.add_argument("--warmup_lr",        type=float, default=None,
                        help="LR for warm-up (defaults to --lr)")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
