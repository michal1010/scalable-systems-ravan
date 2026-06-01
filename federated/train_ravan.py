"""Ravan federated fine-tuning: Gram-Schmidt and SVD warm-up initialization.

Ravan replaces each LoRA update B@A with a sum of frozen-basis heads:
    ΔW = Σ_i  s_i * B_i * H_i * A_i
where B_i, A_i are frozen (shared across clients) and only H_i, s_i are
trained and communicated.  Clients upload s_i * H_i products; the server
averages them exactly, avoiding FedIT's factor-averaging mismatch.

Two initialization modes for the frozen bases B_i, A_i:
  gram_schmidt — QR orthonormalization of random matrices (data-agnostic)
  svd          — singular vectors of a federated LoRA warm-up (data-aware)

Usage:
    # Gram-Schmidt
    python -m federated.train_ravan \\
        --init gram_schmidt --split noniid --seed 0 \\
        --rounds 50 --heads 4 --rank 55

    # SVD warm-up (data-aware)
    python -m federated.train_ravan \\
        --init svd --split noniid --seed 0 \\
        --rounds 50 --heads 4 --rank 55 \\
        --warmup_clients 5 --warmup_steps 50

Cluster (DAIC):
    sbatch jobs/submit_ravan_gs.sh
    sbatch jobs/submit_ravan_svd.sh
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from .client import evaluate, local_train
from .data import build_federated_loaders
from .model import (
    count_communicated_per_round,
    count_params_detailed,
    inject_ravan,
    make_distilbert,
    print_param_summary,
)
from .plot import plot_all, plot_single_run
from .server import ravan_aggregate, ravan_get_upload, ravan_load_global
from .utils import get_git_hash, make_run_dir, make_run_name, save_config, save_results
from .warmup import federated_svd_init


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
    train_head = args.train_classifier_head.lower() not in ("false", "0", "no")

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

    # ── initialize frozen bases ───────────────────────────────────────────────
    warmup_costs: dict = {
        "warmup_clients":              0,
        "warmup_steps":                0,
        "warmup_rank":                 None,
        "warmup_weighting":            None,
        "warmup_trainable_params":     None,
        "warmup_communicated_params":  0,
        "warmup_train_runtime_s":      0.0,
        "warmup_svd_runtime_s":        0.0,
        "warmup_total_runtime_s":      0.0,
    }
    svd_matrices_per_layer = None

    if args.init == "svd":
        print("\nRunning federated SVD warm-up...")
        total_rank = args.heads * args.rank
        warmup_lr  = args.warmup_lr if args.warmup_lr is not None else args.lr

        run_name_for_svd = make_run_name(f"ravan_{args.init}", args.split, args.seed)
        run_dir_for_svd  = make_run_dir(run_name_for_svd, output_dir=args.output_dir or None)

        svd_matrices_per_layer, warmup_costs = federated_svd_init(
            client_loaders=client_loaders,
            warmup_clients=args.warmup_clients,
            total_rank=total_rank,
            warmup_steps=args.warmup_steps,
            lr=warmup_lr,
            device=device,
            seed=args.seed + 9999,
            warmup_weighting=args.warmup_weighting,
            save_singular_values=args.save_singular_values.lower() not in ("false","0","no"),
            run_dir=run_dir_for_svd,
            cache_dir=args.cache_dir or None,
            train_head=train_head,
        )
        # We use run_dir_for_svd as the final run_dir
        run_name = run_name_for_svd
        run_dir  = run_dir_for_svd
    else:
        run_name = make_run_name(f"ravan_{args.init}", args.split, args.seed)
        run_dir  = make_run_dir(run_name, output_dir=args.output_dir or None)

    # ── model ─────────────────────────────────────────────────────────────────
    model = make_distilbert(train_head=train_head, cache_dir=args.cache_dir or None)
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

    param_detail   = count_params_detailed(model)
    comm_per_round = count_communicated_per_round(model)
    total_main_comm = comm_per_round * args.rounds

    print(f"  Communicated / round : {comm_per_round:,}")
    print(f"  NOTE: Ravan aggregation is EXACT — averaging s*H products.\n")

    # Initial global state: H=0, scales=1 from RavanLinear init
    global_upload = ravan_get_upload(model)

    # ── FL loop ───────────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed + 1000)
    history = []

    print(f"Ravan — init={args.init}  split={args.split}  seed={args.seed}  "
          f"rounds={args.rounds}  cpr={args.clients_per_round}  "
          f"steps={args.local_steps}  heads={args.heads}  rank={args.rank}  lr={args.lr}\n")

    for rnd in range(1, args.rounds + 1):
        t_round_start = time.time()

        ravan_load_global(model, global_upload)

        selected = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
        client_uploads = []

        t_train_start = time.time()
        for cid in selected:
            ravan_load_global(model, global_upload)
            local_train(model, client_loaders[cid], args.local_steps, args.lr, device)
            client_uploads.append(ravan_get_upload(model))
        train_runtime_s = time.time() - t_train_start

        global_upload = ravan_aggregate(client_uploads)

        # Evaluate every eval_every rounds
        acc = None
        if rnd % args.eval_every == 0 or rnd == args.rounds:
            ravan_load_global(model, global_upload)
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
    accs = [r["test_acc"] for r in history if r["test_acc"] is not None]

    cfg = vars(args).copy()
    cfg.update({
        **param_detail,
        "communicated_params_per_round": comm_per_round,
        "device": str(device),
        **warmup_costs,
    })
    save_config(cfg, run_dir)

    summary = {
        "run_name":                    run_name,
        "method":                      f"ravan_{args.init}",
        "init":                        args.init,
        "split":                       args.split,
        "seed":                        args.seed,
        "rounds":                      args.rounds,
        "clients":                     args.clients,
        "clients_per_round":           args.clients_per_round,
        "local_steps":                 args.local_steps,
        "rank":                        args.rank,
        "heads":                       args.heads,
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
        **warmup_costs,
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
    parser = argparse.ArgumentParser(description="Ravan federated fine-tuning")

    # Init
    parser.add_argument("--init",  choices=["gram_schmidt", "svd"], default="gram_schmidt")

    # Federated setup
    parser.add_argument("--split",            choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--rounds",           type=int,   default=50)
    parser.add_argument("--clients",          type=int,   default=20)
    parser.add_argument("--clients_per_round",type=int,   default=3)
    parser.add_argument("--local_steps",      type=int,   default=50)
    parser.add_argument("--eval_every",       type=int,   default=1,
                        help="Evaluate on test set every N rounds")

    # Ravan adapter
    parser.add_argument("--heads",  type=int,   default=4)
    parser.add_argument("--rank",   type=int,   default=55,
                        help="Per-head rank (default 55 ≈ budget-match to FedIT rank=8)")

    # Optimisation
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--max_length",       type=int,   default=128)

    # Non-IID
    parser.add_argument("--dirichlet_alpha",  type=float, default=0.3)
    parser.add_argument("--alpha",            type=float, default=None,
                        help="Alias for --dirichlet_alpha (deprecated)")

    # SVD warm-up (only used when --init svd)
    parser.add_argument("--warmup_clients",   type=int,   default=5)
    parser.add_argument("--warmup_steps",     type=int,   default=50)
    parser.add_argument("--warmup_lr",        type=float, default=None,
                        help="LR for warm-up (defaults to --lr)")
    parser.add_argument("--warmup_weighting", choices=["uniform", "examples"], default="uniform",
                        help="How to weight client ΔW products in warm-up aggregation")
    parser.add_argument("--save_singular_values", type=str, default="false",
                        help="Save top singular values per layer to run_dir/singular_values.csv")

    # Infrastructure
    parser.add_argument("--output_dir",       type=str,   default=None)
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--cache_dir",        type=str,   default=None)
    parser.add_argument("--train_classifier_head", type=str, default="true")
    parser.add_argument("--save_checkpoints", type=str,   default="false")

    # Smoke-test helpers (do NOT use for real experiments)
    parser.add_argument("--limit_train_examples", type=int, default=None,
                        help="[smoke test only] Subsample training data to N examples")
    parser.add_argument("--limit_test_examples",  type=int, default=None,
                        help="[smoke test only] Subsample test data to N examples")

    args = parser.parse_args()

    if args.alpha is not None and args.dirichlet_alpha == 0.3:
        args.dirichlet_alpha = args.alpha

    run(args)


if __name__ == "__main__":
    main()
