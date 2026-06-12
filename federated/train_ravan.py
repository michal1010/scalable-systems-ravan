"""Ravan federated fine-tuning: Gram-Schmidt and SVD warm-up initialization.

Ravan replaces each LoRA update B@A with a sum of frozen-basis heads:
    ΔW = Σ_i  s_i * B_i * H_i * A_i
where B_i, A_i are frozen (shared across clients) and only H_i, s_i are
trained and communicated.  Clients upload s_i * H_i products; the server
averages them exactly, avoiding FedIT's factor-averaging mismatch.

Supports --model_type distilbert (default) and --model_type t5.
T5 mode uses T5EncoderModel + mean-pooling classification head (encoder
approximation, not text-to-text).

Usage:
    # DistilBERT Gram-Schmidt
    python -m federated.train_ravan \\
        --init gram_schmidt --split noniid --seed 0 \\
        --rounds 50 --heads 4 --rank 55

    # T5 Gram-Schmidt
    python -m federated.train_ravan \\
        --model_type t5 --init gram_schmidt \\
        --split noniid --seed 0 --rounds 100 \\
        --heads 4 --rank 110 --lr 5e-4 --max_length 256 \\
        --use_amp --grad_accum_steps 2

    # Profile mode
    python -m federated.train_ravan --model_type t5 --profile \\
        --init gram_schmidt --split noniid --seed 0 --rounds 100 \\
        --heads 4 --rank 110 --lr 5e-4 --max_length 256
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch

from .client import evaluate, local_train
from .data import build_federated_loaders
from .plot import plot_all, plot_single_run
from .server import ravan_aggregate, ravan_get_upload, ravan_load_global
from .utils import (
    get_git_hash, make_run_dir, make_run_name,
    save_config, save_profile_report, save_results,
)
from .warmup import federated_svd_init

_TOKENIZER        = {"distilbert": "distilbert-base-uncased", "t5": "t5-base"}
_ADAPTER_TARGETS  = {"distilbert": ["q_lin", "v_lin"],        "t5": ["q", "v"]}
_ADAPTED_MATRICES = {"distilbert": 12,                         "t5": 24}


def _load_model_components(model_type: str):
    """Return model factory and counting helpers for the given model_type."""
    if model_type == "distilbert":
        from .model import (
            make_distilbert as make_model,
            inject_ravan,
            count_params_detailed,
            count_communicated_per_round,
            count_adapter_communicated,
            print_param_summary,
            get_lora_layers,
            inject_lora,
            count_params_detailed as _cpd,
        )
        return dict(
            make_model=make_model,
            inject_ravan=inject_ravan,
            count_params_detailed=count_params_detailed,
            count_communicated_per_round=count_communicated_per_round,
            count_adapter_communicated=count_adapter_communicated,
            print_param_summary=print_param_summary,
            make_model_fn=None,
            inject_lora_fn=None,
            get_lora_layers_fn=None,
            count_params_fn=None,
        )
    else:  # t5
        from .model_t5 import (
            make_t5_encoder as make_model,
            inject_ravan_t5 as inject_ravan,
            count_params_detailed_t5 as count_params_detailed,
            count_communicated_per_round_t5 as count_communicated_per_round,
            count_adapter_communicated_t5 as count_adapter_communicated,
            print_param_summary_t5 as print_param_summary,
            inject_lora_t5 as inject_lora_t5,
            get_lora_layers_t5,
            count_params_detailed_t5 as _cpd,
        )
        return dict(
            make_model=make_model,
            inject_ravan=inject_ravan,
            count_params_detailed=count_params_detailed,
            count_communicated_per_round=count_communicated_per_round,
            count_adapter_communicated=count_adapter_communicated,
            print_param_summary=print_param_summary,
            make_model_fn=lambda th, cd: make_model(train_head=th, cache_dir=cd),
            inject_lora_fn=lambda m, r: inject_lora_t5(m, rank=r),
            get_lora_layers_fn=get_lora_layers_t5,
            count_params_fn=count_params_detailed,
        )


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
        tokenizer_name=_TOKENIZER[args.model_type],
    )

    # ── load model components ─────────────────────────────────────────────────
    comps = _load_model_components(args.model_type)
    make_model               = comps["make_model"]
    inject_ravan             = comps["inject_ravan"]
    count_params_detailed    = comps["count_params_detailed"]
    count_communicated_per_round = comps["count_communicated_per_round"]
    count_adapter_communicated   = comps["count_adapter_communicated"]
    print_param_summary      = comps["print_param_summary"]

    # ── initialize frozen bases ───────────────────────────────────────────────
    warmup_costs: dict = {
        "warmup_clients":             0,
        "warmup_steps":               0,
        "warmup_rank":                None,
        "warmup_weighting":           None,
        "warmup_trainable_params":    None,
        "warmup_communicated_params": 0,
        "warmup_train_runtime_s":     0.0,
        "warmup_svd_runtime_s":       0.0,
        "warmup_total_runtime_s":     0.0,
    }
    svd_matrices_per_layer = None

    prefix = "" if args.model_type == "distilbert" else f"{args.model_type}_"

    if args.init == "svd":
        print("\nRunning federated SVD warm-up...")
        total_rank = args.heads * args.rank
        warmup_lr  = args.warmup_lr if args.warmup_lr is not None else args.lr

        run_name_for_svd = make_run_name(f"{prefix}ravan_{args.init}", args.split, args.seed)
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
            make_model_fn=comps["make_model_fn"],
            inject_lora_fn=comps["inject_lora_fn"],
            get_lora_layers_fn=comps["get_lora_layers_fn"],
            count_params_fn=comps["count_params_fn"],
            use_amp=args.use_amp,
            grad_accum_steps=args.grad_accum_steps,
        )
        run_name = run_name_for_svd
        run_dir  = run_dir_for_svd
    else:
        run_name = make_run_name(f"{prefix}ravan_{args.init}", args.split, args.seed)
        run_dir  = make_run_dir(run_name, output_dir=args.output_dir or None)

    # ── model (re-seed for deterministic init regardless of SVD warm-up) ──────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model = make_model(train_head=train_head, cache_dir=args.cache_dir or None)
    inject_ravan(
        model,
        heads=args.heads,
        rank=args.rank,
        init_method=args.init,
        svd_matrices_per_layer=svd_matrices_per_layer,
    )
    model.to(device)

    method_tag = f"{prefix}ravan_{args.init}"
    print(f"\nParameter summary ({method_tag}):")
    print_param_summary(model)

    param_detail        = count_params_detailed(model)
    comm_per_client     = count_communicated_per_round(model)
    comm_per_round      = comm_per_client * args.clients_per_round
    total_main_comm     = comm_per_round * args.rounds
    adapter_comm_per_client = count_adapter_communicated(model)

    print(f"  Communicated / round : {comm_per_round:,}")
    print(f"  NOTE: Ravan aggregation is EXACT — averaging s*H products.\n")

    # Initial global state
    global_upload = ravan_get_upload(model)

    # ── FL loop ───────────────────────────────────────────────────────────────
    rng     = np.random.default_rng(args.seed + 1000)
    history = []
    profile_client_times: list[float] = []
    profile_peak_gpu_mb:  list[float] = []

    print(f"Ravan — init={args.init}  model={args.model_type}  split={args.split}  "
          f"seed={args.seed}  rounds={args.rounds}  cpr={args.clients_per_round}  "
          f"steps={args.local_steps}  heads={args.heads}  rank={args.rank}  lr={args.lr}\n")

    for rnd in range(1, args.rounds + 1):
        t_round_start = time.time()

        selected = rng.choice(args.clients, size=args.clients_per_round, replace=False).tolist()
        client_uploads = []

        t_train_start = time.time()
        for cid in selected:
            if args.profile and device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            t_client_start = time.time()

            ravan_load_global(model, global_upload)
            local_train(
                model, client_loaders[cid], args.local_steps, args.lr, device,
                use_amp=args.use_amp,
                grad_accum_steps=args.grad_accum_steps,
                use_grad_checkpoint=args.grad_checkpoint,
            )
            client_uploads.append(ravan_get_upload(model))

            if args.profile:
                profile_client_times.append(time.time() - t_client_start)
                if device.type == "cuda":
                    profile_peak_gpu_mb.append(
                        torch.cuda.max_memory_allocated(device) / 1e6
                    )

        train_runtime_s = time.time() - t_train_start

        global_upload = ravan_aggregate(client_uploads)

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
        "optimizer": "AdamW",
        "weight_decay": 0.01,
        "loss": "CrossEntropyLoss",
        "newsgroups_remove": "headers,footers,quotes",
        "lora_scaling": 1.0,
        "train_head": train_head,
        "adapter_targets": _ADAPTER_TARGETS[args.model_type],
        "adapted_matrices": _ADAPTED_MATRICES[args.model_type],
        "warmup_upload": "lora_factors" if args.init == "svd" else None,
        "warmup_aggregation": "product_after_reconstruction" if args.init == "svd" else None,
        "svd_absorb_singular_values": False if args.init == "svd" else None,
    })
    save_config(cfg, run_dir)

    summary = {
        "run_name":                    run_name,
        "method":                      method_tag,
        "init":                        args.init,
        "split":                       args.split,
        "seed":                        args.seed,
        "model_type":                  args.model_type,
        "rounds":                      args.rounds,
        "clients":                     args.clients,
        "clients_per_round":           args.clients_per_round,
        "local_steps":                 args.local_steps,
        "rank":                        args.rank,
        "heads":                       args.heads,
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
        **warmup_costs,
        "total_runtime_s":             round(time.time() - t_run_start, 1),
        "git_commit":                  get_git_hash(),
        "timestamp":                   time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    results_dir = Path(args.output_dir) if args.output_dir else None
    save_results(summary, history, run_dir, results_dir=results_dir)

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
            "init":                   args.init,
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
    parser = argparse.ArgumentParser(description="Ravan federated fine-tuning")

    # Model selection
    parser.add_argument("--model_type", choices=["distilbert", "t5"], default="distilbert",
                        help="Backbone: distilbert (default) or t5 (T5-encoder approximation)")

    # Init
    parser.add_argument("--init",  choices=["gram_schmidt", "svd"], default="gram_schmidt")

    # Federated setup
    parser.add_argument("--split",            choices=["iid", "noniid"], default="noniid")
    parser.add_argument("--seed",             type=int,   default=0)
    parser.add_argument("--rounds",           type=int,   default=50)
    parser.add_argument("--clients",          type=int,   default=20)
    parser.add_argument("--clients_per_round",type=int,   default=3)
    parser.add_argument("--local_steps",      type=int,   default=50)
    parser.add_argument("--eval_every",       type=int,   default=1)

    # Ravan adapter
    parser.add_argument("--heads",  type=int,   default=4)
    parser.add_argument("--rank",   type=int,   default=55,
                        help="Per-head rank (default 55 for distilbert; 110 recommended for t5)")

    # Optimisation
    parser.add_argument("--lr",               type=float, default=5e-4)
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--grad_accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps (effective_bs = batch_size × grad_accum_steps)")
    parser.add_argument("--use_amp",          action="store_true",
                        help="Enable mixed-precision training (float16, CUDA only)")
    parser.add_argument("--grad_checkpoint",  action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory")
    parser.add_argument("--max_length",       type=int,   default=128,
                        help="Tokenizer max sequence length (use 256 for t5)")

    # Non-IID
    parser.add_argument("--dirichlet_alpha",  type=float, default=0.3)
    parser.add_argument("--alpha",            type=float, default=None)

    # SVD warm-up
    parser.add_argument("--warmup_clients",   type=int,   default=5)
    parser.add_argument("--warmup_steps",     type=int,   default=50)
    parser.add_argument("--warmup_lr",        type=float, default=None)
    parser.add_argument("--warmup_weighting", choices=["uniform", "examples"], default="uniform")
    parser.add_argument("--save_singular_values", type=str, default="false")

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
