"""Federated SVD warm-up initialization for Ravan.

Algorithm (Section 3.3 of the report):
  1. Select a subset of clients for the warm-up phase.
  2. Each client trains a temporary LoRA model of total rank R = heads * rank
     for warmup_steps gradient steps.
  3. Each client computes its update product  ΔW_c = B_c @ A_c  per layer.
     The server aggregates PRODUCTS, not factors — avoiding FedIT mismatch.
  4. Server averages: ΔW_warm = mean_c(ΔW_c) per layer (or weighted by examples).
  5. Server runs truncated SVD on each ΔW_warm:
         ΔW_warm ≈ U_R  Σ_R  Vh_R
  6. Singular vectors (not singular values) initialize Ravan's frozen bases:
         B_i = U_R[:, (i-1)*r : i*r],   A_i = Vh_R[(i-1)*r : i*r, :]
     H_i = 0 and s_i = 1 so the adapter starts at zero.

Raw client data stays local; only temporary LoRA update products ΔW_c are
communicated, not raw examples.
"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .client import local_train
from .model import make_distilbert, inject_lora, get_lora_layers, count_params_detailed


def federated_svd_init(
    client_loaders: list[DataLoader],
    warmup_clients: int,
    total_rank: int,
    warmup_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    warmup_weighting: str = "uniform",
    save_singular_values: bool = False,
    run_dir: Path | None = None,
    cache_dir: str | None = None,
    train_head: bool = True,
) -> tuple[
    list[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]],
    dict,
]:
    """Run federated LoRA warm-up and return SVD bases plus cost metadata.

    Args:
        client_loaders    : all client DataLoaders (full pool)
        warmup_clients    : number of clients sampled for warm-up
        total_rank        : temporary LoRA rank R = heads * per_head_rank
        warmup_steps      : local gradient steps per warm-up client
        lr                : learning rate for warm-up
        device            : compute device
        seed              : RNG seed for deterministic client selection
        warmup_weighting  : "uniform" (equal weight) or "examples" (weight by dataset size)
        save_singular_values : if True, save top-R singular values to run_dir
        run_dir           : directory to save optional diagnostics
        cache_dir         : HuggingFace cache directory for model loading
        train_head        : whether to include head in the temporary warm-up model

    Returns:
        svd_per_layer : list of length num_transformer_layers.
                        Each element is (q_svd, v_svd) where each svd is (U_R, Vh_R).
        costs         : dict with all warm-up cost and timing fields.
    """
    t_total_start = time.time()

    rng = np.random.default_rng(seed)
    n_clients = len(client_loaders)
    selected = rng.choice(n_clients, size=min(warmup_clients, n_clients), replace=False).tolist()

    print(f"  [SVD warm-up] clients={len(selected)}, rank={total_rank}, "
          f"steps={warmup_steps}, weighting={warmup_weighting}")

    # Count warm-up trainable params from one temporary model
    _tmp = make_distilbert(train_head=train_head, cache_dir=cache_dir)
    inject_lora(_tmp, rank=total_rank)
    _warmup_detail = count_params_detailed(_tmp)
    warmup_trainable_params = _warmup_detail["total_trainable_params"]
    del _tmp

    delta_W_sum: list[torch.Tensor] | None = None
    weights: list[float] = []

    t_train_start = time.time()

    for cid in selected:
        warmup_model = make_distilbert(train_head=train_head, cache_dir=cache_dir)
        inject_lora(warmup_model, rank=total_rank)

        local_train(warmup_model, client_loaders[cid], warmup_steps, lr, device)

        w = float(len(client_loaders[cid].dataset)) if warmup_weighting == "examples" else 1.0
        weights.append(w)

        layers = list(get_lora_layers(warmup_model))
        delta_Ws: list[torch.Tensor] = []
        for ll in layers:
            with torch.no_grad():
                dW = (ll.lora_B @ ll.lora_A).cpu()
            delta_Ws.append(dW)

        if delta_W_sum is None:
            delta_W_sum = [w * dW.clone() for dW in delta_Ws]
        else:
            for i, dW in enumerate(delta_Ws):
                delta_W_sum[i] = delta_W_sum[i] + w * dW

        del warmup_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    t_train_end = time.time()
    train_runtime_s = t_train_end - t_train_start

    total_weight = sum(weights)
    delta_W_avg = [dW / total_weight for dW in delta_W_sum]

    # Warm-up communicated params: each client uploads one ΔW per adapted layer
    warmup_comm_per_client = sum(dW.numel() for dW in delta_W_avg)
    warmup_communicated_params = warmup_comm_per_client * len(selected)

    t_svd_start = time.time()
    svd_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    singular_values_all: list[list[float]] = []

    for dW in delta_W_avg:
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        R = min(total_rank, U.shape[1], Vh.shape[0])
        if R < total_rank:
            print(f"  [SVD warm-up] Warning: only {R}/{total_rank} singular vectors available")
        svd_results.append((U[:, :R].contiguous(), Vh[:R, :].contiguous()))
        if save_singular_values:
            singular_values_all.append(S[:total_rank].tolist())

    t_svd_end = time.time()
    svd_runtime_s = t_svd_end - t_svd_start
    total_runtime_s = time.time() - t_total_start

    # Group into (q_svd, v_svd) pairs per transformer layer
    n_layers = len(svd_results) // 2
    svd_per_layer = [
        (svd_results[2 * i], svd_results[2 * i + 1])
        for i in range(n_layers)
    ]

    costs = {
        "warmup_clients":            len(selected),
        "warmup_steps":              warmup_steps,
        "warmup_rank":               total_rank,
        "warmup_weighting":          warmup_weighting,
        "warmup_trainable_params":   warmup_trainable_params,
        "warmup_communicated_params": warmup_communicated_params,
        "warmup_train_runtime_s":    round(train_runtime_s, 2),
        "warmup_svd_runtime_s":      round(svd_runtime_s, 4),
        "warmup_total_runtime_s":    round(total_runtime_s, 2),
    }

    if save_singular_values and run_dir is not None and singular_values_all:
        _save_singular_values(singular_values_all, run_dir)

    print(f"  [SVD warm-up] Done — {n_layers} transformer layers  "
          f"(train={train_runtime_s:.1f}s, svd={svd_runtime_s:.3f}s)")
    return svd_per_layer, costs


def _save_singular_values(singular_values_all: list[list[float]], run_dir: Path) -> None:
    """Save per-layer top singular values to run_dir/singular_values.csv."""
    path = run_dir / "singular_values.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer_idx", "projection", "sv_rank", "singular_value"])
        for idx, svs in enumerate(singular_values_all):
            layer_i = idx // 2
            proj    = "q" if idx % 2 == 0 else "v"
            for rank_i, sv in enumerate(svs):
                writer.writerow([layer_i, proj, rank_i, sv])
    print(f"  [SVD warm-up] Singular values → {path}")
