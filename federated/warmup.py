"""Federated SVD warm-up initialization for Ravan.

Algorithm (Section 3.3 of the report):
  1. Select a subset of clients for the warm-up phase.
  2. Each client trains a temporary LoRA model of total rank R = heads * rank
     for warmup_steps gradient steps.
  3. Each client computes its update product  ΔW_c = B_c @ A_c  per layer.
     Importantly, the server aggregates PRODUCTS, not factors.  This avoids
     the FedIT factor-averaging mismatch.
  4. Server averages: ΔW_warm = mean_c(ΔW_c) per layer.
  5. Server runs truncated SVD on each ΔW_warm:
         ΔW_warm ≈ U_R  Σ_R  Vh_R
     with  U_R ∈ R^{d_out × R},  Vh_R ∈ R^{R × d_in}.
  6. Singular vectors (not singular values) initialize Ravan's frozen bases:
         B_i = U_R[:, (i-1)*r : i*r],   A_i = Vh_R[(i-1)*r : i*r, :]
     H_i = 0 and s_i = 1 at the start of main Ravan training, so the initial
     adapter contribution is zero and the model starts from pretrained weights.

This keeps raw client data local: only temporary LoRA updates (ΔW products)
are communicated, not raw examples.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .client import local_train
from .model import make_distilbert, inject_lora, get_lora_layers


def federated_svd_init(
    client_loaders: list[DataLoader],
    warmup_clients: int,
    total_rank: int,
    warmup_steps: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> list[tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]]:
    """Run federated LoRA warm-up and return SVD bases for Ravan initialization.

    Returns:
        svd_per_layer : list of length num_transformer_layers (6 for DistilBERT).
                        Each element is  (q_svd, v_svd)  where
                        q_svd = (U_R, Vh_R) for the query projection and
                        v_svd = (U_R, Vh_R) for the value projection.
                        U_R  : [d_out, total_rank]
                        Vh_R : [total_rank, d_in]
    """
    rng = np.random.default_rng(seed)
    n_clients = len(client_loaders)
    selected = rng.choice(n_clients, size=min(warmup_clients, n_clients), replace=False).tolist()

    print(f"  [SVD warm-up] {len(selected)} clients, rank={total_rank}, steps={warmup_steps}")

    delta_W_sum: list[torch.Tensor] | None = None

    for cid in selected:
        # Fresh LoRA model for this client (temporary; discarded after warm-up)
        warmup_model = make_distilbert()
        inject_lora(warmup_model, rank=total_rank)

        local_train(warmup_model, client_loaders[cid], warmup_steps, lr, device)

        # Compute ΔW_c = B_c @ A_c for every adapted layer (ordered q then v)
        layers = list(get_lora_layers(warmup_model))
        delta_Ws = []
        for ll in layers:
            with torch.no_grad():
                dW = (ll.lora_B @ ll.lora_A).cpu()
            delta_Ws.append(dW)

        if delta_W_sum is None:
            delta_W_sum = [dW.clone() for dW in delta_Ws]
        else:
            for i, dW in enumerate(delta_Ws):
                delta_W_sum[i] = delta_W_sum[i] + dW

        del warmup_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    n = len(selected)
    delta_W_avg = [dW / n for dW in delta_W_sum]

    # SVD per layer, then group into (q_svd, v_svd) pairs
    svd_results: list[tuple[torch.Tensor, torch.Tensor]] = []
    for dW in delta_W_avg:
        U, _, Vh = torch.linalg.svd(dW, full_matrices=False)
        # U:  [d_out, min(d_out, d_in)]
        # Vh: [min(d_out, d_in), d_in]
        R = min(total_rank, U.shape[1], Vh.shape[0])
        if R < total_rank:
            print(f"  [SVD warm-up] Warning: only {R}/{total_rank} singular vectors available")
        svd_results.append((U[:, :R].contiguous(), Vh[:R, :].contiguous()))

    # Interleaved order is: layer0_q, layer0_v, layer1_q, layer1_v, ...
    n_layers = len(svd_results) // 2
    svd_per_layer = [
        (svd_results[2 * i], svd_results[2 * i + 1])
        for i in range(n_layers)
    ]
    print(f"  [SVD warm-up] Done — {len(svd_per_layer)} transformer layers processed")
    return svd_per_layer
