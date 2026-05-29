"""LoRA linear layer for FedIT.

Implements:
    y = frozen_linear(x) + scaling * (x @ A.T @ B.T)

where:
    A : [rank, in_features]  — Kaiming uniform init, trainable
    B : [out_features, rank] — zero init (so initial update is zero), trainable

This is the standard LoRA formulation (Hu et al. 2021).
The frozen base weight W is never modified.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen base linear layer augmented with a low-rank LoRA update.

    Args:
        linear  : the nn.Linear to wrap (its weights are frozen in place)
        rank    : LoRA rank r
        scaling : multiplier on the LoRA update (default 1.0; set to alpha/r
                  for the original LoRA scaling convention if desired)
    """

    def __init__(self, linear: nn.Linear, rank: int, scaling: float = 1.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scaling = scaling

        # Freeze base weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        d_in  = linear.in_features
        d_out = linear.out_features

        # A: randomly initialised (Kaiming uniform, same as nn.Linear default)
        # B: zero-initialised so that the adapter output starts at zero
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # x: [..., d_in]  →  [..., rank]  →  [..., d_out]
        update = (x @ self.lora_A.T) @ self.lora_B.T
        return base + self.scaling * update

    def __repr__(self):
        return (f"LoRALinear(in={self.linear.in_features}, "
                f"out={self.linear.out_features}, rank={self.rank})")
