"""Ravan adapter linear layer.

Forward pass:
    output = frozen_linear(x) + Σ_h  scales[h] * (x @ A[h].T @ H[h].T @ B[h].T)

Dimensions per head h:
    B[h] : [d_out, rank]  — frozen
    A[h] : [rank, d_in]   — frozen
    H[h] : [rank, rank]   — trainable, zero-initialised
    scales[h] : scalar    — trainable, initialised to 1.0

Because H starts at zero the adapter contributes nothing at initialisation,
preserving the pretrained DistilBERT output exactly.

B and A are stored as registered buffers (not parameters) so they are
excluded from optimiser updates but are moved correctly by .to(device).

Two initialisation strategies for B and A:
  gram_schmidt — QR-based orthonormal columns/rows (random but structured)
  svd          — top-R singular vectors of a pre-trained LoRA ΔW
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def gram_schmidt_init(
    d_out: int,
    d_in: int,
    heads: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (B, A) with orthonormal columns/rows via QR decomposition.

    B : [heads, d_out, rank]
    A : [heads, rank, d_in]

    The h-th head occupies the h*rank : (h+1)*rank slice of a globally
    orthogonal R=heads*rank dimensional subspace.
    """
    R = heads * rank

    # --- B: orthonormalise columns ---
    B_rand = torch.randn(d_out, R)
    Q_B, _ = torch.linalg.qr(B_rand)       # Q_B: [d_out, min(d_out, R)]
    # Grab first R columns (always valid when d_out >= R, which holds for 768)
    n_cols = Q_B.shape[1]
    if n_cols >= R:
        B_orth = Q_B[:, :R]
    else:
        # Rare edge case (very small layer): pad with random columns
        pad = torch.randn(d_out, R - n_cols)
        B_orth = torch.cat([Q_B, pad], dim=1)

    # --- A: orthonormalise rows (= orthonormalise columns of A.T) ---
    A_rand = torch.randn(R, d_in)
    Q_A, _ = torch.linalg.qr(A_rand.T)     # Q_A: [d_in, min(d_in, R)]
    A_orth = Q_A.T                           # [min(d_in, R), d_in]
    n_rows = A_orth.shape[0]
    if n_rows >= R:
        A_orth = A_orth[:R, :]
    else:
        pad = torch.randn(R - n_rows, d_in)
        A_orth = torch.cat([A_orth, pad], dim=0)

    # --- Slice into per-head blocks ---
    B = torch.stack([B_orth[:, i * rank : (i + 1) * rank] for i in range(heads)])
    A = torch.stack([A_orth[i * rank : (i + 1) * rank, :]  for i in range(heads)])

    return B, A   # [heads, d_out, rank],  [heads, rank, d_in]


def svd_init(
    U_R: torch.Tensor,
    Vh_R: torch.Tensor,
    heads: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (B, A) from pre-computed truncated SVD of a LoRA ΔW.

    Args:
        U_R  : [d_out, R]  left singular vectors  (R = heads * rank)
        Vh_R : [R, d_in]   right singular vectors (V^H form)
        heads, rank : Ravan head count and per-head rank

    Singular values are NOT absorbed — B_i and A_i span the principal
    subspaces but start with unit scale.  H initialised to zero keeps
    the adapter contribution at zero until training begins.
    """
    R = heads * rank
    assert U_R.shape[1]  >= R, f"U_R has only {U_R.shape[1]} cols, need {R}"
    assert Vh_R.shape[0] >= R, f"Vh_R has only {Vh_R.shape[0]} rows, need {R}"

    B = torch.stack([U_R[:, i * rank : (i + 1) * rank] for i in range(heads)])
    A = torch.stack([Vh_R[i * rank : (i + 1) * rank, :] for i in range(heads)])

    return B, A   # [heads, d_out, rank],  [heads, rank, d_in]


# ---------------------------------------------------------------------------
# Ravan linear layer
# ---------------------------------------------------------------------------

class RavanLinear(nn.Module):
    """Frozen base linear layer augmented with Ravan multi-head adapters.

    Args:
        linear       : the nn.Linear to wrap (weights frozen in place)
        heads        : number of Ravan heads
        rank         : per-head rank  (total rank R = heads * rank)
        init_method  : "gram_schmidt" or "svd"
        svd_matrices : (U_R, Vh_R) tensors required when init_method="svd"
    """

    def __init__(
        self,
        linear: nn.Linear,
        heads: int,
        rank: int,
        init_method: str = "gram_schmidt",
        svd_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.linear = linear
        self.heads  = heads
        self.rank   = rank

        # Freeze base weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

        d_out = linear.out_features
        d_in  = linear.in_features

        if init_method == "gram_schmidt":
            B, A = gram_schmidt_init(d_out, d_in, heads, rank)
        elif init_method == "svd":
            if svd_matrices is None:
                raise ValueError("svd_matrices=(U_R, Vh_R) required for svd init")
            U_R, Vh_R = svd_matrices
            B, A = svd_init(U_R.cpu(), Vh_R.cpu(), heads, rank)
        else:
            raise ValueError(f"Unknown init_method '{init_method}'")

        # Frozen bases — registered as buffers so they travel with .to(device)
        self.register_buffer("B", B)   # [heads, d_out, rank]
        self.register_buffer("A", A)   # [heads, rank, d_in]

        # Trainable adapter parameters
        self.H      = nn.Parameter(torch.zeros(heads, rank, rank))  # zero init
        self.scales = nn.Parameter(torch.ones(heads))               # 1.0 init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        for h in range(self.heads):
            t = x      @ self.A[h].T    # [..., rank]
            t = t      @ self.H[h].T    # [..., rank]
            t = t      @ self.B[h].T    # [..., d_out]
            out = out + self.scales[h] * t
        return out

    def __repr__(self):
        return (f"RavanLinear(in={self.linear.in_features}, "
                f"out={self.linear.out_features}, "
                f"heads={self.heads}, rank={self.rank})")
