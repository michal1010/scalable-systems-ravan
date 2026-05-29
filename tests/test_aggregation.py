"""Correctness tests for aggregation properties.

Tests:
  1. Ravan exact aggregation — averaging s*H products gives the same model
     update as averaging the actual per-client ΔW contributions.
  2. FedIT mismatch — separately averaging B and A does NOT equal averaging
     the products B@A in general.
  3. Gram-Schmidt orthogonality — B_i columns and A_i rows are orthonormal
     across heads.
  4. SVD orthogonality — same check for SVD-initialized bases.
  5. Zero initial update — with H=0, the Ravan adapter contributes nothing.
"""

import torch
import pytest

from federated.ravan import RavanLinear, gram_schmidt_init, svd_init
from federated.lora import LoRALinear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ravan(d_out=32, d_in=16, heads=3, rank=4, init="gram_schmidt"):
    linear = torch.nn.Linear(d_in, d_out, bias=False)
    return RavanLinear(linear, heads=heads, rank=rank, init_method=init)


# ---------------------------------------------------------------------------
# Test 1: Ravan exact aggregation
# ---------------------------------------------------------------------------

def test_ravan_exact_aggregation():
    """
    For frozen B_i, A_i shared across clients:
        mean_c [ Σ_i  B_i (s_{c,i} H_{c,i}) A_i ]
        ==
        Σ_i  B_i ( mean_c [ s_{c,i} H_{c,i} ] ) A_i
    """
    torch.manual_seed(0)
    heads, rank, d_out, d_in = 4, 4, 32, 16
    num_clients = 5

    layer = _make_ravan(d_out=d_out, d_in=d_in, heads=heads, rank=rank)
    B = layer.B.clone()  # [heads, d_out, rank]
    A = layer.A.clone()  # [heads, rank, d_in]

    # Simulate random client H and s values
    client_Hs = [torch.randn(heads, rank, rank) for _ in range(num_clients)]
    client_s  = [torch.rand(heads) + 0.5       for _ in range(num_clients)]

    # LHS: average the per-client full ΔW updates
    lhs = torch.zeros(d_out, d_in)
    for H_c, s_c in zip(client_Hs, client_s):
        dW_c = sum(
            s_c[i] * (B[i] @ H_c[i] @ A[i])
            for i in range(heads)
        )
        lhs = lhs + dW_c
    lhs = lhs / num_clients

    # RHS: first average s*H, then apply frozen bases
    sH_avg = torch.stack([
        s_c[:, None, None] * H_c
        for H_c, s_c in zip(client_Hs, client_s)
    ]).mean(0)  # [heads, rank, rank]

    rhs = sum(B[i] @ sH_avg[i] @ A[i] for i in range(heads))

    assert torch.allclose(lhs, rhs, atol=1e-5), \
        f"Exact aggregation failed: max diff = {(lhs - rhs).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Test 2: FedIT mismatch (expected to fail exact equality)
# ---------------------------------------------------------------------------

def test_fedit_mismatch():
    """
    Separately averaging B and A is generally NOT equal to averaging B@A.
    This test verifies the mismatch exists (not a bug — this is by design).
    """
    torch.manual_seed(42)
    d, rank, num_clients = 32, 4, 5

    Bs = [torch.randn(d, rank) for _ in range(num_clients)]
    As = [torch.randn(rank, d) for _ in range(num_clients)]

    avg_product  = torch.stack([B @ A for B, A in zip(Bs, As)]).mean(0)
    product_avgs = torch.stack(Bs).mean(0) @ torch.stack(As).mean(0)

    diff = (avg_product - product_avgs).abs().max().item()
    assert diff > 1e-4, \
        "Expected FedIT mismatch but got near-zero difference — something is wrong."


# ---------------------------------------------------------------------------
# Test 3: Gram-Schmidt orthogonality
# ---------------------------------------------------------------------------

def test_gram_schmidt_orthogonality():
    """B_i columns and A_i rows should be orthonormal across all heads."""
    torch.manual_seed(1)
    d_out, d_in, heads, rank = 64, 48, 4, 6

    B, A = gram_schmidt_init(d_out, d_in, heads, rank)
    # B: [heads, d_out, rank],  A: [heads, rank, d_in]

    # Stack all head columns of B into one matrix and check orthonormality
    R = heads * rank
    B_all = B.permute(1, 0, 2).reshape(d_out, R)  # [d_out, R]
    BtB   = B_all.T @ B_all                         # [R, R]
    assert torch.allclose(BtB, torch.eye(R), atol=1e-5), \
        f"B columns not orthonormal: max off-diag = {(BtB - torch.eye(R)).abs().max():.2e}"

    # Stack all head rows of A into one matrix and check orthonormality
    A_all = A.reshape(R, d_in)                      # [R, d_in]
    AAt   = A_all @ A_all.T                          # [R, R]
    assert torch.allclose(AAt, torch.eye(R), atol=1e-5), \
        f"A rows not orthonormal: max off-diag = {(AAt - torch.eye(R)).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Test 4: SVD orthogonality
# ---------------------------------------------------------------------------

def test_svd_init_orthogonality():
    """SVD-initialized bases should also be orthonormal (singular vectors are)."""
    torch.manual_seed(2)
    d_out, d_in, heads, rank = 64, 48, 3, 5
    R = heads * rank

    dW = torch.randn(d_out, d_in)
    U, _, Vh = torch.linalg.svd(dW, full_matrices=False)
    U_R  = U[:, :R]
    Vh_R = Vh[:R, :]

    B, A = svd_init(U_R, Vh_R, heads, rank)

    B_all = B.permute(1, 0, 2).reshape(d_out, R)
    BtB   = B_all.T @ B_all
    assert torch.allclose(BtB, torch.eye(R), atol=1e-5), \
        f"SVD B columns not orthonormal"

    A_all = A.reshape(R, d_in)
    AAt   = A_all @ A_all.T
    assert torch.allclose(AAt, torch.eye(R), atol=1e-5), \
        f"SVD A rows not orthonormal"


# ---------------------------------------------------------------------------
# Test 5: Zero initial Ravan update
# ---------------------------------------------------------------------------

def test_ravan_zero_init_output():
    """With H=0, the Ravan adapter adds zero to the frozen linear output."""
    torch.manual_seed(3)
    layer = _make_ravan(d_out=32, d_in=16, heads=3, rank=4)

    # All H matrices must be zero
    assert layer.H.abs().max() == 0.0, "H should be zero-initialized"

    x = torch.randn(8, 16)
    with torch.no_grad():
        base   = layer.linear(x)
        output = layer(x)

    assert torch.allclose(base, output, atol=1e-6), \
        f"Non-zero initial adapter output: max diff = {(base - output).abs().max():.2e}"


# ---------------------------------------------------------------------------
# Test 6: LoRA zero init output
# ---------------------------------------------------------------------------

def test_lora_zero_init_output():
    """With B=0, the LoRA adapter adds zero to the frozen linear output."""
    torch.manual_seed(4)
    linear = torch.nn.Linear(16, 32, bias=False)
    layer  = LoRALinear(linear, rank=4)

    assert layer.lora_B.abs().max() == 0.0, "lora_B should be zero-initialized"

    x = torch.randn(8, 16)
    with torch.no_grad():
        base   = layer.linear(x)
        output = layer(x)

    assert torch.allclose(base, output, atol=1e-6), \
        f"Non-zero initial LoRA output: max diff = {(base - output).abs().max():.2e}"
