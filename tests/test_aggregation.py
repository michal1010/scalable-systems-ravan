"""Correctness tests for aggregation properties.

Tests:
  1. Ravan exact aggregation — averaging s*H products gives the same model
     update as averaging the actual per-client ΔW contributions.
  2. FedIT mismatch — separately averaging B and A does NOT equal averaging
     the products B@A in general.
  3. Gram-Schmidt orthogonality — B_i columns and A_i rows are orthonormal
     across heads.
  4. SVD orthogonality — same check for SVD-initialized bases.
  5. Zero initial Ravan update — with H=0, the adapter contributes nothing.
  6. Zero initial LoRA update — with B=0, the adapter contributes nothing.
  7. Parameter budget counting — FedIT rank=8 and Ravan h=4 r=55 match closely.
  8. Report asset generation with dummy data — tables and figures are produced.
"""

import csv
import json
import tempfile
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Test 7: Parameter budget matching
# ---------------------------------------------------------------------------

def test_param_budget_counting():
    """FedIT rank=8 and Ravan h=4 r=55 should have close adapter param counts.

    12 adapted layers (6 transformer blocks × q + v projections).
    FedIT per layer:  rank * d_in + d_out * rank  = 8*768 + 768*8  = 12,288
    Ravan per layer:  heads * rank^2 + heads      = 4*55^2 + 4     = 12,104
    Total adapter params:  FedIT=147,456   Ravan≈145,248
    Tolerance: within 5% of each other.
    """
    import torch.nn as nn

    d = 768
    fedit_rank = 8
    ravan_heads = 4
    ravan_rank  = 55
    n_layers    = 12  # 6 transformer blocks × (q, v)

    fedit_adapter_per_layer = fedit_rank * d + d * fedit_rank
    fedit_adapter_total     = fedit_adapter_per_layer * n_layers

    # Ravan trainable: H [heads, rank, rank] + scales [heads]
    ravan_adapter_per_layer = ravan_heads * ravan_rank * ravan_rank + ravan_heads
    ravan_adapter_total     = ravan_adapter_per_layer * n_layers

    ratio = fedit_adapter_total / ravan_adapter_total
    assert 0.95 <= ratio <= 1.05, (
        f"Adapter param budgets diverge: FedIT={fedit_adapter_total:,}, "
        f"Ravan={ravan_adapter_total:,}, ratio={ratio:.3f}"
    )

    # Cross-check with actual module instantiation
    linear = nn.Linear(d, d, bias=False)
    lora_layer  = LoRALinear(linear, rank=fedit_rank)
    ravan_layer = RavanLinear(nn.Linear(d, d, bias=False), heads=ravan_heads, rank=ravan_rank)

    lora_params  = sum(p.numel() for p in lora_layer.parameters()  if p.requires_grad)
    ravan_params = sum(p.numel() for p in ravan_layer.parameters() if p.requires_grad)

    assert lora_params  == fedit_adapter_per_layer, (
        f"LoRA module: expected {fedit_adapter_per_layer}, got {lora_params}")
    assert ravan_params == ravan_adapter_per_layer, (
        f"Ravan module: expected {ravan_adapter_per_layer}, got {ravan_params}")


# ---------------------------------------------------------------------------
# Test 8: Report asset generation with dummy data
# ---------------------------------------------------------------------------

def test_result_asset_generation_with_dummy_data(tmp_path):
    """Create minimal dummy results and verify generate_report_assets produces outputs."""
    # ── write dummy results ──────────────────────────────────────────────────
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    methods = [
        ("fedit",              "iid",    0),
        ("fedit",              "noniid", 0),
        ("ravan_gram_schmidt", "iid",    0),
        ("ravan_gram_schmidt", "noniid", 0),
        ("ravan_svd",          "iid",    0),
        ("ravan_svd",          "noniid", 0),
    ]

    all_rows = []
    for method, split, seed in methods:
        run_name = f"{method}_{split}_seed{seed}_20260101_000000"
        run_dir  = results_dir / run_name
        run_dir.mkdir()

        summary = {
            "run_name":                    run_name,
            "method":                      method,
            "init":                        "lora" if method == "fedit" else method.split("_", 1)[1],
            "split":                       split,
            "seed":                        seed,
            "rounds":                      5,
            "clients":                     20,
            "clients_per_round":           3,
            "local_steps":                 5,
            "rank":                        8  if method == "fedit" else 55,
            "heads":                       None if method == "fedit" else 4,
            "lr":                          1e-3,
            "batch_size":                  16,
            "dirichlet_alpha":             0.3,
            "max_length":                  128,
            "final_acc":                   0.30 + 0.05 * (methods.index((method, split, seed))),
            "best_acc":                    0.32 + 0.05 * (methods.index((method, split, seed))),
            "final_test_acc":              0.30 + 0.05 * (methods.index((method, split, seed))),
            "best_test_acc":               0.32,
            "final_loss":                  None,
            "trainable_adapter_params":    147456 if method == "fedit" else 145248,
            "trainable_head_params":       605972,
            "total_trainable_params":      753428 if method == "fedit" else 751220,
            "communicated_adapter_params_per_client": 147456 if method == "fedit" else 145200,
            "communicated_params_per_round": (753428 if method == "fedit" else 751172) * 3,
            "total_main_communication_params": (753428 if method == "fedit" else 751172) * 3 * 5,
            "warmup_clients":              5   if method == "ravan_svd" else None,
            "warmup_steps":                50  if method == "ravan_svd" else None,
            "warmup_rank":                 220 if method == "ravan_svd" else None,
            "warmup_weighting":            "uniform" if method == "ravan_svd" else None,
            "warmup_trainable_params":     None,
            "warmup_communicated_params":  20275200 if method == "ravan_svd" else 0,
            "warmup_train_runtime_s":      12.3 if method == "ravan_svd" else None,
            "warmup_svd_runtime_s":        0.02 if method == "ravan_svd" else None,
            "warmup_total_runtime_s":      12.5 if method == "ravan_svd" else None,
            "total_runtime_s":             60.0,
            "git_commit":                  "abc1234",
            "timestamp":                   "2026-01-01T00:00:00",
        }
        with open(run_dir / "summary.json", "w") as f:
            json.dump(summary, f)

        # per-round CSV
        rounds_path = run_dir / "rounds.csv"
        with open(rounds_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "test_acc", "test_loss",
                                                    "elapsed_seconds", "selected_clients",
                                                    "train_runtime_seconds"])
            writer.writeheader()
            for r in range(1, 6):
                writer.writerow({
                    "round": r,
                    "test_acc": 0.2 + 0.02 * r,
                    "test_loss": None,
                    "elapsed_seconds": 10.0,
                    "selected_clients": "[0, 1, 2]",
                    "train_runtime_seconds": 9.0,
                })
        all_rows.append(summary)

    # Write master CSV
    master = results_dir / "all_results.csv"
    with open(master, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # ── run the asset generator ──────────────────────────────────────────────
    from scripts.generate_report_assets import (
        load_summaries, load_histories, load_singular_values,
        make_main_results_table, make_gap_analysis_table,
        make_init_costs_table, make_setup_summary_table, make_parameter_budget_table,
        fig_learning_curves, fig_final_accuracy_bar,
        fig_iid_vs_noniid_drop, fig_gap_analysis, fig_init_cost_vs_accuracy,
    )

    out_dir     = tmp_path / "paper_assets"
    out_tables  = out_dir / "tables"
    out_figures = out_dir / "figures"
    out_tables.mkdir(parents=True)
    out_figures.mkdir(parents=True)

    summaries = load_summaries(results_dir)
    histories = load_histories(results_dir, summaries)
    sv_df     = load_singular_values(results_dir, summaries)

    assert len(summaries) == 6, f"Expected 6 runs, got {len(summaries)}"
    assert len(histories) == 6, f"Expected 6 round histories, got {len(histories)}"

    make_main_results_table(summaries, out_tables)
    make_gap_analysis_table(summaries, out_tables)
    make_init_costs_table(summaries, out_tables)
    make_setup_summary_table(summaries, out_tables)
    make_parameter_budget_table(summaries, out_tables)

    fig_learning_curves(summaries, histories, out_figures)
    fig_final_accuracy_bar(summaries, out_figures)
    fig_iid_vs_noniid_drop(summaries, out_figures)
    fig_gap_analysis(summaries, out_figures)
    fig_init_cost_vs_accuracy(summaries, out_figures)

    # ── assertions ────────────────────────────────────────────────────────────
    expected_tables = [
        "main_results.csv", "main_results.tex",
        "gap_analysis.csv",
        "init_costs.csv", "init_costs.tex",
        "setup_summary.csv",
        "parameter_budget.csv",
    ]
    for fname in expected_tables:
        p = out_tables / fname
        assert p.exists(), f"Expected table file not found: {p}"
        assert p.stat().st_size > 0, f"Table file is empty: {p}"

    expected_figures = [
        "learning_curves_iid.png",
        "learning_curves_noniid.png",
        "final_accuracy_by_method_split.png",
        "iid_vs_noniid_drop.png",
        "gap_analysis.png",
        "init_cost_vs_accuracy.png",
    ]
    for fname in expected_figures:
        p = out_figures / fname
        assert p.exists(), f"Expected figure not found: {p}"
        assert p.stat().st_size > 0, f"Figure file is empty: {p}"


# ---------------------------------------------------------------------------
# Test 9: Warm-up factor communication count
# ---------------------------------------------------------------------------

def test_warmup_factor_communication_count():
    """Default warm-up (K=5, R=220, 12 layers of 768×768) gives 20,275,200 communicated params.

    Clients upload LoRA factors B_c [d_out×R] and A_c [R×d_in] per adapted layer.
    Per layer per client: R*(d_out + d_in) = 220*(768+768) = 337,920
    Total: 5 clients × 12 layers × 337,920 = 20,275,200
    """
    K_warm = 5
    total_rank = 220   # heads * per_head_rank = 4 * 55
    n_layers = 12
    d = 768            # DistilBERT hidden size (q_lin and v_lin are 768×768)

    factor_comm_per_layer = total_rank * (d + d)          # B_c + A_c
    factor_comm_per_client = n_layers * factor_comm_per_layer
    total = K_warm * factor_comm_per_client

    assert factor_comm_per_layer == 337_920, f"Per layer: {factor_comm_per_layer}"
    assert factor_comm_per_client == 4_055_040, f"Per client: {factor_comm_per_client}"
    assert total == 20_275_200, f"Total: {total}"


# ---------------------------------------------------------------------------
# Test 10: Ravan-SVD head matches pretrained
# ---------------------------------------------------------------------------

def test_ravan_svd_head_matches_pretrained():
    """inject_ravan with SVD init does not modify the classification head.

    The main Ravan model in train_ravan.py is always freshly created via
    make_distilbert(), so the head comes from pretrained weights regardless
    of any warm-up training.  We verify inject_ravan does not touch the head
    by recording head params before and after injection.
    """
    import torch
    torch.manual_seed(0)
    from federated.model import make_distilbert, inject_ravan

    heads, rank = 2, 4
    R = heads * rank

    # Simulate SVD output: random orthonormal U_R, Vh_R for each adapted layer
    d = 768
    dW = torch.randn(d, d)
    U, _, Vh = torch.linalg.svd(dW, full_matrices=False)
    one_svd = (U[:, :R].contiguous(), Vh[:R, :].contiguous())
    # 6 transformer layers × (q_svd, v_svd)
    svd_per_layer = [(one_svd, one_svd) for _ in range(6)]

    head_keys = ["pre_classifier.weight", "pre_classifier.bias",
                 "classifier.weight", "classifier.bias"]

    m_ravan = make_distilbert()
    # Record head params before injection
    params_before = {k: v.clone() for k, v in m_ravan.named_parameters()
                     if k in head_keys}

    inject_ravan(m_ravan, heads=heads, rank=rank, init_method="svd",
                 svd_matrices_per_layer=svd_per_layer)

    params_after = dict(m_ravan.named_parameters())
    for key in head_keys:
        assert torch.allclose(params_before[key], params_after[key]), \
            f"Head param '{key}' was modified by inject_ravan — warm-up state must not leak"


# ---------------------------------------------------------------------------
# Test 11: Non-empty client splits
# ---------------------------------------------------------------------------

def test_nonempty_client_splits():
    """IID and Dirichlet (alpha=0.3) splits produce no empty clients for seeds 0, 1, 2."""
    import numpy as np
    from federated.data import iid_split, dirichlet_split

    # Synthetic dataset mimicking 20 Newsgroups structure
    rng = np.random.default_rng(99)
    n_examples = 11_314
    n_classes = 20
    n_clients = 20
    labels = rng.integers(0, n_classes, size=n_examples)

    for seed in [0, 1, 2]:
        iid_splits = iid_split(n_examples, n_clients, seed)
        assert all(len(s) > 0 for s in iid_splits), \
            f"IID split seed={seed} produced empty client(s)"

        noniid_splits = dirichlet_split(labels, n_clients, alpha=0.3, seed=seed)
        assert all(len(s) > 0 for s in noniid_splits), \
            f"Dirichlet alpha=0.3 seed={seed} produced empty client(s)"


# ---------------------------------------------------------------------------
# Test 12: Parameter accounting
# ---------------------------------------------------------------------------

def test_parameter_accounting():
    """Exact parameter counts must match the report's specified values."""
    import torch
    from federated.model import (
        make_distilbert, inject_lora, inject_ravan,
        count_params_detailed, count_adapter_communicated,
    )

    # FedIT (rank=8, d=768, 12 layers)
    torch.manual_seed(0)
    m_fedit = make_distilbert()
    inject_lora(m_fedit, rank=8)
    d_fedit = count_params_detailed(m_fedit)
    ac_fedit = count_adapter_communicated(m_fedit)

    assert d_fedit["trainable_adapter_params"] == 147_456, \
        f"FedIT adapter trainable: expected 147456, got {d_fedit['trainable_adapter_params']}"
    assert d_fedit["trainable_head_params"] == 605_972, \
        f"FedIT head: expected 605972, got {d_fedit['trainable_head_params']}"
    assert d_fedit["total_trainable_params"] == 753_428, \
        f"FedIT total trainable: expected 753428, got {d_fedit['total_trainable_params']}"
    assert ac_fedit == 147_456, \
        f"FedIT adapter comm: expected 147456, got {ac_fedit}"

    # Ravan (heads=4, rank=55, d=768, 12 layers)
    torch.manual_seed(0)
    m_ravan = make_distilbert()
    inject_ravan(m_ravan, heads=4, rank=55, init_method="gram_schmidt")
    d_ravan = count_params_detailed(m_ravan)
    ac_ravan = count_adapter_communicated(m_ravan)

    assert d_ravan["trainable_adapter_params"] == 145_248, \
        f"Ravan adapter trainable: expected 145248, got {d_ravan['trainable_adapter_params']}"
    assert d_ravan["trainable_head_params"] == 605_972, \
        f"Ravan head: expected 605972, got {d_ravan['trainable_head_params']}"
    assert d_ravan["total_trainable_params"] == 751_220, \
        f"Ravan total trainable: expected 751220, got {d_ravan['total_trainable_params']}"
    assert ac_ravan == 145_200, \
        f"Ravan adapter comm: expected 145200, got {ac_ravan}"
