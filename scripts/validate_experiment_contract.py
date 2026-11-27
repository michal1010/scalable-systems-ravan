"""Validate that the implementation matches the experimental contract in the report.

Run fast (no full training), fail loudly on any mismatch.

Usage:
    python -m scripts.validate_experiment_contract
"""

import sys
import torch

EXPECTED = {
    "model_name":             "distilbert-base-uncased",
    "num_labels":             20,
    "n_transformer_layers":   6,
    "adapted_per_layer":      2,   # q_lin, v_lin
    "adapted_matrices":       12,  # 6 × 2
    "fedit_rank":             8,
    "ravan_heads":            4,
    "ravan_rank":             55,
    "warmup_rank":            220, # heads × rank = 4 × 55
    "warmup_clients":         5,
    "warmup_steps":           50,
    "clients":                20,
    "clients_per_round":      3,
    "local_steps":            50,
    "rounds":                 50,
    "batch_size":             16,
    "dirichlet_alpha":        0.3,
    "max_length":             128,
    "weight_decay":           0.01,
    # exact param counts
    "fedit_adapter_trainable": 147_456,
    "fedit_adapter_comm":      147_456,
    "fedit_head_params":       605_972,
    "fedit_total_trainable":   753_428,
    "ravan_adapter_trainable": 145_248,
    "ravan_adapter_comm":      145_200,
    "ravan_head_params":       605_972,
    "ravan_total_trainable":   751_220,
    "warmup_comm_total":       20_275_200,
}


def _check(name, actual, expected):
    if actual != expected:
        print(f"  FAIL  {name}: expected {expected!r}, got {actual!r}", flush=True)
        return False
    print(f"  ok    {name}", flush=True)
    return True


def main():
    failures = 0

    print("=== validate_experiment_contract ===\n")

    # --- Static checks (no model loading) ---
    print("[1] Static constants")
    from federated.model import MODEL_NAME, NUM_LABELS
    from federated.data import MAX_LENGTH, MODEL_NAME as DATA_MODEL_NAME

    failures += not _check("MODEL_NAME", MODEL_NAME, EXPECTED["model_name"])
    failures += not _check("NUM_LABELS", NUM_LABELS, EXPECTED["num_labels"])
    failures += not _check("MAX_LENGTH", MAX_LENGTH, EXPECTED["max_length"])

    # --- Data preprocessing ---
    print("\n[2] Data preprocessing")
    import inspect
    from federated.data import load_20newsgroups
    src = inspect.getsource(load_20newsgroups)
    has_headers  = '"headers"' in src or "'headers'" in src
    has_footers  = '"footers"' in src or "'footers'" in src
    has_quotes   = '"quotes"'  in src or "'quotes'"  in src
    failures += not _check("remove headers", has_headers, True)
    failures += not _check("remove footers", has_footers, True)
    failures += not _check("remove quotes",  has_quotes,  True)

    # --- Client.py optimizer ---
    print("\n[3] Optimizer settings")
    from federated.client import local_train
    src_client = inspect.getsource(local_train)
    has_adamw       = "AdamW" in src_client
    has_wd          = "weight_decay=0.01" in src_client
    has_cross_ent   = "CrossEntropyLoss" in src_client
    failures += not _check("optimizer=AdamW",       has_adamw,     True)
    failures += not _check("weight_decay=0.01",     has_wd,        True)
    failures += not _check("CrossEntropyLoss",      has_cross_ent, True)

    # --- Warm-up communication mode ---
    print("\n[4] Warm-up communication accounting")
    from federated.warmup import federated_svd_init
    src_warmup = inspect.getsource(federated_svd_init)
    # Protocol: client uploads factors (B_c, A_c); server reconstructs ΔW_c = B_c @ A_c
    has_factor_upload = "client_factors" in src_warmup and "lora_B.detach" in src_warmup
    has_server_recon  = "B_c @ A_c" in src_warmup
    has_factor_comm   = "B_c.numel() + A_c.numel()" in src_warmup
    failures += not _check("client uploads factors (not full ΔW)", has_factor_upload, True)
    failures += not _check("server reconstructs ΔW_c = B_c @ A_c", has_server_recon, True)
    failures += not _check("comm counts factor numel, not dW numel",  has_factor_comm, True)

    warmup_formula = EXPECTED["warmup_clients"] * EXPECTED["adapted_matrices"] * (
        EXPECTED["warmup_rank"] * (768 + 768)
    )
    failures += not _check("warmup_comm formula value", warmup_formula, EXPECTED["warmup_comm_total"])

    # --- Model loading and exact param counts ---
    print("\n[5] DistilBERT adapter injection")
    from federated.model import (
        make_distilbert, inject_lora, inject_ravan,
        count_params_detailed, count_adapter_communicated,
        get_ravan_layers, get_lora_layers,
    )
    from federated.ravan import RavanLinear
    from federated.lora import LoRALinear

    torch.manual_seed(0)
    m_fedit = make_distilbert()
    inject_lora(m_fedit, rank=8)

    # Verify adapter placement
    n_adapted = sum(
        1 for layer in m_fedit.distilbert.transformer.layer
        for m in [layer.attention.q_lin, layer.attention.v_lin]
        if isinstance(m, LoRALinear)
    )
    failures += not _check("adapted_matrices (FedIT)", n_adapted, EXPECTED["adapted_matrices"])

    # FedIT param counts
    d_fedit = count_params_detailed(m_fedit)
    ac_fedit = count_adapter_communicated(m_fedit)
    failures += not _check("fedit_adapter_trainable", d_fedit["trainable_adapter_params"], EXPECTED["fedit_adapter_trainable"])
    failures += not _check("fedit_head_params",       d_fedit["trainable_head_params"],     EXPECTED["fedit_head_params"])
    failures += not _check("fedit_total_trainable",   d_fedit["total_trainable_params"],    EXPECTED["fedit_total_trainable"])
    failures += not _check("fedit_adapter_comm",      ac_fedit,                             EXPECTED["fedit_adapter_comm"])

    torch.manual_seed(0)
    m_ravan = make_distilbert()
    inject_ravan(m_ravan, heads=4, rank=55, init_method="gram_schmidt")

    n_adapted_ravan = sum(
        1 for layer in m_ravan.distilbert.transformer.layer
        for m in [layer.attention.q_lin, layer.attention.v_lin]
        if isinstance(m, RavanLinear)
    )
    failures += not _check("adapted_matrices (Ravan)", n_adapted_ravan, EXPECTED["adapted_matrices"])

    d_ravan = count_params_detailed(m_ravan)
    ac_ravan = count_adapter_communicated(m_ravan)
    failures += not _check("ravan_adapter_trainable", d_ravan["trainable_adapter_params"], EXPECTED["ravan_adapter_trainable"])
    failures += not _check("ravan_head_params",       d_ravan["trainable_head_params"],     EXPECTED["ravan_head_params"])
    failures += not _check("ravan_total_trainable",   d_ravan["total_trainable_params"],    EXPECTED["ravan_total_trainable"])
    failures += not _check("ravan_adapter_comm",      ac_ravan,                             EXPECTED["ravan_adapter_comm"])

    # --- Zero initial adapter output ---
    print("\n[6] Zero initial adapter output")
    x = torch.randn(2, EXPECTED["max_length"], 768)
    for ravan_layer in get_ravan_layers(m_ravan):
        base = ravan_layer.linear(x)
        out  = ravan_layer(x)
        max_diff = (base - out).abs().max().item()
        if max_diff > 1e-5:
            print(f"  FAIL  Ravan initial output non-zero: max_diff={max_diff:.2e}")
            failures += 1
            break
    else:
        print("  ok    initial Ravan update is zero (H=0)")

    # --- SVD singular values not absorbed ---
    print("\n[7] SVD: singular values not absorbed")
    dW_test = torch.randn(32, 24)
    U, S, Vh = torch.linalg.svd(dW_test, full_matrices=False)
    R_test = 4
    from federated.ravan import svd_init
    B_test, A_test = svd_init(U[:, :R_test], Vh[:R_test, :], heads=2, rank=2)
    # B_test shape: [heads, d_out, rank] — check each column (dim=1) has unit norm
    # If singular values were absorbed, B norms would differ from 1
    norms = B_test.norm(dim=1)  # [heads, rank] — column norms per head
    max_norm_dev = (norms - 1.0).abs().max().item()
    if max_norm_dev > 1e-4:
        print(f"  FAIL  B column norms deviate from 1: max dev = {max_norm_dev:.2e}")
        failures += 1
    else:
        print("  ok    singular values not absorbed (B columns are unit-norm)")

    # --- Summary ---
    print(f"\n{'='*40}")
    if failures == 0:
        print(f"ALL CHECKS PASSED ({failures} failures)")
    else:
        print(f"FAILED: {failures} check(s) did not pass")
        sys.exit(1)


if __name__ == "__main__":
    main()
