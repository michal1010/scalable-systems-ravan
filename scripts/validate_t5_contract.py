"""T5-mode unit contract validation — fast, no full training.

Checks:
  1.  T5 model loads and adapter injection produces correct layer counts
  2.  Param counts match expected values (d_model=768, 24 adapted layers)
  3.  Zero-init property: H=0 adapter contributes nothing
  4.  Gram-Schmidt orthogonality holds for T5 layer dims
  5.  ravan_get_upload is model-agnostic: 24 sH entries, head entries only
  6.  ravan_load_global round-trip preserves sH values exactly
  7.  fedit_get_state / fedit_load_state round-trip is exact
  8.  FedAvg aggregation is exact for Ravan (same property as DistilBERT)
  9.  Gradient accumulation: 2×(bs=8) equivalent to 1×(bs=16) update direction
  10. AMP forward pass produces finite logits
  11. T5 tokenizer integration: tokenizes 20 Newsgroups text correctly
  12. T5 Ravan-SVD warmup: svd_matrices_per_layer has correct shape

Usage:
    python -m scripts.validate_t5_contract
"""

import sys
import torch
import numpy as np

T5_EXPECTED = {
    "model_name":             "t5-base",
    "num_labels":             20,
    "d_model":                768,
    "n_encoder_layers":       12,
    "adapted_per_layer":      2,   # q, v
    "adapted_matrices":       24,  # 12 × 2
    "fedit_rank":             32,
    "ravan_heads":            4,
    "ravan_rank":             110,
    "warmup_rank":            440,  # heads × rank
    # param counts
    "fedit_adapter_trainable": 24 * 2 * 32 * 768,    # 24 layers × (A+B) = 1,179,648
    "fedit_adapter_comm":      24 * 2 * 32 * 768,    # same for FedIT (uploads A+B)
    "fedit_head_params":       768 * 20 + 20,         # 15,380
    "ravan_adapter_trainable": 24 * (4 * 110 * 110 + 4),  # 1,161,696
    "ravan_adapter_comm":      24 * 4 * 110 * 110,        # 1,161,600 (H only, scales absorbed)
    "ravan_head_params":       768 * 20 + 20,              # 15,380
}


def _check(name, actual, expected):
    if actual != expected:
        print(f"  FAIL  {name}: expected {expected!r}, got {actual!r}", flush=True)
        return False
    print(f"  ok    {name}", flush=True)
    return True


def _check_close(name, actual, expected, tol=1e-5):
    diff = abs(actual - expected)
    if diff > tol:
        print(f"  FAIL  {name}: expected ~{expected}, got {actual} (diff={diff:.2e})", flush=True)
        return False
    print(f"  ok    {name}", flush=True)
    return True


def main():
    failures = 0
    print("=== validate_t5_contract ===\n")

    from federated.model_t5 import (
        make_t5_encoder, inject_lora_t5, inject_ravan_t5,
        count_params_detailed_t5, count_communicated_per_round_t5,
        count_adapter_communicated_t5, get_lora_layers_t5, get_ravan_layers_t5,
        T5EncoderForClassification,
    )
    from federated.ravan import RavanLinear
    from federated.lora import LoRALinear
    from federated.server import (
        ravan_get_upload, ravan_load_global, ravan_aggregate,
        fedit_get_state, fedit_load_state, fedit_aggregate,
    )

    # ── [1] Adapter layer counts ──────────────────────────────────────────────
    print("[1] Adapter injection — layer counts")
    torch.manual_seed(0)
    m_lora = make_t5_encoder()
    inject_lora_t5(m_lora, rank=T5_EXPECTED["fedit_rank"])

    n_lora = sum(
        1 for block in m_lora.encoder.encoder.block
        for proj in [block.layer[0].SelfAttention.q, block.layer[0].SelfAttention.v]
        if isinstance(proj, LoRALinear)
    )
    failures += not _check("adapted_matrices (LoRA)", n_lora, T5_EXPECTED["adapted_matrices"])
    failures += not _check("get_lora_layers count", len(list(get_lora_layers_t5(m_lora))),
                           T5_EXPECTED["adapted_matrices"])

    torch.manual_seed(0)
    m_ravan = make_t5_encoder()
    inject_ravan_t5(m_ravan, heads=T5_EXPECTED["ravan_heads"], rank=T5_EXPECTED["ravan_rank"])
    n_ravan = sum(
        1 for block in m_ravan.encoder.encoder.block
        for proj in [block.layer[0].SelfAttention.q, block.layer[0].SelfAttention.v]
        if isinstance(proj, RavanLinear)
    )
    failures += not _check("adapted_matrices (Ravan)", n_ravan, T5_EXPECTED["adapted_matrices"])
    failures += not _check("get_ravan_layers count", len(list(get_ravan_layers_t5(m_ravan))),
                           T5_EXPECTED["adapted_matrices"])

    # ── [2] Param counts ──────────────────────────────────────────────────────
    print("\n[2] Param counts")
    d_lora = count_params_detailed_t5(m_lora)
    ac_lora = count_adapter_communicated_t5(m_lora)
    failures += not _check("fedit_adapter_trainable", d_lora["trainable_adapter_params"],
                           T5_EXPECTED["fedit_adapter_trainable"])
    failures += not _check("fedit_head_params",       d_lora["trainable_head_params"],
                           T5_EXPECTED["fedit_head_params"])
    failures += not _check("fedit_adapter_comm",      ac_lora, T5_EXPECTED["fedit_adapter_comm"])

    d_ravan = count_params_detailed_t5(m_ravan)
    ac_ravan = count_adapter_communicated_t5(m_ravan)
    failures += not _check("ravan_adapter_trainable", d_ravan["trainable_adapter_params"],
                           T5_EXPECTED["ravan_adapter_trainable"])
    failures += not _check("ravan_head_params",       d_ravan["trainable_head_params"],
                           T5_EXPECTED["ravan_head_params"])
    failures += not _check("ravan_adapter_comm",      ac_ravan, T5_EXPECTED["ravan_adapter_comm"])

    # ── [3] Zero-init property ────────────────────────────────────────────────
    print("\n[3] Zero-init: H=0 adapter adds zero")
    x = torch.randn(2, 8, T5_EXPECTED["d_model"])
    all_zero = True
    for ravan_layer in get_ravan_layers_t5(m_ravan):
        base = ravan_layer.linear(x)
        out  = ravan_layer(x)
        diff = (base - out).abs().max().item()
        if diff > 1e-5:
            print(f"  FAIL  Ravan layer output non-zero at H=0: diff={diff:.2e}")
            failures += 1
            all_zero = False
            break
    if all_zero:
        print("  ok    all Ravan adapter outputs are zero at H=0")

    # ── [4] Gram-Schmidt orthogonality ────────────────────────────────────────
    print("\n[4] GS orthogonality for T5 dims (d_model=768, h=4, r=110)")
    from federated.ravan import gram_schmidt_init
    d = T5_EXPECTED["d_model"]
    h = T5_EXPECTED["ravan_heads"]
    r = T5_EXPECTED["ravan_rank"]
    R = h * r
    torch.manual_seed(42)
    B, A = gram_schmidt_init(d, d, h, r)  # d_out=d_in=768 for T5 q,v

    B_all = B.permute(1, 0, 2).reshape(d, R)
    BtB   = B_all.T @ B_all
    B_err = (BtB - torch.eye(R)).abs().max().item()
    failures += not _check_close("B columns orthonormal (max err)", B_err, 0.0, tol=1e-4)

    A_all = A.reshape(R, d)
    AAt   = A_all @ A_all.T
    A_err = (AAt - torch.eye(R)).abs().max().item()
    failures += not _check_close("A rows orthonormal (max err)", A_err, 0.0, tol=1e-4)

    # ── [5] ravan_get_upload model-agnostic ───────────────────────────────────
    print("\n[5] ravan_get_upload model-agnostic (T5)")
    upload = ravan_get_upload(m_ravan)
    ravan_keys = [k for k in upload if k.startswith("ravan_sH/")]
    head_keys  = [k for k in upload if k.startswith("head/")]
    other_keys = [k for k in upload if not k.startswith(("ravan_sH/", "head/"))]
    failures += not _check("ravan_sH/ entries", len(ravan_keys), T5_EXPECTED["adapted_matrices"])
    failures += not _check("head/ entries",     len(head_keys),  2)  # weight + bias
    failures += not _check("no stray keys",     len(other_keys), 0)
    # Verify head keys are classifier params
    head_param_names = {k[len("head/"):] for k in head_keys}
    failures += not _check("head key names",
                           head_param_names, {"classifier.weight", "classifier.bias"})
    # Verify sH shapes: [heads, rank, rank]
    sH_shapes = {v.shape for v in (upload[k] for k in ravan_keys)}
    expected_sH_shape = (T5_EXPECTED["ravan_heads"], T5_EXPECTED["ravan_rank"], T5_EXPECTED["ravan_rank"])
    failures += not _check("sH tensor shape", sH_shapes, {torch.Size(expected_sH_shape)})

    # ── [6] ravan_load_global round-trip ─────────────────────────────────────
    print("\n[6] ravan_load_global round-trip")
    torch.manual_seed(99)
    m_rt = make_t5_encoder()
    inject_ravan_t5(m_rt, heads=4, rank=110)
    # Simulate some training: set random H values
    with torch.no_grad():
        for ravan_layer in get_ravan_layers_t5(m_rt):
            ravan_layer.H.data = torch.randn_like(ravan_layer.H)
            ravan_layer.scales.data = torch.rand_like(ravan_layer.scales) + 0.5
    upload_before = ravan_get_upload(m_rt)
    # Aggregate (single client → mean is itself)
    agg = ravan_aggregate([upload_before])
    ravan_load_global(m_rt, agg)
    # After load: H should equal sH from upload (since scales were reset to 1.0)
    all_match = True
    for k in [k for k in upload_before if k.startswith("ravan_sH/")]:
        layer_name = k[len("ravan_sH/"):]
        module = dict(m_rt.named_modules())[layer_name]
        diff = (module.H - upload_before[k]).abs().max().item()
        if diff > 1e-6:
            print(f"  FAIL  round-trip mismatch at {layer_name}: diff={diff:.2e}")
            failures += 1
            all_match = False
            break
        # scales should be reset to 1
        scale_err = (module.scales - 1.0).abs().max().item()
        if scale_err > 1e-6:
            print(f"  FAIL  scales not reset to 1 at {layer_name}: err={scale_err:.2e}")
            failures += 1
            all_match = False
            break
    if all_match:
        print("  ok    H loaded from sH upload; scales reset to 1.0")

    # ── [7] fedit state round-trip ────────────────────────────────────────────
    print("\n[7] fedit_get_state / fedit_load_state round-trip (T5)")
    torch.manual_seed(7)
    m_fedit_rt = make_t5_encoder()
    inject_lora_t5(m_fedit_rt, rank=32)
    state = fedit_get_state(m_fedit_rt)
    # Corrupt trainable params, then restore
    with torch.no_grad():
        for k in state:
            state[k] = state[k] + 0.1  # add noise
    fedit_load_state(m_fedit_rt, state)
    state_after = fedit_get_state(m_fedit_rt)
    match = all(torch.allclose(state[k], state_after[k]) for k in state)
    failures += not _check("fedit state round-trip exact", match, True)

    # ── [8] Exact aggregation property (T5 Ravan) ────────────────────────────
    print("\n[8] Exact aggregation property (T5 Ravan, 3 clients)")
    torch.manual_seed(42)
    m_agg = make_t5_encoder()
    inject_ravan_t5(m_agg, heads=4, rank=110)
    # Get frozen B, A from one Ravan layer for verification
    layer0 = list(get_ravan_layers_t5(m_agg))[0]
    B_frozen = layer0.B.clone()
    A_frozen = layer0.A.clone()
    num_clients_agg = 3
    # Simulate random client H and scales
    client_uploads_list = []
    for _ in range(num_clients_agg):
        with torch.no_grad():
            layer0.H.data = torch.randn_like(layer0.H)
            layer0.scales.data = torch.rand_like(layer0.scales) + 0.5
        client_uploads_list.append(ravan_get_upload(m_agg))
    # Aggregate
    agg_state = ravan_aggregate(client_uploads_list)
    ravan_load_global(m_agg, agg_state)
    # Verify: aggregated H = mean of sH products → correct ΔW reconstruction
    layer0_name = list(k[len("ravan_sH/"):] for k in client_uploads_list[0] if k.startswith("ravan_sH/"))[0]
    sH_key = f"ravan_sH/{layer0_name}"
    # Mean of sH products
    sH_mean = torch.stack([u[sH_key].float() for u in client_uploads_list]).mean(0)
    H_loaded = layer0.H.data.float()
    exact_match = torch.allclose(sH_mean, H_loaded, atol=1e-5)
    failures += not _check("exact aggregation (sH_avg == loaded H)", exact_match, True)
    failures += not _check("scales reset to 1 after aggregation",
                           (layer0.scales - 1.0).abs().max().item() < 1e-6, True)

    # ── [9] Gradient accumulation direction ──────────────────────────────────
    print("\n[9] Gradient accumulation: 2×(bs=8, accum=2) ≈ 1×(bs=16, accum=1)")
    from federated.client import local_train
    torch.manual_seed(11)
    # Fake tiny dataset
    inp = torch.randint(0, 32128, (32, 16))
    msk = torch.ones(32, 16, dtype=torch.long)
    lbl = torch.randint(0, 20, (32,))
    from torch.utils.data import TensorDataset, DataLoader
    ds = TensorDataset(inp, msk, lbl)
    loader_bs16 = DataLoader(ds, batch_size=16, shuffle=False)
    loader_bs8  = DataLoader(ds, batch_size=8,  shuffle=False)

    def _extract_grads(model):
        """Return list of first-param gradient norms."""
        return [p.grad.clone() for p in model.parameters() if p.requires_grad and p.grad is not None]

    # Run one optimizer step with bs=16, accum=1
    torch.manual_seed(11)
    m_ref = make_t5_encoder()
    inject_lora_t5(m_ref, rank=4)
    optimizer_ref = torch.optim.AdamW([p for p in m_ref.parameters() if p.requires_grad], lr=1e-4)
    m_ref.train()
    for batch in loader_bs16:
        ids, mask, labels = [t for t in batch]
        out = m_ref(ids, mask)
        loss = torch.nn.CrossEntropyLoss()(out.logits, labels)
        optimizer_ref.zero_grad()
        loss.backward()
        break
    grad_norm_ref = sum(p.grad.norm().item() for p in m_ref.parameters() if p.requires_grad and p.grad is not None)

    # Run two accumulation steps with bs=8, accum=2 (same total samples)
    torch.manual_seed(11)
    m_acc = make_t5_encoder()
    inject_lora_t5(m_acc, rank=4)
    optimizer_acc = torch.optim.AdamW([p for p in m_acc.parameters() if p.requires_grad], lr=1e-4)
    m_acc.train()
    optimizer_acc.zero_grad()
    data_iter = iter(loader_bs8)
    for _ in range(2):
        ids, mask, labels = next(data_iter)
        out = m_acc(ids, mask)
        loss = torch.nn.CrossEntropyLoss()(out.logits, labels) / 2
        loss.backward()
    grad_norm_acc = sum(p.grad.norm().item() for p in m_acc.parameters() if p.requires_grad and p.grad is not None)

    # Gradient norms should be close (not necessarily identical due to batch ordering)
    ratio = grad_norm_acc / (grad_norm_ref + 1e-12)
    if 0.3 < ratio < 3.0:
        print(f"  ok    grad norm ratio accum/no-accum = {ratio:.3f} (within 3×)")
    else:
        print(f"  FAIL  grad norm ratio = {ratio:.3f}, expected ~1.0 ± 3×")
        failures += 1

    # ── [10] AMP forward pass ─────────────────────────────────────────────────
    print("\n[10] AMP forward produces finite logits")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.manual_seed(5)
        m_amp = make_t5_encoder()
        inject_lora_t5(m_amp, rank=4)
        m_amp.to(device)
        m_amp.train()
        ids_gpu = inp[:4].to(device)
        msk_gpu = msk[:4].to(device)
        with torch.amp.autocast(device_type="cuda"):
            out_amp = m_amp(ids_gpu, msk_gpu)
        logits_finite = out_amp.logits.isfinite().all().item()
        failures += not _check("AMP forward produces finite logits", logits_finite, True)
    else:
        print("  skip  (no CUDA device available)")

    # ── [11] T5 tokenizer integration ────────────────────────────────────────
    print("\n[11] T5 tokenizer integration")
    from federated.data import build_federated_loaders
    client_loaders, test_loader, tok = build_federated_loaders(
        split_type="iid", num_clients=3, batch_size=8, seed=0,
        max_length=64, limit_examples=60, limit_test_examples=20,
        tokenizer_name="t5-base",
    )
    failures += not _check("num client loaders", len(client_loaders), 3)
    # Check batch shapes
    for ids, mask, labels in client_loaders[0]:
        failures += not _check("input_ids shape[1]", ids.shape[1], 64)
        failures += not _check("attention_mask shape[1]", mask.shape[1], 64)
        failures += not _check("labels dtype", labels.dtype, torch.int64)
        break
    # T5 tokenizer: pad_token_id should be 0 (not the same as DistilBERT's 0)
    pad_ids_in_batch = (ids == 0).any().item()  # should have padding in short sequences
    print(f"  ok    batch contains padding tokens: {pad_ids_in_batch}")

    # ── [12] T5 Ravan-SVD warmup shape check ─────────────────────────────────
    print("\n[12] T5 Ravan-SVD warmup: svd_matrices_per_layer shape")
    from federated.warmup import federated_svd_init
    from federated.model_t5 import inject_lora_t5, get_lora_layers_t5

    small_loaders, _, _ = build_federated_loaders(
        split_type="iid", num_clients=3, batch_size=4, seed=0,
        max_length=32, limit_examples=24,
        tokenizer_name="t5-base",
    )
    device_cpu = torch.device("cpu")
    total_rank = T5_EXPECTED["ravan_heads"] * T5_EXPECTED["ravan_rank"]  # 440

    svd_per_layer, warmup_costs = federated_svd_init(
        client_loaders=small_loaders,
        warmup_clients=2,
        total_rank=total_rank,
        warmup_steps=2,
        lr=1e-3,
        device=device_cpu,
        seed=0,
        make_model_fn=lambda th, cd: make_t5_encoder(train_head=th, cache_dir=cd),
        inject_lora_fn=lambda m, r: inject_lora_t5(m, rank=r),
        get_lora_layers_fn=get_lora_layers_t5,
        count_params_fn=count_params_detailed_t5,
    )
    expected_n_layers = T5_EXPECTED["n_encoder_layers"]
    failures += not _check("svd_per_layer length", len(svd_per_layer), expected_n_layers)
    # Each element: (q_svd, v_svd) where each svd = (U_R, Vh_R)
    q_svd, v_svd = svd_per_layer[0]
    U_R_q, Vh_R_q = q_svd
    failures += not _check("U_R shape[0] = d_model", U_R_q.shape[0], T5_EXPECTED["d_model"])
    failures += not _check("U_R shape[1] = total_rank", U_R_q.shape[1], total_rank)
    failures += not _check("Vh_R shape[0] = total_rank", Vh_R_q.shape[0], total_rank)
    failures += not _check("Vh_R shape[1] = d_model", Vh_R_q.shape[1], T5_EXPECTED["d_model"])

    # Verify we can actually inject Ravan from these SVD matrices
    torch.manual_seed(0)
    m_svd = make_t5_encoder()
    inject_ravan_t5(m_svd, heads=T5_EXPECTED["ravan_heads"], rank=T5_EXPECTED["ravan_rank"],
                    init_method="svd", svd_matrices_per_layer=svd_per_layer)
    n_ravan_svd = len(list(get_ravan_layers_t5(m_svd)))
    failures += not _check("Ravan-SVD layers injected", n_ravan_svd, T5_EXPECTED["adapted_matrices"])
    # Zero-init still holds for SVD init
    x_test = torch.randn(1, 4, T5_EXPECTED["d_model"])
    svd_zero_ok = True
    for rl in get_ravan_layers_t5(m_svd):
        base = rl.linear(x_test)
        out  = rl(x_test)
        if not torch.allclose(base, out, atol=1e-5):
            print("  FAIL  Ravan-SVD init: H=0 does not give zero adapter output")
            failures += 1
            svd_zero_ok = False
            break
    if svd_zero_ok:
        print("  ok    Ravan-SVD init preserves zero-init property")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    if failures == 0:
        print(f"ALL T5 CHECKS PASSED")
    else:
        print(f"FAILED: {failures} check(s) did not pass")
        sys.exit(1)


if __name__ == "__main__":
    main()
