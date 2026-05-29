"""Server-side aggregation for FedIT and Ravan.

FedIT (fedit_*):
    Trainable state = {lora_A, lora_B, head params}
    Aggregation     = simple FedAvg (average all trainable tensors).
    This produces the factor-averaging mismatch described in the paper
    because mean(B_c) @ mean(A_c) != mean(B_c @ A_c).

Ravan (ravan_*):
    Upload state = {s_i * H_i per layer, head params}
    Aggregation  = FedAvg on s*H products → new H; scales reset to 1.
    Because B_i and A_i are frozen and shared, averaging the products
    s_i*H_i is an exact aggregation of the true client updates.

Both methods share the same head FedAvg (pre_classifier + classifier).
"""

import torch
import torch.nn as nn

from .ravan import RavanLinear


# ---------------------------------------------------------------------------
# FedIT helpers
# ---------------------------------------------------------------------------

def fedit_get_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a copy of all trainable parameters as a flat dict."""
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k in trainable}


def fedit_load_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Load a FedIT state dict into the model (updates trainable params only)."""
    cur = model.state_dict()
    cur.update(state)
    model.load_state_dict(cur, strict=True)


def fedit_aggregate(client_states: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """FedAvg: element-wise mean of all client state dicts."""
    avg = {}
    for key in client_states[0]:
        stacked = torch.stack([s[key].float() for s in client_states])
        avg[key] = stacked.mean(dim=0).to(client_states[0][key].dtype)
    return avg


# ---------------------------------------------------------------------------
# Ravan helpers
# ---------------------------------------------------------------------------

_RAVAN_PREFIX = "ravan_sH/"
_HEAD_PREFIX  = "head/"


def ravan_get_upload(model: nn.Module) -> dict[str, torch.Tensor]:
    """Construct what a client sends to the server:
    - For each RavanLinear: s_i * H_i products  [heads, rank, rank]
    - Head params: pre_classifier.*, classifier.*
    """
    upload: dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if isinstance(module, RavanLinear):
            # scales: [heads]  →  broadcast to [heads, rank, rank]
            sH = (module.scales[:, None, None] * module.H).detach().clone()
            upload[f"{_RAVAN_PREFIX}{name}"] = sH

    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    sd = model.state_dict()
    for k in sd:
        if ("pre_classifier" in k or k.startswith("classifier")) and k in trainable:
            upload[f"{_HEAD_PREFIX}{k}"] = sd[k].clone()

    return upload


def ravan_aggregate(client_uploads: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """FedAvg on s*H products and head params."""
    avg: dict[str, torch.Tensor] = {}
    for key in client_uploads[0]:
        stacked = torch.stack([u[key].float() for u in client_uploads])
        avg[key] = stacked.mean(dim=0).to(client_uploads[0][key].dtype)
    return avg


def ravan_load_global(model: nn.Module, aggregated: dict[str, torch.Tensor]) -> None:
    """Load aggregated state into model:
    - Ravan layers: H = averaged(s*H), scales reset to 1
    - Head params: FedAvg result loaded directly
    """
    named_modules = dict(model.named_modules())
    named_params  = dict(model.named_parameters())

    for key, val in aggregated.items():
        if key.startswith(_RAVAN_PREFIX):
            name   = key[len(_RAVAN_PREFIX):]
            module = named_modules[name]
            with torch.no_grad():
                module.H.copy_(val)
                module.scales.fill_(1.0)

        elif key.startswith(_HEAD_PREFIX):
            param_name = key[len(_HEAD_PREFIX):]
            if param_name in named_params:
                with torch.no_grad():
                    named_params[param_name].data.copy_(val)
