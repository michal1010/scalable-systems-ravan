"""Client-side local training for the federated simulation.

Both FedIT and Ravan use the same local training procedure: run AdamW
for a fixed number of optimizer steps on the client's DataLoader, then
return the updated model.  The caller is responsible for loading the correct
global state before calling local_train and extracting the updated state after.

Mixed precision (use_amp=True) is supported on CUDA via torch.amp.autocast
and GradScaler.  Gradient accumulation (grad_accum_steps>1) accumulates
gradients over N mini-batches before each optimizer step, so local_steps
always counts optimizer steps regardless of accumulation factor.
"""

import contextlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def local_train(
    model: nn.Module,
    loader: DataLoader,
    local_steps: int,
    lr: float,
    device: torch.device,
    use_amp: bool = False,
    grad_accum_steps: int = 1,
    use_grad_checkpoint: bool = False,
) -> nn.Module:
    """Train model in-place for local_steps optimizer steps.

    Loops over the DataLoader repeatedly until local_steps optimizer steps
    are completed, so the step count is exact regardless of DataLoader length.

    Args:
        model               : model with trainable adapter + head params
        loader              : client DataLoader (input_ids, attention_mask, labels)
        local_steps         : number of optimizer steps (not mini-batch forward passes)
        lr                  : AdamW learning rate
        device              : target device
        use_amp             : enable mixed-precision (float16 autocast, CUDA only)
        grad_accum_steps    : accumulate gradients over N mini-batches per optimizer step
        use_grad_checkpoint : enable gradient checkpointing to reduce activation memory

    Returns the same model object (modified in-place).
    """
    if use_grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    model.train()
    model.to(device)

    use_amp_eff = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp_eff else None
    amp_ctx = (
        torch.amp.autocast(device_type="cuda")
        if use_amp_eff
        else contextlib.nullcontext()
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()

    optimizer.zero_grad()
    step = 0
    accum_count = 0

    while step < local_steps:
        for input_ids, attention_mask, labels in loader:
            if step >= local_steps:
                break

            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels         = labels.to(device)

            with amp_ctx:
                out  = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(out.logits, labels) / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_count += 1
            if accum_count == grad_accum_steps:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1
                if step >= local_steps:
                    break

    return model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Return accuracy on loader. Model is moved to device."""
    model.eval()
    model.to(device)
    correct = total = 0
    for input_ids, attention_mask, labels in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)
        out            = model(input_ids=input_ids, attention_mask=attention_mask)
        preds          = out.logits.argmax(dim=-1)
        correct       += (preds == labels).sum().item()
        total         += labels.size(0)
    return correct / total if total > 0 else 0.0
