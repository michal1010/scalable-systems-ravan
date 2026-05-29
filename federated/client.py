"""Client-side local training for the federated simulation.

Both FedIT and Ravan use the same local training procedure: run SGD/AdamW
for a fixed number of steps on the client's DataLoader, then return the
updated model.  The caller is responsible for loading the correct global
state before calling local_train and extracting the updated state after.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def local_train(
    model: nn.Module,
    loader: DataLoader,
    local_steps: int,
    lr: float,
    device: torch.device,
) -> nn.Module:
    """Train model in-place for local_steps gradient updates.

    Loops over the DataLoader repeatedly until local_steps is reached,
    so the step count is exact regardless of DataLoader length.

    Args:
        model       : model with trainable adapter + head params
        loader      : client's DataLoader (input_ids, attention_mask, labels)
        local_steps : number of gradient steps (not epochs)
        lr          : AdamW learning rate
        device      : target device

    Returns the same model object (modified in-place).
    """
    model.train()
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    while step < local_steps:
        for input_ids, attention_mask, labels in loader:
            if step >= local_steps:
                break
            input_ids     = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels        = labels.to(device)

            optimizer.zero_grad()
            out  = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.logits, labels)
            loss.backward()
            optimizer.step()
            step += 1

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
