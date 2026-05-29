"""DistilBERT model construction and adapter injection.

We use distilbert-base-uncased (66M params, 6 layers, hidden=768).

Frozen:   entire DistilBERT backbone (embeddings + transformer)
Trainable & communicated:
  - adapters injected into q_lin and v_lin of every attention layer
  - pre_classifier  Linear(768, 768)
  - classifier      Linear(768, 20)

Adapted layers per model:
  6 transformer layers × 2 projections (q, v) = 12 adapted linear layers.
"""

import torch.nn as nn
from transformers import DistilBertForSequenceClassification

from .lora  import LoRALinear
from .ravan import RavanLinear

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 20


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_distilbert() -> DistilBertForSequenceClassification:
    """Load DistilBERT, freeze the backbone, keep the head trainable."""
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS
    )
    # Freeze everything first
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze classification head (randomly initialised for this task)
    for p in model.pre_classifier.parameters():
        p.requires_grad_(True)
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    return model


# ---------------------------------------------------------------------------
# Adapter injection
# ---------------------------------------------------------------------------

def inject_lora(model: DistilBertForSequenceClassification, rank: int, scaling: float = 1.0):
    """Replace q_lin and v_lin in every transformer layer with LoRALinear."""
    for layer in model.distilbert.transformer.layer:
        attn = layer.attention
        attn.q_lin = LoRALinear(attn.q_lin, rank=rank, scaling=scaling)
        attn.v_lin = LoRALinear(attn.v_lin, rank=rank, scaling=scaling)


def inject_ravan(
    model: DistilBertForSequenceClassification,
    heads: int,
    rank: int,
    init_method: str = "gram_schmidt",
    svd_matrices_per_layer: list | None = None,
):
    """Replace q_lin and v_lin in every transformer layer with RavanLinear.

    Args:
        svd_matrices_per_layer : list of (q_svd, v_svd) per layer, where each
                                 svd entry is (U_R, Vh_R).  Required when
                                 init_method="svd".
    """
    layers = list(model.distilbert.transformer.layer)

    for i, layer in enumerate(layers):
        attn = layer.attention

        q_svd = None
        v_svd = None
        if svd_matrices_per_layer is not None:
            q_svd, v_svd = svd_matrices_per_layer[i]

        attn.q_lin = RavanLinear(
            attn.q_lin,
            heads=heads,
            rank=rank,
            init_method=init_method,
            svd_matrices=q_svd,
        )
        attn.v_lin = RavanLinear(
            attn.v_lin,
            heads=heads,
            rank=rank,
            init_method=init_method,
            svd_matrices=v_svd,
        )


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_params, total_params)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return trainable, total


def get_lora_layers(model: DistilBertForSequenceClassification):
    """Yield every LoRALinear in the model."""
    for layer in model.distilbert.transformer.layer:
        attn = layer.attention
        if isinstance(attn.q_lin, LoRALinear):
            yield attn.q_lin
        if isinstance(attn.v_lin, LoRALinear):
            yield attn.v_lin


def get_ravan_layers(model: DistilBertForSequenceClassification):
    """Yield every RavanLinear in the model."""
    for layer in model.distilbert.transformer.layer:
        attn = layer.attention
        if isinstance(attn.q_lin, RavanLinear):
            yield attn.q_lin
        if isinstance(attn.v_lin, RavanLinear):
            yield attn.v_lin


def print_param_summary(model: nn.Module):
    trainable, total = count_params(model)
    adapter_params = sum(
        p.numel()
        for layer in model.distilbert.transformer.layer
        for m in [layer.attention.q_lin, layer.attention.v_lin]
        if isinstance(m, (LoRALinear, RavanLinear))
        for p in m.parameters()
        if p.requires_grad
    )
    head_params = (
        sum(p.numel() for p in model.pre_classifier.parameters() if p.requires_grad) +
        sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    )
    print(f"  Trainable params : {trainable:>10,}  /  {total:,} total")
    print(f"    — adapter      : {adapter_params:>10,}")
    print(f"    — head         : {head_params:>10,}")