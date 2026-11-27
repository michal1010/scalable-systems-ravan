"""T5-encoder classification model and adapter injection.

NOTE: T5-encoder classification approximation — uses T5EncoderModel (encoder
half of t5-base) with masked mean pooling and a linear classification head.
This is NOT T5ForConditionalGeneration text-to-text classification.

Frozen:   entire T5 encoder backbone (embeddings + 12 transformer blocks)
Trainable & communicated:
  - adapters injected into q and v of every encoder self-attention layer
  - classifier  Linear(d_model, 20)

Adapted layers per model:
  12 encoder blocks × 2 projections (q, v) = 24 adapted linear layers.

t5-base dims: d_model=512, num_heads=8, d_kv=64, inner_dim=512
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import T5EncoderModel

from .lora import LoRALinear
from .ravan import RavanLinear

MODEL_NAME = "t5-base"
NUM_LABELS = 20


# ---------------------------------------------------------------------------
# Wrapper model
# ---------------------------------------------------------------------------

@dataclass
class ClassifierOutput:
    logits: torch.Tensor


class T5EncoderForClassification(nn.Module):
    """T5 encoder + masked mean pooling + linear classification head.

    NOTE: T5-encoder classification approximation (not text-to-text).
    """

    def __init__(self, encoder: T5EncoderModel, d_model: int, num_labels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> ClassifierOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state  # [B, T, d_model]
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(1)
        return ClassifierOutput(logits=self.classifier(pooled))

    def gradient_checkpointing_enable(self, **kwargs):
        self.encoder.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self, **kwargs):
        self.encoder.gradient_checkpointing_disable(**kwargs)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_t5_encoder(
    train_head: bool = True,
    cache_dir: str | None = None,
) -> T5EncoderForClassification:
    """Load T5EncoderModel, freeze backbone, add classification head."""
    encoder = T5EncoderModel.from_pretrained(MODEL_NAME, cache_dir=cache_dir)
    d_model = encoder.config.d_model
    model = T5EncoderForClassification(encoder, d_model=d_model, num_labels=NUM_LABELS)

    for p in model.encoder.parameters():
        p.requires_grad_(False)

    if train_head:
        for p in model.classifier.parameters():
            p.requires_grad_(True)

    return model


# ---------------------------------------------------------------------------
# Adapter injection
# ---------------------------------------------------------------------------

def _encoder_blocks(model: T5EncoderForClassification):
    """Return the ModuleList of T5Block for the encoder stack."""
    return model.encoder.encoder.block


def inject_lora_t5(
    model: T5EncoderForClassification,
    rank: int,
    scaling: float = 1.0,
) -> None:
    """Replace q and v in every encoder self-attention block with LoRALinear."""
    for block in _encoder_blocks(model):
        attn = block.layer[0].SelfAttention
        attn.q = LoRALinear(attn.q, rank=rank, scaling=scaling)
        attn.v = LoRALinear(attn.v, rank=rank, scaling=scaling)


def inject_ravan_t5(
    model: T5EncoderForClassification,
    heads: int,
    rank: int,
    init_method: str = "gram_schmidt",
    svd_matrices_per_layer: list | None = None,
) -> None:
    """Replace q and v in every encoder self-attention block with RavanLinear."""
    blocks = list(_encoder_blocks(model))
    for i, block in enumerate(blocks):
        attn = block.layer[0].SelfAttention
        q_svd = v_svd = None
        if svd_matrices_per_layer is not None:
            q_svd, v_svd = svd_matrices_per_layer[i]
        attn.q = RavanLinear(
            attn.q, heads=heads, rank=rank,
            init_method=init_method, svd_matrices=q_svd,
        )
        attn.v = RavanLinear(
            attn.v, heads=heads, rank=rank,
            init_method=init_method, svd_matrices=v_svd,
        )


# ---------------------------------------------------------------------------
# Inspection helpers
# ---------------------------------------------------------------------------

def count_params_detailed_t5(model: T5EncoderForClassification) -> dict:
    """Return dict with adapter, head, and total trainable parameter counts."""
    adapter_params = sum(
        p.numel()
        for block in _encoder_blocks(model)
        for m in [block.layer[0].SelfAttention.q, block.layer[0].SelfAttention.v]
        if isinstance(m, (LoRALinear, RavanLinear))
        for p in m.parameters()
        if p.requires_grad
    )
    head_params = sum(
        p.numel() for p in model.classifier.parameters() if p.requires_grad
    )
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "trainable_adapter_params": adapter_params,
        "trainable_head_params":    head_params,
        "total_trainable_params":   total_trainable,
    }


def count_communicated_per_round_t5(model: T5EncoderForClassification) -> int:
    """Return scalars communicated per client per FL round (adapters + head)."""
    comm = 0
    for block in _encoder_blocks(model):
        for m in [block.layer[0].SelfAttention.q, block.layer[0].SelfAttention.v]:
            if isinstance(m, LoRALinear):
                comm += m.lora_A.numel() + m.lora_B.numel()
            elif isinstance(m, RavanLinear):
                comm += m.H.numel()
    comm += sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    return comm


def count_adapter_communicated_t5(model: T5EncoderForClassification) -> int:
    """Return adapter-only parameters communicated per client per round."""
    comm = 0
    for block in _encoder_blocks(model):
        for m in [block.layer[0].SelfAttention.q, block.layer[0].SelfAttention.v]:
            if isinstance(m, LoRALinear):
                comm += m.lora_A.numel() + m.lora_B.numel()
            elif isinstance(m, RavanLinear):
                comm += m.H.numel()
    return comm


def get_lora_layers_t5(model: T5EncoderForClassification):
    """Yield every LoRALinear in the encoder (q0, v0, q1, v1, ...) order."""
    for block in _encoder_blocks(model):
        attn = block.layer[0].SelfAttention
        if isinstance(attn.q, LoRALinear):
            yield attn.q
        if isinstance(attn.v, LoRALinear):
            yield attn.v


def get_ravan_layers_t5(model: T5EncoderForClassification):
    """Yield every RavanLinear in the encoder (q0, v0, q1, v1, ...) order."""
    for block in _encoder_blocks(model):
        attn = block.layer[0].SelfAttention
        if isinstance(attn.q, RavanLinear):
            yield attn.q
        if isinstance(attn.v, RavanLinear):
            yield attn.v


def print_param_summary_t5(model: T5EncoderForClassification) -> None:
    d = count_params_detailed_t5(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {d['total_trainable_params']:>10,}  /  {total:,} total")
    print(f"    — adapter      : {d['trainable_adapter_params']:>10,}")
    print(f"    — head         : {d['trainable_head_params']:>10,}")
