"""
Model adapters that expose a nanochat-compatible causal LM interface.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HFBackedCausalLMAdapter(nn.Module):
    """
    Adapter over HuggingFace AutoModelForCausalLM.

    The adapter intentionally exposes the minimal interface used across nanochat:
    - forward(input_ids, targets=None, loss_reduction='mean')
    - get_device()
    - max_seq_len
    """

    # Engine should use generic full-recompute generation unless a model provides a fast cache path.
    engine_mode = "full_recompute"

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.config = model.config
        self.max_seq_len = getattr(model.config, "max_position_embeddings", None)

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs) -> "HFBackedCausalLMAdapter":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return cls(model)

    def get_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, input_ids, targets=None, kv_cache=None, loss_reduction="mean"):
        # kv_cache is accepted for interface compatibility; full-recompute path ignores it for now.
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss


class Qwen3HFAdapter(HFBackedCausalLMAdapter):
    """
    Thin HF adapter constrained to Qwen3 model family.
    """

    @classmethod
    def from_pretrained(cls, model_id: str = "Qwen/Qwen3-0.6B", **kwargs) -> "Qwen3HFAdapter":
        adapter = super().from_pretrained(model_id, **kwargs)
        model_type = getattr(adapter.config, "model_type", None)
        if model_type != "qwen3":
            raise ValueError(f"Expected a qwen3 model, got model_type={model_type!r} for {model_id}")
        # Re-wrap in the subclass to make isinstance checks explicit.
        return cls(adapter.model)
