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

    def _split_optimizer_params(self):
        named_params = [(n, p) for n, p in self.named_parameters() if p.requires_grad]
        embed_params = [p for n, p in named_params if "embed_tokens" in n]
        lm_head_params = [p for n, p in named_params if "lm_head" in n]
        embed_ids = {id(p) for p in embed_params}
        lm_head_ids = {id(p) for p in lm_head_params}
        rest_params = [p for _, p in named_params if id(p) not in embed_ids and id(p) not in lm_head_ids]
        return embed_params, lm_head_params, rest_params

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.9, 0.95),
        scalar_lr=0.5,
    ):
        # scalar_lr is accepted for interface compatibility.
        del scalar_lr
        embed_params, lm_head_params, rest_params = self._split_optimizer_params()
        param_groups = []
        if lm_head_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=lm_head_params,
                    lr=unembedding_lr,
                    betas=adam_betas,
                    eps=1e-8,
                    weight_decay=0.0,
                )
            )
        if embed_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=embed_params,
                    lr=embedding_lr,
                    betas=adam_betas,
                    eps=1e-8,
                    weight_decay=0.0,
                )
            )
        if rest_params:
            param_groups.append(
                dict(
                    kind="adamw",
                    params=rest_params,
                    lr=matrix_lr,
                    betas=adam_betas,
                    eps=1e-8,
                    weight_decay=weight_decay,
                )
            )
        optimizer = torch.optim.AdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def num_scaling_params(self):
        total = sum(p.numel() for p in self.parameters())
        wte = 0
        if hasattr(self.model, "model") and hasattr(self.model.model, "embed_tokens"):
            wte = self.model.model.embed_tokens.weight.numel()
        lm_head = 0
        if hasattr(self.model, "lm_head"):
            lm_head = self.model.lm_head.weight.numel()
        transformer_matrices = max(total - wte - lm_head, 0)
        return {
            "wte": wte,
            "value_embeds": 0,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": 0,
            "total": total,
        }

    def estimate_flops(self):
        # Generic training FLOPs/token approximation.
        return 6 * sum(p.numel() for p in self.parameters())

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
