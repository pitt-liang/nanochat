"""
Factory utilities for building models/tokenizers from explicit specs.
"""

from __future__ import annotations

import torch

from nanochat.model_adapter import HFBackedCausalLMAdapter, Qwen3HFAdapter
from nanochat.tokenizer import TransformersTokenizer, get_tokenizer


def _parse_torch_dtype(dtype_value):
    if dtype_value is None:
        return None
    if isinstance(dtype_value, torch.dtype):
        return dtype_value
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_value not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_value}")
    return mapping[dtype_value]


def build_model_from_spec(model_spec: dict, device, phase: str = "eval"):
    """
    Build a model adapter from a structured model spec.
    """
    family = model_spec.get("family")
    model_id = model_spec.get("model_id")
    dtype = _parse_torch_dtype(model_spec.get("torch_dtype"))
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    if family == "qwen3_hf":
        model = Qwen3HFAdapter.from_pretrained(model_id=model_id or "Qwen/Qwen3-0.6B", **model_kwargs)
    elif family == "hf_causal_lm":
        if not model_id:
            raise ValueError("hf_causal_lm requires model_spec.model_id")
        model = HFBackedCausalLMAdapter.from_pretrained(model_id=model_id, **model_kwargs)
    else:
        raise ValueError(f"Unsupported model family in model_spec: {family!r}")

    model.to(device)
    if phase == "eval":
        model.eval()
    elif phase == "train":
        model.train()
    else:
        raise ValueError(f"Invalid phase: {phase}")
    return model


def build_tokenizer_from_spec(tokenizer_spec: dict | None):
    """
    Build tokenizer from tokenizer spec.
    """
    if tokenizer_spec is None:
        return get_tokenizer()

    family = tokenizer_spec.get("family")
    if family == "hf_auto":
        tokenizer_id = tokenizer_spec.get("tokenizer_id")
        if not tokenizer_id:
            raise ValueError("hf_auto requires tokenizer_spec.tokenizer_id")
        return TransformersTokenizer.from_pretrained(tokenizer_id)
    if family == "nanochat_rustbpe":
        return get_tokenizer()
    raise ValueError(f"Unsupported tokenizer family in tokenizer_spec: {family!r}")
