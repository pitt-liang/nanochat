"""
Native Qwen3 model implementation in nanochat style.

This file intentionally does not reuse transformers' model blocks/classes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn


@dataclass
class Qwen3Config:
    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    hidden_act: str = "silu"
    max_position_embeddings: int = 40960
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    rope_scaling: dict | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    use_sliding_window: bool = False
    sliding_window: int | None = None
    max_window_layers: int = 28
    layer_types: tuple[str, ...] | None = None
    tie_word_embeddings: bool = True
    bos_token_id: int = 151643
    eos_token_id: int = 151645
    pad_token_id: int | None = None

    def __post_init__(self):
        if self.rope_scaling is not None:
            raise NotImplementedError("Qwen3 rope_scaling is not implemented in Phase C1")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim != self.hidden_size // self.num_attention_heads:
            raise ValueError("head_dim must equal hidden_size // num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.layer_types is None:
            layer_types = []
            for i in range(self.num_hidden_layers):
                if self.use_sliding_window and self.sliding_window is not None and i >= self.max_window_layers:
                    layer_types.append("sliding_attention")
                else:
                    layer_types.append("full_attention")
            self.layer_types = tuple(layer_types)
        else:
            if len(self.layer_types) != self.num_hidden_layers:
                raise ValueError("layer_types length must match num_hidden_layers")
        if "sliding_attention" in self.layer_types and self.sliding_window is None:
            raise ValueError("sliding_attention requires sliding_window to be set")

    @property
    def n_layer(self):
        return self.num_hidden_layers

    @property
    def n_head(self):
        return self.num_attention_heads

    @property
    def n_kv_head(self):
        return self.num_key_value_heads

    @property
    def n_embd(self):
        return self.hidden_size

    @property
    def sequence_len(self):
        return self.max_position_embeddings


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _get_activation_fn(name: str):
    name = name.lower()
    if name == "silu":
        return F.silu
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation for Qwen3: {name}")


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    d = q.shape[-1] // 2
    q1, q2 = q[..., :d], q[..., d:]
    k1, k2 = k[..., :d], k[..., d:]
    q_embed = torch.cat((q1 * cos + q2 * sin, q1 * (-sin) + q2 * cos), dim=-1)
    k_embed = torch.cat((k1 * cos + k2 * sin, k1 * (-sin) + k2 * cos), dim=-1)
    return q_embed, k_embed


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = _get_activation_fn(config.hidden_act)

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)

        if config.layer_types[layer_idx] == "sliding_attention":
            self.window_size = (config.sliding_window, 0)
        else:
            self.window_size = (-1, 0)

    def forward(self, hidden_states, cos_sin, kv_cache):
        B, T, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, T, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, T, self.num_key_value_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = cos_sin
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # FA3 requires fp16/bf16/fp8 inputs on CUDA.
        attn_input_dtype = q.dtype
        if q.device.type == "cuda" and q.dtype not in (torch.float16, torch.bfloat16):
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        if kv_cache is None:
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=self.window_size)
        else:
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=self.window_size,
            )
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        if y.dtype != attn_input_dtype:
            y = y.to(attn_input_dtype)
        y = y.contiguous().view(B, T, -1)
        y = self.o_proj(y)
        if self.training and self.attention_dropout > 0:
            y = F.dropout(y, p=self.attention_dropout, training=True)
        return y


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, cos_sin, kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cos_sin, kv_cache)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.max_seq_len = config.max_position_embeddings
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.register_buffer("cos", torch.empty(0), persistent=False)
        self.register_buffer("sin", torch.empty(0), persistent=False)

    @torch.no_grad()
    def _refresh_rope_cache(self, device=None):
        if device is None:
            device = self.model.embed_tokens.weight.device
        channel_range = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (self.config.rope_theta ** (channel_range / self.config.head_dim))
        positions = torch.arange(self.max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)
        cache_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        cos = freqs.cos().to(cache_dtype)[None, :, None, :]
        sin = freqs.sin().to(cache_dtype)[None, :, None, :]
        self.cos = cos
        self.sin = sin

    def _ensure_rope_cache(self, device):
        if self.cos.numel() == 0 or self.cos.device != device or self.cos.size(1) < self.max_seq_len:
            self._refresh_rope_cache(device=device)

    @torch.no_grad()
    def init_weights(self):
        std = self.config.initializer_range
        torch.nn.init.normal_(self.model.embed_tokens.weight, mean=0.0, std=std)
        if self.model.embed_tokens.padding_idx is not None:
            self.model.embed_tokens.weight[self.model.embed_tokens.padding_idx].zero_()

        for layer in self.model.layers:
            for linear in (
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer.mlp.gate_proj,
                layer.mlp.up_proj,
                layer.mlp.down_proj,
            ):
                torch.nn.init.normal_(linear.weight, mean=0.0, std=std)
                if linear.bias is not None:
                    torch.nn.init.zeros_(linear.bias)
            layer.self_attn.q_norm.weight.fill_(1.0)
            layer.self_attn.k_norm.weight.fill_(1.0)
            layer.input_layernorm.weight.fill_(1.0)
            layer.post_attention_layernorm.weight.fill_(1.0)

        self.model.norm.weight.fill_(1.0)
        if not self.config.tie_word_embeddings:
            torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=std)
        self._refresh_rope_cache()

    def get_device(self):
        return self.model.embed_tokens.weight.device

    def _split_optimizer_params(self):
        unique_named = []
        seen = set()
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in seen:
                continue
            seen.add(id(param))
            unique_named.append((name, param))

        embed_params = [p for n, p in unique_named if n.startswith("model.embed_tokens.")]
        lm_head_params = [p for n, p in unique_named if n.startswith("lm_head.")]
        embed_ids = {id(p) for p in embed_params}
        lm_head_ids = {id(p) for p in lm_head_params}
        rest_params = [p for _, p in unique_named if id(p) not in embed_ids and id(p) not in lm_head_ids]
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
        wte = self.model.embed_tokens.weight.numel()
        lm_head = 0
        if self.lm_head.weight is not self.model.embed_tokens.weight:
            lm_head = self.lm_head.weight.numel()
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
        return 6 * sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()
        device = idx.device
        self._ensure_rope_cache(device)
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        if T0 + T > self.max_seq_len:
            raise ValueError(
                f"Sequence position exceeds max_position_embeddings: {T0 + T} > {self.max_seq_len}"
            )
        cos = self.cos[:, T0:T0 + T]
        sin = self.sin[:, T0:T0 + T]
        # Convert cache dtype to hidden dtype if autocast or params changed it.
        x = self.model.embed_tokens(idx)
        cos, sin = cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

        for layer in self.model.layers:
            x = layer(x, (cos, sin), kv_cache)
        x = self.model.norm(x)

        logits = self.lm_head(x)
        logits = logits.float()
        if targets is None:
            return logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            yield next_ids.item()
