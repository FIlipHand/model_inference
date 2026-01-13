import json
import os
import re
from dataclasses import dataclass, fields
from typing import List

import torch
import torch.nn as nn
from safetensors.torch import load_file

from model_utils import KVCache, ModelOutput


@dataclass
class Qwen3Config:
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int
    head_dim: int
    hidden_act: str
    hidden_size: int
    initializer_range: float
    intermediate_size: int
    max_position_embeddings: int
    max_window_layers: int
    model_type: str
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: float
    rope_theta: int
    sliding_window: bool
    tie_word_embeddings: bool
    torch_dtype: str
    transformers_version: str
    use_cache: bool
    use_sliding_window: bool
    vocab_size: int
    max_seq_len: int

    @classmethod
    def load(cls, path: str) -> "Qwen3Config":
        with open(f"{path}/config.json", "r") as f:
            data = json.load(f)

        allowed = {f.name for f in fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in allowed}

        new_cls = cls(**filtered_data, pad_token_id=151643, max_seq_len=32768)
        return new_cls


def build_rope_cache(max_seq_len, dim, base=1_000_000):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

    t = torch.arange(max_seq_len, device=inv_freq.device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(q, k, cos_cache, sin_cache):
    kv_len = k.shape[1]
    q_len = q.shape[1]

    # Convert cos/sin caches to match the dtype of q and k
    dtype = q.dtype
    cos_k = cos_cache[:kv_len].to(dtype)
    sin_k = sin_cache[:kv_len].to(dtype)

    # With KV cache kv_len - q_len = kv_len - 1 which is correct index for q
    # Without KV cache (or prefill) it is equal to 0 which is also correct
    cos_q = cos_cache[kv_len - q_len : kv_len].to(dtype)
    sin_q = sin_cache[kv_len - q_len : kv_len].to(dtype)

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Expand cos/sin to full head_dim
    cos_q_full = torch.cat((cos_q, cos_q), dim=-1).unsqueeze(0).unsqueeze(2)  # [1, q_len, 1, head_dim]
    sin_q_full = torch.cat((sin_q, sin_q), dim=-1).unsqueeze(0).unsqueeze(2)
    cos_k_full = torch.cat((cos_k, cos_k), dim=-1).unsqueeze(0).unsqueeze(2)  # [1, kv_len, 1, head_dim]
    sin_k_full = torch.cat((sin_k, sin_k), dim=-1).unsqueeze(0).unsqueeze(2)

    q_rot = (q * cos_q_full + rotate_half(q) * sin_q_full).to(dtype)
    k_rot = (k * cos_k_full + rotate_half(k) * sin_k_full).to(dtype)

    return q_rot, k_rot


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, rms_norm) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(hidden_size))
        self.eps = rms_norm

    def forward(self, x: torch.Tensor):
        variance = x.pow(2).mean(-1, keepdim=True)
        norm = torch.rsqrt(variance + self.eps)
        normalized_x = x * norm * self.weights
        return normalized_x


class Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dim = config.hidden_size
        self.attn_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        self.register_buffer(
            "cos_cache",
            build_rope_cache(config.max_seq_len, self.head_dim)[0],
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            build_rope_cache(config.max_seq_len, self.head_dim)[1],
            persistent=False,
        )

        self.cache_k = None
        self.cache_v = None

    def reset_kv_cache(self):
        self.cache_k, self.cache_v = None, None

    def forward(self, x: torch.Tensor, is_causal=True, use_cache=True):
        batch, seq_len, dim = x.shape

        q = self.q_proj(x).reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if use_cache:
            if self.cache_k is None:
                self.cache_k, self.cache_v = k, v
            else:
                self.cache_k = torch.cat([self.cache_k, k], dim=1)
                self.cache_v = torch.cat([self.cache_v, v], dim=1)  # type: ignore
            k, v = self.cache_k, self.cache_v

        q, k = apply_rope(q, k, self.cos_cache, self.sin_cache)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        # Ensure q, k, v have the same dtype before attention (match input dtype)
        target_dtype = x.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return KVCache(k, v), self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(config, layer_idx)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, is_causal=True, use_cache=True):
        past_kv, self_attn = self.self_attn(self.input_layernorm(x), is_causal, use_cache)
        x = x + self_attn
        x = x + self.mlp(self.post_attention_layernorm(x))
        return past_kv, x


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, is_causal=True, use_cache=True) -> ModelOutput:
        x = self.embeddings(x).contiguous()
        past_kv_list = []
        for layer in self.layers:
            past_kv, x = layer(x, is_causal=is_causal, use_cache=use_cache)
            past_kv_list.append(past_kv)
        x = self.norm(x)
        logits = self.lm_head(x)
        return ModelOutput(logits, past_kv_list)

    def update_kv_cache(self, new_cache: List[KVCache]):
        for idx, layer in enumerate(self.layers):
            assert isinstance(layer.self_attn, Attention)
            layer.self_attn.cache_k = new_cache[idx].keys
            layer.self_attn.cache_v = new_cache[idx].values

    def get_kv_cache(self) -> List[KVCache]:
        kv_cache_list = []
        for layer in self.layers:
            kv_cache_list.append(KVCache(layer.self_attn.cache_k, layer.self_attn.cache_v))  # type: ignore
        return kv_cache_list

    def reset_kv_cache(self):
        for layer in self.layers:
            assert isinstance(layer.self_attn, Attention)
            layer.self_attn.reset_kv_cache()
        self.current_pos = 0

    def load_model(self, dir_name: str):
        safetensor_files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if f.endswith(".safetensors")]
        if not safetensor_files:
            raise FileNotFoundError(f"No .safetensors files found in {dir_name}")

        def transform_key(key):
            key = re.sub(r"^model\.", "", key)
            key = key.replace("embed_tokens", "embeddings")
            if "norm" in key:
                key = key.replace("weight", "weights")
            return key

        new_state_dict = {}
        for file_path in safetensor_files:
            weights = load_file(file_path)
            for k, v in weights.items():
                new_state_dict[transform_key(k)] = v

        self.load_state_dict(new_state_dict)
