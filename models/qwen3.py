import json
import re
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


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

    cos_k = cos_cache[:kv_len]
    sin_k = sin_cache[:kv_len]

    # With KV cache kv_len - q_len = kv_len - 1 which is correct index for q
    # Without KV cache (or prefill) it is equal to 0 which is also correct
    cos_q = cos_cache[kv_len - q_len : kv_len]
    sin_q = sin_cache[kv_len - q_len : kv_len]

    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Expand cos/sin to full head_dim
    cos_q_full = torch.cat((cos_q, cos_q), dim=-1).unsqueeze(0).unsqueeze(2)  # [1, q_len, 1, head_dim]
    sin_q_full = torch.cat((sin_q, sin_q), dim=-1).unsqueeze(0).unsqueeze(2)
    cos_k_full = torch.cat((cos_k, cos_k), dim=-1).unsqueeze(0).unsqueeze(2)  # [1, kv_len, 1, head_dim]
    sin_k_full = torch.cat((sin_k, sin_k), dim=-1).unsqueeze(0).unsqueeze(2)

    q_rot = q * cos_q_full + rotate_half(q) * sin_q_full
    k_rot = k * cos_k_full + rotate_half(k) * sin_k_full

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

        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


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
        x = x + self.self_attn(self.input_layernorm(x), is_causal, use_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(config, i) for i in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x, is_causal=True, use_cache=True):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, is_causal=is_causal, use_cache=use_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_kv_cache()  # type: ignore
        self.current_pos = 0

    def load_model(self, dir_name: str):
        qwen3_weights = load_file(f"{dir_name}/model.safetensors")

        def transform_key(key):
            key = re.sub(r"^model\.", "", key)
            key = key.replace("embed_tokens", "embeddings")
            if "norm" in key:
                key = key.replace("weight", "weights")
            return key

        new_state_dict = {transform_key(k): v for k, v in qwen3_weights.items()}
        self.load_state_dict(new_state_dict)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    assert logits.dim() == 2  # batch size, vocab size
    top_k = min(top_k, logits.size(-1))

    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits


@torch.no_grad()
def generate(
    model: Qwen3,
    model_inputs,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    use_cache=True,
):
    model.eval()

    print("=" * 40)
    print(model.__class__.__name__)
    print(f"Using KV cache? {use_cache}")
    print("=" * 40)

    start_event = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event_event = torch.cuda.Event(enable_timing=True)

    if use_cache:
        model.reset_kv_cache()

    generated_ids = model_inputs.clone()
    start_event.record()
    # Prompt processing and KV cache fill
    outputs = model(model_inputs, True, use_cache)
    next_token_logits = outputs[:, -1, :]
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    if do_sample:
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
    else:
        next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    first_token_event.record()

    for _ in range(max_new_tokens):
        if use_cache:
            outputs = model(next_token, False, use_cache)
        else:
            outputs = model(generated_ids, True, use_cache)
        next_token_logits = outputs[:, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        if do_sample:
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == model.config.eos_token_id:
            break
    end_event_event.record()
    torch.cuda.synchronize()

    ttft_ms = start_event.elapsed_time(first_token_event)
    total_time_ms = start_event.elapsed_time(end_event_event)
    num_gen_tokens = len(generated_ids[0] - model_inputs.shape[1])

    if num_gen_tokens > 1 and total_time_ms > ttft_ms:
        generation_time_s = (total_time_ms - ttft_ms) / 1000.0
        num_subsequent_tokens = num_gen_tokens - 1
        tps = num_subsequent_tokens / generation_time_s
    else:
        tps = 0.0

    print("=" * 40)
    print(
        f"    Time to first tokens: {ttft_ms:.4f} ms\n    Tokens per sencond: {tps:.4f}\n    Number of tokens: {num_gen_tokens}"
    )
    print("=" * 40)
    return generated_ids
