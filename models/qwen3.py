import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

HEAD_DIM = 128
HIDDEN_SIZE = 1024
NUM_HIDDEN_LAYERS = 28
NUM_ATTENTION_HEADS = 16
INTERMIDIET_SIZE = 3072
NUM_KEY_VALUE_HEADS = 8
VOCAB_SIZE = 151936
RMS_NORM_EPS = 1e-06
MAX_SEQ_LEN = 32768
PAD_TOKEN_ID = 151643


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
    def __init__(self, dim, eps=RMS_NORM_EPS) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        variance = x.pow(2).mean(-1, keepdim=True)
        norm = torch.rsqrt(variance + self.eps)
        normalized_x = x * norm * self.weights
        return normalized_x


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, layer_idx) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = HEAD_DIM

        self.q_proj = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.register_buffer(
            "cos_cache",
            build_rope_cache(MAX_SEQ_LEN, self.head_dim)[0],
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            build_rope_cache(MAX_SEQ_LEN, self.head_dim)[1],
            persistent=False,
        )

        self.cache_k = None
        self.cache_v = None

    def reset_kv_cache(self):
        self.cache_k, self.cache_v = None, None

    def forward(self, x: torch.Tensor, mask=None, use_cache=True):
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

        # # there is something wrong with my implementation and i do not know what
        # # my code did not produced the same output as the HF implementation
        # attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5)

        # if mask is not None:
        #     attn = attn + mask
        # attn = F.softmax(attn, dim=-1)
        # out = torch.matmul(attn, v)
        # out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        # if self.layer_idx == 0:
        #     print("v: ", v[0, 0, :, 0])
        #     print("k: ", k[0, 0, :, 0])
        #     print("q: ", q[0, 0, :, 0])
        # if self.layer_idx == 0:
        #     print(f"Mask? -> {mask is not None}")
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            # attn_mask=mask,
            dropout_p=0.0,
            is_causal=(mask is None),  # (mask is not None)
        )
        # if self.layer_idx == 0:
        #     print("out: ", out[0, 0, :, 0])
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_idx,
        dim=HIDDEN_SIZE,
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        hidden_dim=INTERMIDIET_SIZE,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(dim, n_heads, n_kv_heads, layer_idx)
        self.mlp = MLP(dim, hidden_dim)
        self.input_layernorm = RMSNorm(dim)
        self.post_attention_layernorm = RMSNorm(dim)

    def forward(self, x, mask=None, use_cache=True):
        x = x + self.self_attn(self.input_layernorm(x), mask, use_cache)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, dim=HIDDEN_SIZE, n_layers=NUM_HIDDEN_LAYERS):
        super().__init__()
        self.pad_token_id = PAD_TOKEN_ID
        self.embeddings = nn.Embedding(vocab_size, dim, self.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock(i) for i in range(n_layers))
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, mask=None, use_cache=True):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask=mask, use_cache=use_cache)
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
    attention_mask,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.9,
    do_sample=True,
    use_cache=True,
):
    model.eval()

    print("="*40)
    print(model.__class__.__name__)
    print(f"Using KV cache? {use_cache}")
    print("="*40)

    start_event = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event_event = torch.cuda.Event(enable_timing=True)

    if use_cache:
        model.reset_kv_cache()

    generated_ids = model_inputs.clone()
    start_event.record()
    # Prompt processing and KV cache fill
    outputs = model(model_inputs, attention_mask, use_cache)
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
            outputs = model(next_token, 123, use_cache)
        else:
            outputs = model(generated_ids, None, use_cache)
        next_token_logits = outputs[:, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        if do_sample:
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        if next_token.item() == 151645:
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
    print(f"    Time to first tokens: {ttft_ms:.4f} ms\n    Tokens per sencond: {tps:.4f}\n    Number of tokens: {num_gen_tokens}")
    print("=" * 40)
    return generated_ids
