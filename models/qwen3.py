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
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim)
    )

    t = torch.arange(max_seq_len, device=inv_freq.device).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(q, k, cos_cache, sin_cache):  # Renamed to avoid confusion
    seq_len = q.shape[1]

    cos = cos_cache[:seq_len, :]
    sin = sin_cache[:seq_len, :]

    # To get the same implementation as HF is using... just to be sure...
    cos_full = torch.cat((cos, cos), dim=-1)
    sin_full = torch.cat((sin, sin), dim=-1)

    # Broadcast across batch and heads: [1, seq_len, 1, dim]
    cos_broadcast = cos_full.unsqueeze(0).unsqueeze(2)
    sin_broadcast = sin_full.unsqueeze(0).unsqueeze(2)

    # Yoink implementation of rotate_half from HF
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    # Apply the rotation in the HF style
    q_rot = (q * cos_broadcast) + (rotate_half(q) * sin_broadcast)
    k_rot = (k * cos_broadcast) + (rotate_half(k) * sin_broadcast)

    return q_rot, k_rot


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=RMS_NORM_EPS) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        norm = torch.rsqrt(variance + self.eps)
        normalized_x = x_fp32 * norm * self.weights
        return normalized_x.to(input_dtype)


class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads) -> None:
        super().__init__()
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

    def forward(self, x: torch.Tensor, mask=None):
        batch, seq_len, dim = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

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

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=(mask is not None)
        )
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
        dim=HIDDEN_SIZE,
        n_heads=NUM_ATTENTION_HEADS,
        n_kv_heads=NUM_KEY_VALUE_HEADS,
        hidden_dim=INTERMIDIET_SIZE,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(dim, n_heads, n_kv_heads)
        self.mlp = MLP(dim, hidden_dim)
        self.input_layernorm = RMSNorm(dim)
        self.post_attention_layernorm = RMSNorm(dim)

    def forward(self, x, mask=None):
        x = x + self.self_attn(self.input_layernorm(x), mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3(nn.Module):
    def __init__(
        self, vocab_size=VOCAB_SIZE, dim=HIDDEN_SIZE, n_layers=NUM_HIDDEN_LAYERS
    ) -> None:
        super().__init__()
        self.pad_token_id = PAD_TOKEN_ID
        self.embeddings = nn.Embedding(vocab_size, dim, self.pad_token_id)
        self.layers = nn.ModuleList(TransformerBlock() for _ in range(n_layers))
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, mask=None):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def load_model(self, filename: str):
        qwen3_weights = load_file(filename)

        def transform_key(key):
            key = re.sub(r"^model\.", "", key)
            key = key.replace("embed_tokens", "embeddings")
            if "norm" in key:
                key = key.replace("weight", "weights")
            return key

        new_state_dict = {transform_key(k): v for k, v in qwen3_weights.items()}
        self.load_state_dict(new_state_dict)
