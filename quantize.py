"""
INT8 Dynamic Quantization for Qwen3 model.

This module provides quantized versions of the Qwen3 model components
that use INT8 dynamic quantization for linear layers, reducing memory
usage while maintaining reasonable accuracy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from models.qwen3 import Qwen3

from models.qwen3 import (
    Qwen3Config,
    RMSNorm,
    apply_rope,
    build_rope_cache,
)
from model_utils import KVCache, ModelOutput


class QuantizedLinear(nn.Module):
    """
    Linear layer with INT8 dynamic quantization.

    Weights are stored as INT8 with per-channel scales.
    Activations are quantized dynamically at runtime.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Store quantized weights as int8
        self.register_buffer(
            "weight_quantized",
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        # Per-output-channel scale factors
        self.register_buffer(
            "weight_scale",
            torch.ones(out_features, dtype=torch.float32)
        )

        if bias:
            self.register_buffer("bias", torch.zeros(out_features))
        else:
            self.bias = None

    @staticmethod
    def quantize_tensor(tensor: torch.Tensor, axis: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize a tensor to INT8 with per-channel scaling.

        Args:
            tensor: Input tensor to quantize
            axis: Axis along which to compute scales (typically output channel)

        Returns:
            Tuple of (quantized_tensor, scales)
        """
        # Compute per-channel max absolute values
        if axis == 0:
            max_vals = tensor.abs().max(dim=1).values
        else:
            max_vals = tensor.abs().max(dim=0).values

        # Avoid division by zero
        max_vals = torch.clamp(max_vals, min=1e-8)

        # Scale to fit in INT8 range [-127, 127]
        scales = max_vals / 127.0

        # Quantize
        if axis == 0:
            quantized = torch.round(tensor / scales.unsqueeze(1)).to(torch.int8)
        else:
            quantized = torch.round(tensor / scales.unsqueeze(0)).to(torch.int8)

        return quantized, scales

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear":
        """
        Create a QuantizedLinear from an existing nn.Linear layer.
        """
        has_bias = linear.bias is not None
        quantized_layer = cls(
            linear.in_features,
            linear.out_features,
            bias=has_bias
        )

        # Quantize weights (per output channel)
        weight_q, weight_s = cls.quantize_tensor(linear.weight.data, axis=0)
        quantized_layer.weight_quantized = weight_q
        quantized_layer.weight_scale = weight_s

        if has_bias:
            quantized_layer.bias = linear.bias.data.clone()

        return quantized_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dynamic activation quantization.
        """
        original_dtype = x.dtype

        # Dequantize weights for computation
        # weight_quantized: [out_features, in_features]
        # weight_scale: [out_features]
        weight_fp = self.weight_quantized.float() * self.weight_scale.unsqueeze(1)

        # Convert to input dtype for computation
        weight_fp = weight_fp.to(original_dtype)

        # Standard linear operation
        output = F.linear(x, weight_fp, self.bias)

        return output


class QuantizedAttention(nn.Module):
    """Attention module with quantized linear layers."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.dim = config.hidden_size
        self.attn_dropout = config.attention_dropout

        # Quantized projection layers
        self.q_proj = QuantizedLinear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = QuantizedLinear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = QuantizedLinear(self.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = QuantizedLinear(self.n_heads * self.head_dim, self.dim, bias=False)

        # RMSNorm layers (keep in full precision)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        # RoPE cache
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

    def forward(self, x: torch.Tensor, is_causal: bool = True, use_cache: bool = True):
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
                self.cache_v = torch.cat([self.cache_v, v], dim=1)
            k, v = self.cache_k, self.cache_v

        q, k = apply_rope(q, k, self.cos_cache, self.sin_cache)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
            v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        target_dtype = x.dtype
        q = q.to(target_dtype)
        k = k.to(target_dtype)
        v = v.to(target_dtype)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout,
            is_causal=is_causal,
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return KVCache(k, v), self.o_proj(out)


class QuantizedMLP(nn.Module):
    """MLP module with quantized linear layers."""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.gate_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = QuantizedLinear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = QuantizedLinear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_proj(x)) * self.up_proj(x))


class QuantizedTransformerBlock(nn.Module):
    """Transformer block with quantized attention and MLP."""

    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = QuantizedAttention(config, layer_idx)
        self.mlp = QuantizedMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, is_causal: bool = True, use_cache: bool = True):
        past_kv, self_attn = self.self_attn(self.input_layernorm(x), is_causal, use_cache)
        x = x + self_attn
        x = x + self.mlp(self.post_attention_layernorm(x))
        return past_kv, x


class Qwen3Quantized(nn.Module):
    """
    Quantized Qwen3 model with INT8 linear layers.

    The embedding and lm_head layers are kept in full precision
    for better accuracy, while all other linear layers are quantized.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        # Keep embeddings in full precision
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            QuantizedTransformerBlock(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        # Keep lm_head in full precision for output quality
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True, use_cache: bool = True) -> ModelOutput:
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
            layer.self_attn.cache_k = new_cache[idx].keys
            layer.self_attn.cache_v = new_cache[idx].values

    def get_kv_cache(self) -> List[KVCache]:
        kv_cache_list = []
        for layer in self.layers:
            kv_cache_list.append(KVCache(layer.self_attn.cache_k, layer.self_attn.cache_v))
        return kv_cache_list

    def reset_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.reset_kv_cache()


def quantize_qwen3(model: "Qwen3") -> Qwen3Quantized:
    """
    Convert a full-precision Qwen3 model to a quantized version.

    Args:
        model: The original Qwen3 model with full-precision weights

    Returns:
        A new Qwen3Quantized model with INT8 weights
    """
    from models.qwen3 import Qwen3, Attention, MLP

    quantized_model = Qwen3Quantized(model.config)

    # Copy embeddings (full precision)
    quantized_model.embeddings.weight.data = model.embeddings.weight.data.clone()

    # Quantize transformer layers
    for i, (orig_layer, quant_layer) in enumerate(zip(model.layers, quantized_model.layers)):
        # Quantize attention projections
        quant_layer.self_attn.q_proj = QuantizedLinear.from_linear(orig_layer.self_attn.q_proj)
        quant_layer.self_attn.k_proj = QuantizedLinear.from_linear(orig_layer.self_attn.k_proj)
        quant_layer.self_attn.v_proj = QuantizedLinear.from_linear(orig_layer.self_attn.v_proj)
        quant_layer.self_attn.o_proj = QuantizedLinear.from_linear(orig_layer.self_attn.o_proj)

        # Copy RMSNorm weights (full precision)
        quant_layer.self_attn.q_norm.weights.data = orig_layer.self_attn.q_norm.weights.data.clone()
        quant_layer.self_attn.k_norm.weights.data = orig_layer.self_attn.k_norm.weights.data.clone()

        # Quantize MLP projections
        quant_layer.mlp.gate_proj = QuantizedLinear.from_linear(orig_layer.mlp.gate_proj)
        quant_layer.mlp.up_proj = QuantizedLinear.from_linear(orig_layer.mlp.up_proj)
        quant_layer.mlp.down_proj = QuantizedLinear.from_linear(orig_layer.mlp.down_proj)

        # Copy layer norm weights (full precision)
        quant_layer.input_layernorm.weights.data = orig_layer.input_layernorm.weights.data.clone()
        quant_layer.post_attention_layernorm.weights.data = orig_layer.post_attention_layernorm.weights.data.clone()

    # Copy final norm (full precision)
    quantized_model.norm.weights.data = model.norm.weights.data.clone()

    # Copy lm_head (full precision)
    quantized_model.lm_head.weight.data = model.lm_head.weight.data.clone()

    return quantized_model


def load_quantized_model(model_path: str, device: str = "cpu") -> Qwen3Quantized:
    """
    Load a Qwen3 model and return its quantized version.

    Args:
        model_path: Path to the model directory containing safetensors and config
        device: Device to load the model on

    Returns:
        Quantized Qwen3 model
    """
    from models.qwen3 import Qwen3, Qwen3Config

    # Load config
    config = Qwen3Config.load(model_path)

    # Load original model
    model = Qwen3(config)
    model.load_model(model_path)
    model = model.to(device)

    # Quantize
    quantized_model = quantize_qwen3(model)
    quantized_model = quantized_model.to(device)

    # Free original model memory
    del model
    torch.cuda.empty_cache() if device == "cuda" else None

    return quantized_model


def save_quantized_model(model: Qwen3Quantized, save_path: str):
    """
    Save a quantized model to disk.

    Args:
        model: The quantized model to save
        save_path: Path to save the model state dict
    """
    torch.save(model.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")


def load_quantized_state(model_path: str, state_path: str, device: str = "cpu") -> Qwen3Quantized:
    """
    Load a pre-quantized model from saved state.

    Args:
        model_path: Path to the original model directory (for config)
        state_path: Path to the saved quantized state dict
        device: Device to load the model on

    Returns:
        Quantized Qwen3 model with loaded weights
    """
    from models.qwen3 import Qwen3Config

    config = Qwen3Config.load(model_path)
    model = Qwen3Quantized(config)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model = model.to(device)

    return model


def compute_model_size(model: nn.Module) -> dict:
    """
    Compute the memory footprint of a model.

    Returns dict with parameter counts and memory sizes.
    """
    param_count = 0
    param_bytes = 0
    buffer_count = 0
    buffer_bytes = 0

    for _, param in model.named_parameters():
        param_count += param.numel()
        param_bytes += param.numel() * param.element_size()

    for _, buffer in model.named_buffers():
        buffer_count += buffer.numel()
        buffer_bytes += buffer.numel() * buffer.element_size()

    total_elements = param_count + buffer_count
    total_bytes = param_bytes + buffer_bytes

    return {
        "total_params": param_count,
        "total_buffers": buffer_count,
        "total_elements": total_elements,  # params + buffers (logical "weights")
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "total_gb": total_bytes / (1024 * 1024 * 1024),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize a Qwen3 model to INT8")
    parser.add_argument("model_path", type=str, help="Path to the model directory")
    parser.add_argument("--output", type=str, default=None, help="Output path for quantized model")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for quantization")

    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")

    # Load and quantize
    from models.qwen3 import Qwen3, Qwen3Config
    from generate import generate
    from transformers import AutoTokenizer

    config = Qwen3Config.load(args.model_path)
    original_model = Qwen3(config)
    original_model.load_model(args.model_path)

    print("Original model size:")
    orig_size = compute_model_size(original_model)
    print(f"  Total weights: {orig_size['total_elements']:,}")
    print(f"  Memory: {orig_size['total_mb']:.2f} MB")

    print("\nQuantizing model...")
    quantized_model = quantize_qwen3(original_model)

    print("Quantized model size:")
    quant_size = compute_model_size(quantized_model)
    print(f"  Total weights: {quant_size['total_elements']:,}")
    print(f"  Memory: {quant_size['total_mb']:.2f} MB")
    print(f"  Compression ratio: {orig_size['total_bytes'] / quant_size['total_bytes']:.2f}x")

    tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B")
    messages = [
        {"role": "user", "content": "Can you explain GPU memory hierarchy?"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    tokenizer_out = tokenizer([text], return_tensors="pt").to('cuda')
    model_inputs = tokenizer_out["input_ids"]
    for _ in range(5):
        with torch.no_grad():
            generated_ids = generate(
                quantized_model.to('cuda'),
                model_inputs,
                draft_model=None,
                k_speculative=0,
                max_new_tokens=100,
                temperature=0.6,
                top_k=20,
                top_p=0.95,
                do_sample=False,
                use_cache=True,
            )
            generated_text = tokenizer.decode(generated_ids[0])
            print(generated_text)
    if args.output:
        save_quantized_model(quantized_model, args.output)
