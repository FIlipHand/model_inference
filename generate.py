from typing import List, Optional

import torch
import torch.nn.functional as F

from model_utils import KVCache
from models.qwen3 import Qwen3


def top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 0.0, filter_value: float = -float("Inf")):
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


def get_tokens_from_logits(logits: torch.Tensor, do_sample: bool = False) -> torch.Tensor:
    if do_sample:
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token

def apply_templerature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature > 0:
        return logits / temperature
    else:
        return logits

def truncate_and_update_cache(working_model: Qwen3, n_accepted: int, k_speculative: int, rejected_token: torch.Tensor):
    num_to_truncate = k_speculative - n_accepted
    if num_to_truncate <= 0:
        return

    current_cache = working_model.get_kv_cache()
    truncated_cache: List[KVCache] = []

    for layer_cache in current_cache:
        original_seq_len = layer_cache.keys.shape[1]  # (batch, seq_len, n_heads, head_dim)
        new_seq_len = original_seq_len - num_to_truncate
        truncated_k = layer_cache.keys[:, :new_seq_len, :, :]
        truncated_v = layer_cache.values[:, :new_seq_len, :, :]

        truncated_cache.append(KVCache(keys=truncated_k, values=truncated_v))

    # Update model's cache with truncated cache
    working_model.update_kv_cache(truncated_cache)
    # Pass new tokens trough model to update KV cache at those positions
    # if rejected_token is not None:
    #     working_model(rejected_token, True, True)


@torch.no_grad()
def generate(
    model: Qwen3,
    model_inputs: torch.Tensor,
    draft_model: Optional[Qwen3],
    k_speculative: Optional[int],
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = True,
    use_cache: bool = True,
):
    device = next(model.parameters()).device
    model.eval()
    use_speculative_decode = draft_model is not None

    if use_speculative_decode:
        draft_model.to(device)

    print("=" * 40)
    print(model.__class__.__name__)
    print(f"Using KV cache? {use_cache}")
    print(f"Using speculative decode? {use_speculative_decode}")
    print("=" * 40)

    start_event = torch.cuda.Event(enable_timing=True)
    first_token_event = torch.cuda.Event(enable_timing=True)
    end_event_event = torch.cuda.Event(enable_timing=True)

    if use_cache:
        model.reset_kv_cache()
        if use_speculative_decode:
            draft_model.reset_kv_cache()

    model.reset_kv_cache()
    if use_speculative_decode:
        draft_model.reset_kv_cache()

    generated_ids = model_inputs.clone()
    start_event.record()

    # Prompt processing and KV cache fill + bonus first token
    outputs = model(model_inputs, True, use_cache).logits
    next_token_logits = outputs[:, -1, :]
    if use_speculative_decode:
        draft_model(model_inputs, True, use_cache)
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    next_token = get_tokens_from_logits(filtered_logits, do_sample)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)
    # next_token can be used as an input to draft_model since we know it is already correct

    # # TODO ugly fix #1 -> pass next_token through target_model to fill kv cache
    # model(next_token, False, True)

    first_token_event.record()

    if use_speculative_decode:
        assert k_speculative is not None
        n_generated = 0
        while n_generated < max_new_tokens:
            draft_tokens: List[torch.Tensor] = []
            draft_probs: List[torch.Tensor] = []
            with torch.no_grad():
                print(f"Draft cache length: {draft_model.get_kv_cache()[0].keys.shape[1]}")
                print(f"Target cache length: {model.get_kv_cache()[0].keys.shape[1]}")
                # print(f"Generated IDs length: {generated_ids.shape[1]}")
                # Autoregressivly geenrate k_speculative tokens
                for i in range(k_speculative):
                    # is_causal=False because next_token is 1D and it shouldn't mask itself
                    logits = draft_model(next_token, False, use_cache).logits[:, -1, :]
                    if temperature > 0:
                        logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    draft_tokens.append(next_token)
                    draft_probs.append(probs)
                print([i.item() for i in draft_tokens])

            with torch.no_grad():
                target_logits = model(torch.cat([next_token] + draft_tokens, dim=-1), True, use_cache).logits
                verify_logits = target_logits[:, -(k_speculative + 1) :, :] 

            # Verification loop
            n_accepted = 0
            rejected_token = None
            for i in range(k_speculative):
                print(f"Verifing {i}")
                if temperature > 0:
                    scaled_logits = verify_logits[:, i, :] / temperature
                else:
                    scaled_logits = verify_logits[:, i, :]
                target_probs = F.softmax(scaled_logits, dim=-1)
                draft_token = draft_tokens[i]

                # Get the probability of the drafted token from each model
                draft_p = draft_probs[i].gather(-1, draft_token)
                target_p = target_probs.gather(-1, draft_token)

                print(draft_probs[i].max())
                accept_prob = torch.min(torch.ones_like(target_p), target_p / (draft_p + 1e-9))

                if torch.rand_like(target_p) < accept_prob:
                    # If we accept
                    generated_ids = torch.cat([generated_ids, draft_token], dim=-1)
                    n_accepted += 1
                    n_generated += 1
                    if n_generated >= max_new_tokens:
                        break
                else:
                    # If we do not accept
                    adjust_dist = torch.clamp(target_probs - draft_probs[i], min=0)
                    denominator = adjust_dist.sum(dim=-1, keepdim=True)
                    adjust_probs = adjust_dist / (denominator + 1e-9)

                    rejected_token = torch.multinomial(adjust_probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, rejected_token], dim=-1)
                    n_generated += 1
                    break

            if rejected_token is not None:
                # After verification loop, update model's kv cache if needed
                truncate_and_update_cache(draft_model, n_accepted, k_speculative, rejected_token)
                truncate_and_update_cache(model, n_accepted, k_speculative, rejected_token)
                next_token = rejected_token
            elif n_accepted == k_speculative and n_generated < max_new_tokens:
                # If all drafted tokens were accepted, sample one more bonus token
                target_probs = F.softmax(verify_logits[:, -1, :] / temperature, dim=-1)
                bonus_token = torch.multinomial(target_probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, bonus_token], dim=-1)
                n_generated += 1
                next_token = bonus_token

                if n_generated >= max_new_tokens:
                    break

                # # Add bonus token to KV cache
                # model(bonus_token, True, True)
                # draft_model(bonus_token, True, True)
    else:
        for _ in range(max_new_tokens):
            if use_cache:
                outputs = model(next_token, False, use_cache).logits
            else:
                outputs = model(generated_ids, True, use_cache).logits
            next_token_logits = outputs[:, -1, :]

            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = get_tokens_from_logits(filtered_logits, do_sample)
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
