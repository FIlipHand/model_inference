from typing import Optional, List

import torch
import torch.nn.functional as F

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


def get_tokens_from_logits(logits: torch.Tensor, do_sample: bool = False):
    if do_sample:
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
    else:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    return next_token


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

    # Prompt processing and KV cache fill
    outputs = model(model_inputs, True, use_cache)
    next_token_logits = outputs[:, -1, :]
    if use_speculative_decode:
        draft_model(model_inputs, True, use_cache)
    if temperature > 0:
        next_token_logits = next_token_logits / temperature
    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
    next_token = get_tokens_from_logits(filtered_logits, do_sample)
    generated_ids = torch.cat([generated_ids, next_token], dim=-1)

    first_token_event.record()

    if use_speculative_decode:
        assert k_speculative is not None
        n_generated = 0
        while n_generated < max_new_tokens:
            draft_tokens: List[torch.Tensor] = []
            draft_probs: List[torch.Tensor] = []
            current_seq = generated_ids
            with torch.no_grad():
                # Autoregressivly geenrate k_speculative tokens
                for i in range(k_speculative):
                    logits = draft_model(current_seq, True, use_cache)[:, -1, :]
                    if temperature > 0:
                        logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    draft_tokens.append(next_token)
                    draft_probs.append(probs)
                    current_seq = torch.cat([current_seq, next_token], dim=-1)
            # Concatenate draft_tokens with prioir context
            verify_seq = torch.cat([generated_ids] + draft_tokens, dim=-1)

            # Get logits from target model for verification
            with torch.no_grad():
                target_logits = model(verify_seq, True, use_cache)
                verify_logits = target_logits[:, -(k_speculative + 1) :, :]

            n_accepted = 0
            for i in range(k_speculative):
                # Use the ith logit vector for verification
                target_probs = F.softmax(verify_logits[:, i, :] / temperature, dim=-1)
                draft_token = draft_tokens[i]

                # Get the probability of the drafted token from each model
                draft_p = draft_probs[i].gather(-1, draft_token)
                target_p = target_probs.gather(-1, draft_token)

                accept_prob = torch.min(torch.ones_like(target_p), target_p / (draft_p + 1e-9))

                if torch.rand_like(target_p) < accept_prob:
                    generated_ids = torch.cat([generated_ids, draft_token], dim=-1)
                    n_accepted += 1
                    n_generated += 1
                    if n_generated >= max_new_tokens:
                        break
                else:
                    adjust_dist = torch.clamp(target_probs - draft_probs[i], min=0)
                    denominator = adjust_dist.sum(dim=-1, keepdim=True)
                    adjust_probs = adjust_dist / (denominator + 1e-9) 

                    rejected_token = torch.multinomial(adjust_probs, num_samples=1)
                    generated_ids = torch.cat([generated_ids, rejected_token], dim=-1)
                    n_generated += 1
                    break

            # If all drafted tokens were accepted, sample one more bonus token
            if n_accepted == k_speculative and n_generated < max_new_tokens:
                # Use the final logit vector for the bonus token
                target_probs = F.softmax(verify_logits[:, -1, :] / temperature, dim=-1)
                bonus_token = torch.multinomial(target_probs, num_samples=1)
                generated_ids = torch.cat([generated_ids, bonus_token], dim=-1)
                n_generated += 1
    else:
        for _ in range(max_new_tokens):
            if use_cache:
                outputs = model(next_token, False, use_cache)
            else:
                outputs = model(generated_ids, True, use_cache)
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
