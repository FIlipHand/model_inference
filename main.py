import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from models.qwen3 import Qwen3


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
def generate(model, model_inputs, attention_mask, max_new_tokens=100, temperature=1.0, top_k=50, top_p=0.9):
    model.eval()
    
    generated_ids = model_inputs
    
    for _ in range(max_new_tokens):
        outputs = model(generated_ids, attention_mask)
        next_token_logits = outputs[:, -1, :]

        if temperature > 0:
            next_token_logits = next_token_logits / temperature

        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)

        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=-1
        )
        
        if  next_token.item() == 151645:
            break
    
    return generated_ids

if __name__ == "__main__":
    model = Qwen3()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_model("./Qwen3-0.6B/model.safetensors")

    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B")

    messages = [
        {"role": "user", "content": "What is SM in nvidia gpus?"},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    tokenizer_out = tokenizer([text], return_tensors="pt").to(device)
    model_inputs = tokenizer_out["input_ids"]
    attention_mask_float = tokenizer_out["attention_mask"].to(torch.float)

    do_sample = True
    temperature = 0.6
    top_k = 20
    top_p = 0.95
    max_new_tokens = 100

    generated_ids = generate(
        model,
        model_inputs,
        attention_mask_float,
        max_new_tokens=100,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)
