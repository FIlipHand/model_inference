import sys

import torch
from transformers import AutoTokenizer

from models.qwen3 import Qwen3, generate

if __name__ == "__main__":
    model = Qwen3()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_model("./Qwen3-0.6B")

    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("./Qwen3-0.6B")

    messages = [
        {"role": "user", "content": "Can you explain GPU memory hierarchy?"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    tokenizer_out = tokenizer([text], return_tensors="pt").to(device)
    model_inputs = tokenizer_out["input_ids"]
    attention_mask_float = tokenizer_out["attention_mask"].to(torch.float)
    
    generated_ids = generate(
        model,
        model_inputs,
        None,
        max_new_tokens=500,
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        do_sample=False,
        use_cache=True,
    )

    with torch.no_grad():
        generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)
