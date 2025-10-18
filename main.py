import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer, Qwen3Model

from generate import generate
from models.qwen3 import Qwen3, Qwen3Config

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Target model
    model_config = Qwen3Config.load("./Qwen3-1.7B")
    model = Qwen3(model_config)
    model.load_model("./Qwen3-1.7B")
    tokenizer = AutoTokenizer.from_pretrained("./Qwen3-1.7B")
    model = model.to(device)

    # Draft model
    draft_model_config = Qwen3Config.load("./Qwen3-0.6B")
    draft_model = Qwen3(draft_model_config)
    draft_model.load_model("./Qwen3-0.6B")
    draft_model = draft_model.to(device)

    messages = [
        {"role": "user", "content": "Can you explain GPU memory hierarchy?"},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    tokenizer_out = tokenizer([text], return_tensors="pt").to(device)
    model_inputs = tokenizer_out["input_ids"]
    attention_mask_float = tokenizer_out["attention_mask"].to(torch.float)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    with torch.no_grad():
        generated_ids = generate(
            model,
            model_inputs,
            draft_model=draft_model,
            k_speculative=5,
            max_new_tokens=50,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            do_sample=False,
            use_cache=True,
        )

    # prof.export_chrome_trace("trace.json")
    generated_text = tokenizer.decode(generated_ids[0])
    print(generated_text)
