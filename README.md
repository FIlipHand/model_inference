## Department of Poorly Optimized GPU Code presents...
# Poorly optimized inference server

## Stats:
All experiments were conducted on generation of 300 next tokens.
- Huggingface implementation 

- **Navive implementation of Qwen3 0.6B model:**
```
========================================
    Time to first tokens: 191.1921 ms
    Tokens per sencond: 10.3811
    VRAM used:  12018MiB / 12288MiB
========================================
```

- **With KV caching**
```
========================================
    Time to first tokens: 182.7574 ms
    Tokens per sencond: 21.9977
    VRAM used: 4434MiB / 12288MiB
========================================
```

- **torch.compile**

I have to be doing something wrong since tps are crazy low.
I am aware that the first run through the compiled model is slower because it's doing the graph capture and optimization in the background.
```
========================================
    Time to first tokens: 21328.3926 ms
    Tokens per sencond: 6.8753
    Number of tokens: 516
========================================
```