## Department of Poorly Optimized GPU Code presents...
# Poorly optimized inference server

## Stats:
All experiments were conducted on generation of 300 next tokens.
Since CUDA is ASYNC we cannot use python `time.time` utility, beacuse it will only measure the time oevrhead to lunch the kernels. Insterad we use `torch.cuda.Event` so that we correctly measure actual time that it takes.

Also (which is not done) CUDA needs to be initialized, so first we need to run some data through model and at the end run `torch.cuda.syncronize()` and then start the measurements. 
- Huggingface implementation 

- **Navive implementation of Qwen3 0.6B model:**
```
========================================
    Time to first tokens: 49.2472 ms
    Tokens per second: 3.6248
========================================
```

- **With KV caching**
```
========================================
    Time to first tokens: 53.4047 ms
    Tokens per sencond: 24.0186
========================================
```

- **torch.compile**
```
========================================
    Time to first tokens: 56.3527 ms
    Tokens per second: 57.3766
========================================
```