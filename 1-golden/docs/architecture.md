# Ornith-Flight Architecture

## Overview

A pure C inference engine for Ornith 35B MoE that streams routed experts from SSD, keeping only non-routed weights resident. Inspired by Colibri for GLM-5.2.

## Ornith 35B MoE Specs

| Property | Value |
|---|---|
| Base model | Qwen 3.5 MoE |
| Total params | 35B |
| Active per token | ~3B |
| Architecture | MoE (256 experts, 8 active + 1 shared per token) |
| Attention | Hybrid: full attention + Gated DeltaNet (linear attention) |
| Context | 262K tokens |
| Non-routed weights | embeddings, norms, QKVO, shared expert, router, LM head |
| Routed experts | 256 experts/layer, ~62 MB each at int4 |

## GPU Abstraction Layer

```c
// gpu.h — thin abstraction, all backends implement these

void   gpu_init(void);
void   gpu_load_kernels(void);
void   gpu_matmul(float* out, void* weights, float* act, int M, int N, int K, int quant_bits);
void   gpu_attention(float* out, float* q, float* k, float* v, int seq_len, int head_dim);
void   gpu_rmsnorm(float* out, float* x, float* weight, int dim);
void   gpu_router(float* gates, float* x, float* router_weight, int n_experts);
void   gpu_expert_mlp(float* out, float* x, void* gate_w, void* up_w, void* down_w);
void   gpu_silu_mul(float* out, float* gate, float* up, int dim);
void   gpu_rope(float* q, float* k, int seq_len, int head_dim, float base);
void   gpu_deltanet(float* out, float* q, float* k, float* v, float* state, int dim);
void   gpu_copy_to_device(void* dst, void* src, size_t bytes);
void   gpu_sync(void);
void   gpu_destroy(void);
```

Metal backend: `gpu_metal.m` — implements each via Metal Performance Shaders or custom MSL.
CUDA backend (future): `gpu_cuda.cu` — implements each via cuBLAS + custom CUDA kernels.

## Memory Hierarchy

```
T0 - RESIDENT (pinned in RAM/VRAM):
  embed_tokens (int8):       ~0.2 GB
  layer_norms (int8):        ~0.1 GB
  attention QKVO (int8):     ~0.5 GB
  shared_expert (int8):      ~0.3 GB
  router_gate (int8):        ~0.1 GB
  lm_head (int8):            ~0.3 GB
  Total:                     ~1.5 GB

T1 - EXPERT LRU CACHE (in RAM):
  Recently-used routed experts
  Configurable: 4 GB on Mac, 6 GB on PC
  ~62 MB per expert at int4
  ~64-96 expert slots

T2 - HOT STORE (pinned in RAM):
  Frequently-routed experts
  Identified via imatrix-style profiling
  ~1 GB, pinned, never evicted
  ~16 expert slots

T3 - DISK (SSD, mmap'd GGUF):
  All routed experts
  ~16 GB at int4
  256 experts/layer × ~32 layers
```

## Async I/O Engine

Background thread with lock-free work queue:

1. Router finishes layer N → pushes expert load requests for layer N+1
2. I/O thread reads from mmap'd GGUF into pre-allocated buffers
3. When GPU finishes layer N, experts for N+1 are already in cache or loading
4. Double-buffered: load buffer A while GPU reads buffer B

```
Timeline (decode step):
  [GPU: layer 0] → [GPU: layer 1] → [GPU: layer 2] → ...
       ↓ prefetch         ↓ prefetch         ↓ prefetch
  [I/O: layer 1]    [I/O: layer 2]    [I/O: layer 3]
```

## Quantization Strategy

| Component | Bits | Reason |
|---|---|---|
| Routed experts | int4 | Bulk of params, sparsity tolerates lower precision |
| Non-routed weights | int8 | Every token uses these, precision matters |
| Router gate | fp16 | Routing accuracy is critical |
| KV cache | fp16 (or int8) | Compressed via DeltaNet linear attention |
| Activations | fp16 | GPU native, no dequant overhead |

## Inference Loop

```c
// Prefill (prompt processing)
for each layer:
    load non-routed weights (already resident)
    for each expert selected by router:
        if expert not in cache:
            async_load_from_disk(expert_id)
    compute attention
    compute shared expert
    compute routed experts
    update KV cache

// Decode (token generation)
for each token:
    for each layer:
        wait_for_experts(layer)     // I/O wait if miss
        compute attention
        compute shared expert
        compute routed experts
        prefetch_next_layer()
    sample next token
```

## Performance Model

```
TPS = 1 / (compute_time + io_overhead)

compute_time ≈ active_params / gpu_bandwidth
             ≈ 3 GB / 100 GB/s = 0.03 sec (M2)

io_overhead = (1 - hit_rate) × experts_per_token × expert_size / ssd_bandwidth
            = (1 - h) × 256 × 62 MB / 3000 MB/s
            = (1 - h) × 5.3 sec

hit_rate  | TPS (M2) | TPS (PC)
100%      | ~30      | ~40
80%       | ~8       | ~15
50%       | ~2       | ~4
```

Warm cache with hot-store achieves ~50-70% hit rate after 10+ tokens of context.

## File Layout (1-golden/)

```
src/
├── gpu.h              # GPU abstraction interface
├── gpu_metal.m        # Metal backend implementation
├── gguf.h             # GGUF loader interface
├── gguf.c             # GGUF parser + memory-mapped tensor access
├── memory.h           # tiered memory + LRU cache interface
├── memory.c           # expert cache, hot-store, async prefetch
├── model.h            # Ornith 35B forward pass interface
├── model.c            # Qwen MoE forward pass implementation
├── inference.h        # prefill + decode loop interface
├── inference.c        # token generation loop
├── tokenizer.h        # BPE tokenizer interface
├── tokenizer.c        # Qwen tokenizer (BPE)
├── server.c           # HTTP server (OpenAI-compatible API)
├── main.c             # CLI entry point (chat, interactive)
├── metal/
│   ├── quant.metal    # int4/int8 dequant + matmul kernels
│   ├── attention.metal # scaled dot-product attention
│   ├── mlp.metal      # SiLU-gated MLP, MoE routing
│   └── norm.metal     # RMSNorm, RoPE
└── third_party/
    └── gguf.h          # GGUF format constants (from llama.cpp)
```
