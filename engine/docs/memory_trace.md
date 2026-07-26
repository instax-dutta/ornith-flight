# Memory Allocation Trace — Ornith 35B (Q4_K_M)

## model_load() — Memory at Each Step

| Step | Allocation | Virtual | Resident | Notes |
|------|-----------|---------|----------|-------|
| `gguf_open()` | mmap 20,446,756,864 bytes (19.7 GB GGUF) | **19.7 GB** | ~0 GB | MAP_PRIVATE — lazy page-in, no RSS until accessed |
| `read_config()` | reads 44 metadata entries from mmap | 0 | ~64 KB | small metadata pages |
| `gpu_init()` | `gpu_context` struct ~64 bytes | 0 | ~64 KB | negligible |
| `memory_init()` | `memory_manager` struct ~20 KB | 0 | ~64 KB | slot arrays embedded in struct; **no large allocation** |
| **model_load total** | | **19.7 GB** | **~200 KB** | |

The memory manager (`memory_init`) allocates **zero large buffers** — the hot-store and LRU slots are `slot hot_store[256]` and `slot lru_cache[512]` embedded directly in the `memory_manager` struct (~12 KB combined). The `expert_size_bytes = 62 MB` in the config is **metadata only** — actual expert weight data would be loaded from the mmap'd GGUF on demand, not pre-allocated.

## model_forward() — Peak Memory Per Layer

### SSM Layer (layers 0, 1, 2 — 3 out of every 4)

| Buffer | Size (elements) | Size (bytes) | Location |
|--------|----------------|-------------|----------|
| `x` | 2048 | 8 KB | heap |
| `norm_w` | 2048 | 8 KB | stack |
| `qkv_w` (dequant) | 2048 × 8192 = 16,777,216 | **64 MB** | heap |
| `gate_w` (dequant) | 2048 × 4096 = 8,388,608 | **32 MB** | heap |
| `out_w` (dequant) | 4096 × 2048 = 8,388,608 | **32 MB** | heap |
| `conv_w` (dequant) | 4 × 8192 = 32,768 | **128 KB** | stack |
| `a_diag` | 32 | 128 B | stack |
| `dt_b` | 32 | 128 B | stack |
| `alpha_w` | 2048 × 32 = 65,536 | **256 KB** | stack |
| `beta_w` | 2048 × 32 = 65,536 | **256 KB** | stack |
| `norm_s_w` | 128 | 512 B | stack |
| SSM internal `alloca` | 8192 + 4096 + 4096 + 2048 | **~72 KB** | stack |
| Shared expert `s_gate_w` | 2048 × 512 | **4 MB** | heap |
| Shared expert `s_up_w` | 2048 × 512 | **4 MB** | heap |
| Shared expert `s_down_w` | 512 × 2048 | **4 MB** | heap |
| **SSM layer total** | | **~140 MB** | (freed at end of layer) |

### Attention Layer (layer 3 — every 4th layer)

| Buffer | Size (elements) | Size (bytes) | Location |
|--------|----------------|-------------|----------|
| `norm_w` | 2048 | 8 KB | stack |
| `q_w` (dequant) | 2048 × 8192 = 16,777,216 | **64 MB** | heap |
| `k_w` (dequant) | 2048 × 512 = 1,048,576 | **4 MB** | heap |
| `v_w` (dequant) | 2048 × 512 = 1,048,576 | **4 MB** | heap |
| `o_w` (dequant) | 8192 × 2048 = 16,777,216 | **64 MB** | heap |
| Shared expert (same as SSM) | | **12 MB** | heap |
| **Attention layer total** | | **~148 MB** | (freed at end of layer) |

### Final Norm + LM Head

| Buffer | Size (elements) | Size (bytes) | Location |
|--------|----------------|-------------|----------|
| `final_norm_buf` | 2048 | 8 KB | stack |
| `logits` | min(248320, 256) = 256 | 1 KB | heap (from caller) |
| **Final total** | | **~9 KB** | |

## Peak Resident Memory

- **GGUF mmap (resident portion)**: ~70 MB per SSM layer × 4 layers ≈ **280 MB** (Q4_K quantized weights accessed during dequant)
- **Dequantized float buffers**: **140 MB** peak (freed per layer)
- **Stack**: **~600 KB** (safe within 8 MB macOS limit)
- **Code + library**: ~2 MB

**Total peak RSS: ~420 MB** — well within 16 GB M2 Mac.

## Root Cause of Historical OOM (Killed: 9)

The `o_w` buffer was originally allocated as:
```c
float *o_w = calloc(n_heads * head_dim * D, sizeof(float));
// = 16 * 128 * 2048 = 4,194,304 elements = 16 MB
```

But `attn_output.weight` in the Ornith GGUF has dimensions `[8192, 2048]` = 16,777,216 elements, requiring **64 MB**. The dequant wrote 48 MB past the buffer end, corrupting the heap. This caused `free()` or subsequent `malloc()` to segfault, which the OS reported as "Killed: 9" (SIGKILL from memory pressure).

**Fix**: All weight buffers now use `tensor->n_elems * sizeof(float)` instead of computed config sizes.

## What Does NOT Cause OOM

- **`memory_init`**: Zero large allocations (~20 KB struct with embedded slot arrays)
- **20 GB GGUF mmap**: MAP_PRIVATE lazy, no RSS until data accessed
- **Per-layer dequant**: Freed before next layer; peak ~140 MB
- **Stack arrays**: ~600 KB total, within 8 MB macOS limit
