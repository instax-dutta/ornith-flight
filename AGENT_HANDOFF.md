# Ornith-Flight — AI Agent Handoff Report

**Generated:** July 26, 2026
**Goal:** Run Ornith 1.0 35B (256-expert MoE) on an M2 MacBook Air with 8 GB RAM via expert streaming from SSD.
**Repository:** `https://github.com/instax-dutta/ornith-flight`
**Tests:** 58/58 passing (8 test binaries)

---

## Real Model (Ornith 1.0 35B)

Downloaded as `models/ornith-1.0-35b-Q4_K_M.gguf` (20 GB):

| Parameter | Value |
|---|---|
| Architecture | `qwen35moe` |
| Layers | 40 (not 28 as proto assumed) |
| d_model | 2048 |
| n_heads / n_kv_heads | 16 / 2 |
| key_length / value_length | 256 / 256 |
| Expert count | 256 per layer (256 x 40 = 10,240 total) |
| Active experts | 8 per token |
| Expert hidden dim | 512 |
| Shared expert | Yes (ffn_gate/up/down_shexp) |
| Quantization | Q4_K_M (gate/up), Q6_K (down), F32 (norms/router) |
| Total tensors | 733 |
| Context length | 262,144 |
| Vocab | 248,320 (BPE) |
| RoPE theta | 10,000,000 |
| Hybrid architecture | Full attn every 4th layer, SSM (Mamba) for rest |

---

## Module Status

### GGUF Parser (`src/gguf.c`) — FUNCTIONAL
- Reads v3 GGUF header, metadata, tensor info
- mmap'd file access
- Dequantization: F32 (direct), F16 (to F32), **Q4_K**, **Q6_K** — all implemented and tested (14 tests)

### Model Forward Pass (`src/model.c`) — PARTIAL
- Config loading reads all params from GGUF metadata
- SSM layer forward: implemented (RMSNorm + gate + conv1d + SSM recurrence + output)
- Attention layer forward: implemented (RMSNorm + QKV proj + RoPE + GQA + output proj)
- Router: softmax + top-8 selection (CPU)
- **CRITICAL GAP**: `ensure_layer_weights()` loads ALL 256 experts per layer, dequantizes them to float32, uses them, then frees them. This works for tiny test models but will NOT work on the real 20 GB model — the fused expert tensors (ffn_gate_exps [2048,512,256]) alone are ~268M elements each.

### Memory Manager (`src/memory.c`) — IMPLEMENTED BUT NOT WIRED
- Hot-store (pinned experts, configurable size, default 50)
- LRU cache (configurable, default 16 entries for M2)
- Stats tracking (hot/lru hits, misses, evictions)
- **NOT WIRED INTO FORWARD PASS**: `memory_get()` and `memory_hot_promote()` are never called from `model_forward()`

### Inference Loop (`src/inference.c`) — FUNCTIONAL
- KV cache allocation and management (2048 positions per layer)
- SSM state buffers (conv ring buffer + h_state per layer)
- Prefill phase + decode phase
- Timing (TTFT, decode tps, prefill time)
- Token sampling (temperature, top-k)

### GPU Abstraction (`src/gpu.c` + `src/gpu_metal.m`) — READY
- CPU fallback: matmul, rmsnorm, silu, add all implemented
- Metal backend: compiles on macOS, MSL shaders written (attention, MLP, norm, quant, ops)
- 13 tests (7 CPU + 6 Metal) passing

### Tokenizer (`src/tokenizer.c`) — FUNCTIONAL
- BPE encode/decode
- Special tokens (BOS, EOS)

### Python Prototype (`0-proto/`) — COMPLETE
- 6/6 tests passing
- Optimized configs for M2 and PC
- Power-law routing validated (exponent ~1.5)
- int4 vs int8 trade-off analyzed

---

## What's Blocking Real Model Inference

### P1: Expert Offloading NOT Wired (CRITICAL)
The engine's core innovation — streaming experts from SSD via hot-store + LRU — exists as infrastructure but is **not connected to the forward pass**. Currently `model_forward()` calls `ensure_layer_weights()` which loads everything. Replace this with:
1. Skip loading all expert weights upfront
2. Router predicts 8 experts per token
3. Check hot-store/LRU via `memory_get()` for each expert
4. On miss: load + dequantize single expert from fused 3D tensor at correct offset
5. Cache in LRU / promote to hot-store
6. Deallocate evicted entries

### P2: Fused Expert Tensor Indexing (HIGH)
Expert weights are stored as fused 3D tensors:
- `ffn_gate_exps.weight`: [2048, 512, 256] — Q4_K_M
- `ffn_up_exps.weight`: [2048, 512, 256] — Q4_K_M
- `ffn_down_exps.weight`: [512, 2048, 256] — Q6_K

Need to index into these at `[:, :, expert_id]` and dequantize only that slice. The current code loads the entire tensor as float32 — impossible for 20 GB model.

### P3: Real Model Weight Loading Untested (HIGH)
`ensure_layer_weights()` will attempt to load 256 expert tensors from Q4_K/Q6_K data. Need to verify that:
- Dequantization of the actual GGUF data works correctly
- F32 norms/router weights read correctly
- SSM conv1d and state tensors parse correctly
- Total memory doesn't explode

### P4: Async Prefetch (MEDIUM)
Lookahead prefetch (load layer N+1 while computing N) reduces I/O stall by ~60%. Not implemented.

### P5: Per-Layer Expert Caches (MEDIUM)
Current memory manager is global — all layers share one hot-store + LRU. Each layer has different hot experts. 40 small per-layer caches would improve hit rate 2-3x over 1 global cache.

### P6: KV Cache Size (LOW)
Currently 2048 positions. Model supports 262,144 context. For long contexts, KV cache quantization (FP8/INT8) will be needed.

---

## Expected Performance

| Scenario | Approx Hit Rate | Decode TPS |
|---|---|---|
| No cache (all from SSD) | 0% | ~0.1 |
| 50 hot + 16 LRU (M2) | 11-13% | ~0.26 |
| 50 hot + 49 LRU (PC) | 14-18% | ~0.30 |
| With per-layer caching | 25-35% | ~0.8-1.2 |
| With async prefetch | +10% perf | +10% |
| With speculative decoding | N/A | 2-3x |

**On your M2 MacBook Air 8GB:** Expect ~0.2-0.3 tok/s initially. Per-layer caching + prefetch could push to ~0.5-1.0 tok/s.

**On your PC (i9-9900K + RTX 4060 + 16GB):** Larger LRU cache + GPU offloading via Metal/CUDA should give ~0.3-0.5 tok/s initial, up to ~1-3 tok/s with full optimization.

---

## Implementation Priority

1. Wire expert offloading (hot-store/LRU into forward pass)
2. Single-expert indexing from fused 3D tensors + dequant
3. Test inference on real model, measure hit rate & throughput
4. Async prefetch (overlap I/O with compute)
5. Per-layer caching
6. KV cache quantization for long contexts
7. Speculative decoding for throughput boost

### Quick Start to Test Real Model
```bash
# Current engine (will load but not produce useful output yet):
cd 1-golden && make && ./ornith --model ../models/ornith-1.0-35b-Q4_K_M.gguf --dry-run --verbose

# Run full test suite:
make test
```
