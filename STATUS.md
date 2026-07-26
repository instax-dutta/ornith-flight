# Ornith-Flight — Project Status Report

> **Generated:** July 26, 2026
> **Repository:** https://github.com/instax-dutta/ornith-flight
> **Tests:** **68/68 passing** (CPU + Metal GPU)
> **Model:** Ornith 1.0 35B — `models/ornith-1.0-35b-Q4_K_M.gguf` (20 GB GGUF, Q4_K_M)

---

## Overview

ornith-flight is an expert-streaming inference engine for **Ornith 1.0 35B** — a 256-expert MoE model with hybrid attention (full attention + SSM/Mamba layers). The engine streams experts from SSD to RAM, making 35B inference feasible on 8 GB consumer hardware.

The engine supports **Q4_K and Q6_K quantized tensors** via per-super-block element extraction, **async I/O prefetch** with a background thread pool, and both **CPU fallback** and **Metal GPU** backends (via Apple MPS and custom Metal shaders).

---

## ✅ What's Done

### Model Downloaded

| Item | Value |
|------|-------|
| Source | [`deepreinforce-ai/Ornith-1.0-35B-GGUF`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF) (Hugging Face) |
| File | `models/ornith-1.0-35b-Q4_K_M.gguf` |
| Size | 20 GB |
| Quant | Q4_K (gate/up), Q6_K (down), F32 (norms/router) |
| Auth | HF token configured |

### Verified Model Architecture (via `--dry-run -v`)

| Parameter | Value | Source |
|-----------|-------|--------|
| Architecture | `qwen35moe` | GGUF metadata |
| Layers | 40 | GGUF |
| d_model | 2048 | GGUF |
| Heads / KV heads | 16 / 2 | GGUF |
| Head dim | 128 | Computed: 2048/16 |
| Key / Value length | 256 / 256 | GGUF |
| RoPE dims / theta | 64 / 10,000,000 | GGUF |
| Experts/layer | 256 | GGUF |
| Active experts | 8 | GGUF |
| Shared experts | 1 | GGUF |
| Shared FFN dim | 512 | GGUF |
| Expert FFN dim | 512 | GGUF |
| **Vocab size** | **248,320** ✅ | GGUF (fixed `read_config`) |
| Max seq len | 262,144 | GGUF |
| Full attn interval | 4 | GGUF |
| Expert size | 62.0 MB | Computed |
| **Total tensors** | **733** | GGUF |

> ✅ **Dry-run validation**: `./ornith --dry-run -v --model models/ornith-1.0-35b-Q4_K_M.gguf` loads config from GGUF metadata, prints all parameters, and exits without allocating weights. Confirmed all 733 tensors found, architecture correctly identified.

### Phase 0: Python Prototype — COMPLETE

All 15 modules implemented and tested (parameter tuning, cache simulation, routing profiles, etc.).

### Phase 1: C Inference Engine — **68/68 Tests Passing**

| Module | Status | Tests |
|--------|--------|-------|
| CLI argument parsing (`t_cli`) | ✅ | 7/7 |
| GGUF parser + Q4_K/Q6_K dequant (`t_gguf`) | ✅ | 16/16 |
| GPU abstraction — CPU fallback (`t_gpu`) | ✅ | 7/7 |
| GPU abstraction — Metal shaders (`t_gpu_metal`) | ✅ | 6/6 |
| Memory manager + prefetch (`t_memory`) | ✅ | 12/12 |
| Model forward pass (`t_model`) | ✅ | 7/7 |
| Inference loop (KV cache, decode) (`t_inference`) | ✅ | 6/6 |
| Tokenizer (BPE) (`t_tokenizer`) | ✅ | 7/7 |
| **Total** | **✅** | **68/68** |

### Phase B: Expert Slice Dequantization from GGUF — COMPLETE

**Problem**: The model stores all 256 experts per layer in fused 3D tensors (e.g. `ffn_gate_exps.weight` with shape `[2048, 512, 256]`). Dequantizing the entire tensor per layer required ~3 GB of F32 buffers — impossible on 8 GB RAM.

**Solution**: Extract single elements directly from the mmap'd Q4_K/Q6_K super-blocks without materializing the full dequantized tensor.

| Function | File | Purpose |
|----------|------|---------|
| `dequantize_q4_K_one(block, pos)` | `gguf.c` | Extract 1 element from a Q4_K super-block at position 0–255 |
| `dequantize_q6_K_one(block, pos)` | `gguf.c` | Extract 1 element from a Q6_K super-block at position 0–255 |
| `gguf_dequantize_expert_slice()` | `gguf.c` | Dequantize only one expert's slice from a fused 3D quantized tensor |
| `cache_layer_experts()` | `model.c` | Uses `gguf_dequantize_expert_slice()` instead of full F32 dequant |
| `ensure_layer_weights()` | `model.c` | No longer loads fused 3D tensors — saves ~3 GB per layer |

**Impact**: Peak per-layer memory dropped from ~3 GB to ~0 MB for the fused tensors. Each expert's weights are extracted on-demand from the mmap'd GGUF data.

| Metric | Before | After |
|--------|--------|-------|
| Peak per-layer F32 dequant | ~3 GB (3 fused tensors) | ~0 MB (skipped entirely) |
| Expert slice dequant | F32 strided copy + free | Direct Q4_K element extraction from mmap |
| Cache storage per layer | ~12 MB | ~12 MB (unchanged) |

### Phase C: Async I/O Prefetch — COMPLETE

**Problem**: Per-layer weight dequant and expert extraction from the 20 GB GGUF file is I/O bound when expert caching misses. The forward pass stalls waiting for mmap page faults.

**Solution**: Background thread pool submits lookahead prefetch jobs for upcoming layers while the current layer computes.

| Component | File | What it does |
|-----------|------|--------------|
| `prefetch_job` struct | `memory.h` | Job queue entry with `func`/`arg`/`completed`/`active` fields |
| `prefetch_func_t` typedef | `memory.h` | `void (*)(void*)` — job function signature |
| `memory_prefetch_submit()` | `memory.c` | Enqueue a job, signal a worker thread |
| `memory_prefetch_poll()` | `memory.c` | Non-blocking check if job completed |
| `memory_prefetch_wait()` | `memory.c` | Spin-wait with `usleep(100)` until job completes |
| `memory_prefetch_pending()` | `memory.c` | Count of active jobs in the queue |
| `memory_async_enabled()` / `memory_lookahead_layers()` | `memory.c` | Getter functions for opaque `memory_manager` |
| `prefetch_worker()` | `memory.c` | pthread worker: dequeue, execute, mark completed, release slot |
| `prefetch_layer_job()` | `model.c` | Calls `ensure_layer_weights()` in background for a given layer |
| `model_forward()` prefetch wiring | `model.c` | Submits lookahead jobs for next layers while computing current layer |

**Architecture**:
- Thread pool created at `memory_init`, destroyed at `memory_destroy`
- Ring buffer job queue (256 slots) with `pthread_mutex_t` + `pthread_cond_t`
- Worker copies func/arg to locals under lock; defers `job_count--` until after execution (prevents slot-reuse race)
- Configurable via `config_m2_final.json`: `async_io_threads`, `prefetch_queue_depth`, `lookahead_layers`
- Model forward pass: submits prefetch for layer 0 before main loop, submits `lookahead_layers` ahead at end of each iteration
- Falls back gracefully (0 threads, queue full, or disabled) — `memory_prefetch_submit` returns `-1`, forward loop proceeds synchronously

**Tests**: 3 new prefetch tests — submit/poll, multiple concurrent jobs, invalid args.

### Metal GPU Backend

| Component | Status |
|-----------|--------|
| ObjC bridge (`gpu_metal.m`) | Compiles, runs on M2 (6/6 tests pass) |
| MSL shaders (quant, attention, MLP, norm) | Written and tested |
| MPS matrix multiplication | Wired through `metal_backend_gpu.h` |
| Auto-run after CPU tests | ✅ `make test` runs both CPU + Metal |

### CLI Flags

| Flag | Purpose |
|------|---------|
| `--model <path>` | Path to GGUF model file |
| `-v` / `--verbose` | Print model config before allocating weights |
| `--dry-run` | Load config, print, exit — no weight allocation |
| `-b` / `--benchmark` | Run in benchmark mode |
| `-n <tokens>` | Number of tokens to generate |
| `-p <prompt>` | Input prompt string |

---

## 📋 What Remains

### Priority: Run Real Inference on 8 GB M2 MacBook

The 20 GB model on 8 GB RAM hits severe memory pressure during weight loading. Key strategies to address:

| Strategy | Status | Impact |
|----------|--------|--------|
| ✅ **Expert slice dequant** (Phase B) | Done | No full-tensor F32 dequant |
| ✅ **Async I/O prefetch** (Phase C) | Done | Page faults hidden behind computation |
| 🔲 **Streaming layer I/O** — map+unmap per-layer from file | Next | Peak memory per-layer ~60 MB instead of 20 GB |
| 🔲 **Memory-mapped expert cache** — `MAP_POPULATE` | Next | Warm up expert pages on demand |
| 🔲 **Tiered inference** — process 1–2 layers at a time, swap rest to disk page cache | Next | OS handles eviction automatically |

### Priority: Fused Expert Weight Indexing

`gguf_dequantize_expert_slice()` can extract any expert at any position — but the router needs to return expert indices that map to the correct fused tensor offsets. This is already done in the routing path.

### Priority: Full Hybrid Attention

Full attention layers require handling KV cache at `key_length=256` instead of `head_dim=128`. The KV cache struct already supports this — needs integration with the cache allocator.

### Priority: SSM Layer

SSM layers (30 of 40) need conv1d + gating + state-space recurrence wired into the forward pass. The weight tensors (`ssm_conv1d`, `ssm_a`, `ssm_dt_bias`, `ssm_alpha`, `ssm_beta`) are laid out in the GGUF — just need the compute loop.

### Priority: Tokenizer (BPE) Integration

Tokenizer is implemented and tested (7/7) but needs to be wired into the inference loop for prompt processing and output decoding.

---

## Quick Reference

```bash
# Build & test engine (CPU + Metal)
cd 1-golden && make && make test

# Inspect model config (no weight allocation)
cd 1-golden && ./ornith --dry-run -v --model ../models/ornith-1.0-35b-Q4_K_M.gguf

# Run benchmark inference
cd 1-golden && ./ornith --model ../models/ornith-1.0-35b-Q4_K_M.gguf -b -n 3

# Check SSD speed (affects prefetch tuning)
fio --name=seq_read --rw=read --bs=1M --size=4G

# Check memory pressure
memory_pressure
```

## Test Results (68/68)

```
Module           Passed   Failed
──────────────────────────────────
t_cli              7        0
t_gguf            16        0    ← +2 expert slice dequant tests
t_gpu              7        0
t_gpu_metal        6        0    ← Metal shader tests
t_inference        6        0
t_memory          12        0    ← +3 async prefetch tests
t_model            7        0
t_tokenizer        7        0
──────────────────────────────────
Total:            68        0
```
