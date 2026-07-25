# ornith-flight

**Expert-streaming inference engine for Ornith 1.0 35B MoE — runs on consumer hardware.**

[![C 99](https://img.shields.io/badge/C-99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-47%2F47-passing-green)]()

> **Inspired by [Colibri's](https://github.com/OpenBMB/MiniCPM) expert streaming for GLM-5.2.**  
> Goal: run Ornith 1.0 35B — a 256-expert MoE — on a $999 MacBook Air M2 with 8 GB RAM.

---

## The Model: Ornith 1.0 35B

| Spec | Value |
|------|-------|
| **Architecture** | qwen35moe (MoE) |
| **Parameters** | ~35B total, ~3.7B active per token |
| **Layers** | 40 |
| **Hidden dim** | 2048 |
| **Attention heads** | 16 Q heads × 256 dim → 2 KV heads (GQA) |
| **Attention type** | Hybrid — full attention every 4th layer, SSM (Mamba-style) for the rest |
| **Experts** | 256 per layer, 8 active per token |
| **Expert hidden dim** | 512 (gated SiLU MLP) |
| **Vocab** | 248,320 tokens (BPE) |
| **Context** | 262,144 tokens |
| **Quantization** | Q4_K_M (gate/up), Q6_K (down) — **20 GB file** |
| **Source** | [`deepreinforce-ai/Ornith-1.0-35B-GGUF`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF) |

The model has a hybrid architecture:
- **Every 4th layer** (blk.3, 7, 11, …): standard full attention with QKV projections
- **Other layers**: SSM-based (Mamba-style with conv1d + gating + state-space model)
- **All layers**: 256 MoE experts with shared expert + top-8 routing

---

## The Problem

Ornith 1.0 35B has **10,240 experts** (256 × 40 layers). At Q4_K_M quantization:
- **All experts on disk**: ~18 GB  
- **Non-routed weights (always needed)**: ~2 GB  
- **One expert**: ~1.75 MB  

To load everything into RAM: **~20 GB required**. An 8 GB MacBook Air cannot fit it.

## The Solution — Expert Streaming

Instead of loading all experts, the engine keeps a small cache in RAM and streams the rest from SSD on demand:

```
┌────────────────────────────────────────────┐
│ T0: RESIDENT (~2 GB)                      │ ← Embeddings, attention, SSM,
│     Always loaded, never evicted           │   shared experts, LM head
├────────────────────────────────────────────┤
│ T1: HOT-STORE (pinned experts)            │ ← Most-used experts (never evicted)
│     Size configurable (e.g., 50 experts)  │
├────────────────────────────────────────────┤
│ T2: LRU CACHE (dynamic)                   │ ← Recently used experts
│     Evicted when full                     │
├────────────────────────────────────────────┤
│ T3: SSD (~18 GB)                          │ ← All 10,240 experts
│     Streamed on demand at 1.71 GB/s       │
└────────────────────────────────────────────┘
```

### Expert Loading Pipeline

```
Token → Router predicts 8 experts
         │
         ├── Hot-store hit?  → Use immediately (~0.1 µs)
         ├── LRU cache hit?  → Use immediately (~0.5 µs)
         └── SSD miss?       → Load ~1.75 MB from SSD (~1 ms)
                               Then dequantize + run expert MLP
```

---

## Hardware Reality (MacBook Air M2, 8 GB)

| Resource | Available | Ornith-35B Requirement |
|----------|-----------|----------------------|
| **RAM** | 8 GB | ~2 GB resident + expert cache |
| **SSD** | 1.71 GB/s (measured) | ~18 GB model file fits |
| **Disk free** | 45 GB | 20 GB model file ✅ |

### Measured SSD Bandwidth

```
fio --name=seq_read --rw=read --bs=1M --size=4G
  → 1,632 MiB/s = 1.71 GB/s
```

This is the **critical bottleneck**. Each SSD-expert fetch takes ~1 ms per expert for 8 experts = ~8 ms per token minimum I/O time.

---

## C Inference Engine — Status

The engine is written in C99 with 47 unit tests passing. It implements:

| Module | File | Status |
|--------|------|--------|
| GGUF v3 parser (mmap'd) | `src/gguf.c` | ✅ Reads header, metadata, tensor info (F32 only) |
| Forward pass | `src/model.c` | ✅ Attention, MoE routing, expert MLP, residuals, sampling |
| Inference loop | `src/inference.c` | ✅ KV cache, prefill, decode, timing |
| Memory manager | `src/memory.c` | ✅ LRU cache, hot-store pinning, stats |
| Tokenizer (BPE) | `src/tokenizer.c` | ✅ Encode/decode, special tokens |
| GPU abstraction (CPU) | `src/gpu.c` | ✅ Buffer management, CPU matmul fallback |
| Metal backend | `src/gpu_metal.m` | ⚠️ Compiles, MSL kernels written, needs Xcode for metallib |

### Test Results: 47/47 Passing

```
Module            Pass  Fail
t_cli                7     0  CLI argument parsing
t_gguf               9     0  GGUF header, metadata, tensor info
t_gpu                7     0  Buffer lifecycle, copy, matmul
t_inference          6     0  KV cache, prefill, decode, timing
t_memory             6     0  LRU eviction, hot-store, stats
t_model              5     0  Config, routing, sampling
t_tokenizer          7     0  Encode/decode, special tokens
Total               47     0
```

---

## What's Needed to Run Ornith 1.0 35B

### 1. Extend GGUF Parser for Quantized Tensors (🏗️ IN PROGRESS)

The downloaded `ornith-1.0-35b-Q4_K_M.gguf` uses **Q4_K_M** and **Q6_K** quantization block formats. Our parser currently only reads F32 tensors. We need to add dequantization for:

| Format | Blocks | Used for |
|--------|--------|---------|
| **Q4_K** | 4-bit K-quant (super-block) | Expert gate/up weights, attention projections |
| **Q6_K** | 6-bit K-quant | Expert down weights, some attention QKV |
| **F32** | 32-bit float | Norms, router weights, biases |

### 2. Adapt Forward Pass for Ornith's Architecture

The current forward pass was written for a simplified vanilla MoE arch. The real Ornith model needs:

| Feature | Current Engine | Ornith Needs |
|---------|---------------|--------------|
| Attention | Standard QKV | Q/K/V with separate norms, **key_length=256 ≠ head_count×head_dim** |
| SSM layers | Not implemented | Conv1d + gating + state-space recurrence |
| Hybrid pattern | Sequential layers | Full attention every 4th layer, SSM for others |
| Expert weight layout | Per-expert tensors | **Fused** gate/up/down_exps.weight with [dim, hidden, n_experts] |
| Head dim | head_dim = d_model/n_heads | key_length=256 (independent of head_count) |

### 3. Wire in Expert Offloading

The memory manager (hot-store + LRU cache) is built but not connected to the forward pass. Currently the engine reads from mmap'd tensor data. Real offloading requires:

- Load individual experts from the fused weight tensors at the right offset
- Dequantize on the fly
- Cache in hot-store / LRU
- Async prefetch for next layer's experts

---

## Getting Started

### Build & Run Tests

```bash
cd 1-golden
make                    # Build CPU-fallback engine
make test               # Run all 47 tests
```

### Inspect the Downloaded Model

```bash
# Show full metadata
gguf-dump models/ornith-1.0-35b-Q4_K_M.gguf | head -50

# Show tensor list
gguf-dump models/ornith-1.0-35b-Q4_K_M.gguf | grep -c "blk\."

# File size
ls -lh models/ornith-1.0-35b-Q4_K_M.gguf
```

### Try Loading with Current Parser (will fail on quantized tensors)

```bash
cd 1-golden
./ornith --model ../models/ornith-1.0-35b-Q4_K_M.gguf --config
```

### Benchmark Your SSD

```bash
fio --name=seq_read --rw=read --bs=1M --size=4G
```

---

## Project Structure

```
ornith-flight/
├── models/                ← Downloaded Ornith-1.0-35B GGUF (20 GB)
├── 0-proto/               Python prototype — param optimization & tuning
├── 1-golden/              C inference engine
│   ├── src/
│   │   ├── model.c        Forward pass (attention, MoE, residuals)
│   │   ├── inference.c    KV cache, prefill, decode, timing
│   │   ├── gguf.c         GGUF v3 parser (mmap'd)
│   │   ├── gpu.c          CPU-fallback GPU abstraction
│   │   ├── gpu_metal.m    Metal GPU backend
│   │   ├── memory.c       Tiered LRU cache + hot-store
│   │   └── metal/         Metal shader source files
│   ├── test/              47 unit tests
│   └── Makefile
├── STATUS.md              Current project status
└── README.md
```

---

## Roadmap

| Step | What | Priority |
|------|------|----------|
| **1** | Extend GGUF parser: add Q4_K/Q6_K dequantization | 🔴 Critical |
| **2** | Add SSM layer support (conv1d + state-space) | 🔴 Critical |
| **3** | Adapt attention for separate key/value lengths + Q/K norms | 🔴 Critical |
| **4** | Wire expert offloading (hot-store + LRU into forward pass) | 🔴 Critical |
| **5** | Handle fused expert weight tensors (gate/up/down_exps) | 🟡 High |
| **6** | Test inference on real Ornith model, measure throughput | 🟡 High |
| **7** | Async prefetch (overlap I/O with compute) | 🟢 Medium |
| **8** | Per-layer caching (different hot experts per layer) | 🟢 Medium |

---

## Contributing

Key areas: CUDA backend, SSM kernel optimization, async I/O prefetch, per-layer caching, server API.

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).

---

**Model:** [Ornith 1.0 35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF)  
**Engine:** 47/47 tests passing, needs parser extension for Q4_K/Q6_K support  
**Repo:** https://github.com/instax-dutta/ornith-flight
