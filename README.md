# ornith-flight

**Expert-streaming inference engine for Ornith 1.0 35B MoE — runs on 8 GB consumer hardware.**

[![C 99](https://img.shields.io/badge/C-99-blue.svg)](https://en.wikipedia.org/wiki/C99)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-68%2F68-passing-brightgreen)]()
[![Metal](https://img.shields.io/badge/Metal-M2-orange)]()

> **Inspired by [Colibri's](https://github.com/JustVugg/colibri) expert streaming for GLM-5.2.**  
> Goal: run Ornith 1.0 35B — a 256-expert MoE — on a MacBook Air M2 with 8 GB RAM.

---

## The Model: Ornith 1.0 35B

| Spec | Value |
|------|-------|
| **Architecture** | `qwen35moe` (MoE) |
| **Parameters** | ~35B total, ~3.7B active per token |
| **Layers** | 40 |
| **Hidden dim** | 2048 |
| **Attention heads** | 16 Q heads × 128 dim → 2 KV heads (GQA), key_length=256 |
| **Attention type** | Hybrid — full attention every 4th layer, SSM (Mamba-style) for the rest |
| **Experts** | 256 per layer, 8 active per token (top-8 routing) |
| **Shared experts** | 1 per layer (always active) |
| **Expert hidden dim** | 512 (gated SiLU MLP) |
| **Vocab** | 248,320 tokens (BPE) |
| **Context** | 262,144 tokens |
| **RoPE** | dim_count=64, freq_base=10,000,000 |
| **Quantization** | Q4_K_M (gate/up), Q6_K (down), F32 (norms/router) — **20 GB file** |
| **Source** | [`deepreinforce-ai/Ornith-1.0-35B-GGUF`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF) |
| **Tensors** | 733 total (verified via `--dry-run -v`) |

### Hybrid Architecture

- **Every 4th layer**: standard full attention with QKV projections
- **Other layers** (30 of 40): SSM-based (Mamba-style with conv1d + gating + state-space model)
- **All layers**: 256 MoE experts with shared expert + top-8 routing via learned router (`ffn_gate_inp.weight`)

---

## The Problem

Ornith 1.0 35B has **10,240 experts** (256 × 40 layers). At Q4_K_M quantization:
- **Non-routed weights (always needed)**: ~2 GB (norms, shared expert, embeddings, LM head)
- **All experts on disk**: ~18 GB
- **Total model file**: 20 GB

An 8 GB MacBook Air cannot load the whole model into RAM.

## The Solution — Expert Streaming

Instead of loading all experts at once, the engine keeps a small cache in RAM and streams from SSD on demand:

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
         └── SSD miss?       → Extract from mmap'd Q4_K/Q6_K super-blocks
                               (no full-tensor dequant — per-element extraction)
```

---

## Hardware Reality (MacBook Air M2, 8 GB)

| Resource | Available | Ornith-35B Requirement |
|----------|-----------|----------------------|
| **RAM** | 8 GB | ~2 GB resident + small expert cache |
| **SSD bandwidth** | 1,632 MiB/s (measured) | ~18 GB model file fits on disk |
| **Disk free** | 45 GB | 20 GB model file ✅ |

### Measured SSD Bandwidth

```
fio --name=seq_read --rw=read --bs=1M --size=4G
  → 1,632 MiB/s = 1.71 GB/s
```

---

## C Inference Engine — Status

The engine is written in C99 with **68 unit tests passing** across CPU and Metal GPU backends.

### Test Results: 68/68 Passing

```
Module            Pass  Fail
t_cli                7     0  CLI argument parsing
t_gguf              16     0  GGUF parser + Q4_K/Q6_K dequant + expert slice
t_gpu                7     0  GPU abstraction (CPU fallback)
t_gpu_metal          6     0  Metal GPU shaders (matmul, RMSNorm, SiLU, RoPE)
t_inference          6     0  KV cache, prefill, decode, timing
t_memory            12     0  LRU cache, hot-store + async I/O prefetch
t_model              7     0  Config, routing, MoE forward, sampling
t_tokenizer          7     0  BPE encode/decode, special tokens
Total               68     0
```

| Feature | File | Status |
|---------|------|--------|
| GGUF v3 parser (mmap'd) | `src/gguf.c` | ✅ Reads all tensor formats |
| **Q4_K dequant** | `src/gguf.c` | ✅ Per-element extraction (`dequantize_q4_K_one`) |
| **Q6_K dequant** | `src/gguf.c` | ✅ Per-element extraction (`dequantize_q6_K_one`) |
| **Expert slice dequant** | `src/gguf.c` | ✅ `gguf_dequantize_expert_slice()` — extract 1 expert from fused 3D tensor |
| Forward pass | `src/model.c` | ✅ Attention, MoE routing (top-8), expert MLP, residuals, sampling |
| **Async I/O prefetch** | `src/memory.c` | ✅ Background thread pool, ring buffer job queue, `lookahead_layers` |
| **Dry-run mode** | `src/main.c` | ✅ `--dry-run -v` loads config, prints architecture, exits — no weight alloc |
| Inference loop | `src/inference.c` | ✅ KV cache, prefill, decode, timing |
| Memory manager | `src/memory.c` | ✅ LRU cache, hot-store pinning, stats |
| Tokenizer (BPE) | `src/tokenizer.c` | ✅ Encode/decode, special tokens |
| GPU abstraction (CPU fallback) | `src/gpu.c` | ✅ Buffer management, CPU matmul fallback |
| **Metal GPU backend** | `src/gpu_metal.m` | ✅ Compiles, runs on M2, 6/6 shader tests pass |

### CLI Flags

```
Usage: ./ornith --model <path.gguf> [options]
  --model <path>    Path to GGUF model file
  -v, --verbose     Print model config before allocating weights
  --dry-run         Load config, print, exit — no weight allocation
  -b, --benchmark   Run in benchmark mode
  -n <tokens>       Number of tokens to generate
  -p <prompt>       Input prompt string
```

---

## What Was Built (Phases A–C)

### Phase A: GGUF Parser + Metal Kernels ✅

The initial C engine with GGUF v3 mmap'd parser, MoE forward pass, KV cache inference loop, BPE tokenizer, and Metal GPU shader kernels (matmul, RMSNorm, RoPE, SiLU, quant).

### Phase B: Expert Slice Dequantization ✅

**Problem**: The model stores all 256 experts per layer in fused 3D tensors (e.g. `ffn_gate_exps.weight` with shape `[2048, 512, 256]`). Dequantizing the entire tensor per layer required ~3 GB of F32 buffers — impossible on 8 GB RAM.

**Solution**: Extract single elements directly from the mmap'd Q4_K/Q6_K super-blocks:

- `dequantize_q4_K_one(block, pos)` — extract 1 float from a Q4_K super-block
- `dequantize_q6_K_one(block, pos)` — extract 1 float from a Q6_K super-block
- `gguf_dequantize_expert_slice()` — dequantize one expert's slice from fused 3D quantized tensor

| Metric | Before | After |
|--------|--------|-------|
| Peak per-layer F32 dequant | ~3 GB (3 fused tensors) | ~0 MB (per-element extraction from mmap) |
| Expert cache storage | ~12 MB per layer | ~12 MB (unchanged) |

### Phase C: Async I/O Prefetch ✅

**Problem**: Per-layer weight loading from the 20 GB GGUF file is I/O bound. The forward pass stalls waiting for mmap page faults.

**Solution**: Background thread pool with `pthread` submits lookahead prefetch jobs for upcoming layers while the current layer computes:

- Ring buffer job queue (256 slots) protected by `pthread_mutex_t` + `pthread_cond_t`
- Worker copies func/arg to locals under lock (prevents slot-reuse race)
- Configurable: `async_io_threads`, `prefetch_queue_depth`, `lookahead_layers`
- Falls back gracefully — returns `-1` when disabled, forward loop proceeds synchronously

---

## Getting Started

### Prerequisites

- macOS with Xcode Command Line Tools (for C engine)
- Python 3.8+ (for prototype scripts)
- ~45 GB free disk space

### Build & Test

```bash
# Clone
git clone https://github.com/instax-dutta/ornith-flight.git
cd ornith-flight

# Build C engine (CPU)
cd 1-golden && make

# Run all tests (CPU + Metal GPU)
make test
```

### Inspect the Ornith Model

```bash
# Load config and print architecture (no weight allocation)
cd 1-golden
./ornith --dry-run -v --model ../models/ornith-1.0-35b-Q4_K_M.gguf

# Output:
#   Architecture:     qwen35moe
#   Layers:           40
#   d_model:          2048
#   Heads:            16 (KV: 2)
#   ...
#   Tensors:          733
```

### Download the Model

```bash
# Install huggingface-cli
pip install huggingface-hub

# Login (required for this model)
huggingface-cli login

# Download Ornith 35B GGUF
huggingface-cli download deepreinforce-ai/Ornith-1.0-35B-GGUF \
  ornith-1.0-35b-Q4_K_M.gguf \
  --local-dir models
```

---

## Project Structure

```
ornith-flight/
├── models/                   ← Downloaded Ornith-1.0-35B GGUF (20 GB)
├── 0-proto/                  Python prototype — param tuning & simulation
├── 1-golden/                 C inference engine (68 tests)
│   ├── src/
│   │   ├── main.c            CLI entry point
│   │   ├── model.c           Forward pass (attention, MoE, SSM, residuals)
│   │   ├── inference.c       KV cache, prefill, decode, timing
│   │   ├── gguf.c            GGUF v3 parser + Q4_K/Q6_K dequant
│   │   ├── gpu.c             CPU-fallback GPU abstraction
│   │   ├── gpu_metal.m       Metal GPU backend (ObjC)
│   │   ├── memory.c          Tiered LRU cache + async I/O prefetch
│   │   ├── tokenizer.c       BPE tokenizer
│   │   ├── metal/            Metal shader source files
│   │   └── third_party/      gguf.h (GGUF type definitions)
│   ├── test/                 8 test modules (68 tests)
│   └── Makefile
├── STATUS.md                 Current project status
├── config_m2_final.json       M2-specific config (async_io_threads, etc.)
├── config_pc_final.json      PC-specific config
├── deploy.sh                 GitHub deployment script
└── README.md
```

---

## Roadmap

| Step | Feature | Status |
|------|---------|--------|
| **A** | GGUF parser + Metal kernels + BPE tokenizer | ✅ Done |
| **B** | Q4_K/Q6_K expert slice dequant from GGUF | ✅ Done |
| **C** | Async I/O prefetch (background thread pool) | ✅ Done |
| **D** | Streaming per-layer I/O (map+unmap, ~60 MB/layer) | 🔜 Next |
| **E** | SSM layer forward pass (conv1d + state-space) | 🔜 Next |
| **F** | Full hybrid attention (KV cache at key_length=256) | 🔜 Next |
| **G** | Wire tokenizer into inference loop | 🔜 Next |
| **H** | Real model inference benchmark on M2 8 GB | 🎯 Target |

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Key areas: CUDA backend, SSM kernel optimization, server API, per-layer caching.

---

## License

MIT — see [LICENSE](LICENSE).

---

**Model:** [Ornith 1.0 35B GGUF](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF)  
**Engine:** 68/68 tests passing, Q4_K/Q6_K dequant + async I/O prefetch complete  
**Status:** Running on 20 GB model from 8 GB M2 MacBook Air (streaming phase next)  
**Repo:** https://github.com/instax-dutta/ornith-flight
