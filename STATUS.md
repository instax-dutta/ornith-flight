# Ornith-Flight — Project Status Report

> **Generated:** July 24, 2026
> **Repository:** https://github.com/instax-dutta/ornith-flight
> **Tests:** 47/47 passing

---

## Overview

Ornith-flight streams large MoE model experts from SSD to consumer RAM, making inference feasible on 8 GB machines. The C inference engine is complete and running — the next step is validating against a real model.

---

## ✅ What's Done

### Phase 0: Python Prototype — COMPLETE

All 15 modules implemented and tested:
- Model layers (MoE, attention, router), config, GGUF format
- Routing profile analysis, throughput simulation
- LRU cache + prefetch strategy simulation
- Parameter tuning (cache size, hot-store, quantization, power-law)
- Comprehensive test suite (5 categories), config verification
- Two production configs: `config_m2_final.json` and `config_pc_final.json`

### Phase 1: C Inference Engine — BUILDING & TESTING

| Module | File(s) | Status | Tests |
|--------|---------|--------|-------|
| CLI argument parsing | `src/cli.c` | ✅ | 7/7 |
| GGUF parser (v3, mmap'd) | `src/gguf.c` | ✅ | 9/9 |
| GPU abstraction (CPU) | `src/gpu.c` | ✅ | 7/7 |
| Memory manager (tiered) | `src/memory.c` | ✅ | 6/6 |
| Model forward pass | `src/model.c` | ✅ Full pass | 5/5 |
| Inference engine | `src/inference.c` | ✅ KV cache + decode | 6/6 |
| Tokenizer (BPE) | `src/tokenizer.c` | ✅ | 7/7 |
| **Total** | | | **47/47** |

#### Forward Pass Implementation (model.c)

The forward pass implements the full Qwen2MoE computation:
1. Token embedding lookup
2. Per-layer: RMSNorm → QKV projection → RoPE → GQA attention → residual
3. Per-layer: RMSNorm → router (top-k softmax) → shared expert → routed experts → residual
4. Final RMSNorm → LM head → logits → temperature sampling

#### Real Inference Performance (MacBook Air M2)

```
Tiny model (D=64, 2 layers, 4 experts, 2 active):
  Prefill:   1.6 ms for 7 prompt tokens
  Decode:    3.5 ms for 16 generated tokens
  Throughput: 4,541 tok/s
  Benchmark:  3,129 tok/s avg over 3 iterations
```

### Metal GPU Backend

| Component | Status |
|-----------|--------|
| `gpu_metal.m` (ObjC bridge) | ✅ Compiles, runs on M2 |
| `quant.metal` (int4/int8 dequant) | ✅ Written |
| `attention.metal` (SDPA, DeltaNet) | ✅ Written |
| `mlp.metal` (SiLU, expert MLP, router) | ✅ Written |
| `norm.metal` (RMSNorm, RoPE) | ✅ Written |
| Runtime MSL compilation | ✅ Falls back from metallib |

Note: `xcrun metal` requires full Xcode.app (not just CLT) to compile `.metal` → `.metallib`. The ObjC bridge falls back to runtime `newLibraryWithSource:` compilation.

### Real Hardware Benchmarking

| Metric | Measured | Previous Assumption | Delta |
|--------|----------|-------------------|-------|
| SSD sequential read | **1.71 GB/s** | 3.0 GB/s | **-43%** |
| RAM | 8 GB | 8 GB | Match |
| GPU | Apple M2 (Metal 4) | Apple M2 GPU | Match |

Performance projections in `config_m2_final.json` have been revised to reflect the real 1.71 GB/s bandwidth.

---

## 📋 What Remains

### Critical Path

| Item | Priority | Details |
|------|----------|---------|
| **Download real GGUF model** | High | Qwen2-57B-A14B-Instruct Q4_K_M (~32 GB) from Hugging Face |
| **Extend GGUF parser for quantized formats** | High | Current parser only handles F32. Need Q4_K_M, Q8_0, etc. |
| **Expert offloading from SSD** | High | Wire hot-store + LRU into actual model_forward (currently always fetches from mmap) |
| **Validate end-to-end on real model** | High | Run full-scale 28-layer inference on M2 |

### Near-Term

| Item | Priority | Details |
|------|----------|---------|
| **Per-layer caching** | Medium | Each layer has different hot experts; 28 small caches could improve hit rate 2-3× |
| **Async prefetch** | Medium | Lookahead 1 layer: overlap I/O with computation (60-70% stall reduction) |
| **Metal verification** | Medium | Need full Xcode to compile metallib and test MSL on GPU |
| **CUDA backend** | Medium | `gpu_cuda.cu` for NVIDIA GPUs |
| **HTTP server** | Medium | OpenAI-compatible API (`server.c`) |

### Performance Optimizations

| Item | Impact | Details |
|------|--------|---------|
| **RMSNorm kernel** | 2-5× faster | Replace O(dim²) with simd_sum threadgroup reduction |
| **Top-k sort** | 10-100× faster | Replace bubble sort with parallel radix top-k |
| **Expert pruning** | 33% better cache | Remove bottom 25% least-used experts |
| **2-bit quant** | 4× more experts | Acceptable quality for cached experts |
| **Per-layer caching** | 2-3× hit rate | 28 independent hot-stores instead of 1 global |

---

## Quick Reference

```bash
# Build CPU-fallback engine
cd 1-golden && make

# Run all 47 tests
cd 1-golden && make test

# Build with Metal (requires Xcode)
cd 1-golden && make metal

# Generate test GGUF and run inference
python3 /tmp/create_tiny_gguf.py /tmp/tiny_model.gguf
cd 1-golden && ./ornith --model /tmp/tiny_model.gguf -p "Hello" -n 32

# Benchmark mode
cd 1-golden && ./ornith --model /tmp/tiny_model.gguf -b -n 10

# Run Python prototype tests
cd 0-proto && python3 test_suite.py --test all

# SSD benchmark
fio --name=seq_read --rw=read --bs=1M --size=4G
```
