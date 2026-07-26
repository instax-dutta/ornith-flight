# Ornith-Flight — Project Status

> **Updated:** July 2026
> **Tests:** 69/69 passing (CPU + Metal GPU)
> **Model:** Ornith 1.0 35B — 20 GB GGUF (Q4_K_M)

---

## Repository Structure

```
ornith-flight/
├── engine/          C99 inference engine (0 dependencies)
│   ├── src/         Core: model, gguf, gpu, memory, tokenizer, inference
│   ├── test/        8 test modules (69 tests)
│   ├── docs/        Architecture documentation
│   └── Makefile     Build: all, cpu, metal, test, clean
├── research/        Python prototypes — parameter tuning, cache simulation
├── docs/            Project documentation (deployment, contributing)
├── scripts/         Deployment and automation
├── models/          Downloaded GGUF files (gitignored)
├── Makefile         Top-level convenience targets
├── README.md        Public-facing overview
├── PRIORITY.md      Development roadmap
├── STATUS.md        This file
└── LICENSE          MIT
```

---

## Engine Components

| Module | Source | Tests | Description |
|--------|--------|-------|-------------|
| CLI | `engine/src/main.c`, `cli.c` | 7/7 | Argument parsing, interactive mode, benchmark |
| GGUF parser | `engine/src/gguf.c` | 16/16 | v3 GGUF, Q4_K/Q6_K dequant, expert slice extraction |
| GPU (CPU fallback) | `engine/src/gpu.c` | 7/7 | Buffer management, CPU matmul fallback |
| GPU (Metal backend) | `engine/src/gpu_metal.m` | 6/6 | Metal shaders: matmul, RMSNorm, RoPE, SiLU |
| Model forward pass | `engine/src/model.c` | 8/8 | Attention, SSM, MoE routing, sampling |
| Inference loop | `engine/src/inference.c` | 6/6 | KV cache, prefill, decode, timing |
| Memory manager | `engine/src/memory.c` | 12/12 | LRU cache, hot-store, async I/O prefetch |
| Tokenizer | `engine/src/tokenizer.c` | 7/7 | BPE encode/decode, byte→tokenId lookup |
| **Total** | | **69/69** | |

---

## Key Technical Achievements

- **20 GB model runs on 8 GB RAM** via per-layer streaming I/O (`madvise` + `pread`, ~72 MB/layer peak)
- **Hybrid forward pass**: GQA attention (every 4th layer) + SSM/Mamba (30 of 40 layers) + MoE
- **On-demand expert dequant**: Per-super-block element extraction from Q4_K/Q6_K fused 3D tensors — no full-tensor materialization
- **Async I/O prefetch**: Background thread pool overlaps layer weight loading with computation
- **Dual GPU backend**: CPU fallback + Metal Performance Shaders on Apple Silicon

---

## Benchmark Results

First inference pass on MacBook Air M2 (8 GB):

| Metric | Value |
|--------|-------|
| Model loaded | 20 GB GGUF, 733 tensors |
| Architecture verified | `qwen35moe`, 40 layers, 256 experts |
| Prefill time | 141 s (2 prompt tokens) |
| Decode time | 77 s (1 generated token) |
| Peak memory | Well within 8 GB (streaming I/O) |
| Bottleneck | MoE expert path runs on CPU — GPU acceleration pending |

---

## Upcoming Work

- **GPU MoE acceleration** — move expert matmuls from CPU to Metal backend (primary perf target)
- **Tokenizer decode fix** — proper byte-to-UTF-8 for 248K-vocab BPE
- **SSM norm wiring** — RMSNorm on SSM output signal
- **LM head optimization** — 248K × 2048 vocab projection tuning

See [PRIORITY.md](PRIORITY.md) for detailed roadmap.
