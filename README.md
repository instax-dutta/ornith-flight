<p align="center">
<img src="ornith_flight_logo.png" width="320" alt="ornith-flight — expert-streaming inference engine">
</p>
<p align="center">
<a href="https://github.com/instax-dutta/ornith-flight"><img src="https://img.shields.io/badge/repo-ornith--flight-2ea043" alt="ornith-flight"></a>
<a href="https://github.com/instax-dutta/ornith-flight/releases/tag/v0.1.0"><img src="https://img.shields.io/github/v/release/instax-dutta/ornith-flight" alt="Release v0.1.0"></a>
<a href="STATUS.md"><img src="https://img.shields.io/badge/tests-69%2F69-passing-brightgreen" alt="Tests 69/69 passing"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-yellow" alt="License MIT"></a>
<a href="https://github.com/instax-dutta/ornith-flight"><img src="https://img.shields.io/github/stars/instax-dutta/ornith-flight?style=social" alt="Stars"></a>
</p>
<p align="center">
<a href="https://github.com/instax-dutta/ornith-flight"><b>GitHub</b></a> ·
<a href="https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF"><b>Model</b></a> ·
<a href="PRIORITY.md"><b>Roadmap</b></a>
</p>

**ornith-flight** is a C99 inference engine for [Ornith 1.0 35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF), a 256-expert MoE model with hybrid attention (GQA + SSM). It streams experts from SSD to RAM, making 35B-parameter inference feasible on **8 GB consumer hardware** with zero dependencies.

Inspired by [Colibri's](https://github.com/JustVugg/colibri) expert streaming approach.

---

## Key Achievements

- **20 GB model on 8 GB hardware** — Ornith 35B loads and runs on an 8 GB MacBook Air M2 via per-layer streaming I/O, with peak memory under 3 GB.
- **On-demand expert dequant** — Q4_K/Q6_K fused 3D tensors (256 experts per layer) are never fully materialized. Individual expert slices are extracted per-super-block, saving ~3 GB per layer.
- **Hybrid forward pass** — Full GQA attention (every 4th layer) and SSM/Mamba recurrence (30 of 40 layers) both execute correctly, with top-8 expert routing via learned router.
- **Dual GPU backend** — CPU fallback and Metal Performance Shaders backends with 69 unit tests passing across both.
- **Async I/O prefetch** — Background thread pool overlaps per-layer weight loading with computation, keeping forward pass I/O-bound instead of memory-bound.

---

## The Model: Ornith 1.0 35B

| Spec | Value |
|------|-------|
| Architecture | `qwen35moe` (MoE) |
| Parameters | ~35B total / ~3.7B active per token |
| Layers | 40 — every 4th: full attention, rest: SSM (Mamba-style) |
| Hidden dim | 2048 |
| Heads | 16 Q × 128 dim → 2 KV heads (GQA) |
| Experts | 256 per layer, 8 active (top-8 routing) + 1 shared expert |
| Vocab | 248,320 tokens (BPE) |
| Context | 262,144 tokens |
| Model file | 20 GB GGUF (Q4_K_M / Q6_K quant) |

---

## How It Works

A 20 GB MoE model cannot fit in 8 GB RAM. Instead of loading all experts at once, the engine keeps a small resident cache and streams from SSD on demand:

```
Token → Router selects 8/256 experts
         │
         ├── Hot-store hit?   → Use immediately (~0.1 µs)
         ├── LRU cache hit?   → Use immediately (~0.5 µs)
         └── SSD miss?        → Dequant expert slice from mmap'd Q4_K blocks
                                (no full-tensor materialization)
```

This design keeps peak memory at ~2.6 GB — well within the 8 GB budget.

---

## Engine Status

| Module | Tests | Status |
|--------|-------|--------|
| GGUF v3 parser + Q4_K/Q6_K dequant + expert slice | 16/16 | ✅ |
| GPU abstraction (CPU fallback) | 7/7 | ✅ |
| Metal GPU shaders (matmul, RMSNorm, RoPE, SiLU) | 6/6 | ✅ |
| Model forward pass (attention, SSM, MoE routing) | 8/8 | ✅ |
| Inference loop (KV cache, prefill, decode) | 6/6 | ✅ |
| Memory manager (LRU cache, hot-store, async prefetch) | 12/12 | ✅ |
| BPE tokenizer (encode/decode, byte lookup) | 7/7 | ✅ |
| CLI argument parsing | 7/7 | ✅ |
| **Total** | **69/69** | ✅ |

---

## Getting Started

### Prerequisites

- macOS with Xcode Command Line Tools
- 45 GB free disk space

### Build & Test

```bash
git clone https://github.com/instax-dutta/ornith-flight.git
cd ornith-flight

# Build engine (Metal on macOS, CPU elsewhere)
make

# Run all tests
make test
```

### Download the Model

```bash
pip install huggingface-hub
huggingface-cli login
huggingface-cli download deepreinforce-ai/Ornith-1.0-35B-GGUF \
  ornith-1.0-35b-Q4_K_M.gguf --local-dir models
```

### Inspect Without Loading Weights

```bash
cd engine
./ornith --dry-run -v --model ../models/ornith-1.0-35b-Q4_K_M.gguf
```

---

## Project Structure

```
ornith-flight/
├── engine/          C inference engine (69 tests, zero dependencies)
├── research/        Python research — parameter tuning, cache modeling
├── docs/            Architecture, deployment, contributing guides
├── scripts/         Deployment and automation
├── models/          Downloaded GGUF files (gitignored)
├── Makefile         Top-level convenience targets
├── PRIORITY.md      Development roadmap
└── LICENSE          MIT
```

---

## Upcoming Work

- **GPU MoE acceleration** — move expert matmuls from CPU to Metal backend
- **Tokenizer decode** — proper byte-to-UTF-8 conversion for 248K vocab
- **SSM norm wiring** — RMSNorm on SSM output signal
- **Performance tuning** — LM head optimization, expert count tuning

See [PRIORITY.md](PRIORITY.md) for details.

---

## License

MIT — see [LICENSE](LICENSE).

---

**Model:** [Ornith 1.0 35B GGUF](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B-GGUF)  
**Engine:** 69/69 tests passing, runs on 8 GB hardware via per-layer streaming I/O  
**Repo:** https://github.com/instax-dutta/ornith-flight
