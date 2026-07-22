# ornith-flight

A Colibri-inspired inference engine for running Ornith 35B MoE on consumer hardware via expert streaming from disk.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: Google](https://img.shields.io/badge/code%20style-google-blueviolet.svg)](https://google.github.io/styleguide/pyguide.html)

## Overview

Ornith-flight enables running the 35B parameter Ornith MoE model on weak consumer hardware (8-16GB RAM) by streaming experts from disk instead of keeping all 7,168 experts in memory. Inspired by [Colibri](https://github.com/OpenBMB/MiniCPM) for GLM-5.2.

### Target Hardware

| Device | Chip | RAM | Storage | GPU | Expected Performance |
|--------|------|-----|---------|-----|---------------------|
| MacBook Air M2 | Apple M2 | 8 GB | 256 GB SSD | Metal | 0.26 tok/s, 5.5s TTFT |
| Gaming PC | i9-9900K | 16 GB | NVMe SSD | RTX 4060 8GB | 0.30 tok/s, 2.5s TTFT |

## Key Features

- **Expert Streaming:** Load only active experts (66-99 of 7,168) from disk on-demand
- **Two-Tier Caching:** Hot-store (pinned) + LRU cache achieves 11-18% hit rate
- **Aggressive Quantization:** int4 routed experts, int8 non-routed weights
- **Async Prefetch:** Lookahead prefetching overlaps I/O with GPU compute
- **Test-Backed Optimization:** All parameters validated through comprehensive simulation

## Project Structure

```
ornith-flight/
├── 0-proto/              # Python prototype & parameter optimization
│   ├── constants.py      # Configuration constants
│   ├── utils.py          # Helper utilities
│   ├── tuner.py          # Core tuning algorithms
│   ├── reporters.py      # Output formatting
│   ├── test_suite.py     # Comprehensive validation tests
│   ├── tune_parameters.py # Parameter optimization CLI
│   ├── bench/            # Performance simulation
│   ├── model/            # Model architecture definitions
│   ├── streaming/        # Cache and prefetch implementations
│   └── docs/             # Detailed documentation
├── 1-golden/             # C implementation (Metal backend)
│   └── docs/             # Architecture documentation
├── 2-cuda/               # CUDA implementation (future)
└── README.md             # This file
```

## Quick Start

### Prerequisites

```bash
python3 --version  # 3.8 or higher required
```

### Installation

```bash
git clone https://github.com/yourusername/ornith-flight.git
cd ornith-flight
cd 0-proto
```

No additional dependencies needed for simulation - uses only Python standard library.

### Running Parameter Optimization

```bash
# Full optimization suite for your device
python3 tune_parameters.py --device m2    # For MacBook Air M2
python3 tune_parameters.py --device pc    # For gaming PC

# Individual component tuning
python3 tune_parameters.py --device m2 --component cache
python3 tune_parameters.py --device m2 --component hotstore
python3 tune_parameters.py --device m2 --component power
python3 tune_parameters.py --device m2 --component quant
python3 tune_parameters.py --device m2 --component ssd
```

### Running Tests

```bash
# Run all tests
python3 test_suite.py --test all

# Run specific tests
python3 test_suite.py --test cache        # Cache hit rate validation
python3 test_suite.py --test power-law    # Routing distribution
python3 test_suite.py --test hotstore     # Hot-store effectiveness
python3 test_suite.py --test prefetch     # Prefetch strategies
python3 test_suite.py --test quant        # Quantization trade-offs
python3 test_suite.py --test edge         # Edge case handling
```

## Documentation

- **[Quick Reference](0-proto/QUICK_REFERENCE.md)** - Parameter lookup guide
- **[Optimization Summary](0-proto/OPTIMIZATION_SUMMARY.md)** - Complete optimization report
- **[Optimized Parameters](0-proto/OPTIMIZED_PARAMETERS.md)** - Detailed analysis
- **[Architecture](1-golden/docs/architecture.md)** - System design
- **[Code Review Fixes](0-proto/CODE_REVIEW_FIXES.md)** - Implementation notes

## Performance Results

### MacBook Air M2 (8GB RAM)
- **Cache:** 4GB (50 hot-store + 16 LRU experts)
- **Quantization:** int4 routed, int8 non-routed
- **Hit Rate:** 11-13% (warm)
- **Performance:** 0.26 tok/s, 5.5s TTFT
- **Memory:** 5.6 GB total footprint

### PC (RTX 4060, 16GB RAM)
- **Cache:** 6GB (50 hot-store + 49 LRU experts)
- **Quantization:** int4 routed, int8 non-routed
- **Hit Rate:** 14-18% (warm)
- **Performance:** 0.30 tok/s, 2.5s TTFT
- **Memory:** 7.6 GB total footprint

## Current Status

| Phase | Status | Description |
|-------|--------|-------------|
| 0 - Prototype | ✅ Complete | Python simulation & parameter optimization |
| 1 - Golden | ⬜ Not Started | C engine with Metal backend |
| 2 - CUDA | ⬜ Not Started | CUDA backend for NVIDIA GPUs |

## Development Phases

### Phase 0: Prototype (Complete)
Python-based simulation and parameter optimization. All parameters validated through comprehensive testing.

**Key Achievements:**
- Test-backed parameter optimization
- Realistic performance modeling (11-18% hit rates)
- Comprehensive edge case coverage
- Production-ready configurations

### Phase 1: Golden (Next)
Pure C inference engine with Metal backend for macOS.

**Planned Features:**
- GGUF memory-mapped file loading
- Two-tier cache (hot-store + LRU)
- Async I/O with lookahead prefetch
- Metal kernels for int4 dequant + matmul

### Phase 2: CUDA (Future)
Port to CUDA for NVIDIA GPUs.

## Benchmarking Your Hardware

```bash
# Benchmark SSD sequential read speed
fio --name=seq_read --rw=read --bs=1M --size=4G --numjobs=1

# Target: 3+ GB/s minimum, 7+ GB/s ideal
```

## Architecture Highlights

### Memory Hierarchy
```
T0 - RESIDENT (1.5 GB in RAM/VRAM)
  └─ Non-routed weights (embeddings, attention, norms, LM head)

T1 - HOT-STORE (3.1 GB pinned in RAM)
  └─ Top 50 most-used experts (never evicted)

T2 - LRU CACHE (1-3 GB in RAM)
  └─ Recently-used experts (dynamic eviction)

T3 - DISK (16 GB mmap'd GGUF)
  └─ All 7,168 experts at int4
```

### Async I/O Pipeline
```
Token N decode:
  [GPU: layer 0] → [GPU: layer 1] → [GPU: layer 2] → ...
       ↓ prefetch      ↓ prefetch      ↓ prefetch
  [I/O: layer 1]  [I/O: layer 2]  [I/O: layer 3]
```

## Contributing

Contributions welcome! Areas of interest:
- C implementation (Phase 1)
- CUDA backend (Phase 2)
- Per-layer caching strategies
- Expert pruning techniques
- Better quantization methods (3-bit, 2-bit)

Please follow [Google's code review guidelines](https://google.github.io/eng-practices/review/).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use this work, please cite:

```bibtex
@software{ornith_flight_2026,
  title = {Ornith-Flight: Expert Streaming for Large MoE Models},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/ornith-flight}
}
```

## Acknowledgments

- Inspired by [Colibri](https://github.com/OpenBMB/MiniCPM) expert streaming approach
- Based on [Ornith 35B MoE](https://huggingface.co/ornith) architecture
- Built on [GGUF](https://github.com/ggerganov/llama.cpp) quantization format

## Related Projects

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - LLM inference in C/C++
- [Colibri](https://github.com/OpenBMB/MiniCPM) - Expert streaming for GLM
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm) - Universal LLM deployment

---

**Status:** Prototype complete, C implementation ready to begin.
