# ornith-flight

A Colibri-inspired inference engine for running Ornith 35B MoE on weak laptops/PCs via expert streaming from disk.

## Target Hardware

| Device | Chip | RAM | Storage | GPU |
|---|---|---|---|---|
| MacBook Air M2 | Apple M2 | 8 GB unified | 256 GB SSD | Metal (unified memory) |
| Windows PC | i9-9900K | 16 GB | - | RTX 4060 8GB (CUDA) |

## Phases

| Phase | Directory | Goal |
|---|---|---|
| 0 - Prototype | `0-proto/` | Validate architecture in Python before writing C |
| 1 - Golden | `1-golden/` | C engine with Metal backend for Mac |
| 2 - CUDA | `2-cuda/` | Port to CUDA for PC |

## Key Numbers

| Metric | MacBook M2 | PC RTX 4060 |
|---|---|---|
| Non-routed resident | ~1.5 GB (int8) | 1.5 GB in VRAM |
| Expert cache | ~4 GB LRU | ~6 GB in sys RAM |
| Cold TTFT | ~5 sec | ~2 sec |
| Steady TPS (warm) | 3-8 tok/s | 10-20 tok/s |
