# Prototype Phase Findings

## From Simulation (bench/simulate.py)

### MacBook Air M2 (8GB, 256GB SSD)

| Hit Rate | TTFT | Decode TPS | Cache Fit |
|---|---|---|---|
| 50% | 6.2s | 0.4 tok/s | No |
| 70% | 5.3s | 0.7 tok/s | No |
| 85% | 4.5s | 1.4 tok/s | Marginal |
| 95% | 4.1s | 4.1 tok/s | OK |
| 99% | 3.9s | 20.6 tok/s | Yes |

### PC (i9-9900K, RTX 4060, 16GB RAM)

| Hit Rate | TTFT | Decode TPS | Cache Fit |
|---|---|---|---|
| 95% | 1.8s | 4.1 tok/s | Yes |
| 99% | 1.6s | 20.6 tok/s | Yes |

## Key Insights

1. **95%+ hit rate is the threshold for usability** (>3 tok/s)
2. **M2 likely achieves 85-92% typical** (only ~66 expert slots with 4GB LRU)
3. **PC likely achieves 92-98%** (bigger cache, ~99 expert slots with 6GB LRU)
4. **Hot-store pinning** (top 50 experts) transforms the hit rate curve
5. **PC advantage is not raw speed — it's cache size** (16GB RAM vs 8GB)
6. **SSD bandwidth is the bottleneck** everywhere (3 GB/s)

## What to Validate in Python Before C

1. Real routing distributions (are expert loads power-law?)
2. LRU cache hit rate over real Ornith routing traces
3. int4 quantization accuracy impact on expert outputs
4. Lookahead prefetch schedule (eager vs lazy vs hybrid)
5. Hot-store candidate identification from imatrix-like profiling
