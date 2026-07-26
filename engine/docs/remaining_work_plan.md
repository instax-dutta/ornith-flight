# Remaining Work Plan: Running Real Inference on 8 GB M2

> **Updated:** July 26, 2026
> **Tests:** 69/69 passing (CPU + Metal)
> **Model:** Ornith 1.0 35B (20 GB GGUF at `models/ornith-1.0-35b-Q4_K_M.gguf`)

---

## Current State Summary

| Phase | Feature | Status | Tests |
|-------|---------|--------|-------|
| A | GGUF parser + MoE forward + Metal kernels | ✅ Done | 7 + 6 = 13 |
| B | Q4_K/Q6_K expert slice dequant | ✅ Done | +2 (16 gguf) |
| C | Async I/O prefetch (pthread pool) | ✅ Done | +3 (12 memory) |
| D | Streaming per-layer I/O (madvise + pread) | ✅ Done | — |
| E | SSM layer forward pass + norm | ✅ Done | +1 (8 model) |
| F | BPE tokenizer wiring (no double mmap, byte lookup) | ✅ Done | — |
| — | Inference loop (KV cache, decode, timing) | ✅ Done | 6 |
| — | CLI (--dry-run, -v, -b, -n, -p) | ✅ Done | 7 |
| — | Metal GPU shaders (matmul, RMSNorm, RoPE, SiLU) | ✅ Done | 6 |
| **Total** | | **69/69** | |

---

## Memory Budget Analysis

### Peak Memory per Component

| Component | Memory |
|-----------|--------|
| OS + other apps | ~2 GB |
| GGUF metadata mmap (freed by madvise) | ~1 MB |
| Tokenizer (248k vocab) | ~36 MB |
| K/V caches (40 layers × 2048 pos) | ~336 MB |
| SSM conv/h states (40 layers) | ~4 MB |
| Per-layer compute (temp, freed each layer) | ~50 MB |
| Expert cache (50 hot + 16 LRU) | ~198 MB |
| **Total** | **~2.6 GB — fits in 8 GB** |

---

## Next Steps

### Step 1: Run Real Inference

```bash
cd engine
./ornith --model ../models/ornith-1.0-35b-Q4_K_M.gguf -v -n 1
```

**What to expect**: Config should print immediately (same as `--dry-run -v`). Then:
1. `model_load()` creates inference engine (allocates KV caches)
2. Tokenizer opens from model's GGUF handle (no second mmap)
3. `inference_generate()` encodes prompt "Hello" → token IDs
4. Per-layer forward pass (40 layers): pread + dequant + compute + free
5. LM head evaluates logits → sample token → decode → print

**If it still hangs**: The culprit is likely `inference_init()` allocating 336 MB of KV caches via calloc, or the overhead of the first `ensure_layer_weights()` call (40 layers × multiple pread ops).

### Step 2: Debug if Needed

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| No output at all | Binary issue or stdout buffering | Add `fflush(stdout)` before key operations |
| Hangs during tokenizer load | Large vocab string parsing (248k tokens) | Optimize string processing in `tokenizer_create_from_gguf` |
| Hangs during KV cache alloc | 336 MB calloc on memory-constrained system | Reduce `KV_CACHE_MAX_LEN` or lazy-allocate per layer |
| Hangs during first forward pass | ~12 MB pread per layer × 40 layers | Check `pread()` return values, add timeout |

### Step 3: Performance Optimization

| Optimization | Expected Gain | Effort |
|-------------|--------------|--------|
| Metal GPU for matmul | 2-3x (GPU vs CPU) | Medium — ensure `gpu_matmul_vec_wrap` works end-to-end |
| LM head optimization | ~10x (streaming vs full eval) | Low — already streaming 1 row at a time via pread |
| Reduce KV cache size | ~336 MB → ~84 MB (4x reduction) | Low — trim KV_CACHE_MAX_LEN to 512 |
| Tune prefetch config | Latency hiding | Low — adjust lookahead_layers, async_io_threads |

### Step 4: Functional Correctness

- [ ] Verify output text is valid (tokenizer decode works)
- [ ] Compare with llama.cpp output for same prompt
- [ ] Test temperature=0 (deterministic) vs temperature=0.7
- [ ] Test multi-token generation (e.g., `-n 10`)
- [ ] Test with custom prompt (`-p "The capital of France is"`)

---

## Potential Blockers and Solutions

### Blocker 1: 20 GB mmap still causes swap under load
Even with `madvise(MADV_DONTNEED)`, macOS may page-in parts of the mmap. **Already solved**: All tensor data access uses `pread()`, not mmap. Only metadata (few MB) stays mapped.

### Blocker 2: CPU matmul is too slow (~550M MACs/token)
LM head dominates at 508M MACs. The per-layer matmuls (attention/SSM + MoE) add ~50M MACs. On M2 CPU (~100 GFLOP/s), expect ~5-10 ms per token. Metal GPU should improve this.

### Blocker 3: Second GGUF open for tokenizer causes hang
**Already solved in Phase F**: Tokenizer now uses `model_get_gguf()` — no second 20 GB mmap.

### Blocker 4: KV cache at 2048 positions overflows for long contexts
For short prompts (< 100 tokens), 2048 is fine. Fix: dynamic KV cache expansion or ring buffer.

---

## Command Reference

```bash
# Dry run (config only, no memory alloc)
./ornith --dry-run -v --model ../models/ornith-1.0-35b-Q4_K_M.gguf

# Verbose inference (should print config + per-layer progress)
./ornith -v --model ../models/ornith-1.0-35b-Q4_K_M.gguf -n 1

# Benchmark mode (3 runs, timing stats)
./ornith -b -n 3 --model ../models/ornith-1.0-35b-Q4_K_M.gguf

# Custom prompt
./ornith -p "The meaning of life is" -n 20 --model ../models/ornith-1.0-35b-Q4_K_M.gguf
```
