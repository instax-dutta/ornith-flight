# Development Roadmap

> **ornith-flight** — Expert-streaming inference engine for Ornith 1.0 35B MoE.

---

## Current Status

The engine successfully loads and runs the full Ornith 1.0 35B model (20 GB GGUF) on 8 GB MacBook Air M2 hardware. Key capabilities demonstrated:

- **Per-layer streaming I/O** with madvise + pread (~72 MB peak per layer)
- **Hybrid forward pass**: GQA attention (every 4th layer) + SSM/Mamba (30 of 40 layers)
- **MoE routing**: Top-8 expert selection from 256 per layer, on-demand slice dequant
- **Dual backend**: CPU fallback + Metal GPU (shaders for matmul, RMSNorm, RoPE, SiLU)
- **69/69 unit tests** passing across both backends

---

## Upcoming Work

### GPU Acceleration of MoE Expert Path
The routed expert path currently runs entirely on CPU (dequant + matmul). Moving these operations to the Metal GPU backend is the primary performance target.

### Tokenizer Decoder Fix
Byte-token decoding for the 248K-vocab BPE tokenizer requires proper `<0xNN>` byte-to-UTF-8 conversion.

### Additional Targets
- SSM norm wiring (RMSNorm on SSM output signal)
- LM head optimization (248K × 2048 vocab projection)
- Active expert count tuning (quality vs speed tradeoff)

---

*For detailed project status, see [STATUS.md](STATUS.md).*
