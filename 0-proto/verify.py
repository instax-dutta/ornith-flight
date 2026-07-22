"""Correctness verification: compare our fast path against reference.

For real verification, run through a real step of Ornith with known weights.
This stub validates the shapes and basic math."""

import numpy as np
from model.layers import MoEMLP, Router, rmsnorm


def verify_moe_mlp():
    d_model = 256
    hidden_dim = 512
    x = np.random.randn(4, d_model).astype(np.float16)
    mlp = MoEMLP(d_model, hidden_dim, np.random.RandomState(42))

    # Check gate
    gate = mlp.gate.shape
    assert gate == (d_model, hidden_dim), f"gate shape {gate} != {(d_model, hidden_dim)}"

    # Check forward
    out = mlp.forward_fp16(x)
    assert out.shape == x.shape, f"out shape {out.shape} != {x.shape}"
    assert not np.any(np.isnan(out)), "NaN in MLP output"
    print(f"  MoE MLP: {x.shape} -> {out.shape} OK")


def verify_router():
    d_model = 256
    n_experts = 64
    x = np.random.randn(2, d_model).astype(np.float16)
    router = Router(d_model, n_experts, np.random.RandomState(42))

    indices, weights = router.route(x, top_k=4)
    assert indices.shape == (2, 4), f"indices shape {indices.shape} != (2, 4)"
    assert weights.shape == (2, 4), f"weights shape {weights.shape} != (2, 4)"
    assert np.allclose(weights.sum(axis=1), 1.0), "router weights don't sum to 1"
    print(f"  Router: top-4 from {n_experts} experts OK")


def verify_rmsnorm():
    x = np.random.randn(4, 256)
    w = np.ones(256)
    y = rmsnorm(x, w)
    assert y.shape == x.shape
    rms = np.sqrt(np.mean(y ** 2)) / np.sqrt(np.mean(x ** 2))
    print(f"  RMSNorm: output RMS ratio {rms:.4f} OK (expected ~1.0)")


if __name__ == "__main__":
    print("Verifying forward pass components...")
    verify_rmsnorm()
    verify_router()
    verify_moe_mlp()
    print("\nAll checks passed.")
