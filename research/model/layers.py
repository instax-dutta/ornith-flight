"""MoE layer and forward pass prototype in numpy."""

import numpy as np


def rmsnorm(x, weight, eps=1e-6):
    """RMS normalization."""
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def silu(x):
    """SiLU activation."""
    return x / (1 + np.exp(-x))


class MoEMLP:
    """Routed expert MLP with SiLU-gated activation."""

    def __init__(self, d_model, hidden_dim, rng):
        self.gate = rng.randn(d_model, hidden_dim).astype(np.float16)
        self.up = rng.randn(d_model, hidden_dim).astype(np.float16)
        self.down = rng.randn(hidden_dim, d_model).astype(np.float16)

    def forward_fp16(self, x):
        x = x.astype(np.float16)
        gate = silu(x @ self.gate)
        up = x @ self.up
        return (gate * up) @ self.down


class SharedExpert:
    """Shared expert used by every token."""

    def __init__(self, d_model, hidden_dim, rng):
        self.gate = rng.randn(d_model, hidden_dim).astype(np.float16)
        self.up = rng.randn(d_model, hidden_dim).astype(np.float16)
        self.down = rng.randn(hidden_dim, d_model).astype(np.float16)

    def forward_fp16(self, x):
        return MoEMLP.forward_fp16(self, x)


class Router:
    """Top-k expert router."""

    def __init__(self, d_model, n_experts, rng):
        self.weight = rng.randn(d_model, n_experts).astype(np.float16)

    def route(self, x, top_k=8):
        logits = x.astype(np.float16) @ self.weight
        scores = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        scores = scores / np.sum(scores, axis=-1, keepdims=True)
        indices = np.argsort(-scores, axis=-1)[:, :top_k]
        weights = np.take_along_axis(scores, indices, axis=-1)
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return indices, weights


class Attention:
    """Hybrid attention: full attention + Gated DeltaNet."""

    def __init__(self, d_model, n_heads, n_kv_heads, rng):
        self.q_proj = rng.randn(d_model, n_heads * 128).astype(np.float16)
        self.k_proj = rng.randn(d_model, n_kv_heads * 128).astype(np.float16)
        self.v_proj = rng.randn(d_model, n_kv_heads * 128).astype(np.float16)
        self.o_proj = rng.randn(n_heads * 128, d_model).astype(np.float16)
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads

    def forward(self, x, kv_cache=None):
        q = x @ self.q_proj
        k = x @ self.k_proj
        v = x @ self.v_proj
        # Full attention on first token, DeltaNet recurrence on subsequent
        if kv_cache is None:
            return self._full_attention(q, k, v)
        else:
            return self._deltanet(q, k, v, kv_cache)

    def _full_attention(self, q, k, v):
        scale = 1.0 / np.sqrt(q.shape[-1])
        attn = (q @ k.T) * scale
        attn = np.exp(attn - np.max(attn, axis=-1, keepdims=True))
        attn = attn / np.sum(attn, axis=-1, keepdims=True)
        return attn @ v

    def _deltanet(self, q, k, v, state):
        # Simplified Gated DeltaNet recurrence
        gate = 0.9
        state = gate * state + (1 - gate) * (k.T @ v)
        return q @ state


class MoELayer:
    """Single MoE layer with attention + shared expert + routed experts."""

    def __init__(self, layer_id, config, rng):
        self.layer_id = layer_id
        self.router = Router(config.d_model, config.n_experts, rng)
        self.shared_expert = SharedExpert(config.d_model, config.expert_hidden_dim, rng)
        self.attention = Attention(config.d_model, config.n_heads, config.n_kv_heads, rng)
        self.input_norm = rng.randn(config.d_model).astype(np.float16)
        self.post_attn_norm = rng.randn(config.d_model).astype(np.float16)

    def forward(self, x, experts, kv_cache=None, top_k=8):
        # Attention with residual
        residual = x
        x = rmsnorm(x, self.input_norm)
        attn_out = self.attention.forward(x, kv_cache)
        x = residual + attn_out @ self.attention.o_proj

        # MoE with residual
        residual = x
        x = rmsnorm(x, self.post_attn_norm)
        shared_out = self.shared_expert.forward_fp16(x)

        indices, weights = self.router.route(x, top_k=top_k)
        routed_out = np.zeros_like(x)
        for i, idx in enumerate(indices[0]):
            if idx in experts:
                expert_out = experts[idx].forward_fp16(x)
                routed_out += weights[0, i] * expert_out

        x = residual + shared_out + routed_out
        return x
