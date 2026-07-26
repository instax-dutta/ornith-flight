// SPDX-License-Identifier: MIT
// Metal Shading Language — SiLU-gated MLP, MoE expert MLP, router kernels.

#include <metal_stdlib>
using namespace metal;

// ── SiLU activation ──────────────────────────────────────────────────────────
// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))

kernel void silu_activation(
    device const float *x      [[buffer(0)]],
    device       float *output [[buffer(1)]],
    constant     int   &n      [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;
    float v = x[tid];
    output[tid] = v / (1.0f + exp(-v));
}

// ── SiLU-gated multiply ──────────────────────────────────────────────────────
// out = silu(gate) * up  (element-wise)
// This is the core gating of the MoE MLP.

kernel void silu_mul(
    device const float *gate   [[buffer(0)]],
    device const float *up     [[buffer(1)]],
    device       float *output [[buffer(2)]],
    constant     int   &n      [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;
    float g = gate[tid];
    float silu = g / (1.0f + exp(-g));
    output[tid] = silu * up[tid];
}

// ── Expert MLP forward (single expert, float32) ─────────────────────────────
// out = (silu(x @ gate_w) * (x @ up_w)) @ down_w
// Where:
//   x:       [d_model]
//   gate_w:  [d_model × hidden_dim]
//   up_w:    [d_model × hidden_dim]
//   down_w:  [hidden_dim × d_model]
//   output:  [d_model]
//
// This is a large kernel — each thread computes one element of the output.

kernel void expert_mlp_forward(
    device const float *x        [[buffer(0)]],  // [d_model]
    device const float *gate_w   [[buffer(1)]],  // [d_model, hidden_dim]
    device const float *up_w     [[buffer(2)]],  // [d_model, hidden_dim]
    device const float *down_w   [[buffer(3)]],  // [hidden_dim, d_model]
    device       float *output   [[buffer(4)]],  // [d_model]
    constant     int   &d_model  [[buffer(5)]],
    constant     int   &hidden   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)d_model) return;

    // Thread tid computes output[tid]
    // First compute the intermediate hidden vector h (size hidden_dim)
    // h[j] = silu(x · gate_w[:,j]) * (x · up_w[:,j])
    // Then output[tid] = sum_j h[j] * down_w[j][tid]

    // Since we only have one thread per output element, we must iterate over
    // the full hidden dimension. For production, use a tile-based approach.
    float sum = 0.0f;
    for (int j = 0; j < hidden; j++) {
        // Compute gate_gate = x · gate_w[:,j]
        float gate_gate = 0.0f;
        float up_gate   = 0.0f;
        for (int i = 0; i < d_model; i++) {
            float xi = x[i];
            gate_gate += xi * gate_w[i * hidden + j];  // row-major: [i][j]
            up_gate   += xi * up_w[i * hidden + j];
        }

        // SiLU gate
        float silu_g = gate_gate / (1.0f + exp(-gate_gate));
        float h_j = silu_g * up_gate;

        // Accumulate into output[tid] via down_w
        sum += h_j * down_w[j * d_model + tid];
    }
    output[tid] = sum;
}

// ── Router forward ──────────────────────────────────────────────────────────
// logits = x × router_weight
// x: [d_model], router_weight: [d_model × n_experts]
// logits: [n_experts] — one thread per expert

kernel void router_forward(
    device const float *x              [[buffer(0)]],  // [d_model]
    device const float *router_weight  [[buffer(1)]],  // [d_model, n_experts]
    device       float *logits         [[buffer(2)]],  // [n_experts]
    constant     int   &d_model        [[buffer(3)]],
    constant     int   &n_experts      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n_experts) return;

    float sum = 0.0f;
    for (int i = 0; i < d_model; i++) {
        sum += x[i] * router_weight[i * n_experts + tid];
    }
    logits[tid] = sum;
}

// ── Top-k softmax ────────────────────────────────────────────────────────────
// Select top-k logits and apply softmax over them.
// logits: [n_experts] (will be modified)
// indices: [top_k] output
// weights: [top_k] output

kernel void topk_softmax(
    device       float   *logits   [[buffer(0)]],  // in/out: [n_experts]
    device       uint    *indices  [[buffer(1)]],  // out: [top_k]
    device       float   *weights  [[buffer(2)]],  // out: [top_k]
    constant     int     &n_experts [[buffer(3)]],
    constant     int     &top_k    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1) return;  // single-threaded (simplified — use parallel sort for prod)

    // Build index array
    // NOTE: in production, use a parallel radix sort for top-k.
    // For now, a simple sequential selection.

    // Copy scores and build index pairs
    float scores[1024];  // max 1024 experts
    int idx[1024];
    int n = n_experts < 1024 ? n_experts : 1024;

    for (int i = 0; i < n; i++) {
        scores[i] = logits[i];
        idx[i] = i;
    }

    // Simple bubble-sort (descending) — use proper sort in production
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (scores[j] > scores[i]) {
                float tmp_s = scores[i]; scores[i] = scores[j]; scores[j] = tmp_s;
                int tmp_i = idx[i]; idx[i] = idx[j]; idx[j] = tmp_i;
            }
        }
    }

    // Softmax over top-k
    float max_val = scores[0];
    float sum_exp = 0.0f;
    int k = top_k < n ? top_k : n;

    for (int i = 0; i < k; i++) {
        float e = exp(scores[i] - max_val);
        weights[i] = e;
        indices[i] = (uint)idx[i];
        sum_exp += e;
    }
    for (int i = 0; i < k; i++) {
        weights[i] /= sum_exp;
    }
}

// The shared (dense) expert uses the same forward pass as a routed expert.
// Use expert_mlp_forward for both; they share the same kernel structure.
// (shared_expert_forward is identical to expert_mlp_forward — aliased at
//  the pipeline state level if separate PSO names are needed.)
