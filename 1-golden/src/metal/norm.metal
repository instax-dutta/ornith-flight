// SPDX-License-Identifier: MIT
// Metal Shading Language — RMSNorm and RoPE rotary position embedding kernels.

#include <metal_stdlib>
using namespace metal;

// ── RMSNorm ──────────────────────────────────────────────────────────────────
// y = x / sqrt(mean(x^2) + eps) * weight
// rows: number of vectors to normalize (batch × seq_len)
// dim:  dimension of each vector

kernel void rmsnorm_forward(
    device const float *x       [[buffer(0)]],  // [rows × dim]
    device const float *weight  [[buffer(1)]],  // [dim]
    device       float *output  [[buffer(2)]],  // [rows × dim]
    constant     int   &rows    [[buffer(3)]],
    constant     int   &dim     [[buffer(4)]],
    constant     float &eps     [[buffer(5)]],  // 1e-6
    uint2 pos [[thread_position_in_grid]]
) {
    int row = (int)pos.x;
    int col = (int)pos.y;
    if (row >= rows || col >= dim) return;

    // Each thread handles one element of output[row][col]
    // First compute RMS across the row using threadgroup memory

    // Compute sum of squares for this row
    // We need cross-thread reduction within a row.
    // For simplicity: each thread independently sums the full row's squares.
    // (Production: use threadgroup reduce for better perf)

    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float v = x[row * dim + d];
        sum_sq += v * v;
    }

    float rms = sqrt(sum_sq / (float)dim + eps);
    output[row * dim + col] = (x[row * dim + col] / rms) * weight[col];
}

// ── Single-row RMSNorm (for decode: one vector at a time) ───────────────────

kernel void rmsnorm_single(
    device const float *x       [[buffer(0)]],  // [dim]
    device const float *weight  [[buffer(1)]],  // [dim]
    device       float *output  [[buffer(2)]],  // [dim]
    constant     int   &dim     [[buffer(3)]],
    constant     float &eps     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)dim) return;

    // Compute sum of squares across the vector
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) {
        float v = x[d];
        sum_sq += v * v;
    }

    float rms = sqrt(sum_sq / (float)dim + eps);
    output[tid] = (x[tid] / rms) * weight[tid];
}

// ── RoPE (rotary position embedding) ────────────────────────────────────────
// Applies rotary embeddings to Q and K tensors.
// For each pair of dimensions (2d, 2d+1):
//   q[2d]     = q[2d]   * cos(θ) - q[2d+1] * sin(θ)
//   q[2d+1]   = q[2d]   * sin(θ) + q[2d+1] * cos(θ)
// Where θ = pos / base^(2d/head_dim)
//
// pos:      position in sequence
// head_dim: dimension of each head (must be even)
// base:     RoPE frequency base (typically 10000.0)

kernel void rope_forward(
    device       float *Q        [[buffer(0)]],  // [seq_len, n_heads, head_dim]
    device       float *K        [[buffer(1)]],  // [seq_len, n_kv_heads, head_dim]
    constant     int   &seq_len  [[buffer(2)]],
    constant     int   &head_dim [[buffer(3)]],
    constant     int   &n_heads  [[buffer(4)]],
    constant     int   &n_kv_heads [[buffer(5)]],
    constant     float &base     [[buffer(6)]],
    constant     int   &pos_start [[buffer(7)]],  // starting position (0 for prefill)
    uint3 pos [[thread_position_in_grid]]
) {
    int seq_pos = (int)pos.x;
    int head    = (int)pos.y;
    int d2      = (int)pos.z;  // dimension pair index (0..head_dim/2)

    if (seq_pos >= seq_len) return;
    int actual_pos = pos_start + seq_pos;

    int d = 2 * d2;
    if (d >= head_dim) return;

    // Compute θ = pos / base^(2d/head_dim)
    float theta = (float)actual_pos / pow(base, (float)(2 * d2) / (float)head_dim);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Apply to Q
    int q_offset = seq_pos * (n_heads * head_dim) + head * head_dim + d;
    float q0 = Q[q_offset];
    float q1 = Q[q_offset + 1];
    Q[q_offset]     = q0 * cos_theta - q1 * sin_theta;
    Q[q_offset + 1] = q0 * sin_theta + q1 * cos_theta;

    // Apply to K (if head index < n_kv_heads)
    if (head < n_kv_heads) {
        int k_offset = seq_pos * (n_kv_heads * head_dim) + head * head_dim + d;
        float k0 = K[k_offset];
        float k1 = K[k_offset + 1];
        K[k_offset]     = k0 * cos_theta - k1 * sin_theta;
        K[k_offset + 1] = k0 * sin_theta + k1 * cos_theta;
    }
}

// ── RoPE single-step (for decode: one position at a time) ───────────────────

kernel void rope_single(
    device       float *Q        [[buffer(0)]],  // [n_heads, head_dim]
    device       float *K        [[buffer(1)]],  // [n_kv_heads, head_dim]
    constant     int   &pos      [[buffer(2)]],  // current position
    constant     int   &head_dim [[buffer(3)]],
    constant     int   &n_heads  [[buffer(4)]],
    constant     int   &n_kv_heads [[buffer(5)]],
    constant     float &base     [[buffer(6)]],
    uint2 pos_idx [[thread_position_in_grid]]
) {
    int head = (int)pos_idx.x;
    int d2   = (int)pos_idx.y;  // dimension pair

    if (head >= n_heads) return;
    int d = 2 * d2;
    if (d >= head_dim) return;

    float theta = (float)pos / pow(base, (float)(2 * d2) / (float)head_dim);
    float cos_theta = cos(theta);
    float sin_theta = sin(theta);

    // Apply to Q
    int q_idx = head * head_dim + d;
    float q0 = Q[q_idx];
    float q1 = Q[q_idx + 1];
    Q[q_idx]     = q0 * cos_theta - q1 * sin_theta;
    Q[q_idx + 1] = q0 * sin_theta + q1 * cos_theta;

    // Apply to K
    if (head < n_kv_heads) {
        int k_idx = head * head_dim + d;
        float k0 = K[k_idx];
        float k1 = K[k_idx + 1];
        K[k_idx]     = k0 * cos_theta - k1 * sin_theta;
        K[k_idx + 1] = k0 * sin_theta + k1 * cos_theta;
    }
}
