// SPDX-License-Identifier: MIT
// Metal Shading Language — scaled dot-product attention + Gated DeltaNet kernel.
//
// Scaled dot-product attention: O = softmax(Q × K^T / sqrt(d)) × V
// Gated DeltaNet: state = gate * state + (1-gate) * (k^T × v); output = q × state

#include <metal_stdlib>
using namespace metal;

// ── Scaled dot-product attention (single head) ───────────────────────────────
// Each thread block handles one head.
// Q: [seq_len × head_dim]  (one query vector per thread)
// K: [seq_len × head_dim]  (transposed already or cached)
// V: [seq_len × head_dim]
// O: [seq_len × head_dim]

kernel void attention_forward(
    device const float *Q          [[buffer(0)]],  // [seq_len, head_dim]
    device const float *K          [[buffer(1)]],  // [seq_len, head_dim]
    device const float *V          [[buffer(2)]],  // [seq_len, head_dim]
    device       float *output     [[buffer(3)]],  // [seq_len, head_dim]
    constant     int   &seq_len    [[buffer(4)]],
    constant     int   &head_dim   [[buffer(5)]],
    constant     float &scale      [[buffer(6)]],  // 1/sqrt(head_dim)
    uint2       pos [[thread_position_in_grid]]
) {
    int row = (int)pos.x;  // which query position
    if (row >= seq_len) return;

    // Shared memory for attention scores (per query)
    // We compute S[row][t] = Q[row] · K[t] * scale for all t < seq_len
    // Then softmax, then weighted sum of V

    // Thread col = which element of V to contribute to output
    int col = (int)pos.y;
    if (col >= head_dim) return;

    // Compute attention scores for this query
    // For each key position t, compute Q[row] · K[t]
    // Since we're in a 2D grid, accumulate over t with thread per (row, col)

    // Step 1: compute scores = softmax(Q[row] · K[:] * scale) using thread memory
    // Each thread handles one (query_pos, head_dim) output element

    float sum_weights = 0.0f;
    float output_val = 0.0f;

    for (int t = 0; t < seq_len; t++) {
        // Compute Q[row] · K[t] dot product (across head_dim)
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[row * head_dim + d] * K[t * head_dim + d];
        }
        score *= scale;
        // Softmax numerator: exp(score)
        float weight = exp(score);
        sum_weights += weight;
        // Accumulate weighted V contribution for this head_dim element
        output_val += weight * V[t * head_dim + col];
    }

    // Normalize by sum of weights
    if (sum_weights > 1e-10f) {
        output_val /= sum_weights;
    }

    output[row * head_dim + col] = output_val;
}

// ── Gated DeltaNet step ─────────────────────────────────────────────────────
// state = gate * state + (1 - gate) * outer_product(k, v)
// output = q × state (matrix-vector multiply)
//
// state: [head_dim × head_dim] — the recurrent state
// gate: scalar in (0,1), e.g. 0.9
// k, v: [head_dim] vectors
// q:    [head_dim] vector
// output: [head_dim]

kernel void deltanet_step(
    device       float *state     [[buffer(0)]],  // in/out: [dim × dim]
    device const float *Q          [[buffer(1)]],  // [dim]
    device const float *K          [[buffer(2)]],  // [dim]
    device const float *V          [[buffer(3)]],  // [dim]
    device       float *output     [[buffer(4)]],  // [dim]
    constant     int   &dim        [[buffer(5)]],
    constant     float &gate       [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)dim) return;

    // Each thread updates output[tid] and the tid-th row of state

    // 1. Compute state update: state[i][j] += (1-gate) * k[j] * v[i]
    //    Only for the tid-th column of state
    for (int j = 0; j < dim; j++) {
        int idx = j * dim + (int)tid;          // state[j][tid]
        state[idx] = gate * state[idx] + (1.0f - gate) * K[(int)tid] * V[j];
    }

    // 2. Compute output[tid] = sum_j Q[j] * state[j][tid]
    float sum = 0.0f;
    for (int j = 0; j < dim; j++) {
        sum += Q[j] * state[j * dim + (int)tid];
    }
    output[tid] = sum;
}

// ── KV cache update (append new K, V to cache) ──────────────────────────────
// For decode phase: append single new K,V vector to the cached sequence.
// cache_offset: position to write to in the cache

kernel void kv_cache_append(
    device       float *K_cache   [[buffer(0)]],  // [max_seq, head_dim]
    device       float *V_cache   [[buffer(1)]],  // [max_seq, head_dim]
    device const float *K_new     [[buffer(2)]],  // [head_dim]
    device const float *V_new     [[buffer(3)]],  // [head_dim]
    constant     int   &pos       [[buffer(4)]],  // position to write
    constant     int   &head_dim  [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)head_dim) return;
    K_cache[pos * head_dim + tid] = K_new[tid];
    V_cache[pos * head_dim + tid] = V_new[tid];
}
