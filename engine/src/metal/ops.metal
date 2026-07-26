// SPDX-License-Identifier: MIT
// Metal compute shaders for Ornith 35B MoE inference engine.
// Compiled at runtime by gpu_metal.m via newLibraryWithSource:.

#include <metal_stdlib>
using namespace metal;

// ═════════════════════════════════════════════════════════════════════════════
// Vector-Matrix Multiply: out[M] = vec[K] @ mat[K][M]  (row-major)
// M threads, each computes one output element.
// ═════════════════════════════════════════════════════════════════════════════

kernel void matmul_vec_kernel(
    device const float *vec   [[buffer(0)]],
    device const float *mat   [[buffer(1)]],
    device float       *out   [[buffer(2)]],
    constant int       &K     [[buffer(3)]],
    constant int       &M     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)M) return;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        sum += vec[i] * mat[i * M + gid];
    }
    out[gid] = sum;
}

// ═════════════════════════════════════════════════════════════════════════════
// RMSNorm — first pass: compute partial sum of squares per threadgroup.
// Each threadgroup (256 threads) covers a tile, writes 1 partial sum.
// ═════════════════════════════════════════════════════════════════════════════

kernel void rmsnorm_partial_kernel(
    device const float *x             [[buffer(0)]],
    device float       *partial_sums  [[buffer(1)]],
    constant int       &dim           [[buffer(2)]],
    threadgroup float  *shared        [[threadgroup(0)]],
    uint gid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint gidx [[threadgroup_position_in_grid]],
    uint tps  [[threads_per_threadgroup]])
{
    // Each thread processes one element (or zero if beyond dim)
    float val = (gid < (uint)dim) ? x[gid] : 0.0f;
    shared[lid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Threadgroup reduction
    for (uint stride = tps / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 writes partial sum
    if (lid == 0) {
        partial_sums[gidx] = shared[0];
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// RMSNorm — second pass: apply normalization.
// Each thread applies: out[i] = weight[i] * x[i] * inv_rms
// inv_rms is precomputed on CPU from partial sums.
// ═════════════════════════════════════════════════════════════════════════════

kernel void rmsnorm_apply_kernel(
    device const float *x       [[buffer(0)]],
    device const float *weight  [[buffer(1)]],
    device float       *out     [[buffer(2)]],
    constant float     &inv_rms [[buffer(3)]],
    constant int       &dim     [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= (uint)dim) return;
    out[gid] = weight[gid] * (x[gid] * inv_rms);
}

// ═════════════════════════════════════════════════════════════════════════════
// SiLU (in-place): data[i] = data[i] / (1 + exp(-data[i]))
// ═════════════════════════════════════════════════════════════════════════════

kernel void silu_kernel(
    device float *data [[buffer(0)]],
    uint gid [[thread_position_in_grid]])
{
    float x = data[gid];
    data[gid] = x / (1.0f + exp(-x));
}

// ═════════════════════════════════════════════════════════════════════════════
// SiLU + Multiply (out-of-place): out[i] = gate[i] * SiLU(up[i])
// ═════════════════════════════════════════════════════════════════════════════

kernel void silu_mul_kernel(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device float       *out  [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    float g = gate[gid];
    float u = up[gid];
    out[gid] = (g / (1.0f + exp(-g))) * u;
}

// ═════════════════════════════════════════════════════════════════════════════
// RoPE: apply rotary position embedding to Q and K vectors.
// Each thread processes one pair of elements (2i, 2i+1) for a given head.
// ═════════════════════════════════════════════════════════════════════════════

kernel void rope_kernel(
    device float       *qk        [[buffer(0)]],
    constant int       &head_dim  [[buffer(1)]],
    constant int       &n_heads   [[buffer(2)]],
    constant int       &rot_dims  [[buffer(3)]],
    constant int       &pos       [[buffer(4)]],
    constant float     &theta     [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    // gid = head_index * (rot_dims/2) + pair_index
    int head = gid / (rot_dims / 2);
    int pair = gid % (rot_dims / 2);
    if (head >= n_heads) return;
    
    int i = pair * 2;
    if (i >= rot_dims) return;
    
    float inv_freq = 1.0f / pow(theta, (float)i / (float)rot_dims);
    float cos_v = cos((float)pos * inv_freq);
    float sin_v = sin((float)pos * inv_freq);
    
    device float *h_vec = qk + head * head_dim;
    float a = h_vec[i];
    float b = h_vec[i + 1];
    h_vec[i]     = a * cos_v - b * sin_v;
    h_vec[i + 1] = a * sin_v + b * cos_v;
}

// ═════════════════════════════════════════════════════════════════════════════
// Element-wise add (in-place): a[i] += b[i]
// ═════════════════════════════════════════════════════════════════════════════

kernel void add_kernel(
    device float       *a    [[buffer(0)]],
    device const float *b    [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    a[gid] += b[gid];
}

// ═════════════════════════════════════════════════════════════════════════════
// Softmax: compute softmax over the last dimension of a 2D tensor
// Used for attention softmax over key-value positions.
// Each threadgroup handles one row (one Q head's scores over KV positions).
// ═════════════════════════════════════════════════════════════════════════════

kernel void softmax_kernel(
    device float       *scores      [[buffer(0)]],
    constant int       &rows        [[buffer(1)]],
    constant int       &cols        [[buffer(2)]],
    threadgroup float  *shared      [[threadgroup(0)]],
    uint gid  [[thread_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]])
{
    if (gid >= (uint)(rows * cols)) return;
    
    int row = gid / cols;
    int col = gid % cols;
    
    device float *row_data = scores + row * cols;
    
    // Find max — threadgroup reduction
    float val = row_data[col];
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = 256 / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = fmax(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float max_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Compute exp and sum
    float exp_v = exp(val - max_val);
    shared[lid] = exp_v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for (uint stride = 256 / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_exp = shared[0];
    
    // Normalize
    row_data[col] = exp_v / sum_exp;
}
