// SPDX-License-Identifier: MIT
// Metal Shading Language — int4/int8 dequantization + quantized matmul kernels.
//
// int4 packing: each byte stores two 4-bit values [low4 | high4].
// int8: standard 8-bit signed integer.
// Group quantization: a scale factor per N elements (group_size = 32).

#include <metal_stdlib>
using namespace metal;

// ── Constants ────────────────────────────────────────────────────────────────

constant int GROUP_SIZE [[function_constant(0)]]; // default 32
constant int QUANT_Q4 [[function_constant(1)]];   // 1 = int4, 0 = int8

// ── Helper: dequantize a single int4 pair ────────────────────────────────────

static float dequantize_q4(uchar packed, int idx_in_pair, float scale) {
    int lo = (int)(packed & 0x0F);       // low nibble (first element in pair)
    int hi = (int)((packed >> 4) & 0x0F); // high nibble (second element in pair)
    // Subtract 8 to center around 0 (signed int4 range: -8..7)
    float v = (float)((idx_in_pair == 0 ? lo : hi) - 8);
    return v * scale;
}

static float dequantize_q8(char val, float scale) {
    return (float)val * scale;
}

// ── Kernel: int4 → float dequant ─────────────────────────────────────────────
// Each thread dequantizes one output float element.
// Input:  packed int4 data  (nibbles packed in uchar)
//         scale factors     (1 per GROUP_SIZE elements)
// Output: float32 array

kernel void dequantize_q4_kernel(
    device const uchar  *packed  [[buffer(0)]],
    device const float  *scales  [[buffer(1)]],
    device       float  *output  [[buffer(2)]],
    constant     int    &n       [[buffer(3)]],  // number of output elements
    constant     int    &group   [[buffer(4)]],  // group size (32)
    uint        tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;

    // Each uchar holds two int4 values
    int packed_idx = tid / 2;
    int in_pair    = tid % 2;
    int group_idx  = tid / group;

    float scale = scales[group_idx];
    uchar p = packed[packed_idx];

    output[tid] = dequantize_q4(p, in_pair, scale);
}

// ── Kernel: int4 × float16 matmul (A_quant is int4, B is fp16) ──────────────
// C[M×N] = dequant(A)[M×K] × B[K×N]
// Each thread computes one element of C.

kernel void matmul_q4_f16(
    device const uchar  *A_quant  [[buffer(0)]],  // int4 packed, M×K elements
    device const float  *A_scale  [[buffer(1)]],  // scales per group
    device const half   *B_f16    [[buffer(2)]],  // fp16, K×N
    device       float  *C        [[buffer(3)]],  // output, M×N
    constant     int    &M        [[buffer(4)]],
    constant     int    &N        [[buffer(5)]],
    constant     int    &K        [[buffer(6)]],
    constant     int    &group    [[buffer(7)]],  // group size
    uint2       pos [[thread_position_in_grid]]
) {
    int row = (int)pos.x;
    int col = (int)pos.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // Dequantize A[row][k]
        int idx = row * K + k;
        int packed_idx = idx / 2;
        int in_pair    = idx % 2;
        int group_idx  = idx / group;
        float a_val = dequantize_q4(A_quant[packed_idx], in_pair, A_scale[group_idx]);

        // B is fp16, K×N (row-major: B[k][col])
        sum += a_val * (float)B_f16[k * N + col];
    }
    C[row * N + col] = sum;
}

// ── Kernel: int8 × float16 matmul ───────────────────────────────────────────
// C[M×N] = dequant(A_int8)[M×K] × B_f16[K×N]

kernel void matmul_q8_f16(
    device const char   *A_quant  [[buffer(0)]],  // int8, M×K
    device const float  *A_scale  [[buffer(1)]],  // scales per group
    device const half   *B_f16    [[buffer(2)]],  // fp16, K×N
    device       float  *C        [[buffer(3)]],  // output, M×N
    constant     int    &M        [[buffer(4)]],
    constant     int    &N        [[buffer(5)]],
    constant     int    &K        [[buffer(6)]],
    constant     int    &group    [[buffer(7)]],  // group size
    uint2       pos [[thread_position_in_grid]]
) {
    int row = (int)pos.x;
    int col = (int)pos.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        int idx = row * K + k;
        int group_idx = idx / group;
        float a_val = dequantize_q8(A_quant[idx], A_scale[group_idx]);
        sum += a_val * (float)B_f16[k * N + col];
    }
    C[row * N + col] = sum;
}

// ── Kernel: fp16 × fp16 matmul (fallback when no quantization) ─────────────
// C[M×N] = A[M×K] × B[K×N]

kernel void matmul_f16_f16(
    device const half *A     [[buffer(0)]],
    device const half *B     [[buffer(1)]],
    device       float *C    [[buffer(2)]],
    constant     int   &M    [[buffer(3)]],
    constant     int   &N    [[buffer(4)]],
    constant     int   &K    [[buffer(5)]],
    uint2 pos [[thread_position_in_grid]]
) {
    int row = (int)pos.x;
    int col = (int)pos.y;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += (float)A[row * K + k] * (float)B[k * N + col];
    }
    C[row * N + col] = sum;
}

// ── Kernel: int4 → fp16 dequant (for converting quantized weights to fp16) ──

kernel void dequant_to_f16(
    device const uchar  *packed   [[buffer(0)]],
    device const float  *scales   [[buffer(1)]],
    device       half   *output   [[buffer(2)]],
    constant     int    &n        [[buffer(3)]],
    constant     int    &group    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;

    int packed_idx = tid / 2;
    int in_pair    = tid % 2;
    int group_idx  = tid / group;
    float scale = scales[group_idx];

    output[tid] = (half)dequantize_q4(packed[packed_idx], in_pair, scale);
}
