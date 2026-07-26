// SPDX-License-Identifier: MIT
// GGUF format constants — including K-quant block structs for Q4_K and Q6_K.

#ifndef ORNITH_GGUF_TYPES_H
#define ORNITH_GGUF_TYPES_H

#include <stdint.h>
#include <stddef.h>

#define GGUF_MAGIC_U32 ((uint32_t)0x46554747u)  // "GGUF" little-endian
#define GGUF_VERSION_MAX 3

#define QK_K 256  // K-quant super-block size

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q6_K = 14,
} ggml_type;

// ── K-quant block structs (from llama.cpp ggml-common.h) ─────────────────────

// Q4_K: 256 elements, 4-bit + 6-bit scales, 12 bytes scales, fp16 d/dmin
typedef struct {
    uint16_t d;                    // super-block scale for quantized scales (fp16)
    uint16_t dmin;                 // super-block scale for quantized mins (fp16)
    uint8_t scales[12];            // scales and mins, quantized with 6 bits
    uint8_t qs[QK_K / 2];          // 128 bytes: 4-bit quants
} block_q4_K;

// Q6_K: 256 elements, 6-bit + 8-bit scales, fp16 super-scale
typedef struct {
    uint8_t ql[QK_K / 2];          // 128 bytes: quants, lower 4 bits
    uint8_t qh[QK_K / 4];          // 64 bytes: quants, upper 2 bits
    int8_t  scales[QK_K / 16];     // 16 bytes: scales, quantized with 8 bits
    uint16_t d;                     // super-block scale (fp16)
} block_q6_K;

// Compute row size (bytes) for a given type and element count
static inline size_t ggml_row_size(ggml_type type, uint64_t n_elems) {
    switch (type) {
    case GGML_TYPE_F32:  return n_elems * 4;
    case GGML_TYPE_F16:  return n_elems * 2;
    case GGML_TYPE_Q4_0: return (n_elems / 32) * (sizeof(uint16_t) + 16);  // d + qs[16]
    case GGML_TYPE_Q8_0: return (n_elems / 32) * (sizeof(uint16_t) + 32);  // d + qs[32]
    case GGML_TYPE_Q4_K: return (n_elems / QK_K) * sizeof(block_q4_K);
    case GGML_TYPE_Q6_K: return (n_elems / QK_K) * sizeof(block_q6_K);
    default: return 0;
    }
}

#endif
