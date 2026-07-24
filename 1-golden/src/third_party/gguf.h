// SPDX-License-Identifier: MIT
// GGUF format constants — minimal set for test passing.

#ifndef ORNITH_GGUF_TYPES_H
#define ORNITH_GGUF_TYPES_H

#include <stdint.h>

#define GGUF_MAGIC_U32 ((uint32_t)0x46554747u)  // "GGUF" little-endian
#define GGUF_VERSION_MAX 3

typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q8_0 = 8,
} ggml_type;

#endif
