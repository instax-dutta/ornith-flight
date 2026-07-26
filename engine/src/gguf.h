// SPDX-License-Identifier: MIT
// GGUF loader interface — header + metadata + tensor info.

#ifndef ORNITH_GGUF_H
#define ORNITH_GGUF_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "third_party/gguf.h"

typedef struct gguf_model gguf_model;

// ── Tensor descriptor ────────────────────────────────────────────────────────
typedef struct {
    const char  *name;
    ggml_type    type;
    uint32_t     n_dims;
    uint64_t     dims[4];
    size_t       offset;
    size_t       size_bytes;
    size_t       n_elems;
} gguf_tensor_info;

// ── Metadata value ───────────────────────────────────────────────────────────
typedef enum {
    GGUF_VALUE_NONE   = 0,
    GGUF_VALUE_UINT8  = 0,
    GGUF_VALUE_INT8   = 1,
    GGUF_VALUE_UINT16 = 2,
    GGUF_VALUE_INT16  = 3,
    GGUF_VALUE_UINT32 = 4,
    GGUF_VALUE_INT32  = 5,
    GGUF_VALUE_FLOAT32= 6,
    GGUF_VALUE_BOOL   = 7,
    GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY  = 9,
    GGUF_VALUE_UINT64 = 10,
    GGUF_VALUE_INT64  = 11,
    GGUF_VALUE_FLOAT64= 12,
} gguf_value_type;

typedef struct {
    gguf_value_type type;
    union {
        uint8_t  uint8;
        int8_t   int8;
        uint16_t uint16;
        int16_t  int16;
        uint32_t uint32;
        int32_t  int32;
        uint64_t uint64;
        int64_t  int64;
        float    float32;
        double   float64;
        bool     boolean;
        struct {
            const char *data;
            size_t      len;
        } string;
        struct {
            gguf_value_type elem_type;
            size_t          count;
            const void     *data;  // raw pointer into mmap for scalar arrays
        } array;
    } value;
} gguf_metadata_value;

// ── API ──────────────────────────────────────────────────────────────────────

gguf_model *gguf_open(const char *filepath, char *err_buf, size_t err_buf_size);
void gguf_close(gguf_model *model);

uint32_t gguf_version(const gguf_model *model);
uint64_t gguf_tensor_count(const gguf_model *model);
uint64_t gguf_metadata_count(const gguf_model *model);

const char *gguf_architecture(const gguf_model *model);
size_t gguf_alignment(const gguf_model *model);

bool gguf_find_metadata(const gguf_model *model, const char *key,
                        gguf_metadata_value *out);

const gguf_tensor_info *gguf_find_tensor(const gguf_model *model, const char *name);
uint64_t gguf_list_tensors(const gguf_model *model,
                           const gguf_tensor_info **tensors,
                           uint64_t max_count);
const void *gguf_tensor_data(const gguf_model *model, const char *name);
const void *gguf_tensor_data_from_info(const gguf_model *model,
                                       const gguf_tensor_info *info);

// ── Streaming / File I/O ───────────────────────────────────────────────────

// Read raw (possibly quantized) tensor data from the GGUF file into a buffer.
// The buffer must be at least tensor->size_bytes large.
// Returns the number of bytes read, or 0 on failure.
size_t gguf_read_tensor_data(const gguf_model *model,
                             const gguf_tensor_info *tensor,
                             void *buffer, size_t buffer_size);

// Read a byte-range slice of tensor data from the GGUF file.
// byte_offset is relative to the start of the tensor's data in the file.
// Useful for reading individual rows of large 2D tensors.
// Returns number of bytes read, or 0 on failure.
size_t gguf_read_tensor_bytes(const gguf_model *model,
                              const gguf_tensor_info *tensor,
                              size_t byte_offset,
                              void *buffer,
                              size_t buffer_size);

// ── Dequantization API (for quantized tensor types) ────────────────────────

// Dequantize a tensor from an in-memory buffer of raw quantized bytes.
// output must have room for tensor->n_elems * sizeof(float).
// data points to the raw quantized bytes (same layout as in GGUF file).
bool gguf_dequantize_tensor_from_buf(const void *data,
                                     const gguf_tensor_info *tensor,
                                     float *output);

// Dequantize a single expert's slice from a fused 3D quantized tensor in memory.
// The tensor shape must have the last (fastest-varying) dimension = n_experts.
// expert_idx selects which expert to extract (0 <= expert_idx < n_experts).
// output must have room for (tensor->n_elems / n_experts) floats.
bool gguf_dequantize_expert_slice_from_buf(const void *data,
                                           const gguf_tensor_info *tensor,
                                           int expert_idx,
                                           float *output);

// Original convenience wrappers (read from mmap — may return false for large
// models where tensor data is not in the metadata-only mapping).
bool gguf_dequantize_tensor(const gguf_model *model,
                            const gguf_tensor_info *tensor,
                            float *output);
bool gguf_dequantize_expert_slice(const gguf_model *model,
                                  const gguf_tensor_info *tensor,
                                  int expert_idx,
                                  float *output);

// Convenience: architecture-prefixed parameter lookup
bool gguf_get_param_u32(const gguf_model *model, const char *param, uint32_t *out);
bool gguf_get_param_f32(const gguf_model *model, const char *param, float *out);

#endif
