// SPDX-License-Identifier: MIT
// GPU abstraction layer — CPU-fallback backend for cross-platform testing.
// Metal backend (gpu_metal.m) will be a drop-in replacement on macOS.

#ifndef ORNITH_GPU_H
#define ORNITH_GPU_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct gpu_context gpu_context;
typedef struct gpu_buffer  gpu_buffer;

typedef enum {
    QUANT_NONE = 0,
    QUANT_FP16,
    QUANT_INT8,
    QUANT_INT4,
} quant_mode;

typedef enum {
    GPU_BUF_DEFAULT  = 0,
    GPU_BUF_SHARED   = 1 << 0,
    GPU_BUF_CONSTANT = 1 << 1,
} gpu_buffer_flags;

// ── Lifecycle ────────────────────────────────────────────────────────────────
gpu_context *gpu_init(void);
void gpu_destroy(gpu_context *ctx);

// ── Buffer management ────────────────────────────────────────────────────────
gpu_buffer *gpu_alloc(gpu_context *ctx, size_t size, gpu_buffer_flags flags);
void gpu_free(gpu_context *ctx, gpu_buffer *buf);
void *gpu_get_cpu_ptr(gpu_context *ctx, gpu_buffer *buf);
size_t gpu_buffer_size(gpu_context *ctx, const gpu_buffer *buf);

void gpu_copy_to_device(gpu_context *ctx, gpu_buffer *dst,
                        const void *src, size_t size);
void gpu_copy_to_host(gpu_context *ctx, void *dst,
                      const gpu_buffer *src, size_t size);
void gpu_copy_buffer(gpu_context *ctx, gpu_buffer *dst,
                     const gpu_buffer *src, size_t size);

// ── Compute ──────────────────────────────────────────────────────────────────
void gpu_matmul(gpu_context *ctx, gpu_buffer *C,
                const gpu_buffer *A, const gpu_buffer *B,
                int M, int N, int K,
                quant_mode a_quant, quant_mode b_quant);

void gpu_matmul_int4(gpu_context *ctx, gpu_buffer *C,
                     const gpu_buffer *A_quant, const gpu_buffer *A_scale,
                     int M, int N, int K);

void gpu_rmsnorm(gpu_context *ctx, gpu_buffer *out,
                 const gpu_buffer *x, const gpu_buffer *weight,
                 int rows, int dim);

void gpu_silu(gpu_context *ctx, gpu_buffer *out,
              const gpu_buffer *x, int n);

void gpu_silu_mul(gpu_context *ctx, gpu_buffer *out,
                  const gpu_buffer *gate, const gpu_buffer *up, int n);

void gpu_rope(gpu_context *ctx, gpu_buffer *Q, gpu_buffer *K,
              int seq_len, int head_dim, int n_heads, int n_kv_heads,
              float base, int pos_start);

void gpu_attention(gpu_context *ctx, gpu_buffer *out,
                   const gpu_buffer *Q, const gpu_buffer *K_cache,
                   const gpu_buffer *V_cache, int seq_len, int head_dim);

void gpu_deltanet_step(gpu_context *ctx, gpu_buffer *state,
                       const gpu_buffer *Q, const gpu_buffer *K,
                       const gpu_buffer *V, int dim);

void gpu_expert_mlp(gpu_context *ctx, gpu_buffer *out,
                    const gpu_buffer *x, const gpu_buffer *gate_w,
                    const gpu_buffer *up_w, const gpu_buffer *down_w,
                    int d_model, int hidden_dim, quant_mode quant);

void gpu_router(gpu_context *ctx, gpu_buffer *out,
                const gpu_buffer *x, const gpu_buffer *router_weight,
                int d_model, int n_experts);

void gpu_topk_softmax(gpu_context *ctx, uint32_t *indices, float *weights,
                      const gpu_buffer *logits, int n_experts, int top_k);

void gpu_add(gpu_context *ctx, gpu_buffer *out,
             const gpu_buffer *a, const gpu_buffer *b, int n);

// ── Sync ─────────────────────────────────────────────────────────────────────
void gpu_sync(gpu_context *ctx);
size_t gpu_memory_used(gpu_context *ctx);
void gpu_print_info(gpu_context *ctx);

#endif
