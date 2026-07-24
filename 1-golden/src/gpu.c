// SPDX-License-Identifier: MIT
// CPU-fallback GPU backend — real matmul, buffer ops on CPU.
// Drop-in replacement for gpu_metal.m on non-Metal platforms.

#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct gpu_context {
    size_t used_memory;
    bool   initialized;
};

struct gpu_buffer {
    void             *cpu_ptr;
    size_t            size;
    gpu_buffer_flags  flags;
};

// ── Lifecycle ────────────────────────────────────────────────────────────────

gpu_context *gpu_init(void) {
    gpu_context *ctx = (gpu_context *)calloc(1, sizeof(gpu_context));
    if (!ctx) return NULL;
    ctx->initialized = true;
    ctx->used_memory = 0;
    return ctx;
}

void gpu_destroy(gpu_context *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    free(ctx);
}

// ── Buffer management ────────────────────────────────────────────────────────

gpu_buffer *gpu_alloc(gpu_context *ctx, size_t size, gpu_buffer_flags flags) {
    if (!ctx || size == 0) return NULL;

    gpu_buffer *buf = (gpu_buffer *)calloc(1, sizeof(gpu_buffer));
    if (!buf) return NULL;

    buf->cpu_ptr = calloc(1, size);
    if (!buf->cpu_ptr) {
        free(buf);
        return NULL;
    }

    buf->size = size;
    buf->flags = flags;
    ctx->used_memory += size;
    return buf;
}

void gpu_free(gpu_context *ctx, gpu_buffer *buf) {
    if (!ctx || !buf) return;
    ctx->used_memory -= buf->size;
    free(buf->cpu_ptr);
    memset(buf, 0, sizeof(*buf));
    free(buf);
}

void *gpu_get_cpu_ptr(gpu_context *ctx, gpu_buffer *buf) {
    (void)ctx;
    return buf ? buf->cpu_ptr : NULL;
}

size_t gpu_buffer_size(gpu_context *ctx, const gpu_buffer *buf) {
    (void)ctx;
    return buf ? buf->size : 0;
}

void gpu_copy_to_device(gpu_context *ctx, gpu_buffer *dst,
                        const void *src, size_t size) {
    (void)ctx;
    if (!dst || !src) return;
    size_t copy = size < dst->size ? size : dst->size;
    memcpy(dst->cpu_ptr, src, copy);
}

void gpu_copy_to_host(gpu_context *ctx, void *dst,
                      const gpu_buffer *src, size_t size) {
    (void)ctx;
    if (!src || !dst) return;
    size_t copy = size < src->size ? size : src->size;
    memcpy(dst, src->cpu_ptr, copy);
}

void gpu_copy_buffer(gpu_context *ctx, gpu_buffer *dst,
                     const gpu_buffer *src, size_t size) {
    (void)ctx;
    if (!dst || !src) return;
    size_t copy = size;
    if (copy > dst->size) copy = dst->size;
    if (copy > src->size) copy = src->size;
    memcpy(dst->cpu_ptr, src->cpu_ptr, copy);
}

// ── Compute: CPU matmul (row-major) ──────────────────────────────────────────

static void cpu_matmul(float *C, const float *A, const float *B,
                        int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void gpu_matmul(gpu_context *ctx, gpu_buffer *C,
                const gpu_buffer *A, const gpu_buffer *B,
                int M, int N, int K,
                quant_mode a_quant, quant_mode b_quant) {
    (void)a_quant;
    (void)b_quant;
    if (!ctx || !C || !A || !B) return;

    cpu_matmul((float *)C->cpu_ptr,
               (const float *)A->cpu_ptr,
               (const float *)B->cpu_ptr,
               M, N, K);
}

void gpu_matmul_int4(gpu_context *ctx, gpu_buffer *C,
                     const gpu_buffer *A_quant, const gpu_buffer *A_scale,
                     int M, int N, int K) {
    // CPU fallback: just zero C (int4 dequant not implemented)
    (void)A_quant;
    (void)A_scale;
    (void)K;
    if (!ctx || !C) return;
    memset(C->cpu_ptr, 0, (size_t)M * N * sizeof(float));
}

// ── Stub compute ops (return immediately) ────────────────────────────────────

void gpu_rmsnorm(gpu_context *ctx, gpu_buffer *out,
                 const gpu_buffer *x, const gpu_buffer *weight,
                 int rows, int dim) {
    (void)ctx; (void)out; (void)x; (void)weight; (void)rows; (void)dim;
}

void gpu_silu(gpu_context *ctx, gpu_buffer *out,
              const gpu_buffer *x, int n) {
    (void)ctx; (void)out; (void)x; (void)n;
}

void gpu_silu_mul(gpu_context *ctx, gpu_buffer *out,
                  const gpu_buffer *gate, const gpu_buffer *up, int n) {
    (void)ctx; (void)out; (void)gate; (void)up; (void)n;
}

void gpu_rope(gpu_context *ctx, gpu_buffer *Q, gpu_buffer *K,
              int seq_len, int head_dim, int n_heads, int n_kv_heads,
              float base, int pos_start) {
    (void)ctx; (void)Q; (void)K; (void)seq_len; (void)head_dim;
    (void)n_heads; (void)n_kv_heads; (void)base; (void)pos_start;
}

void gpu_attention(gpu_context *ctx, gpu_buffer *out,
                   const gpu_buffer *Q, const gpu_buffer *K_cache,
                   const gpu_buffer *V_cache, int seq_len, int head_dim) {
    (void)ctx; (void)out; (void)Q; (void)K_cache; (void)V_cache;
    (void)seq_len; (void)head_dim;
}

void gpu_deltanet_step(gpu_context *ctx, gpu_buffer *state,
                       const gpu_buffer *Q, const gpu_buffer *K,
                       const gpu_buffer *V, int dim) {
    (void)ctx; (void)state; (void)Q; (void)K; (void)V; (void)dim;
}

void gpu_expert_mlp(gpu_context *ctx, gpu_buffer *out,
                    const gpu_buffer *x, const gpu_buffer *gate_w,
                    const gpu_buffer *up_w, const gpu_buffer *down_w,
                    int d_model, int hidden_dim, quant_mode quant) {
    (void)ctx; (void)out; (void)x; (void)gate_w; (void)up_w; (void)down_w;
    (void)d_model; (void)hidden_dim; (void)quant;
}

void gpu_router(gpu_context *ctx, gpu_buffer *out,
                const gpu_buffer *x, const gpu_buffer *router_weight,
                int d_model, int n_experts) {
    (void)ctx; (void)out; (void)x; (void)router_weight;
    (void)d_model; (void)n_experts;
}

void gpu_topk_softmax(gpu_context *ctx, uint32_t *indices, float *weights,
                      const gpu_buffer *logits, int n_experts, int top_k) {
    (void)ctx; (void)indices; (void)weights; (void)logits;
    (void)n_experts; (void)top_k;
}

void gpu_add(gpu_context *ctx, gpu_buffer *out,
             const gpu_buffer *a, const gpu_buffer *b, int n) {
    // CPU fallback: element-wise add
    (void)ctx;
    if (!ctx || !out || !a || !b) return;
    float *op = (float *)out->cpu_ptr;
    float *ap = (float *)a->cpu_ptr;
    float *bp = (float *)b->cpu_ptr;
    for (int i = 0; i < n; i++) op[i] = ap[i] + bp[i];
}

// ── Sync / Info ──────────────────────────────────────────────────────────────

void gpu_sync(gpu_context *ctx) {
    (void)ctx;
    // CPU fallback: operations are synchronous by nature
}

size_t gpu_memory_used(gpu_context *ctx) {
    return ctx ? ctx->used_memory : 0;
}

void gpu_print_info(gpu_context *ctx) {
    if (!ctx) return;
    printf("=== GPU Info (CPU fallback) ===\n");
    printf("Backend:  CPU (no Metal acceleration)\n");
    printf("Memory:   %zu bytes\n", ctx->used_memory);
}
