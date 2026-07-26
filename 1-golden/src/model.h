// SPDX-License-Identifier: MIT
// Model forward pass — config loading, weight access, routing.

#ifndef ORNITH_MODEL_H
#define ORNITH_MODEL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "gguf.h"
#include "gpu.h"
#include "memory.h"

// SSM layer constants (from Ornith 1.0 35B GGUF)
#define ORNITH_SSM_TIME_STEP_RANK 32
#define ORNITH_SSM_CONV_KERNEL     4
#define ORNITH_SSM_STATE_SIZE    128
#define ORNITH_SSM_INNER_SIZE   4096

typedef struct ornith_model ornith_model;

typedef struct {
    uint32_t n_layers;
    uint32_t n_experts_per_layer;
    uint32_t n_active_experts;
    uint32_t n_shared_experts;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;          // Q head dimension = d_model / n_heads
    uint32_t key_length;        // KV cache dimension per head (may differ from head_dim)
    uint32_t value_length;      // Value dimension per head
    uint32_t rope_dim_count;    // RoPE dimension count
    uint32_t hidden_dim;        // FFN intermediate dimension (shared expert)
    uint32_t expert_hidden_dim; // Routed expert hidden dimension
    uint32_t router_hidden_dim; // Router hidden dimension
    uint32_t vocab_size;
    float    rope_theta;
    uint32_t max_seq_len;
    uint32_t full_attention_interval;  // Full attention every N layers (0 = all SSM)
    size_t   expert_size_bytes;
    size_t   non_routed_bytes;
    bool     has_shared_expert;
} model_config;

ornith_model *model_load(const char *gguf_path,
                         const memory_config *mem_cfg,
                         char *err_buf, size_t err_buf_size,
                         bool verbose, bool dry_run);
void model_unload(ornith_model *model);

const model_config *model_get_config(const ornith_model *model);
gpu_context *model_get_gpu(ornith_model *model);
memory_manager *model_get_memory(ornith_model *model);

int model_get_routing(ornith_model *model,
                      const float *x, int layer_id,
                      uint32_t *indices, float *weights, int max_experts);

void model_update_hotstore(ornith_model *model,
                           const uint32_t *expert_ids, int count);

// ── State buffers passed to model_forward ───────────────────────────────────

typedef struct {
    void    **kv_caches;          // [n_layers] GPU buffer handles (deprecated, for Metal path)
    float   **ssm_conv_states;    // [n_layers][(kernel-1)*conv_dim] conv ring buffers
    float   **ssm_h_states;       // [n_layers][d_state] SSM recurrent state vectors
    float   **k_cache;            // [n_layers][max_cache_len * n_kv_heads * key_length]
    float   **v_cache;            // [n_layers][max_cache_len * n_kv_heads * value_length]
    int      max_cache_len;       // allocated cache length
} model_buffers;

// Determine if a layer uses full attention or SSM
static inline int layer_uses_full_attention(int layer_id, uint32_t interval) {
    if (interval == 0) return 0;  // all SSM
    return (layer_id % (int)interval == (int)interval - 1);
}

void model_forward(ornith_model *model, uint32_t token_id, float *output,
                   const model_buffers *buffers, int seq_len, int pos);

uint32_t model_sample(ornith_model *model, float *logits,
                      float temperature, int top_k);

#endif
