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

typedef struct ornith_model ornith_model;

typedef struct {
    uint32_t n_layers;
    uint32_t n_experts_per_layer;
    uint32_t n_active_experts;
    uint32_t n_shared_experts;
    uint32_t d_model;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t hidden_dim;
    uint32_t expert_hidden_dim;
    uint32_t router_hidden_dim;
    uint32_t vocab_size;
    float    rope_theta;
    uint32_t max_seq_len;
    size_t   expert_size_bytes;
    size_t   non_routed_bytes;
    bool     has_shared_expert;
} model_config;

ornith_model *model_load(const char *gguf_path,
                         const memory_config *mem_cfg,
                         char *err_buf, size_t err_buf_size);
void model_unload(ornith_model *model);

const model_config *model_get_config(const ornith_model *model);
gpu_context *model_get_gpu(ornith_model *model);
memory_manager *model_get_memory(ornith_model *model);

int model_get_routing(ornith_model *model,
                      const float *x, int layer_id,
                      uint32_t *indices, float *weights, int max_experts);

void model_update_hotstore(ornith_model *model,
                           const uint32_t *expert_ids, int count);

void model_forward(ornith_model *model, uint32_t token_id, float *output,
                   void **kv_caches, int seq_len, int pos);

uint32_t model_sample(ornith_model *model, float *logits,
                      float temperature, int top_k);

#endif
