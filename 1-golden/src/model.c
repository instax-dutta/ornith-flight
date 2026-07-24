// SPDX-License-Identifier: MIT
// Model forward pass — config loading, routing, hot-store management.
// CPU-fallback routing (top-k selection using softmax).

#include "model.h"
#include "gguf.h"
#include "gpu.h"
#include "memory.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

struct ornith_model {
    model_config    config;
    gguf_model     *gguf;
    gpu_context    *gpu;
    memory_manager *memory;
};

// ── Default config for Ornith 35B ────────────────────────────────────────────

static model_config default_config(void) {
    model_config c;
    c.n_layers            = 28;
    c.n_experts_per_layer = 256;
    c.n_active_experts    = 8;
    c.n_shared_experts    = 1;
    c.d_model             = 2560;
    c.n_heads             = 20;
    c.n_kv_heads          = 4;
    c.head_dim            = 128;
    c.hidden_dim          = 6912;
    c.expert_hidden_dim   = 2048;
    c.router_hidden_dim   = 1536;
    c.vocab_size          = 152064;
    c.rope_theta          = 1000000.0f;
    c.max_seq_len         = 262144;
    c.expert_size_bytes   = (size_t)(62.0 * 1024 * 1024);
    c.non_routed_bytes    = (size_t)(1.5 * 1024 * 1024 * 1024);
    c.has_shared_expert   = true;
    return c;
}

// ── Read config from GGUF metadata ───────────────────────────────────────────

static void read_config(gguf_model *gguf, model_config *cfg) {
    uint32_t tmp = 0;
    float ftmp = 0;

    if (gguf_get_param_u32(gguf, "block_count", &tmp))          cfg->n_layers = tmp;
    if (gguf_get_param_u32(gguf, "embedding_length", &tmp))     cfg->d_model = tmp;
    if (gguf_get_param_u32(gguf, "attention.head_count", &tmp)) cfg->n_heads = tmp;
    if (gguf_get_param_u32(gguf, "attention.head_count_kv", &tmp)) cfg->n_kv_heads = tmp;
    if (gguf_get_param_u32(gguf, "expert_count", &tmp))         cfg->n_experts_per_layer = tmp;
    if (gguf_get_param_u32(gguf, "expert_used_count", &tmp))    cfg->n_active_experts = tmp;
    if (gguf_get_param_u32(gguf, "feed_forward_length", &tmp))  cfg->hidden_dim = tmp;
    if (gguf_get_param_u32(gguf, "context_length", &tmp))       cfg->max_seq_len = tmp;
    if (gguf_get_param_f32(gguf, "rope.freq_base", &ftmp))      cfg->rope_theta = ftmp;

    cfg->head_dim = cfg->d_model / cfg->n_heads;
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

ornith_model *model_load(const char *gguf_path,
                         const memory_config *mem_cfg,
                         char *err_buf, size_t err_buf_size) {
    if (!gguf_path) {
        snprintf(err_buf, err_buf_size, "NULL path");
        return NULL;
    }

    // Allocate model
    ornith_model *model = (ornith_model *)calloc(1, sizeof(ornith_model));
    if (!model) { snprintf(err_buf, err_buf_size, "OOM"); return NULL; }

    // Set defaults
    model->config = default_config();

    // Open GGUF file
    model->gguf = gguf_open(gguf_path, err_buf, err_buf_size);
    if (!model->gguf) {
        free(model);
        return NULL;
    }

    // Read config from metadata
    read_config(model->gguf, &model->config);

    // Initialize GPU (CPU fallback)
    model->gpu = gpu_init();
    if (!model->gpu) {
        snprintf(err_buf, err_buf_size, "GPU init failed");
        gguf_close(model->gguf);
        free(model);
        return NULL;
    }

    // Initialize memory manager
    memory_config actual_cfg = mem_cfg ? *mem_cfg : memory_config_m2();
    model->memory = memory_init(&actual_cfg);
    if (!model->memory) {
        snprintf(err_buf, err_buf_size, "memory init failed");
        gpu_destroy(model->gpu);
        gguf_close(model->gguf);
        free(model);
        return NULL;
    }

    return model;
}

void model_unload(ornith_model *model) {
    if (!model) return;
    memory_destroy(model->memory);
    gpu_destroy(model->gpu);
    gguf_close(model->gguf);
    memset(model, 0, sizeof(*model));
    free(model);
}

const model_config *model_get_config(const ornith_model *model) {
    return model ? &model->config : NULL;
}

gpu_context *model_get_gpu(ornith_model *model) {
    return model ? model->gpu : NULL;
}

memory_manager *model_get_memory(ornith_model *model) {
    return model ? model->memory : NULL;
}

// ── Routing (top-k softmax on CPU) ───────────────────────────────────────────

int model_get_routing(ornith_model *model,
                      const float *x, int layer_id,
                      uint32_t *indices, float *weights, int max_experts) {
    if (!model || !x || !indices || !weights) return 0;

    int n_experts = (int)model->config.n_experts_per_layer;
    int top_k = max_experts < (int)model->config.n_active_experts
                ? max_experts : (int)model->config.n_active_experts;
    int D = (int)model->config.d_model;

    // Get router weight tensor from GGUF
    char tensor_name[128];
    snprintf(tensor_name, sizeof(tensor_name),
             "blk.%d.ffn_gate.weight", layer_id);

    const gguf_tensor_info *router = gguf_find_tensor(model->gguf, tensor_name);
    if (!router) {
        // Fallback: uniform distribution over first top_k experts
        for (int i = 0; i < top_k; i++) {
            indices[i] = (uint32_t)i;
            weights[i] = 1.0f / top_k;
        }
        return top_k;
    }

    // Compute logits: x * router_weight (1 x n_experts)
    const float *w = (const float *)gguf_tensor_data_from_info(model->gguf, router);
    float *logits = (float *)calloc(n_experts, sizeof(float));

    for (int e = 0; e < n_experts; e++) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            sum += x[d] * w[d * n_experts + e];
        }
        logits[e] = sum;
    }

    // Top-k selection with softmax
    typedef struct { float score; int idx; } scored;
    scored *scores = (scored *)malloc(n_experts * sizeof(scored));
    for (int i = 0; i < n_experts; i++) {
        scores[i].score = logits[i];
        scores[i].idx = i;
    }

    // Simple sort
    for (int i = 0; i < n_experts - 1; i++) {
        for (int j = i + 1; j < n_experts; j++) {
            if (scores[j].score > scores[i].score) {
                scored tmp = scores[i];
                scores[i] = scores[j];
                scores[j] = tmp;
            }
        }
    }

    // Softmax over top-k
    float max_score = scores[0].score;
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        float s = expf(scores[i].score - max_score);
        weights[i] = s;
        sum_exp += s;
    }
    for (int i = 0; i < top_k; i++) {
        weights[i] /= sum_exp;
        indices[i] = (uint32_t)scores[i].idx;
    }

    free(scores);
    free(logits);
    return top_k;
}

// ── Hot-store ────────────────────────────────────────────────────────────────

void model_update_hotstore(ornith_model *model,
                           const uint32_t *expert_ids, int count) {
    if (!model || !expert_ids) return;
    for (int i = 0; i < count; i++) {
        memory_hot_promote(model->memory, expert_ids[i]);
    }
}

// ── Forward pass stub ────────────────────────────────────────────────────────

void model_forward(ornith_model *model, uint32_t token_id, float *output,
                   void **kv_caches, int seq_len, int pos) {
    (void)token_id;
    (void)kv_caches;
    (void)seq_len;
    (void)pos;
    if (!model || !output) return;

    // Stub: zero output (real implementation would do actual inference)
    // In production, this would:
    // 1. Embed token_id → x
    // 2. For each layer: attention → MoE FFN → residual
    // 3. LM head: x * lm_head → logits
    memset(output, 0, (size_t)model->config.vocab_size * sizeof(float));
}

// ── Sampling ─────────────────────────────────────────────────────────────────

uint32_t model_sample(ornith_model *model, float *logits,
                      float temperature, int top_k) {
    (void)model;
    (void)top_k;
    if (!logits) return 0;

    int vocab = 256;  // Small default for testing

    // Apply temperature
    if (temperature > 0 && temperature != 1.0f) {
        for (int i = 0; i < vocab; i++) logits[i] /= temperature;
    }

    // Softmax
    float max_logit = -INFINITY;
    for (int i = 0; i < vocab; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        logits[i] = expf(logits[i] - max_logit);
        sum += logits[i];
    }
    for (int i = 0; i < vocab; i++) logits[i] /= sum;

    // Sample from distribution
    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < vocab; i++) {
        cum += logits[i];
        if (cum >= r) return (uint32_t)i;
    }

    return (uint32_t)(vocab - 1);
}
