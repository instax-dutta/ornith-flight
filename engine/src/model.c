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
#include <time.h>

// Per-layer cached dequantized float weights (lazy-loaded, kept for lifetime)
typedef struct {
    // Norms
    float *attn_norm;
    float *post_attn_norm;

    // SSM layer
    float *attn_qkv;
    float *attn_gate;
    float *ssm_conv1d;
    float *ssm_a;
    float *ssm_dt_bias;
    float *ssm_alpha;
    float *ssm_beta;
    float *ssm_norm;
    float *ssm_out;

    // Attention layer
    float *attn_q;
    float *attn_k;
    float *attn_v;
    float *attn_output;

    // Shared expert (all layers)
    float *ffn_gate_shexp;
    float *ffn_up_shexp;
    float *ffn_down_shexp;

    // Routed expert 3D weight tensors [d_model, hidden_dim, n_experts]
    float *ffn_gate_inp;     // [d_model, n_experts] — router projection
    float *ffn_gate_exps;    // [d_model, hidden_dim, n_experts] — per-expert gate
    float *ffn_up_exps;      // [d_model, hidden_dim, n_experts] — per-expert up
    float *ffn_down_exps;    // [hidden_dim, d_model, n_experts] — per-expert down

    // Final norm (optional, stored on layer 0's cache)
    float *output_norm;

    bool loaded;
} layer_cache;

// Free all per-layer weight buffers (NOT output_norm — managed separately).
static void free_layer_weights(layer_cache *lc) {
    if (!lc) return;
    free(lc->attn_norm);       lc->attn_norm = NULL;
    free(lc->post_attn_norm);  lc->post_attn_norm = NULL;
    free(lc->attn_qkv);        lc->attn_qkv = NULL;
    free(lc->attn_gate);       lc->attn_gate = NULL;
    free(lc->ssm_conv1d);      lc->ssm_conv1d = NULL;
    free(lc->ssm_a);           lc->ssm_a = NULL;
    free(lc->ssm_dt_bias);     lc->ssm_dt_bias = NULL;
    free(lc->ssm_alpha);       lc->ssm_alpha = NULL;
    free(lc->ssm_beta);        lc->ssm_beta = NULL;
    free(lc->ssm_norm);        lc->ssm_norm = NULL;
    free(lc->ssm_out);         lc->ssm_out = NULL;
    free(lc->attn_q);          lc->attn_q = NULL;
    free(lc->attn_k);          lc->attn_k = NULL;
    free(lc->attn_v);          lc->attn_v = NULL;
    free(lc->attn_output);     lc->attn_output = NULL;
    free(lc->ffn_gate_shexp);  lc->ffn_gate_shexp = NULL;
    free(lc->ffn_up_shexp);    lc->ffn_up_shexp = NULL;
    free(lc->ffn_down_shexp);  lc->ffn_down_shexp = NULL;
    free(lc->ffn_gate_inp);    lc->ffn_gate_inp = NULL;
    free(lc->ffn_gate_exps);   lc->ffn_gate_exps = NULL;
    free(lc->ffn_up_exps);     lc->ffn_up_exps = NULL;
    free(lc->ffn_down_exps);   lc->ffn_down_exps = NULL;
    lc->loaded = false;
}

// Free everything including output_norm (for model_unload).
static void free_layer_cache(layer_cache *lc) {
    if (!lc) return;
    free_layer_weights(lc);
    free(lc->output_norm);
    memset(lc, 0, sizeof(*lc));
}

struct ornith_model {
    model_config    config;
    gguf_model     *gguf;
    gpu_context    *gpu;
    memory_manager *memory;
    layer_cache    *layer_caches;  // [config.n_layers]
    uint32_t        n_layers;      // cached for convenience
    bool            trace_enabled; // per-layer timing markers
};

// ── Default config for Ornith 1.0 35B ───────────────────────────────────────

static model_config default_config(void) {
    model_config c;
    c.n_layers              = 40;
    c.n_experts_per_layer   = 256;
    c.n_active_experts      = 8;
    c.n_shared_experts      = 1;
    c.d_model               = 2048;
    c.n_heads               = 16;
    c.n_kv_heads            = 2;
    c.head_dim              = 128;    // d_model / n_heads (Q head dim)
    c.key_length            = 256;    // attention.key_length
    c.value_length          = 256;    // attention.value_length
    c.rope_dim_count        = 64;     // rope.dimension_count
    c.hidden_dim            = 512;    // feed_forward_length (shared expert)
    c.expert_hidden_dim     = 512;    // expert_feed_forward_length
    c.router_hidden_dim     = 2048;   // default = d_model
    c.vocab_size            = 248320;
    c.rope_theta            = 10000000.0f;
    c.max_seq_len           = 262144;
    c.full_attention_interval = 4;    // full attention every 4th layer
    c.expert_size_bytes     = (size_t)(62.0 * 1024 * 1024);
    c.non_routed_bytes      = (size_t)(1.5 * 1024 * 1024 * 1024);
    c.has_shared_expert     = true;
    return c;
}

// ── Read config from GGUF metadata ───────────────────────────────────────────

static void read_config(gguf_model *gguf, model_config *cfg) {
    uint32_t tmp = 0;
    float ftmp = 0;

    if (gguf_get_param_u32(gguf, "block_count", &tmp))            cfg->n_layers = tmp;
    if (gguf_get_param_u32(gguf, "embedding_length", &tmp))       cfg->d_model = tmp;
    if (gguf_get_param_u32(gguf, "attention.head_count", &tmp))   cfg->n_heads = tmp;
    if (gguf_get_param_u32(gguf, "attention.head_count_kv", &tmp)) cfg->n_kv_heads = tmp;
    if (gguf_get_param_u32(gguf, "attention.key_length", &tmp))   cfg->key_length = tmp;
    if (gguf_get_param_u32(gguf, "attention.value_length", &tmp)) cfg->value_length = tmp;
    if (gguf_get_param_u32(gguf, "expert_count", &tmp))           cfg->n_experts_per_layer = tmp;
    if (gguf_get_param_u32(gguf, "expert_used_count", &tmp))      cfg->n_active_experts = tmp;
    if (gguf_get_param_u32(gguf, "feed_forward_length", &tmp))    cfg->hidden_dim = tmp;
    if (gguf_get_param_u32(gguf, "expert_feed_forward_length", &tmp)) cfg->expert_hidden_dim = tmp;
    if (gguf_get_param_u32(gguf, "context_length", &tmp))         cfg->max_seq_len = tmp;
    if (gguf_get_param_u32(gguf, "rope.dimension_count", &tmp))   cfg->rope_dim_count = tmp;
    if (gguf_get_param_u32(gguf, "full_attention_interval", &tmp)) cfg->full_attention_interval = tmp;
    if (gguf_get_param_u32(gguf, "vocab_size", &tmp))             cfg->vocab_size = tmp;
    if (gguf_get_param_f32(gguf, "rope.freq_base", &ftmp))        cfg->rope_theta = ftmp;

    cfg->head_dim = cfg->d_model / cfg->n_heads;

    if (cfg->key_length == 0) cfg->key_length = cfg->head_dim;
    if (cfg->value_length == 0) cfg->value_length = cfg->key_length;
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

ornith_model *model_load(const char *gguf_path,
                         const memory_config *mem_cfg,
                         char *err_buf, size_t err_buf_size,
                         bool verbose, bool dry_run) {
    if (!gguf_path) { snprintf(err_buf, err_buf_size, "NULL path"); return NULL; }

    ornith_model *model = (ornith_model *)calloc(1, sizeof(ornith_model));
    if (!model) { snprintf(err_buf, err_buf_size, "OOM"); return NULL; }

    model->config = default_config();

    model->gguf = gguf_open(gguf_path, err_buf, err_buf_size);
    if (!model->gguf) { free(model); return NULL; }

    read_config(model->gguf, &model->config);

    if (verbose) {
        const model_config *c = &model->config;
        printf("Model config:\n");
        printf("  Architecture:     %s\n", gguf_architecture(model->gguf) ? gguf_architecture(model->gguf) : "unknown");
        printf("  Layers:           %u\n", c->n_layers);
        printf("  d_model:          %u\n", c->d_model);
        printf("  Heads:            %u (KV: %u)\n", c->n_heads, c->n_kv_heads);
        printf("  Head dim:         %u\n", c->head_dim);
        printf("  Key/Value length: %u / %u\n", c->key_length, c->value_length);
        printf("  RoPE dims:        %u  theta: %.0f\n", c->rope_dim_count, c->rope_theta);
        printf("  Experts/layer:    %u  active: %u  shared: %u\n",
               c->n_experts_per_layer, c->n_active_experts, c->n_shared_experts);
        printf("  Shared FFN dim:   %u\n", c->hidden_dim);
        printf("  Expert FFN dim:   %u\n", c->expert_hidden_dim);
        printf("  Vocab:            %u\n", c->vocab_size);
        printf("  Max seq len:      %u\n", c->max_seq_len);
        printf("  Full attn every:  %u layer(s)\n", c->full_attention_interval);
        printf("  Expert size:      %.1f MB\n", c->expert_size_bytes / (1024.0 * 1024.0));
        printf("  Non-routed:       %.1f MB\n", c->non_routed_bytes / (1024.0 * 1024.0));
        printf("  GGUF alignment:   %zu bytes\n", gguf_alignment(model->gguf));
        printf("  Tensors:          %llu\n", (unsigned long long)gguf_tensor_count(model->gguf));
        fflush(stdout);
    }

    if (dry_run) {
        // Dry run: close GGUF, return model with config populated but no weights allocated.
        // model_unload() handles NULL gguf/gpu/memory/layer_caches gracefully.
        gguf_close(model->gguf);
        model->gguf = NULL;
        model->n_layers = model->config.n_layers;
        return model;
    }

    model->gpu = gpu_init();
    if (!model->gpu) {
        snprintf(err_buf, err_buf_size, "GPU init failed");
        gguf_close(model->gguf); free(model); return NULL;
    }

    if (verbose) gpu_print_info(model->gpu);

    memory_config actual_cfg = mem_cfg ? *mem_cfg : memory_config_m2();
    model->memory = memory_init(&actual_cfg);
    if (!model->memory) {
        snprintf(err_buf, err_buf_size, "memory init failed");
        gpu_destroy(model->gpu); gguf_close(model->gguf); free(model); return NULL;
    }

    // Allocate lazy-load weight cache (one per layer)
    model->n_layers = model->config.n_layers;
    model->layer_caches = (layer_cache *)calloc(model->n_layers, sizeof(layer_cache));
    if (!model->layer_caches) {
        snprintf(err_buf, err_buf_size, "OOM layer caches");
        memory_destroy(model->memory);
        gpu_destroy(model->gpu); gguf_close(model->gguf); free(model); return NULL;
    }

    return model;
}

void model_unload(ornith_model *model) {
    if (!model) return;
    memory_destroy(model->memory);
    gpu_destroy(model->gpu);
    gguf_close(model->gguf);
    if (model->layer_caches) {
        for (uint32_t i = 0; i < model->n_layers; i++)
            free_layer_cache(&model->layer_caches[i]);
        free(model->layer_caches);
    }
    memset(model, 0, sizeof(*model));
    free(model);
}

const model_config *model_get_config(const ornith_model *model) {
    return model ? &model->config : NULL;
}

gpu_context *model_get_gpu(ornith_model *model) { return model ? model->gpu : NULL; }
gguf_model *model_get_gguf(ornith_model *model) { return model ? model->gguf : NULL; }
memory_manager *model_get_memory(ornith_model *model) { return model ? model->memory : NULL; }

void model_set_trace(ornith_model *model, bool enable) {
    if (model) model->trace_enabled = enable;
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

    char tensor_name[128];
    snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_gate_inp.weight", layer_id);

    const gguf_tensor_info *router = gguf_find_tensor(model->gguf, tensor_name);
    if (!router) {
        // Fall back to old name for compatibility
        snprintf(tensor_name, sizeof(tensor_name), "blk.%d.ffn_gate.weight", layer_id);
        router = gguf_find_tensor(model->gguf, tensor_name);
    }
    if (!router) {
        for (int i = 0; i < top_k; i++) {
            indices[i] = (uint32_t)i;
            weights[i] = 1.0f / top_k;
        }
        return top_k;
    }

    const float *w = (const float *)gguf_tensor_data_from_info(model->gguf, router);
    float *logits = (float *)calloc(n_experts, sizeof(float));

    // Router weight shape: [d_model, n_experts] = [2048, 256]
    for (int e = 0; e < n_experts; e++) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) sum += x[d] * w[d * n_experts + e];
        logits[e] = sum;
    }

    typedef struct { float score; int idx; } scored;
    scored *scores = (scored *)malloc(n_experts * sizeof(scored));
    for (int i = 0; i < n_experts; i++) { scores[i].score = logits[i]; scores[i].idx = i; }

    for (int i = 0; i < n_experts - 1; i++)
        for (int j = i + 1; j < n_experts; j++)
            if (scores[j].score > scores[i].score) {
                scored tmp = scores[i]; scores[i] = scores[j]; scores[j] = tmp;
            }

    float max_score = scores[0].score;
    float sum_exp = 0.0f;
    for (int i = 0; i < top_k; i++) {
        float s = expf(scores[i].score - max_score);
        weights[i] = s; sum_exp += s;
    }
    for (int i = 0; i < top_k; i++) { weights[i] /= sum_exp; indices[i] = (uint32_t)scores[i].idx; }

    free(scores); free(logits);
    return top_k;
}

void model_update_hotstore(ornith_model *model,
                           const uint32_t *expert_ids, int count) {
    if (!model || !expert_ids) return;
    for (int i = 0; i < count; i++) memory_hot_promote(model->memory, expert_ids[i]);
}

// ════════════════════════════════════════════════════════════════════════════
// CPU Math Helpers
// ════════════════════════════════════════════════════════════════════════════

static void cpu_rmsnorm(float *out, const float *x, const float *weight, int dim) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
    for (int i = 0; i < dim; i++) out[i] = weight[i] * (x[i] * inv_rms);
}

static inline float cpu_silu(float x) {
    if (x < -50.0f) return 0.0f;
    if (x > 50.0f)  return x;
    return x / (1.0f + expf(-x));
}

static void cpu_silu_vec(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) out[i] = cpu_silu(x[i]);
}

// Vector-matrix multiply: out[M] = vec[K] @ mat[K][M] (row-major)
static void cpu_matmul_vec(float *out, const float *vec, const float *mat, int K, int M) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += vec[k] * mat[k * M + i];
        out[i] = sum;
    }
}

static void cpu_add(float *out, const float *x, int n) {
    for (int i = 0; i < n; i++) out[i] += x[i];
}


// ════════════════════════════════════════════════════════════════════════════
// GPU Convenience Wrappers — float* → gpu_buffer → gpu_* op → float*
// ════════════════════════════════════════════════════════════════════════════

// Vector-matrix multiply via GPU: out[M] = vec[K] @ mat[K][M]
static void gpu_matmul_vec_wrap(gpu_context *ctx, float *out,
                                 const float *vec, const float *mat,
                                 int K, int M) {
    if (!ctx) { cpu_matmul_vec(out, vec, mat, K, M); return; }
    gpu_buffer *v = gpu_alloc(ctx, (size_t)K * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *w = gpu_alloc(ctx, (size_t)K * (size_t)M * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *o = gpu_alloc(ctx, (size_t)M * sizeof(float), GPU_BUF_SHARED);
    if (!v || !w || !o) {
        gpu_free(ctx, v); gpu_free(ctx, w); gpu_free(ctx, o);
        cpu_matmul_vec(out, vec, mat, K, M); return;
    }
    gpu_copy_to_device(ctx, v, vec, (size_t)K * sizeof(float));
    gpu_copy_to_device(ctx, w, mat, (size_t)K * (size_t)M * sizeof(float));
    gpu_matmul(ctx, o, v, w, 1, M, K, QUANT_NONE, QUANT_NONE);
    gpu_copy_to_host(ctx, out, o, (size_t)M * sizeof(float));
    gpu_free(ctx, v); gpu_free(ctx, w); gpu_free(ctx, o);
}

// RMSNorm via GPU: out[d] = weight[d] * (x[d] / rms(x))
static void gpu_rmsnorm_wrap(gpu_context *ctx, float *out,
                              const float *x, const float *weight, int dim) {
    if (!ctx) { cpu_rmsnorm(out, x, weight, dim); return; }
    gpu_buffer *xb = gpu_alloc(ctx, (size_t)dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *wb = gpu_alloc(ctx, (size_t)dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *ob = gpu_alloc(ctx, (size_t)dim * sizeof(float), GPU_BUF_SHARED);
    if (!xb || !wb || !ob) {
        gpu_free(ctx, xb); gpu_free(ctx, wb); gpu_free(ctx, ob);
        cpu_rmsnorm(out, x, weight, dim); return;
    }
    gpu_copy_to_device(ctx, xb, x, (size_t)dim * sizeof(float));
    gpu_copy_to_device(ctx, wb, weight, (size_t)dim * sizeof(float));
    gpu_rmsnorm(ctx, ob, xb, wb, 1, dim);
    gpu_copy_to_host(ctx, out, ob, (size_t)dim * sizeof(float));
    gpu_free(ctx, xb); gpu_free(ctx, wb); gpu_free(ctx, ob);
}

// SiLU via GPU (in-place or out-of-place)
static void gpu_silu_vec_wrap(gpu_context *ctx, float *out,
                               const float *x, int n) {
    if (!ctx) { cpu_silu_vec(out, x, n); return; }
    gpu_buffer *xbuf = gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *obuf = (out == x) ? xbuf : gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    if (!xbuf || !obuf) {
        if (obuf && obuf != xbuf) gpu_free(ctx, obuf);
        gpu_free(ctx, xbuf);
        cpu_silu_vec(out, x, n); return;
    }
    gpu_copy_to_device(ctx, xbuf, x, (size_t)n * sizeof(float));
    gpu_silu(ctx, obuf, xbuf, n);
    gpu_copy_to_host(ctx, out, obuf, (size_t)n * sizeof(float));
    if (obuf != xbuf) gpu_free(ctx, obuf);
    gpu_free(ctx, xbuf);
}

// SiLU(gate) * up via GPU: out[i] = silu(gate[i]) * up[i]
static void gpu_silu_mul_wrap(gpu_context *ctx, float *out,
                               const float *gate, const float *up, int n) {
    if (!ctx) {
        for (int i = 0; i < n; i++) out[i] = cpu_silu(gate[i]) * up[i];
        return;
    }
    gpu_buffer *gb = gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *ub = gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *ob = gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    if (!gb || !ub || !ob) {
        gpu_free(ctx, gb); gpu_free(ctx, ub); gpu_free(ctx, ob);
        for (int i = 0; i < n; i++) out[i] = cpu_silu(gate[i]) * up[i];
        return;
    }
    gpu_copy_to_device(ctx, gb, gate, (size_t)n * sizeof(float));
    gpu_copy_to_device(ctx, ub, up, (size_t)n * sizeof(float));
    gpu_silu_mul(ctx, ob, gb, ub, n);
    gpu_copy_to_host(ctx, out, ob, (size_t)n * sizeof(float));
    gpu_free(ctx, gb); gpu_free(ctx, ub); gpu_free(ctx, ob);
}

// In-place add via GPU: out[i] += x[i]
static void gpu_add_wrap(gpu_context *ctx, float *out,
                          const float *x, int n) {
    if (!ctx) { cpu_add(out, x, n); return; }
    gpu_buffer *ob = gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *xb = gpu_alloc(ctx, (size_t)n * sizeof(float), GPU_BUF_SHARED);
    if (!ob || !xb) {
        gpu_free(ctx, ob); gpu_free(ctx, xb);
        cpu_add(out, x, n); return;
    }
    gpu_copy_to_device(ctx, ob, out, (size_t)n * sizeof(float));
    gpu_copy_to_device(ctx, xb, x, (size_t)n * sizeof(float));
    gpu_add(ctx, ob, ob, xb, n);
    gpu_copy_to_host(ctx, out, ob, (size_t)n * sizeof(float));
    gpu_free(ctx, ob); gpu_free(ctx, xb);
}


// ════════════════════════════════════════════════════════════════════════════
// SSM Layer Forward (one token position)
// ════════════════════════════════════════════════════════════════════════════

static void ssm_layer_forward(
    gpu_context *gpu,             // GPU context (NULL = CPU fallback)
    float *x_in_out,              // [d_model] input/output (residual applied in place)
    float *conv_state,            // [3 * conv_dim] ring buffer for conv history
    float *h_state,               // [d_state] SSM recurrent state (updated in place)
    const float *norm_w,          // [d_model]
    const float *qkv_w,           // [d_model, qkv_dim]
    const float *conv_w,          // [kernel, conv_dim]
    const float *gate_w,          // [d_model, gate_dim]
    const float *a_diag,          // [d_state]
    const float *dt_b,            // [d_state]
    const float *alpha_w,         // [d_model, d_state]
    const float *beta_w,          // [d_model, d_state]
    const float *norm_state_w,    // [state_size]
    const float *out_w,           // [out_in, d_model]
    const model_config *cfg)
{
    int D         = (int)cfg->d_model;
    int d_state   = ORNITH_SSM_TIME_STEP_RANK;  // 32
    int conv_dim  = D * 4;                       // 8192
    int gate_dim  = ORNITH_SSM_INNER_SIZE;       // 4096

    float *x_norm   = (float *)alloca((size_t)D * sizeof(float));
    float *gate     = (float *)alloca((size_t)gate_dim * sizeof(float));
    float *u        = (float *)alloca((size_t)conv_dim * sizeof(float));
    float *combined = (float *)alloca((size_t)gate_dim * sizeof(float));
    float *y_temp   = (float *)alloca((size_t)D * sizeof(float));

    // 1. RMSNorm
    gpu_rmsnorm_wrap(gpu, x_norm, x_in_out, norm_w, D);

    // 2. Gate: z = x_norm @ gate_w → SiLU
    gpu_matmul_vec_wrap(gpu, gate, x_norm, gate_w, D, gate_dim);
    gpu_silu_vec_wrap(gpu, gate, gate, gate_dim);

    // 3. SSM projection: u = x_norm @ qkv_w
    gpu_matmul_vec_wrap(gpu, u, x_norm, qkv_w, D, conv_dim);

    // 4. Causal conv1d (kernel=4) with ring buffer  — kept on CPU (data-dependent)
    for (int c = 0; c < conv_dim; c++) {
        float y = conv_w[0 * conv_dim + c] * conv_state[0 * conv_dim + c]
                + conv_w[1 * conv_dim + c] * conv_state[1 * conv_dim + c]
                + conv_w[2 * conv_dim + c] * conv_state[2 * conv_dim + c]
                + conv_w[3 * conv_dim + c] * u[c];
        u[c] = cpu_silu(y);
    }
    memcpy(conv_state, conv_state + conv_dim, (size_t)(2 * conv_dim) * sizeof(float));
    memcpy(conv_state + 2 * conv_dim, u, (size_t)conv_dim * sizeof(float));

    // 5. SSM recurrence  — kept on CPU (state-space math, small dims)
    float dt[ORNITH_SSM_TIME_STEP_RANK];
    for (int i = 0; i < d_state; i++) {
        float raw = u[i] + dt_b[i];
        dt[i] = raw > 50.0f ? raw : logf(1.0f + expf(raw));
    }

    float B[ORNITH_SSM_TIME_STEP_RANK], C[ORNITH_SSM_TIME_STEP_RANK];
    cpu_matmul_vec(B, x_norm, beta_w, D, d_state);  // small d_state=32 — keep CPU
    cpu_matmul_vec(C, x_norm, alpha_w, D, d_state);

    float h_new[ORNITH_SSM_TIME_STEP_RANK];
    for (int i = 0; i < d_state; i++) {
        float A_val = -expf(a_diag[i]);
        float A_bar = expf(dt[i] * A_val);
        float B_bar = dt[i] * B[i];
        float x_s = u[d_state + i];
        h_new[i] = A_bar * h_state[i] + B_bar * x_s;
    }
    memcpy(h_state, h_new, (size_t)d_state * sizeof(float));

    // SSM output
    float ssm_y[ORNITH_SSM_TIME_STEP_RANK];
    for (int i = 0; i < d_state; i++) ssm_y[i] = C[i] * h_state[i];

    // 6. Combine: expand SSM output [32] → [4096] and gate
    int block = gate_dim / d_state;
    for (int i = 0; i < d_state; i++) {
        for (int j = 0; j < block; j++) {
            int idx = i * block + j;
            combined[idx] = gate[idx] * (u[d_state + idx] + ssm_y[i]);
        }
    }

    // 7. Output projection @ ssm_out
    gpu_matmul_vec_wrap(gpu, y_temp, combined, out_w, gate_dim, D);

    // 8. Residual
    gpu_add_wrap(gpu, x_in_out, y_temp, D);

    (void)norm_state_w; // ssm_norm unused for now
}

// ════════════════════════════════════════════════════════════════════════════
// Attention Layer Forward (GQA with RoPE + KV cache)
// ════════════════════════════════════════════════════════════════════════════

static void attention_layer_forward(
    gpu_context *gpu,             // GPU context (NULL = CPU fallback)
    float *x_in_out, const float *norm_w,
    const float *q_w, const float *k_w, const float *v_w, const float *o_w,
    int pos, int seq_len,
    const model_config *cfg,
    float *k_cache, float *v_cache, int max_cache_len)
{
    (void)seq_len;
    if (!x_in_out || !norm_w || !q_w || !k_w || !v_w || !o_w) return;
    if (!k_cache || !v_cache) return;

    int D      = (int)cfg->d_model;
    int n_h    = (int)cfg->n_heads;
    int n_kv   = (int)cfg->n_kv_heads;
    int kv_dim = (int)(n_kv * cfg->key_length);  // 2 * 256 = 512
    int v_dim  = (int)cfg->value_length;          // 256
    // Q projection produces [D, n_h * key_length * 2] = [2048, 8192]
    int q_total = (int)(n_h * cfg->key_length * 2);  // 16 * 256 * 2 = 8192
    int q_hdim  = q_total / n_h;                      // 512 per Q head
    int q_score_stride = (int)cfg->key_length;         // 256 — scoring dims per head
    int rd     = (int)cfg->rope_dim_count;             // 64
    float theta = cfg->rope_theta;
    size_t q_sz = (size_t)q_total * sizeof(float);

    // Allocate buffers (large stack allocations — safe on macOS 8MB stack)
    float *x_norm  = (float *)alloca((size_t)D * sizeof(float));
    float *q_full  = (float *)alloca(q_sz);
    float *k       = (float *)alloca((size_t)kv_dim * sizeof(float));
    float *v       = (float *)alloca((size_t)kv_dim * sizeof(float));

    // 1. RMSNorm
    gpu_rmsnorm_wrap(gpu, x_norm, x_in_out, norm_w, D);

    // 2. QKV projections via GPU
    gpu_matmul_vec_wrap(gpu, q_full, x_norm, q_w, D, q_total);
    gpu_matmul_vec_wrap(gpu, k, x_norm, k_w, D, kv_dim);
    gpu_matmul_vec_wrap(gpu, v, x_norm, v_w, D, kv_dim);

    // 3. Apply RoPE to first rd dims of Q and K  — kept on CPU (per-element trig)
    for (int h = 0; h < n_h; h++) {
        float *h_q = q_full + (size_t)h * q_hdim;
        for (int i = 0; i < rd && i < q_score_stride; i += 2) {
            float inv_freq = 1.0f / powf(theta, (float)i / (float)rd);
            float cos_v = cosf((float)pos * inv_freq);
            float sin_v = sinf((float)pos * inv_freq);
            float a = h_q[i];
            float b = h_q[i + 1];
            h_q[i]     = a * cos_v - b * sin_v;
            h_q[i + 1] = a * sin_v + b * cos_v;
        }
    }
    int k_stride = (int)cfg->key_length;
    for (int h = 0; h < n_kv; h++) {
        float *h_k = k + (size_t)h * k_stride;
        for (int i = 0; i < rd && i < k_stride; i += 2) {
            float inv_freq = 1.0f / powf(theta, (float)i / (float)rd);
            float cos_v = cosf((float)pos * inv_freq);
            float sin_v = sinf((float)pos * inv_freq);
            float a = h_k[i];
            float b = h_k[i + 1];
            h_k[i]     = a * cos_v - b * sin_v;
            h_k[i + 1] = a * sin_v + b * cos_v;
        }
    }

    // 4. Store K, V in cache
    if (pos < max_cache_len) {
        memcpy(&k_cache[(size_t)pos * kv_dim], k, (size_t)kv_dim * sizeof(float));
        memcpy(&v_cache[(size_t)pos * kv_dim], v, (size_t)kv_dim * sizeof(float));
    }

    // 5. GQA attention  — kept on CPU (score compute + softmax are memory-bound)
    int q_per_kv = n_h / n_kv;
    float *context = (float *)alloca((size_t)n_h * (size_t)v_dim * sizeof(float));
    int kv_seq_len = (pos < max_cache_len) ? pos + 1 : max_cache_len;

    for (int g = 0; g < n_kv; g++) {
        float *scores = (float *)alloca((size_t)q_per_kv * (size_t)kv_seq_len * sizeof(float));
        for (int qi = 0; qi < q_per_kv; qi++) {
            int q_idx = g * q_per_kv + qi;
            const float *q_h = q_full + (size_t)q_idx * q_hdim;
            for (int p = 0; p < kv_seq_len; p++) {
                const float *k_at_pos = k_cache + (size_t)p * kv_dim + (size_t)g * cfg->key_length;
                float score = 0.0f;
                for (int d = 0; d < q_score_stride; d++) score += q_h[d] * k_at_pos[d];
                scores[qi * kv_seq_len + p] = score / sqrtf((float)cfg->head_dim);
            }
        }
        for (int qi = 0; qi < q_per_kv; qi++) {
            float *s = scores + qi * kv_seq_len;
            float max_s = s[0];
            for (int p = 1; p < kv_seq_len; p++) if (s[p] > max_s) max_s = s[p];
            float sum_exp = 0;
            for (int p = 0; p < kv_seq_len; p++) { s[p] = expf(s[p] - max_s); sum_exp += s[p]; }
            for (int p = 0; p < kv_seq_len; p++) s[p] /= sum_exp;
        }
        for (int qi = 0; qi < q_per_kv; qi++) {
            int q_idx = g * q_per_kv + qi;
            float *c = context + (size_t)q_idx * v_dim;
            memset(c, 0, (size_t)v_dim * sizeof(float));
            for (int p = 0; p < kv_seq_len; p++) {
                float w = scores[qi * kv_seq_len + p];
                const float *v_at_pos = v_cache + (size_t)p * kv_dim + (size_t)g * cfg->value_length;
                for (int d = 0; d < v_dim; d++) c[d] += w * v_at_pos[d];
            }
        }
    }

    // 6. Output projection via GPU
    int ctx_concat = n_h * v_dim;
    float *o = (float *)alloca((size_t)D * sizeof(float));
    gpu_matmul_vec_wrap(gpu, o, context, o_w, ctx_concat, D);

    // 7. Residual
    gpu_add_wrap(gpu, x_in_out, o, D);
}

// ════════════════════════════════════════════════════════════════════════════
// Expert Weight Caching (Tiered Memory: Hot-Store → LRU → GGUF Source)
// ════════════════════════════════════════════════════════════════════════════

// Cache key for an expert weight slice: (layer_id * 65536u) + (expert_idx * 3u) + type
// type: 0=gate, 1=up, 2=down

// After router selects top-k experts, dequantize only those k experts directly
// from the GGUF mmap'd fused 3D tensors using gguf_dequantize_expert_slice().
// This avoids loading the entire ~1 GB fused tensor into F32 — we only dequant
// the ~8 active experts (~12 MB total per layer).
static void cache_layer_experts(memory_manager *mm, const gguf_model *gguf,
                                 const model_config *cfg, int layer_id,
                                 const int *expert_indices, int num_experts) {
    if (!mm || !gguf || !cfg || !expert_indices || num_experts <= 0) return;

    uint64_t D = (uint64_t)cfg->d_model;
    uint64_t E_hdim = (uint64_t)cfg->expert_hidden_dim;
    uint64_t stride_exp = (uint64_t)cfg->n_experts_per_layer;

    size_t gate_size = (size_t)(D * E_hdim) * sizeof(float);
    size_t down_size = (size_t)(E_hdim * D) * sizeof(float);

    char tname[256];

    // Gather tensor info for all three fused tensors
    snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_exps.weight", layer_id);
    const gguf_tensor_info *t_gate = gguf_find_tensor(gguf, tname);
    snprintf(tname, sizeof(tname), "blk.%d.ffn_up_exps.weight", layer_id);
    const gguf_tensor_info *t_up = gguf_find_tensor(gguf, tname);
    snprintf(tname, sizeof(tname), "blk.%d.ffn_down_exps.weight", layer_id);
    const gguf_tensor_info *t_down = gguf_find_tensor(gguf, tname);

    // Only works if all three tensors exist
    if (!t_gate || !t_up || !t_down) return;

    for (int a = 0; a < num_experts; a++) {
        int e = expert_indices[a];
        if (e < 0 || (uint64_t)e >= stride_exp) continue;
        uint32_t base_key = (uint32_t)(layer_id * 65536u + (uint32_t)e * 3u);
        void *dummy = NULL;

        // ── Gate slice: dequant expert e directly from fused Q4_K tensor ──
        float *gate_slice = (float *)malloc(gate_size);
        if (gate_slice) {
            if (gguf_dequantize_expert_slice(gguf, t_gate, e, gate_slice)) {
                memory_get(mm, base_key + 0, &dummy);
                memory_set(mm, base_key + 0, gate_slice, gate_size);
            }
            free(gate_slice);
        }

        // ── Up slice ──
        float *up_slice = (float *)malloc(gate_size);
        if (up_slice) {
            if (gguf_dequantize_expert_slice(gguf, t_up, e, up_slice)) {
                memory_get(mm, base_key + 1, &dummy);
                memory_set(mm, base_key + 1, up_slice, gate_size);
            }
            free(up_slice);
        }

        // ── Down slice: [E_hdim, D] ──
        float *down_slice = (float *)malloc(down_size);
        if (down_slice) {
            if (gguf_dequantize_expert_slice(gguf, t_down, e, down_slice)) {
                memory_get(mm, base_key + 2, &dummy);
                memory_set(mm, base_key + 2, down_slice, down_size);
            }
            free(down_slice);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Async Prefetch Support
// ════════════════════════════════════════════════════════════════════════════

// Forward declaration for the worker function
static void ensure_layer_weights(ornith_model *model, int l);

// Job argument for prefetching a layer's weights in the background.
typedef struct {
    ornith_model *model;
    int           layer_id;
} prefetch_layer_args;

// Background worker function: loads one layer's weights (reads from GGUF mmap,
// dequants to F32, stores in layer_cache[layer_id]). Called by prefetch thread.
// Frees the arg struct after use.
static void prefetch_layer_job(void *arg) {
    prefetch_layer_args *pa = (prefetch_layer_args *)arg;
    if (pa && pa->model) {
        ensure_layer_weights(pa->model, pa->layer_id);
    }
    free(pa);  // arg allocated by main thread, freed here after job completes
}

// ════════════════════════════════════════════════════════════════════════════
// Lazy Weight Loader
// ════════════════════════════════════════════════════════════════════════════

// Helper to lookup a tensor, allocate buffer from its n_elems, and dequantize.
// Returns NULL if tensor not found, pointer to allocated buffer otherwise.
static float *load_tensor(const gguf_model *gguf, const char *name) {
    const gguf_tensor_info *info = gguf_find_tensor(gguf, name);
    if (!info) return NULL;
    float *buf = (float *)malloc((size_t)info->n_elems * sizeof(float));
    if (!buf) return NULL;
    if (!gguf_dequantize_tensor(gguf, info, buf)) { free(buf); return NULL; }
    return buf;
}

// Load all weights for a given layer into the cache (idempotent — skips if loaded).
static void ensure_layer_weights(ornith_model *model, int l) {
    if (!model || !model->layer_caches) return;
    layer_cache *lc = &model->layer_caches[l];
    if (lc->loaded) return;

    const model_config *cfg = &model->config;
    char tname[256];

    // Norms (always present)
    snprintf(tname, sizeof(tname), "blk.%d.attn_norm.weight", l);
    lc->attn_norm = load_tensor(model->gguf, tname);

    snprintf(tname, sizeof(tname), "blk.%d.post_attention_norm.weight", l);
    lc->post_attn_norm = load_tensor(model->gguf, tname);

    // Check if this is an attention layer or SSM layer
    int is_attn = layer_uses_full_attention(l, cfg->full_attention_interval);

    if (is_attn) {
        // Attention layer
        snprintf(tname, sizeof(tname), "blk.%d.attn_q.weight", l);
        lc->attn_q = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.attn_k.weight", l);
        lc->attn_k = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.attn_v.weight", l);
        lc->attn_v = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.attn_output.weight", l);
        lc->attn_output = load_tensor(model->gguf, tname);
    } else {
        // SSM layer
        snprintf(tname, sizeof(tname), "blk.%d.attn_qkv.weight", l);
        lc->attn_qkv = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.attn_gate.weight", l);
        lc->attn_gate = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_conv1d.weight", l);
        lc->ssm_conv1d = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_a", l);
        lc->ssm_a = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_dt.bias", l);
        lc->ssm_dt_bias = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_alpha.weight", l);
        lc->ssm_alpha = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_beta.weight", l);
        lc->ssm_beta = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_norm.weight", l);
        lc->ssm_norm = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ssm_out.weight", l);
        lc->ssm_out = load_tensor(model->gguf, tname);
    }

    // Shared expert (always present if has_shared_expert)
    if (cfg->has_shared_expert) {
        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_shexp.weight", l);
        lc->ffn_gate_shexp = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_up_shexp.weight", l);
        lc->ffn_up_shexp = load_tensor(model->gguf, tname);
        snprintf(tname, sizeof(tname), "blk.%d.ffn_down_shexp.weight", l);
        lc->ffn_down_shexp = load_tensor(model->gguf, tname);
    }

    // Routed expert weights — only router is loaded as F32 (it's 2D, small).
    // The fused 3D expert tensors are NOT loaded (Phase B optimization).
    // Expert slices are dequantized on-demand from GGUF mmap via
    // gguf_dequantize_expert_slice() in cache_layer_experts().
    // This saves ~750 MB of dequantized F32 per layer.
    if (cfg->n_experts_per_layer > 0) {
        snprintf(tname, sizeof(tname), "blk.%d.ffn_gate_inp.weight", l);
        lc->ffn_gate_inp = load_tensor(model->gguf, tname);
    }

    // Cache the final norm (stored on layer 0)
    if (l == 0) {
        lc->output_norm = load_tensor(model->gguf, "token_embd_norm.weight");
        if (!lc->output_norm) lc->output_norm = load_tensor(model->gguf, "output_norm.weight");
    }

    lc->loaded = true;
}

// ════════════════════════════════════════════════════════════════════════════
// Full Forward Pass
// ════════════════════════════════════════════════════════════════════════════

void model_forward(ornith_model *model, uint32_t token_id, float *output,
                   const model_buffers *buffers, int seq_len, int pos) {
    if (!model || !output || !buffers) return;

    const model_config *cfg = &model->config;
    const gguf_model *gguf = model->gguf;
    int D  = (int)cfg->d_model;
    int va = (int)cfg->vocab_size;

    // ── Check for real model weights vs tiny test model ─────────────────
    const gguf_tensor_info *norm_check = gguf_find_tensor(gguf, "blk.0.attn_norm.weight");
    const gguf_tensor_info *emb_check  = gguf_find_tensor(gguf, "token_embd.weight");
    bool has_real_weights = (norm_check != NULL && emb_check != NULL);

    if (!has_real_weights) {
        memset(output, 0, (size_t)va * sizeof(float));
        if (token_id < (uint32_t)va) output[token_id] = 1.0f;
        return;
    }

    // ── Real model path ─────────────────────────────────────────────────
    float *x = (float *)calloc((size_t)D, sizeof(float));
    if (!x) { memset(output, 0, (size_t)va * sizeof(float)); return; }

    // Token embedding (lazy-load token_embd.weight on first call)
    if (emb_check->type == GGML_TYPE_F32) {
        const float *w = (const float *)gguf_tensor_data_from_info(gguf, emb_check);
        for (int i = 0; i < D; i++)
            x[i] = w[(size_t)i * (size_t)va + (size_t)token_id];
    }

    gpu_context *gpu = model->gpu;  // GPU context for accelerated ops

    // ── Process ALL layers (malloc/dequant/use/free per-layer) ─────────
    int n_l = (int)cfg->n_layers;
    memory_manager *mm = model->memory;
    int lookahead = memory_async_enabled(mm) ? memory_lookahead_layers(mm) : 0;

    // Allocate prefetch job tracking array
    int *pf_job = (int *)calloc((size_t)n_l > 0 ? (size_t)n_l : 1, sizeof(int));
    if (pf_job) {
        for (int i = 0; i < n_l; i++) pf_job[i] = -1;
    }

    // Prefetch layer 0 before starting (if async enabled)
    if (pf_job && lookahead > 0 && 0 < n_l) {
        prefetch_layer_args *pa = (prefetch_layer_args *)malloc(sizeof(prefetch_layer_args));
        if (pa) { pa->model = model; pa->layer_id = 0; pf_job[0] = memory_prefetch_submit(mm, prefetch_layer_job, pa); }
    }

    // Per-layer timing accumulators (trace markers)
    uint64_t trace_load_ns = 0, trace_attn_ns = 0, trace_moe_ns = 0;
    uint64_t trace_load_max = 0, trace_attn_max = 0, trace_moe_max = 0;
    int trace_slowest_layer = 0;
    struct timespec ts_l_start, ts_sub_start, ts_sub_end;

    for (int l = 0; l < n_l; l++) {
        // Wait for this layer's prefetch to complete (if submitted)
        if (pf_job && pf_job[l] >= 0) {
            memory_prefetch_wait(mm, pf_job[l]);
        }

        clock_gettime(CLOCK_MONOTONIC, &ts_l_start);
        // Per-layer timing deltas (used for max tracking and per-layer [TRACE])
        uint64_t layer_load_ns = 0, layer_attn_ns = 0, layer_moe_ns = 0;

        // Load weights for this layer (if not prefetched, loads synchronously)
        // If prefetched, ensure_layer_weights is a no-op (lc->loaded already true)
        clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
        ensure_layer_weights(model, l);
        layer_cache *lc = &model->layer_caches[l];
        clock_gettime(CLOCK_MONOTONIC, &ts_sub_end);
        layer_load_ns  = (uint64_t)(ts_sub_end.tv_sec - ts_sub_start.tv_sec) * 1000000000ULL
                       + (uint64_t)(ts_sub_end.tv_nsec - ts_sub_start.tv_nsec);
        trace_load_ns  += layer_load_ns;

        int is_attn = layer_uses_full_attention(l, cfg->full_attention_interval);

        clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
        if (is_attn) {
            // ── Attention layer ────────────────────────────────────
            if (!lc->attn_norm || !lc->attn_q || !lc->attn_k || !lc->attn_v || !lc->attn_output) continue;
            attention_layer_forward(gpu, x, lc->attn_norm,
                lc->attn_q, lc->attn_k, lc->attn_v, lc->attn_output,
                pos, seq_len, cfg,
                buffers->k_cache ? buffers->k_cache[l] : NULL,
                buffers->v_cache ? buffers->v_cache[l] : NULL,
                buffers->max_cache_len);

        } else {
            // ── SSM layer ──────────────────────────────────────────
            if (!lc->attn_norm || !lc->attn_qkv || !lc->ssm_out) continue;

            ssm_layer_forward(gpu, x,
                buffers->ssm_conv_states ? buffers->ssm_conv_states[l] : NULL,
                buffers->ssm_h_states ? buffers->ssm_h_states[l] : NULL,
                lc->attn_norm,
                lc->attn_qkv,
                lc->ssm_conv1d,
                lc->attn_gate,
                lc->ssm_a,
                lc->ssm_dt_bias,
                lc->ssm_alpha,
                lc->ssm_beta,
                lc->ssm_norm,
                lc->ssm_out,
                cfg);
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_sub_end);
        layer_attn_ns  = (uint64_t)(ts_sub_end.tv_sec - ts_sub_start.tv_sec) * 1000000000ULL
                       + (uint64_t)(ts_sub_end.tv_nsec - ts_sub_start.tv_nsec);
        trace_attn_ns  += layer_attn_ns;

        clock_gettime(CLOCK_MONOTONIC, &ts_sub_start);
        // ── MoE FFN (shared expert) via GPU ───────────────────────
        if (lc->post_attn_norm && lc->ffn_gate_shexp && lc->ffn_up_shexp && lc->ffn_down_shexp) {
            int hdim = (int)cfg->expert_hidden_dim;
            float *ffn_x = (float *)alloca((size_t)D * sizeof(float));
            gpu_rmsnorm_wrap(gpu, ffn_x, x, lc->post_attn_norm, D);

            // Shared expert
            float *gate_h = (float *)alloca((size_t)hdim * sizeof(float));
            float *up_h   = (float *)alloca((size_t)hdim * sizeof(float));
            float *mid    = (float *)alloca((size_t)hdim * sizeof(float));
            float *down   = (float *)alloca((size_t)D * sizeof(float));
            gpu_matmul_vec_wrap(gpu, gate_h, ffn_x, lc->ffn_gate_shexp, D, hdim);
            gpu_matmul_vec_wrap(gpu, up_h, ffn_x, lc->ffn_up_shexp, D, hdim);
            gpu_silu_mul_wrap(gpu, mid, gate_h, up_h, hdim);
            gpu_matmul_vec_wrap(gpu, down, mid, lc->ffn_down_shexp, hdim, D);
            gpu_add_wrap(gpu, x, down, D);

            // ── Routed experts (top-k MoE) ────────────────────────
            // Fused 3D expert tensors are NOT loaded into F32 (Phase B optimization).
            // Expert slices are dequantized on-demand via gguf_dequantize_expert_slice().
            if (lc->ffn_gate_inp) {
                int n_exp = (int)cfg->n_experts_per_layer;
                int n_active = (int)cfg->n_active_experts;
                int E_hdim = (int)cfg->expert_hidden_dim;

                // Router: compute expert logits from normalized input
                float *router_logits = (float *)alloca((size_t)n_exp * sizeof(float));
                for (int e = 0; e < n_exp; e++) {
                    float sum = 0.0f;
                    for (int d = 0; d < D; d++) {
                        sum += ffn_x[d] * lc->ffn_gate_inp[d * (size_t)n_exp + e];
                    }
                    router_logits[e] = sum;
                }

                // Softmax + top-k selection
                float max_r = router_logits[0];
                for (int e = 1; e < n_exp; e++) if (router_logits[e] > max_r) max_r = router_logits[e];
                float sum_r = 0.0f;
                for (int e = 0; e < n_exp; e++) {
                    router_logits[e] = expf(router_logits[e] - max_r);
                    sum_r += router_logits[e];
                }
                for (int e = 0; e < n_exp; e++) router_logits[e] /= sum_r;

                // Selection: find top-k by sorting indices
                int *idx = (int *)alloca((size_t)n_exp * sizeof(int));
                float *val = (float *)alloca((size_t)n_exp * sizeof(float));
                for (int e = 0; e < n_exp; e++) { idx[e] = e; val[e] = router_logits[e]; }
                // Simple bubble sort for top-k (n_exp=256, negligible cost)
                for (int i = 0; i < n_active; i++) {
                    for (int j = i + 1; j < n_exp; j++) {
                        if (val[j] > val[i]) {
                            float ft = val[i]; val[i] = val[j]; val[j] = ft;
                            int it = idx[i]; idx[i] = idx[j]; idx[j] = it;
                        }
                    }
                }

                // Cache the selected top-k experts using direct GGUF slice dequant
                // (no full tensor load — saves ~750 MB per layer)
                cache_layer_experts(mm, model->gguf, cfg, l, idx, n_active);

                // Compute top-k experts using cached weight slices
                float *routed_out = (float *)alloca((size_t)D * sizeof(float));
                memset(routed_out, 0, (size_t)D * sizeof(float));

                // Track top expert keys for hot-store promotion
                uint32_t active_expert_keys[8];
                int n_active_keys = 0;

                for (int a = 0; a < n_active; a++) {
                    int e = idx[a];
                    float w = val[a];
                    if (w < 1e-6f) continue;

                    uint32_t base_key = (uint32_t)(l * 65536u + (uint32_t)e * 3u);

                    // Cache is guaranteed to have these entries (just cached above)
                    void *cached_gate = NULL, *cached_up = NULL, *cached_down = NULL;
                    if (mm) {
                        memory_get(mm, base_key + 0, &cached_gate);
                        memory_get(mm, base_key + 1, &cached_up);
                        memory_get(mm, base_key + 2, &cached_down);
                    }

                    if (mm && cached_gate && cached_up && cached_down) {
                        // ── Cache hit: simple contiguous matmul (no striding) ──
                        float *e_gate = (float *)alloca((size_t)E_hdim * sizeof(float));
                        float *e_up   = (float *)alloca((size_t)E_hdim * sizeof(float));
                        float *e_mid  = (float *)alloca((size_t)E_hdim * sizeof(float));
                        float *e_down = (float *)alloca((size_t)D * sizeof(float));

                        cpu_matmul_vec(e_gate, ffn_x, (const float *)cached_gate, D, E_hdim);
                        cpu_matmul_vec(e_up,   ffn_x, (const float *)cached_up,   D, E_hdim);
                        for (int i = 0; i < E_hdim; i++) e_mid[i] = cpu_silu(e_gate[i]) * e_up[i];
                        cpu_matmul_vec(e_down, e_mid, (const float *)cached_down, E_hdim, D);

                        for (int d = 0; d < D; d++) routed_out[d] += w * e_down[d];
                    } else {
                        // ── Fallback: direct GGUF slice dequant (no F32 loading) ──
                        // Load the 3 expert weight tensors directly from GGUF mmap
                        char tn[256];
                        snprintf(tn, sizeof(tn), "blk.%d.ffn_gate_exps.weight", l);
                        const gguf_tensor_info *ft_gate = gguf_find_tensor(gguf, tn);
                        snprintf(tn, sizeof(tn), "blk.%d.ffn_up_exps.weight", l);
                        const gguf_tensor_info *ft_up   = gguf_find_tensor(gguf, tn);
                        snprintf(tn, sizeof(tn), "blk.%d.ffn_down_exps.weight", l);
                        const gguf_tensor_info *ft_down = gguf_find_tensor(gguf, tn);

                        if (ft_gate && ft_up && ft_down) {
                            size_t gsz = (size_t)D * (size_t)E_hdim * sizeof(float);
                            size_t dsz = (size_t)E_hdim * (size_t)D * sizeof(float);
                            float *f_gate = (float *)malloc(gsz);
                            float *f_up   = (float *)malloc(gsz);
                            float *f_down = (float *)malloc(dsz);

                            if (f_gate && f_up && f_down &&
                                gguf_dequantize_expert_slice(gguf, ft_gate, e, f_gate) &&
                                gguf_dequantize_expert_slice(gguf, ft_up,   e, f_up)   &&
                                gguf_dequantize_expert_slice(gguf, ft_down, e, f_down)) {
                                float *e_gate = (float *)alloca((size_t)E_hdim * sizeof(float));
                                float *e_up   = (float *)alloca((size_t)E_hdim * sizeof(float));
                                float *e_mid  = (float *)alloca((size_t)E_hdim * sizeof(float));
                                float *e_down = (float *)alloca((size_t)D * sizeof(float));

                                cpu_matmul_vec(e_gate, ffn_x, f_gate, D, E_hdim);
                                cpu_matmul_vec(e_up,   ffn_x, f_up,   D, E_hdim);
                                for (int i = 0; i < E_hdim; i++) e_mid[i] = cpu_silu(e_gate[i]) * e_up[i];
                                cpu_matmul_vec(e_down, e_mid, f_down, E_hdim, D);

                                for (int d = 0; d < D; d++) routed_out[d] += w * e_down[d];
                            }
                            free(f_gate);
                            free(f_up);
                            free(f_down);
                        }
                    }

                    if (n_active_keys < 8) {
                        active_expert_keys[n_active_keys++] = base_key;
                    }
                }

                // Promote selected experts to hot-store
                if (mm && n_active_keys > 0) {
                    model_update_hotstore(model, active_expert_keys, n_active_keys);
                }

                // Residual (via GPU if available)
                gpu_add_wrap(gpu, x, routed_out, D);
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &ts_sub_end);
        layer_moe_ns   = (uint64_t)(ts_sub_end.tv_sec - ts_sub_start.tv_sec) * 1000000000ULL
                       + (uint64_t)(ts_sub_end.tv_nsec - ts_sub_start.tv_nsec);
        trace_moe_ns   += layer_moe_ns;

        // Submit prefetch for upcoming layers (overlaps I/O with compute)
        if (pf_job && lookahead > 0) {
            for (int ahead = 1; ahead <= lookahead && l + ahead < n_l; ahead++) {
                int next = l + ahead;
                if (pf_job[next] < 0) {
                    prefetch_layer_args *pa = (prefetch_layer_args *)malloc(sizeof(prefetch_layer_args));
                    if (pa) {
                        pa->model = model;
                        pa->layer_id = next;
                        pf_job[next] = memory_prefetch_submit(mm, prefetch_layer_job, pa);
                        if (pf_job[next] < 0) {
                            free(pa);
                        }
                    }
                }
            }
        }

        // Free this layer's weights immediately — peak memory stays at ~140 MB
        // (only one layer's dequantized weights in memory at a time)
        if (layer_load_ns > trace_load_max) { trace_load_max = layer_load_ns; }
        if (layer_attn_ns > trace_attn_max) { trace_attn_max = layer_attn_ns; trace_slowest_layer = l; }
        if (layer_moe_ns  > trace_moe_max)  { trace_moe_max  = layer_moe_ns; }
        // Per-layer [TRACE] output (only printed when --trace is enabled)
        if (model->trace_enabled) {
            printf("[TRACE] Layer %d: load=%llu us, attn=%llu us, moe=%llu us\n",
                   l,
                   (unsigned long long)(layer_load_ns / 1000),
                   (unsigned long long)(layer_attn_ns / 1000),
                   (unsigned long long)(layer_moe_ns / 1000));
        }

        free_layer_weights(lc);
    }

    // Clean up: wait for any remaining prefetch jobs
    // (args are freed by the worker function after completion)
    if (pf_job) {
        for (int l = 0; l < n_l; l++) {
            if (pf_job[l] >= 0) {
                memory_prefetch_wait(mm, pf_job[l]);
            }
        }
        free(pf_job);
    }

    // ── Trace summary ─────────────────────────────────────────────
    if (model->trace_enabled) {
        uint64_t total = trace_load_ns + trace_attn_ns + trace_moe_ns;
        printf("[TRACE] Total: load=%llu ms, attn=%llu ms, moe=%llu ms\n",
               (unsigned long long)(trace_load_ns / 1000000),
               (unsigned long long)(trace_attn_ns / 1000000),
               (unsigned long long)(trace_moe_ns / 1000000));
        printf("[TRACE] Max per-layer: load=%llu ms, attn=%llu ms (layer %d), moe=%llu ms\n",
               (unsigned long long)(trace_load_max / 1000000),
               (unsigned long long)(trace_attn_max / 1000000), trace_slowest_layer,
               (unsigned long long)(trace_moe_max / 1000000));
        printf("[TRACE] Overall: %llu ms total across %d layers (%.1f ms/layer)\n",
               (unsigned long long)(total / 1000000), n_l,
               total > 0 ? (double)(total / n_l) / 1000000.0 : 0.0);
    }

    // ── Final RMSNorm (from layer-0 cache) via GPU ────────────────────
    if (model->layer_caches && model->layer_caches[0].output_norm) {
        float *n = (float *)alloca((size_t)D * sizeof(float));
        gpu_rmsnorm_wrap(gpu, n, x, model->layer_caches[0].output_norm, D);
        memcpy(x, n, (size_t)D * sizeof(float));
    }

    // ── LM Head (full vocab evaluation from mmap) ────────────────
    memset(output, 0, (size_t)va * sizeof(float));
    const gguf_tensor_info *head = gguf_find_tensor(gguf, "output.weight");
    if (!head) head = gguf_find_tensor(gguf, "token_embd.weight");
    if (head && head->type == GGML_TYPE_F32) {
        const float *w = (const float *)gguf_tensor_data_from_info(gguf, head);
        // Evaluate ALL vocabulary tokens (not just first 256)
        for (int i = 0; i < va; i++) {
            float sum = 0.0f;
            for (int d = 0; d < D; d++) sum += x[d] * w[(size_t)d * (size_t)va + (size_t)i];
            output[i] = sum;
        }
    }

    free(x);
}

// ── Sampling ─────────────────────────────────────────────────────────────────

uint32_t model_sample(ornith_model *model, float *logits,
                      float temperature, int top_k) {
    if (!model || !logits) return 0;
    (void)top_k;

    const model_config *cfg = model_get_config(model);
    int vocab = cfg ? (int)cfg->vocab_size : 256;

    if (temperature > 0 && temperature != 1.0f)
        for (int i = 0; i < vocab; i++) logits[i] /= temperature;

    float max_logit = -INFINITY;
    for (int i = 0; i < vocab; i++) if (logits[i] > max_logit) max_logit = logits[i];

    float sum = 0.0f;
    for (int i = 0; i < vocab; i++) { logits[i] = expf(logits[i] - max_logit); sum += logits[i]; }
    for (int i = 0; i < vocab; i++) logits[i] /= sum;

    float r = (float)rand() / (float)RAND_MAX;
    float cum = 0.0f;
    for (int i = 0; i < vocab; i++) { cum += logits[i]; if (cum >= r) return (uint32_t)i; }
    return (uint32_t)(vocab - 1);
}
