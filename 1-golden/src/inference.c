// SPDX-License-Identifier: MIT
// Inference loop — token generation with KV cache management, timing.

#include "inference.h"
#include "model.h"
#include "memory.h"
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct inference_engine {
    ornith_model   *model;
    void          **kv_caches;    // per-layer GPU buffer handles
    int             seq_len;
    int             n_layers;
    uint64_t        t_start_ns;
    uint64_t        t_prefill_end_ns;
    uint64_t        t_first_token_ns;
    bool            has_prefilled;
    unsigned int    rng_state;
};

// ── Timing ───────────────────────────────────────────────────────────────────

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

// ── KV cache allocation ──────────────────────────────────────────────────────

static void **alloc_kv_caches(ornith_model *model) {
    const model_config *cfg = model_get_config(model);
    gpu_context *gpu = model_get_gpu(model);
    if (!cfg || !gpu) return NULL;

    uint32_t n_layers = cfg->n_layers;
    int kv_dim = (int)(cfg->n_kv_heads * cfg->head_dim);

    void **caches = (void **)calloc(n_layers, sizeof(void *));
    if (!caches) return NULL;

    for (uint32_t i = 0; i < n_layers; i++) {
        // Minimal KV cache: one float per head*dim per layer
        // In production: this would be circular buffer for long context
        size_t kv_size = (size_t)kv_dim * sizeof(float);
        gpu_buffer *buf = gpu_alloc(gpu, kv_size, GPU_BUF_SHARED);
        if (buf) {
            caches[i] = (void *)buf;
        }
    }

    return caches;
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

inference_engine *inference_init(ornith_model *model) {
    if (!model) return NULL;

    inference_engine *engine = (inference_engine *)calloc(1, sizeof(inference_engine));
    if (!engine) return NULL;

    engine->model = model;
    engine->rng_state = (unsigned int)time(NULL);

    const model_config *cfg = model_get_config(model);
    engine->n_layers = cfg ? (int)cfg->n_layers : 4;

    engine->kv_caches = alloc_kv_caches(model);
    engine->seq_len = 0;

    return engine;
}

void inference_destroy(inference_engine *engine) {
    if (!engine) return;

    if (engine->kv_caches) {
        gpu_context *gpu = model_get_gpu(engine->model);
        for (int i = 0; i < engine->n_layers; i++) {
            if (engine->kv_caches[i]) {
                gpu_free(gpu, (gpu_buffer *)engine->kv_caches[i]);
            }
        }
        free(engine->kv_caches);
    }

    memset(engine, 0, sizeof(*engine));
    free(engine);
}

void inference_reset(inference_engine *engine) {
    if (!engine) return;
    engine->seq_len = 0;
    engine->has_prefilled = false;
    engine->t_start_ns = 0;
    engine->t_prefill_end_ns = 0;
    engine->t_first_token_ns = 0;

    // Clear KV caches
    gpu_context *gpu = model_get_gpu(engine->model);
    if (gpu && engine->kv_caches) {
        for (int i = 0; i < engine->n_layers; i++) {
            if (engine->kv_caches[i]) {
                gpu_buffer *buf = (gpu_buffer *)engine->kv_caches[i];
                size_t sz = gpu_buffer_size(gpu, buf);
                void *ptr = gpu_get_cpu_ptr(gpu, buf);
                if (ptr) memset(ptr, 0, sz);
            }
        }
    }

    // Reset memory manager stats
    memory_manager *mem = model_get_memory(engine->model);
    if (mem) memory_reset_stats(mem);
}

// ── Tokenization (minimal placeholder) ───────────────────────────────────────

static int simple_tokenize(const char *text, uint32_t *tokens, int max_tokens) {
    if (!text) return 0;
    int count = 0;
    const char *p = text;
    while (*p && count < max_tokens) {
        // Rough estimate: one token per ~4 characters
        // For testing, just return a simple tokenization
        if (count == 0) {
            if (tokens) tokens[count] = 1;  // BOS-like
            count++;
        }
        // Skip past a word
        while (*p == ' ') p++;
        if (!*p) break;
        while (*p && *p != ' ') p++;
        if (tokens && count < max_tokens) {
            tokens[count] = (uint32_t)(count * 10 + 5);  // deterministic IDs
        }
        count++;
    }
    return count;
}

// ── Generation ───────────────────────────────────────────────────────────────

generation_result *inference_generate(inference_engine *engine,
                                      const char *prompt,
                                      const generation_params *params) {
    if (!engine || !prompt) return NULL;

    // Simplified: treat prompt as rough token count
    int n_prompt = simple_tokenize(prompt, NULL, 1024);
    if (n_prompt <= 0) return NULL;

    uint32_t *prompt_tokens = (uint32_t *)calloc((size_t)n_prompt, sizeof(uint32_t));
    if (!prompt_tokens) return NULL;

    simple_tokenize(prompt, prompt_tokens, n_prompt);

    generation_result *result = inference_generate_tokens(
        engine, prompt_tokens, n_prompt, params);

    free(prompt_tokens);
    return result;
}

generation_result *inference_generate_tokens(inference_engine *engine,
                                             const uint32_t *prompt_tokens,
                                             int n_prompt_tokens,
                                             const generation_params *params) {
    if (!engine || !prompt_tokens || n_prompt_tokens <= 0) return NULL;

    generation_params gp = params ? *params : generation_params_default();

    // Allocate result
    generation_result *result = (generation_result *)calloc(1, sizeof(generation_result));
    if (!result) return NULL;

    int max_total = gp.max_tokens + n_prompt_tokens;
    result->tokens = (uint32_t *)calloc((size_t)max_total, sizeof(uint32_t));
    if (!result->tokens) { free(result); return NULL; }

    memcpy(result->tokens, prompt_tokens, (size_t)n_prompt_tokens * sizeof(uint32_t));
    result->n_prompt_tokens = n_prompt_tokens;
    result->n_tokens = n_prompt_tokens;

    // Seed RNG
    if (gp.seed >= 0) {
        engine->rng_state = (unsigned int)gp.seed;
        srand((unsigned int)gp.seed);
    } else {
        srand((unsigned int)time(NULL));
    }

    // ── Prefill phase ────────────────────────────────────────────────────────
    inference_reset(engine);
    engine->t_start_ns = now_ns();

    const model_config *cfg = model_get_config(engine->model);
    int vocab = cfg ? (int)cfg->vocab_size : 256;
    float *logits = (float *)calloc((size_t)vocab, sizeof(float));
    if (!logits) { inference_result_free(result); return NULL; }

    // Process prompt tokens (prefill)
    for (int i = 0; i < n_prompt_tokens; i++) {
        model_forward(engine->model, prompt_tokens[i], logits,
                     engine->kv_caches, i + 1, i);
    }

    engine->t_prefill_end_ns = now_ns();
    engine->has_prefilled = true;
    engine->seq_len = n_prompt_tokens;

    // ── Decode phase ─────────────────────────────────────────────────────────
    uint32_t eos_token = inference_eos_token(engine);
    int gen_count = 0;

    for (int i = 0; i < gp.max_tokens; i++) {
        // Sample next token
        uint32_t next_token = model_sample(engine->model, logits,
                                           gp.temperature, gp.top_k);

        if (i == 0) engine->t_first_token_ns = now_ns();

        if (next_token == eos_token) break;

        // Store token
        result->tokens[result->n_tokens++] = next_token;
        gen_count++;

        // Forward pass for next token
        model_forward(engine->model, next_token, logits,
                     engine->kv_caches, engine->seq_len, engine->seq_len);
        engine->seq_len++;
    }

    // ── Compute timing ───────────────────────────────────────────────────────
    uint64_t t_end_ns = now_ns();

    result->t_prefill_ms = (double)(engine->t_prefill_end_ns - engine->t_start_ns) / 1e6;
    result->ttft_ms = (double)(engine->t_first_token_ns - engine->t_start_ns) / 1e6;
    result->t_decode_ms = (double)(t_end_ns - engine->t_prefill_end_ns) / 1e6;
    result->tokens_per_sec = gen_count > 0
        ? (double)gen_count / (result->t_decode_ms / 1000.0) : 0.0;
    result->truncated = (gen_count >= gp.max_tokens);

    // Minimal detokenization: token IDs as text
    char buf[256] = {0};
    size_t pos = 0;
    for (int i = 0; i < gen_count && pos < 250; i++) {
        pos += snprintf(buf + pos, sizeof(buf) - pos, "%u ", result->tokens[n_prompt_tokens + i]);
    }
    result->text = strdup(buf);

    free(logits);
    return result;
}

// ── Result management ────────────────────────────────────────────────────────

void inference_result_free(generation_result *result) {
    if (!result) return;
    free(result->tokens);
    free(result->text);
    memset(result, 0, sizeof(*result));
    free(result);
}

void inference_print_stats(const generation_result *result) {
    if (!result) return;
    printf("=== Generation Stats ===\n");
    printf("Prompt tokens:  %d\n", result->n_prompt_tokens);
    printf("Generated:      %d tokens\n", result->n_tokens - result->n_prompt_tokens);
    printf("Prefill time:   %.1f ms\n", result->t_prefill_ms);
    printf("TTFT:           %.1f ms\n", result->ttft_ms);
    printf("Decode time:    %.1f ms\n", result->t_decode_ms);
    printf("Throughput:     %.2f tok/s\n", result->tokens_per_sec);
    printf("Truncated:      %s\n", result->truncated ? "yes" : "no");
}

uint32_t inference_eos_token(inference_engine *engine) {
    (void)engine;
    return 151645;  // Qwen EOS default
}
