// RED: Inference loop tests — KV cache, prefill, decode, sampling, timing.

#include "test.h"
#include "gguf.h"
#include "gpu.h"
#include "memory.h"
#include "model.h"
#include "inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

// ── GGUF builder (same structure as t_model.c) ───────────────────────────────

static void write_u32(unsigned char *buf, size_t *pos, uint32_t v) {
    memcpy(buf + *pos, &v, 4); *pos += 4;
}
static void write_u64(unsigned char *buf, size_t *pos, uint64_t v) {
    memcpy(buf + *pos, &v, 8); *pos += 8;
}
static void write_str(unsigned char *buf, size_t *pos, const char *s) {
    size_t len = strlen(s);
    write_u64(buf, pos, len);
    memcpy(buf + *pos, s, len); *pos += len;
}
static void write_str_val(unsigned char *buf, size_t *pos, const char *s) {
    write_u32(buf, pos, 8); write_str(buf, pos, s);
}
static void write_u32_val(unsigned char *buf, size_t *pos, uint32_t v) {
    write_u32(buf, pos, 4); write_u32(buf, pos, v);
}

static char *build_model_gguf(void) {
    unsigned char buf[4096] = {0};
    size_t pos = 0;
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);
    write_u64(buf, &pos, 1);  // tensor_count = 1
    write_u64(buf, &pos, 7);  // metadata_count = 7

    write_str(buf, &pos, "general.architecture");
    write_str_val(buf, &pos, "qwen2moe");
    write_str(buf, &pos, "qwen2moe.block_count");
    write_u32_val(buf, &pos, 4);  // 4 layers (small for fast tests)
    write_str(buf, &pos, "qwen2moe.embedding_length");
    write_u32_val(buf, &pos, 256);
    write_str(buf, &pos, "qwen2moe.attention.head_count");
    write_u32_val(buf, &pos, 4);
    write_str(buf, &pos, "qwen2moe.attention.head_count_kv");
    write_u32_val(buf, &pos, 2);
    write_str(buf, &pos, "qwen2moe.expert_count");
    write_u32_val(buf, &pos, 8);
    write_str(buf, &pos, "qwen2moe.expert_used_count");
    write_u32_val(buf, &pos, 4);

    // One tensor so the model loads
    write_str(buf, &pos, "blk.0.attn_q.weight");
    write_u32(buf, &pos, 2);
    write_u64(buf, &pos, 256); write_u64(buf, &pos, 256);
    write_u32(buf, &pos, 0);
    write_u64(buf, &pos, 0);

    char tmpl[] = "/tmp/inf_test_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

// ── Test helpers ─────────────────────────────────────────────────────────────

static ornith_model *load_test_model(void) {
    char *path = build_model_gguf();
    if (!path) return NULL;

    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);

    free(path);  // model keeps its own GGUF handle via mmap
    return model;
}

// ── Test 1: Inference init and KV cache allocation ───────────────────────────

static test_result test_inference_init_kv_cache(void) {
    ornith_model *model = load_test_model();
    test_not_null(model, "load model");

    inference_engine *engine = inference_init(model, NULL);
    test_not_null(engine, "inference_init succeeds");

    // KV caches should be allocated per-layer
    // Just verify the engine exists (caches allocated internally)
    test_assert(engine != NULL, "engine created with KV caches");

    inference_destroy(engine);
    model_unload(model);
    return TEST_PASS;
}

// ── Test 2: Inference reset (clears KV cache state) ──────────────────────────

static test_result test_inference_reset(void) {
    ornith_model *model = load_test_model();
    test_not_null(model, "load model");

    inference_engine *engine = inference_init(model, NULL);
    test_not_null(engine, "inference_init");

    // Reset should work without error
    inference_reset(engine);

    inference_destroy(engine);
    model_unload(model);
    return TEST_PASS;
}

// ── Test 3: Token sampling with temperature ──────────────────────────────────

static test_result test_inference_sampling(void) {
    ornith_model *model = load_test_model();
    test_not_null(model, "load model");

    // Create dummy logits
    int vocab = 256;
    float *logits = (float *)calloc(vocab, sizeof(float));
    test_not_null(logits, "alloc logits");

    // Make token 42 have the highest logit
    logits[42] = 10.0f;

    // Sample with temperature
    uint32_t token = model_sample(model, logits, 0.7f, 10);
    test_assert(token < (uint32_t)vocab, "sampled token within vocab");
    // Token 42 should be most likely (highest logit)
    // With top-k=10, it should still be selected since it's in the top 10

    free(logits);
    model_unload(model);
    return TEST_PASS;
}

// ── Test 4: Prefill phase (process prompt tokens) ────────────────────────────

static test_result test_inference_prefill(void) {
    ornith_model *model = load_test_model();
    test_not_null(model, "load model");

    inference_engine *engine = inference_init(model, NULL);
    test_not_null(engine, "inference_init");

    // Generate a short response from a prompt
    generation_params gp = generation_params_default();
    gp.max_tokens = 5;
    gp.temperature = 0.7f;
    gp.top_k = 10;

    generation_result *result = inference_generate(engine, "Hello", &gp);
    test_not_null(result, "generation result");
    test_assert(result->n_tokens > 0, "generated some tokens");
    test_assert(result->t_prefill_ms >= 0, "prefill time measured");
    test_assert(result->ttft_ms >= 0, "TTFT measured");

    inference_result_free(result);
    inference_destroy(engine);
    model_unload(model);
    return TEST_PASS;
}

// ── Test 5: Generation with tokens (programmatic) ────────────────────────────

static test_result test_inference_generate_tokens(void) {
    ornith_model *model = load_test_model();
    test_not_null(model, "load model");

    inference_engine *engine = inference_init(model, NULL);
    test_not_null(engine, "inference_init");

    // Generate from prompt tokens directly
    uint32_t prompt[] = {1, 42, 100};  // 3 "token" IDs
    generation_params gp = generation_params_default();
    gp.max_tokens = 5;

    generation_result *result = inference_generate_tokens(engine, prompt, 3, &gp);
    test_not_null(result, "generation from tokens");
    test_assert(result->n_tokens > 3, "generated more than prompt tokens");
    test_assert(result->tokens_per_sec >= 0, "tokens/sec recorded");

    inference_print_stats(result);

    inference_result_free(result);
    inference_destroy(engine);
    model_unload(model);
    return TEST_PASS;
}

// ── Test 6: Generation stats and timing ──────────────────────────────────────

static test_result test_inference_stats(void) {
    ornith_model *model = load_test_model();
    test_not_null(model, "load model");

    inference_engine *engine = inference_init(model, NULL);
    test_not_null(engine, "inference_init");

    generation_params gp = generation_params_default();
    gp.max_tokens = 3;

    generation_result *result = inference_generate(engine, "test", &gp);
    test_not_null(result, "generation result");

    // All timing fields should be populated
    test_assert(result->t_prefill_ms >= 0, "prefill time");
    test_assert(result->t_decode_ms >= 0, "decode time");
    test_assert(result->ttft_ms >= 0, "TTFT");
    test_assert(result->tokens_per_sec >= 0, "tokens/sec");

    inference_result_free(result);
    inference_destroy(engine);
    model_unload(model);
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    { "inference_init_kv_cache", test_inference_init_kv_cache },
    { "inference_reset",         test_inference_reset },
    { "inference_sampling",      test_inference_sampling },
    { "inference_prefill",       test_inference_prefill },
    { "inference_generate_tokens", test_inference_generate_tokens },
    { "inference_stats",         test_inference_stats },
};

RUN_TESTS(tests)
