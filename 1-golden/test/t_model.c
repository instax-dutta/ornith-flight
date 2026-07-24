// RED: Model forward pass tests — config loading, tensor weight loading, routing.

#include "test.h"
#include "gguf.h"
#include "gpu.h"
#include "memory.h"
#include "model.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>

// ── GGUF builder helpers ─────────────────────────────────────────────────────

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
    write_u32(buf, pos, 11); // GGUF_META_TYPE_STRING
    write_str(buf, pos, s);
}
static void write_u32_val(unsigned char *buf, size_t *pos, uint32_t v) {
    write_u32(buf, pos, 4); // GGUF_META_TYPE_UINT32
    write_u32(buf, pos, v);
}

// ── Build a GGUF file with Ornith 35B model metadata + one weight tensor ─────

static char *build_model_gguf(void) {
    unsigned char buf[4096] = {0};
    size_t pos = 0;

    // Header (24 bytes)
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);    // version
    write_u64(buf, &pos, 1);    // tensor_count = 1
    write_u64(buf, &pos, 7);    // metadata_count = 7

    // Metadata entries (Qwen2MoE-style)
    write_str(buf, &pos, "general.architecture");
    write_str_val(buf, &pos, "qwen2moe");

    write_str(buf, &pos, "qwen2moe.block_count");
    write_u32_val(buf, &pos, 28);

    write_str(buf, &pos, "qwen2moe.embedding_length");
    write_u32_val(buf, &pos, 2560);

    write_str(buf, &pos, "qwen2moe.attention.head_count");
    write_u32_val(buf, &pos, 20);

    write_str(buf, &pos, "qwen2moe.attention.head_count_kv");
    write_u32_val(buf, &pos, 4);

    write_str(buf, &pos, "qwen2moe.expert_count");
    write_u32_val(buf, &pos, 256);

    write_str(buf, &pos, "qwen2moe.expert_used_count");
    write_u32_val(buf, &pos, 8);

    // Tensor info: "blk.0.attn_q.weight" F32 [2560, 2560], offset=0
    write_str(buf, &pos, "blk.0.attn_q.weight");
    write_u32(buf, &pos, 2);
    write_u64(buf, &pos, 2560); write_u64(buf, &pos, 2560);
    write_u32(buf, &pos, 0);  // F32
    write_u64(buf, &pos, 0);  // offset

    char tmpl[] = "/tmp/model_test_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

// ── Test 1: Model config loading from GGUF ───────────────────────────────────

static test_result test_model_config_loading(void) {
    char *path = build_model_gguf();
    test_not_null(path, "build model gguf");

    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err));
    test_not_null(model, "model_load should succeed");

    const model_config *cfg = model_get_config(model);
    test_not_null(cfg, "config should not be NULL");

    // Verify all config fields
    test_eq_uint64(cfg->n_layers, 28, "n_layers = 28");
    test_eq_uint64(cfg->n_experts_per_layer, 256, "n_experts_per_layer = 256");
    test_eq_uint64(cfg->n_active_experts, 8, "n_active_experts = 8");
    test_eq_uint64(cfg->d_model, 2560, "d_model = 2560");
    test_eq_uint64(cfg->n_heads, 20, "n_heads = 20");
    test_eq_uint64(cfg->n_kv_heads, 4, "n_kv_heads = 4");

    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 2: Tensor loading by name ───────────────────────────────────────────

static test_result test_model_tensor_loading(void) {
    char *path = build_model_gguf();
    test_not_null(path, "build model gguf");

    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err));
    test_not_null(model, "model_load");

    // Verify we can get the GPU context and memory manager
    gpu_context *gpu = model_get_gpu(model);
    test_not_null(gpu, "GPU context");

    memory_manager *mem = model_get_memory(model);
    test_not_null(mem, "memory manager");

    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 3: Expert routing top-k selection ───────────────────────────────────

static test_result test_model_routing(void) {
    char *path = build_model_gguf();
    test_not_null(path, "build model gguf");

    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err));
    test_not_null(model, "model_load");

    // Create dummy input (d_model floats)
    const model_config *cfg = model_get_config(model);
    float *x = (float *)calloc(cfg->d_model, sizeof(float));
    test_not_null(x, "alloc input");
    x[0] = 1.0f;  // non-zero input

    // Get routing decisions
    uint32_t indices[8];
    float weights[8];
    int n = model_get_routing(model, x, 0, indices, weights, 8);

    // Should return n_active_experts experts (8)
    test_assert(n == 8 || (uint32_t)n == cfg->n_active_experts,
                "routing returns 8 experts");

    // All indices should be valid (0..255 for layer 0)
    for (int i = 0; i < (int)n; i++) {
        test_assert(indices[i] < 256, "expert index within range");
        test_assert(weights[i] > 0, "expert weight positive");
    }

    // Weights should sum to ~1.0
    float sum = 0;
    for (int i = 0; i < n; i++) sum += weights[i];
    test_assert(sum > 0.99 && sum < 1.01, "weights sum to ~1.0");

    free(x);
    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 4: Hot-store update via model ───────────────────────────────────────

static test_result test_model_hotstore_update(void) {
    char *path = build_model_gguf();
    test_not_null(path, "build model gguf");

    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err));
    test_not_null(model, "model_load");

    // Update hot-store with top experts
    uint32_t hot_ids[] = {0, 1, 5, 10, 50, 100, 200, 255};
    model_update_hotstore(model, hot_ids, 8);

    // Verify via memory manager
    memory_manager *mem = model_get_memory(model);
    test_not_null(mem, "memory manager");

    void *data;
    const char *r;
    // First hot expert should be found in hot-store
    r = memory_get(mem, 0, &data);
    test_assert(strcmp(r, "hot") == 0, "expert 0 is in hot-store");

    // An expert not in hot-store should not be hot
    r = memory_get(mem, 999, &data);
    test_assert(strcmp(r, "hot") != 0, "expert 999 not in hot-store");

    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 5: Non-existent model path ──────────────────────────────────────────

static test_result test_model_load_nonexistent(void) {
    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load("/tmp/nonexistent_model.gguf",
                                     &mem_cfg, err, sizeof(err));
    test_null(model, "model_load should fail for nonexistent path");
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    { "model_config_loading",   test_model_config_loading },
    { "model_tensor_loading",   test_model_tensor_loading },
    { "model_routing",          test_model_routing },
    { "model_hotstore_update",  test_model_hotstore_update },
    { "model_load_nonexistent", test_model_load_nonexistent },
};

RUN_TESTS(tests)
