// RED: Model forward pass tests — config loading, tensor weight loading, routing.
// TDD: Write failing test first, then implement to make it pass.

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
    write_u32(buf, pos, 8); // GGUF_META_TYPE_STRING
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
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
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
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
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
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
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
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
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

// ═════════════════════════════════════════════════════════════════════════════
// Tiny GGUF Builder for Forward-Pass Tests
// ═════════════════════════════════════════════════════════════════════════════
// Builds a minimal GGUF with F32 tensor data for a 1-layer MoE model.
// Dimensions: d_model=4, hdim=4, n_exp=2, n_heads=2, n_kv=1, vocab=4

#define TINY_D      4
#define TINY_HDIM   4
#define TINY_NEXP   2
#define TINY_NHEAD  2
#define TINY_NKV    1
#define TINY_VOCAB  4

// Write a tensor info entry: name, n_dims, dims[], type=F32, offset
static void write_tinfo(unsigned char *buf, size_t *pos,
                         const char *name, int nd,
                         uint64_t d0, uint64_t d1, uint64_t d2,
                         uint64_t offset) {
    write_str(buf, pos, name);
    write_u32(buf, pos, (uint32_t)nd);
    write_u64(buf, pos, d0);
    if (nd >= 2) write_u64(buf, pos, d1);
    if (nd >= 3) write_u64(buf, pos, d2);
    write_u32(buf, pos, 0);  // type = F32
    write_u64(buf, pos, offset);
}

// Build a complete GGUF with all tensors needed for a realistic model_forward().
// All weight values are 0.5 for deterministic output.
static char *build_tiny_moe_gguf(void) {
    unsigned char buf[16384];
    size_t pos = 0;
    memset(buf, 0, sizeof(buf));

    // ── Header ──
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);           // version = 3
    size_t pos_ntensors = pos;
    write_u64(buf, &pos, 0);           // tensor_count placeholder
    size_t pos_nmeta = pos;
    write_u64(buf, &pos, 0);           // metadata_count placeholder

    // ── 17 metadata entries ──
    int meta_count = 17;

    write_str(buf, &pos, "general.architecture");
    write_str_val(buf, &pos, "qwen2moe");

    write_str(buf, &pos, "qwen2moe.block_count");
    write_u32_val(buf, &pos, 1);

    write_str(buf, &pos, "qwen2moe.embedding_length");
    write_u32_val(buf, &pos, TINY_D);

    write_str(buf, &pos, "qwen2moe.attention.head_count");
    write_u32_val(buf, &pos, TINY_NHEAD);

    write_str(buf, &pos, "qwen2moe.attention.head_count_kv");
    write_u32_val(buf, &pos, TINY_NKV);

    write_str(buf, &pos, "qwen2moe.expert_count");
    write_u32_val(buf, &pos, TINY_NEXP);

    write_str(buf, &pos, "qwen2moe.expert_used_count");
    write_u32_val(buf, &pos, 2);

    write_str(buf, &pos, "qwen2moe.feed_forward_length");
    write_u32_val(buf, &pos, TINY_HDIM);

    write_str(buf, &pos, "qwen2moe.expert_feed_forward_length");
    write_u32_val(buf, &pos, TINY_HDIM);

    write_str(buf, &pos, "qwen2moe.context_length");
    write_u32_val(buf, &pos, 64);

    write_str(buf, &pos, "qwen2moe.vocab_size");
    write_u32_val(buf, &pos, TINY_VOCAB);

    write_str(buf, &pos, "qwen2moe.attention.key_length");
    write_u32_val(buf, &pos, 2);

    write_str(buf, &pos, "qwen2moe.attention.value_length");
    write_u32_val(buf, &pos, 2);

    write_str(buf, &pos, "qwen2moe.rope.dimension_count");
    write_u32_val(buf, &pos, 2);

    write_str(buf, &pos, "qwen2moe.rope.freq_base");
    write_u32(buf, &pos, 6);  // FLOAT32 type
    { float v = 10000.0f; memcpy(buf + pos, &v, 4); pos += 4; }

    write_str(buf, &pos, "qwen2moe.full_attention_interval");
    write_u32_val(buf, &pos, 1);

    write_str(buf, &pos, "general.alignment");
    write_u32_val(buf, &pos, 32);

    // ── 16 tensor info entries ──
    typedef struct { const char *name; int nd; uint64_t d[3]; } TDesc;
    TDesc tensors[] = {
        {"token_embd.weight",             2, {TINY_D, TINY_VOCAB, 0}},
        {"blk.0.attn_norm.weight",        1, {TINY_D, 0, 0}},
        {"blk.0.attn_q.weight",           2, {TINY_D, TINY_NHEAD * TINY_D, 0}},
        {"blk.0.attn_k.weight",           2, {TINY_D, TINY_NKV * 2, 0}},
        {"blk.0.attn_v.weight",           2, {TINY_D, TINY_NKV * 2, 0}},
        {"blk.0.attn_output.weight",      2, {TINY_NHEAD * 2, TINY_D, 0}},
        {"blk.0.post_attention_norm.weight", 1, {TINY_D, 0, 0}},
        {"blk.0.ffn_gate_shexp.weight",   2, {TINY_D, TINY_HDIM, 0}},
        {"blk.0.ffn_up_shexp.weight",     2, {TINY_D, TINY_HDIM, 0}},
        {"blk.0.ffn_down_shexp.weight",   2, {TINY_HDIM, TINY_D, 0}},
        {"blk.0.ffn_gate_inp.weight",     2, {TINY_D, TINY_NEXP, 0}},
        {"blk.0.ffn_gate_exps.weight",    3, {TINY_D, TINY_HDIM, TINY_NEXP}},
        {"blk.0.ffn_up_exps.weight",      3, {TINY_D, TINY_HDIM, TINY_NEXP}},
        {"blk.0.ffn_down_exps.weight",    3, {TINY_HDIM, TINY_D, TINY_NEXP}},
        {"output_norm.weight",            1, {TINY_D, 0, 0}},
        {"output.weight",                 2, {TINY_D, TINY_VOCAB, 0}},
    };
    int n_tensors = sizeof(tensors) / sizeof(tensors[0]);

    uint64_t data_off = 0;
    for (int i = 0; i < n_tensors; i++) {
        uint64_t n_elems = 1;
        for (int d = 0; d < tensors[i].nd; d++) n_elems *= tensors[i].d[d];
        write_tinfo(buf, &pos, tensors[i].name, tensors[i].nd,
                    tensors[i].d[0], tensors[i].d[1], tensors[i].d[2],
                    data_off);
        data_off += n_elems * 4;
    }

    // ── Patch header counts ──
    size_t saved = pos;
    pos = pos_ntensors;    write_u64(buf, &pos, (uint64_t)n_tensors);
    pos = pos_nmeta;       write_u64(buf, &pos, (uint64_t)meta_count);
    pos = saved;

    // ── Pad to alignment ──
    // GGUF alignment is 32 (from general.alignment metadata)
    while (pos % 32 != 0) { buf[pos++] = 0; }

    // ── Tensor data: all 0.5 ──
    if (pos + data_off > sizeof(buf)) return NULL;
    for (uint64_t i = 0; i < data_off; i += 4) {
        float v = 0.5f;
        memcpy(buf + pos + i, &v, 4);
    }
    pos += data_off;

    // ── Write to temp file ──
    char tmpl[] = "/tmp/tiny_moe_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

// ═════════════════════════════════════════════════════════════════════════════
// Forward-Pass Buffer Helpers
// ═════════════════════════════════════════════════════════════════════════════

// Per-layer pointer arrays (heap-allocated so addresses survive function scope)
// Test model has 1 layer, so each array has 1 element.
static float **make_layer_ptr_arr(const float *data) {
    float **arr = (float **)calloc(1, sizeof(float *));
    if (arr) arr[0] = (float *)data;  // const-cast: data is heap-allocated
    return arr;
}

// Allocate minimal model_buffers for the tiny model (1 layer, tiny dims)
static void alloc_tiny_buffers(model_buffers *bufs) {
    memset(bufs, 0, sizeof(*bufs));
    int kv_dim = TINY_NKV * 2;  // n_kv_heads * key_length
    int conv_dim = TINY_D * 4;

    float *k = (float *)calloc((size_t)64 * (size_t)kv_dim, sizeof(float));
    float *v = (float *)calloc((size_t)64 * (size_t)kv_dim, sizeof(float));
    float *cs = (float *)calloc((size_t)(4-1) * (size_t)conv_dim, sizeof(float));
    float *hs = (float *)calloc(32, sizeof(float));

    bufs->k_cache = make_layer_ptr_arr(k);
    bufs->v_cache = make_layer_ptr_arr(v);
    bufs->ssm_conv_states = make_layer_ptr_arr(cs);
    bufs->ssm_h_states = make_layer_ptr_arr(hs);
    bufs->max_cache_len = 64;
}

static void free_tiny_buffers(model_buffers *bufs) {
    if (!bufs) return;
    if (bufs->k_cache)  { free(bufs->k_cache[0]); free(bufs->k_cache); }
    if (bufs->v_cache)  { free(bufs->v_cache[0]); free(bufs->v_cache); }
    if (bufs->ssm_conv_states) { free(bufs->ssm_conv_states[0]); free(bufs->ssm_conv_states); }
    if (bufs->ssm_h_states)    { free(bufs->ssm_h_states[0]);    free(bufs->ssm_h_states); }
    memset(bufs, 0, sizeof(*bufs));
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 6: Expert offloading via memory cache
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_model_expert_offload(void) {
    char *path = build_tiny_moe_gguf();
    test_not_null(path, "build tiny moe gguf");

    memory_config mem_cfg;
    mem_cfg.hot_store_experts    = 0;
    mem_cfg.lru_capacity         = 10;     // > 2 exp * 3 weights = 6 entries
    mem_cfg.expert_size_bytes    = (size_t)TINY_D * (size_t)TINY_HDIM * 4;
    mem_cfg.non_routed_size      = 1024;
    mem_cfg.async_io_threads     = 0;
    mem_cfg.prefetch_queue_depth = 0;
    mem_cfg.double_buffering     = false;
    mem_cfg.lookahead_layers     = 0;

    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
    test_not_null(model, "model_load succeeds");

    memory_manager *mm = model_get_memory(model);
    test_not_null(mm, "memory manager");

    model_buffers bufs;
    alloc_tiny_buffers(&bufs);

    float *output = (float *)calloc(TINY_VOCAB, sizeof(float));
    model_forward(model, 0, output, &bufs, 0, 0);

    // After model_forward, the cache should have LRU hits from expert lookups
    cache_stats stats = memory_get_stats(mm);
    test_assert(stats.lru_hits > 0, "expert cache had >=1 LRU hit");

    // Output should have non-zero logits
    bool has_nonzero = false;
    for (int i = 0; i < TINY_VOCAB; i++) {
        if (output[i] != 0.0f) { has_nonzero = true; break; }
    }
    test_assert(has_nonzero, "non-zero output logits");

    free(output);
    free_tiny_buffers(&bufs);
    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 7: Forward pass with hot-store promotion
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_model_forward_hotstore(void) {
    char *path = build_tiny_moe_gguf();
    test_not_null(path, "build tiny moe gguf");

    memory_config mem_cfg;
    mem_cfg.hot_store_experts    = 10;     // hot-store enabled
    mem_cfg.lru_capacity         = 10;
    mem_cfg.expert_size_bytes    = (size_t)TINY_D * (size_t)TINY_HDIM * 4;
    mem_cfg.non_routed_size      = 1024;
    mem_cfg.async_io_threads     = 0;
    mem_cfg.prefetch_queue_depth = 0;
    mem_cfg.double_buffering     = false;
    mem_cfg.lookahead_layers     = 0;

    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
    test_not_null(model, "model_load succeeds");

    memory_manager *mm = model_get_memory(model);

    model_buffers bufs;
    alloc_tiny_buffers(&bufs);

    float *output = (float *)calloc(TINY_VOCAB, sizeof(float));
    model_forward(model, 0, output, &bufs, 0, 0);

    // Check that some cache activity occurred (LRU or hot)
    cache_stats stats = memory_get_stats(mm);
    test_assert(stats.total_hit_rate > 0.0, "positive cache hit rate");

    free(output);
    free_tiny_buffers(&bufs);
    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 5: Non-existent model path
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_model_load_nonexistent(void) {
    memory_config mem_cfg = memory_config_m2();
    char err[256];
    ornith_model *model = model_load("/tmp/nonexistent_model.gguf",
                                     &mem_cfg, err, sizeof(err), false, false);
    test_null(model, "model_load should fail for nonexistent path");
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Tiny SSM GGUF Builder
// ═════════════════════════════════════════════════════════════════════════════

#define SSM_D        8       // small d_model
#define SSM_HDIM     4       // shared expert hidden
#define SSM_NEXP     2       // experts per layer
#define SSM_DSTATE   32      // must match ORNITH_SSM_TIME_STEP_RANK (= 32)
#define SSM_INNER    32      // inner size (gate_dim)
#define SSM_CONVDIM  (SSM_D * 4)  // 32

static char *build_ssm_gguf(void) {
    unsigned char buf[65536];
    size_t pos = 0;
    memset(buf, 0, sizeof(buf));

    // Header
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);
    size_t pos_ntensors = pos;
    write_u64(buf, &pos, 0);
    size_t pos_nmeta = pos;
    write_u64(buf, &pos, 0);

    // Use 2 layers: layer 0 = attention, layer 1 = SSM
    // Set full_attention_interval=2 so layer 0 (0%2==0) is attention, layer 1 (1%2!=0) is SSM
    int meta_count = 14;
    write_str(buf, &pos, "general.architecture");
    write_str_val(buf, &pos, "qwen2moe");
    write_str(buf, &pos, "qwen2moe.block_count");
    write_u32_val(buf, &pos, 2);   // 2 layers
    write_str(buf, &pos, "qwen2moe.embedding_length");
    write_u32_val(buf, &pos, SSM_D);
    write_str(buf, &pos, "qwen2moe.attention.head_count");
    write_u32_val(buf, &pos, 4);
    write_str(buf, &pos, "qwen2moe.attention.head_count_kv");
    write_u32_val(buf, &pos, 2);
    write_str(buf, &pos, "qwen2moe.expert_count");
    write_u32_val(buf, &pos, SSM_NEXP);
    write_str(buf, &pos, "qwen2moe.expert_used_count");
    write_u32_val(buf, &pos, 2);
    write_str(buf, &pos, "qwen2moe.feed_forward_length");
    write_u32_val(buf, &pos, SSM_HDIM);
    write_str(buf, &pos, "qwen2moe.expert_feed_forward_length");
    write_u32_val(buf, &pos, SSM_HDIM);
    write_str(buf, &pos, "qwen2moe.context_length");
    write_u32_val(buf, &pos, 64);
    write_str(buf, &pos, "qwen2moe.vocab_size");
    write_u32_val(buf, &pos, 8);
    write_str(buf, &pos, "qwen2moe.attention.key_length");
    write_u32_val(buf, &pos, 4);
    write_str(buf, &pos, "qwen2moe.attention.value_length");
    write_u32_val(buf, &pos, 4);
    write_str(buf, &pos, "qwen2moe.full_attention_interval");
    write_u32_val(buf, &pos, 2);  // layer 0 (0%2==0) = attention, layer 1 (1%2!=0) = SSM

    // Tensors: layer 0 (attention) + layer 1 (SSM) + global
    typedef struct { const char *name; int nd; uint64_t d[3]; } TDesc;
    TDesc tensors[] = {
        // Global
        {"token_embd.weight",             2, {SSM_D, 8, 0}},
        {"output_norm.weight",            1, {SSM_D, 0, 0}},
        {"output.weight",                 2, {SSM_D, 8, 0}},
        // Layer 0: full attention (blk.0.attn_q, attn_k, attn_v, attn_output)
        {"blk.0.attn_norm.weight",        1, {SSM_D, 0, 0}},
        {"blk.0.attn_q.weight",           2, {SSM_D, 4 * 4 * 2, 0}},  // n_heads=4, key_len=4, *2 for size
        {"blk.0.attn_k.weight",           2, {SSM_D, 2 * 4, 0}},      // n_kv=2, key_len=4
        {"blk.0.attn_v.weight",           2, {SSM_D, 2 * 4, 0}},
        {"blk.0.attn_output.weight",      2, {4 * 4, SSM_D, 0}},
        {"blk.0.post_attention_norm.weight", 1, {SSM_D, 0, 0}},
        {"blk.0.ffn_gate_shexp.weight",   2, {SSM_D, SSM_HDIM, 0}},
        {"blk.0.ffn_up_shexp.weight",     2, {SSM_D, SSM_HDIM, 0}},
        {"blk.0.ffn_down_shexp.weight",   2, {SSM_HDIM, SSM_D, 0}},
        {"blk.0.ffn_gate_inp.weight",     2, {SSM_D, SSM_NEXP, 0}},
        {"blk.0.ffn_gate_exps.weight",    3, {SSM_D, SSM_HDIM, SSM_NEXP}},
        {"blk.0.ffn_up_exps.weight",      3, {SSM_D, SSM_HDIM, SSM_NEXP}},
        {"blk.0.ffn_down_exps.weight",    3, {SSM_HDIM, SSM_D, SSM_NEXP}},
        // Layer 1: SSM (blk.1.attn_qkv, ssm_conv1d, ssm_a, etc.)
        {"blk.1.attn_norm.weight",        1, {SSM_D, 0, 0}},
        {"blk.1.attn_qkv.weight",         2, {SSM_D, SSM_CONVDIM, 0}},
        {"blk.1.attn_gate.weight",        2, {SSM_D, SSM_INNER, 0}},
        {"blk.1.ssm_conv1d.weight",       2, {4, SSM_CONVDIM, 0}},
        {"blk.1.ssm_a",                   1, {SSM_DSTATE, 0, 0}},
        {"blk.1.ssm_dt.bias",             1, {SSM_DSTATE, 0, 0}},
        {"blk.1.ssm_alpha.weight",        2, {SSM_D, SSM_DSTATE, 0}},
        {"blk.1.ssm_beta.weight",         2, {SSM_D, SSM_DSTATE, 0}},
        {"blk.1.ssm_norm.weight",         1, {SSM_INNER, 0, 0}},
        {"blk.1.ssm_out.weight",          2, {SSM_INNER, SSM_D, 0}},
        {"blk.1.post_attention_norm.weight", 1, {SSM_D, 0, 0}},
        {"blk.1.ffn_gate_shexp.weight",   2, {SSM_D, SSM_HDIM, 0}},
        {"blk.1.ffn_up_shexp.weight",     2, {SSM_D, SSM_HDIM, 0}},
        {"blk.1.ffn_down_shexp.weight",   2, {SSM_HDIM, SSM_D, 0}},
        {"blk.1.ffn_gate_inp.weight",     2, {SSM_D, SSM_NEXP, 0}},
        {"blk.1.ffn_gate_exps.weight",    3, {SSM_D, SSM_HDIM, SSM_NEXP}},
        {"blk.1.ffn_up_exps.weight",      3, {SSM_D, SSM_HDIM, SSM_NEXP}},
        {"blk.1.ffn_down_exps.weight",    3, {SSM_HDIM, SSM_D, SSM_NEXP}},
    };
    int n_tensors = sizeof(tensors) / sizeof(tensors[0]);

    uint64_t data_off = 0;
    for (int i = 0; i < n_tensors; i++) {
        uint64_t n_elems = 1;
        for (int d = 0; d < tensors[i].nd; d++) n_elems *= tensors[i].d[d];
        write_tinfo(buf, &pos, tensors[i].name, tensors[i].nd,
                    tensors[i].d[0], tensors[i].d[1], tensors[i].d[2],
                    data_off);
        data_off += n_elems * 4;
    }

    size_t saved = pos;
    pos = pos_ntensors; write_u64(buf, &pos, (uint64_t)n_tensors);
    pos = pos_nmeta;    write_u64(buf, &pos, (uint64_t)meta_count);
    pos = saved;

    while (pos % 32 != 0) buf[pos++] = 0;

    // Fill tensor data with 0.5 (deterministic)
    if (pos + data_off > sizeof(buf)) return NULL;
    for (uint64_t i = 0; i < data_off; i += 4) {
        float v = 0.5f;
        memcpy(buf + pos + i, &v, 4);
    }
    pos += data_off;

    char tmpl[] = "/tmp/ssm_test_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 8: SSM layer forward pass
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_model_ssm_forward(void) {
    char *path = build_ssm_gguf();
    test_not_null(path, "build ssm gguf");

    memory_config mem_cfg;
    memset(&mem_cfg, 0, sizeof(mem_cfg));
    mem_cfg.hot_store_experts    = 10;
    mem_cfg.lru_capacity         = 10;
    mem_cfg.expert_size_bytes    = (size_t)SSM_D * (size_t)SSM_HDIM * 4;
    mem_cfg.non_routed_size      = 1024;
    mem_cfg.async_io_threads     = 0;
    mem_cfg.prefetch_queue_depth = 0;
    mem_cfg.double_buffering     = false;
    mem_cfg.lookahead_layers     = 0;

    char err[256];
    ornith_model *model = model_load(path, &mem_cfg, err, sizeof(err), false, false);
    test_not_null(model, "ssm model_load succeeds");

    // Build buffers for 2-layer model (layer 0 = attention, layer 1 = SSM)
    model_buffers bufs;
    memset(&bufs, 0, sizeof(bufs));
    int kv_dim = 2 * 4;
    int conv_dim = SSM_D * 4;
    {
        float **kp = (float **)calloc(2, sizeof(float*));
        float **vp = (float **)calloc(2, sizeof(float*));
        float **csp = (float **)calloc(2, sizeof(float*));
        float **hsp = (float **)calloc(2, sizeof(float*));
        for (int i = 0; i < 2; i++) {
            kp[i] = (float *)calloc((size_t)64 * (size_t)kv_dim, sizeof(float));
            vp[i] = (float *)calloc((size_t)64 * (size_t)kv_dim, sizeof(float));
            csp[i] = (float *)calloc((size_t)(4-1) * (size_t)conv_dim, sizeof(float));
            hsp[i] = (float *)calloc((size_t)SSM_DSTATE, sizeof(float));
        }
        bufs.k_cache = kp;
        bufs.v_cache = vp;
        bufs.ssm_conv_states = csp;
        bufs.ssm_h_states = hsp;
        bufs.max_cache_len = 64;
    }

    float *output = (float *)calloc(8, sizeof(float));
    model_forward(model, 0, output, &bufs, 0, 0);

    // Should have non-zero output
    bool has_nonzero = false;
    for (int i = 0; i < 8; i++) {
        if (output[i] != 0.0f) { has_nonzero = true; break; }
    }
    test_assert(has_nonzero, "ssm forward pass produces non-zero logits");

    // Cleanup buffers (2 layers)
    if (bufs.k_cache)  { for (int i = 0; i < 2; i++) free(bufs.k_cache[i]);  free(bufs.k_cache); }
    if (bufs.v_cache)  { for (int i = 0; i < 2; i++) free(bufs.v_cache[i]);  free(bufs.v_cache); }
    if (bufs.ssm_conv_states) { for (int i = 0; i < 2; i++) free(bufs.ssm_conv_states[i]); free(bufs.ssm_conv_states); }
    if (bufs.ssm_h_states)    { for (int i = 0; i < 2; i++) free(bufs.ssm_h_states[i]);    free(bufs.ssm_h_states); }

    free(output);
    model_unload(model);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    { "model_config_loading",    test_model_config_loading },
    { "model_tensor_loading",    test_model_tensor_loading },
    { "model_routing",           test_model_routing },
    { "model_hotstore_update",   test_model_hotstore_update },
    { "model_load_nonexistent",  test_model_load_nonexistent },
    { "model_expert_offload",    test_model_expert_offload },
    { "model_forward_hotstore",  test_model_forward_hotstore },
    { "model_ssm_forward",       test_model_ssm_forward },
};

RUN_TESTS(tests)
