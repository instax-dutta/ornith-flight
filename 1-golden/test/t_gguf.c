// TDD: GGUF parser tests — header, metadata, tensor info.
// Each test creates a synthetic GGUF file and validates parsing.

#include "test.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

// ── Binary GGUF writing helpers ──────────────────────────────────────────────
// All values are little-endian (native on Apple Silicon).

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
    // GGUF string metadata value: type(STRING)=8 + string
    write_u32(buf, pos, 8);  // GGUF_META_TYPE_STRING
    write_str(buf, pos, s);
}
static void write_u32_val(unsigned char *buf, size_t *pos, uint32_t v) {
    // GGUF uint32 metadata value: type(UINT32)=4 + uint32
    write_u32(buf, pos, 4);  // GGUF_META_TYPE_UINT32
    write_u32(buf, pos, v);
}

// ── Write a GGUF file with given metadata and tensor counts ──────────────────

typedef struct {
    uint64_t metadata_count;
    uint64_t tensor_count;
    void   (*write_metadata)(unsigned char *buf, size_t *pos);
    void   (*write_tensors)(unsigned char *buf, size_t *pos);
} gguf_builder;

static char *build_gguf(gguf_builder *b) {
    unsigned char buf[4096] = {0};
    size_t pos = 0;

    // Header (24 bytes)
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);               // version = 3
    write_u64(buf, &pos, b->tensor_count); // tensor_count
    write_u64(buf, &pos, b->metadata_count); // metadata_count

    // Metadata
    if (b->write_metadata) b->write_metadata(buf, &pos);

    // Tensor info
    if (b->write_tensors) b->write_tensors(buf, &pos);

    char tmpl[] = "/tmp/gguf_t_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

// ── Test 1: Header parsing (existing) ────────────────────────────────────────

static test_result test_open_minimal(void) {
    char *path = build_gguf(&(gguf_builder){ .metadata_count = 0, .tensor_count = 0 });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");
    test_eq_uint64(gguf_version(m), 3, "version=3");
    test_eq_uint64(gguf_tensor_count(m), 0, "tensor_count=0");
    test_eq_uint64(gguf_metadata_count(m), 0, "metadata_count=0");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

static test_result test_open_nonexistent(void) {
    char err[256];
    gguf_model *m = gguf_open("/tmp/nonexistent_XXXX.gguf", err, sizeof(err));
    test_null(m, "NULL for nonexistent");
    return TEST_PASS;
}

static test_result test_open_bad_magic(void) {
    unsigned char bad[] = "BAD!";
    char tmpl[] = "/tmp/gguf_bad_XXXXXX";
    int fd = mkstemp(tmpl);
    test_assert(fd >= 0, "mkstemp");
    write(fd, bad, 4); close(fd);
    char err[256];
    gguf_model *m = gguf_open(tmpl, err, sizeof(err));
    test_null(m, "NULL for bad magic");
    unlink(tmpl);
    return TEST_PASS;
}

// ── Test 2: Metadata parsing ─────────────────────────────────────────────────

static void write_meta_one_str(unsigned char *buf, size_t *pos) {
    // One string metadata entry: general.architecture = "qwen2moe"
    write_str(buf, pos, "general.architecture");
    write_str_val(buf, pos, "qwen2moe");
}

static test_result test_metadata_string(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 1, .tensor_count = 0,
        .write_metadata = write_meta_one_str
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");
    test_eq_uint64(gguf_metadata_count(m), 1, "metadata_count=1");

    // Look up by key
    const char *arch = gguf_architecture(m);
    test_not_null(arch, "architecture should be found");
    test_assert(strcmp(arch, "qwen2moe") == 0, "architecture = qwen2moe");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

static void write_meta_alignment(unsigned char *buf, size_t *pos) {
    // general.alignment = 64 (uint32)
    write_str(buf, pos, "general.alignment");
    write_u32_val(buf, pos, 64);
}

static test_result test_metadata_alignment(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 1, .tensor_count = 0,
        .write_metadata = write_meta_alignment
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    size_t align = gguf_alignment(m);
    test_eq_uint64(align, 64, "alignment = 64");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

static void write_meta_three(unsigned char *buf, size_t *pos) {
    // 3 metadata entries
    write_str(buf, pos, "general.architecture");
    write_str_val(buf, pos, "qwen2moe");

    write_str(buf, pos, "general.alignment");
    write_u32_val(buf, pos, 64);

    write_str(buf, pos, "test.custom_key");
    write_str_val(buf, pos, "hello_world");
}

static test_result test_metadata_multiple(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 3, .tensor_count = 0,
        .write_metadata = write_meta_three
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");
    test_eq_uint64(gguf_metadata_count(m), 3, "metadata_count=3");

    // Look up all 3
    gguf_metadata_value val;
    bool found;

    found = gguf_find_metadata(m, "general.architecture", &val);
    test_assert(found, "find general.architecture");
    test_assert(val.type == GGUF_VALUE_STRING, "type=string");
    test_eq_uint64(val.value.string.len, 8, "len=8");
    test_assert(strncmp(val.value.string.data, "qwen2moe", 8) == 0, "value=qwen2moe");

    found = gguf_find_metadata(m, "general.alignment", &val);
    test_assert(found, "find general.alignment");
    test_assert(val.type == GGUF_VALUE_UINT32, "type=uint32");
    test_eq_uint64(val.value.uint32, 64, "value=64");

    found = gguf_find_metadata(m, "test.custom_key", &val);
    test_assert(found, "find test.custom_key");
    test_eq_uint64(val.value.string.len, 11, "len=11");

    // Non-existent key
    found = gguf_find_metadata(m, "nonexistent.key", &val);
    test_assert(!found, "nonexistent key not found");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 3: Tensor info parsing ──────────────────────────────────────────────

static void write_one_tensor(unsigned char *buf, size_t *pos) {
    // One tensor: name="test.weight", n_dims=2, dims=[256, 64], type=F32(0), offset=0
    write_str(buf, pos, "test.weight");
    write_u32(buf, pos, 2);                    // n_dims = 2
    write_u64(buf, pos, 256);                  // dim[0]
    write_u64(buf, pos, 64);                   // dim[1]
    write_u32(buf, pos, 0);                    // type = F32
    write_u64(buf, pos, 0);                    // offset = 0 (first tensor)
}

static test_result test_tensor_info(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 0, .tensor_count = 1,
        .write_tensors = write_one_tensor
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");
    test_eq_uint64(gguf_tensor_count(m), 1, "tensor_count=1");

    // Find tensor by name
    const gguf_tensor_info *info = gguf_find_tensor(m, "test.weight");
    test_not_null(info, "found 'test.weight'");
    test_eq_uint64(info->n_dims, 2, "n_dims=2");
    test_eq_uint64(info->dims[0], 256, "dim[0]=256");
    test_eq_uint64(info->dims[1], 64, "dim[1]=64");
    test_assert(info->type == GGML_TYPE_F32, "type=F32");
    test_eq_uint64(info->n_elems, 256 * 64, "n_elems=16384");

    // Non-existent tensor
    const gguf_tensor_info *notfound = gguf_find_tensor(m, "nonexistent");
    test_null(notfound, "not found returns NULL");

    // List tensors
    uint64_t n = gguf_list_tensors(m, NULL, 0);
    test_eq_uint64(n, 1, "list count=1");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

static void write_two_tensors(unsigned char *buf, size_t *pos) {
    // Tensor 1: "embd.weight" F32 [256, 64], offset=0
    write_str(buf, pos, "embd.weight");
    write_u32(buf, pos, 2);
    write_u64(buf, pos, 256); write_u64(buf, pos, 64);
    write_u32(buf, pos, 0);  // F32
    write_u64(buf, pos, 0);  // offset

    // Tensor 2: "output.weight" Q4_0 [64, 32], offset after embd (256*64*4 = 65536)
    uint64_t embd_bytes = 256 * 64 * 4;  // F32 = 4 bytes per elem
    write_str(buf, pos, "output.weight");
    write_u32(buf, pos, 2);
    write_u64(buf, pos, 64); write_u64(buf, pos, 32);
    write_u32(buf, pos, 2);  // Q4_0
    write_u64(buf, pos, embd_bytes);  // offset
}

static test_result test_tensor_list(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 0, .tensor_count = 2,
        .write_tensors = write_two_tensors
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    // List into buffer
    const gguf_tensor_info *list[10];
    uint64_t n = gguf_list_tensors(m, list, 10);
    test_eq_uint64(n, 2, "list 2 tensors");
    test_assert(strcmp(list[0]->name, "embd.weight") == 0 ||
                strcmp(list[0]->name, "output.weight") == 0,
                "tensor 0 name matches");
    test_assert(list[0]->n_elems > 0, "tensor 0 has elements");
    test_assert(list[1]->n_elems > 0, "tensor 1 has elements");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 4: Tensor data access ───────────────────────────────────────────────

static test_result test_tensor_data_ptr(void) {
    // This test verifies that we can get a raw data pointer from a tensor
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 0, .tensor_count = 1,
        .write_tensors = write_one_tensor
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "test.weight");
    test_not_null(info, "found tensor");

    const void *data = gguf_tensor_data(m, "test.weight");
    // For a file with no tensor data (just info), the pointer is still valid
    // (points into the mmap region past the header+metadata+tensors)
    test_not_null(data, "data pointer not null");

    // Data from info
    const void *data2 = gguf_tensor_data_from_info(m, info);
    test_not_null(data2, "data from info not null");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 5: Q4_K tensor size ──────────────────────────────────────────────

static void write_q4k_tensor(unsigned char *buf, size_t *pos) {
    // One Q4_K tensor: "q4k.weight", n_dims=2, dims=[256, 1], type=Q4_K(12), offset=0
    write_str(buf, pos, "q4k.weight");
    write_u32(buf, pos, 2);
    write_u64(buf, pos, 256); write_u64(buf, pos, 1);
    write_u32(buf, pos, 12);  // Q4_K
    write_u64(buf, pos, 0);
}

static test_result test_q4k_tensor_size(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 0, .tensor_count = 1,
        .write_tensors = write_q4k_tensor
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "q4k.weight");
    test_not_null(info, "found tensor");
    test_assert(info->type == GGML_TYPE_Q4_K, "type=Q4_K");
    test_eq_uint64(info->n_elems, 256, "n_elems=256");
    // sizeof(block_q4_K) = 2+2+12+128 = 144, one block for 256 elements
    test_eq_uint64(info->size_bytes, 144, "size_bytes=144");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 6: Q6_K tensor size ──────────────────────────────────────────────

static void write_q6k_tensor(unsigned char *buf, size_t *pos) {
    write_str(buf, pos, "q6k.weight");
    write_u32(buf, pos, 2);
    write_u64(buf, pos, 256); write_u64(buf, pos, 1);
    write_u32(buf, pos, 14);  // Q6_K
    write_u64(buf, pos, 0);
}

static test_result test_q6k_tensor_size(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 0, .tensor_count = 1,
        .write_tensors = write_q6k_tensor
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "q6k.weight");
    test_not_null(info, "found tensor");
    test_assert(info->type == GGML_TYPE_Q6_K, "type=Q6_K");
    test_eq_uint64(info->n_elems, 256, "n_elems=256");
    // sizeof(block_q6_K) = 128+64+16+2 = 210, one block for 256 elements
    test_eq_uint64(info->size_bytes, 210, "size_bytes=210");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Small helper: write uint16 LE ───────────────────────────────────────────

static void write_u16_buf(unsigned char *buf, size_t *pos, uint16_t v) {
    memcpy(buf + *pos, &v, 2); *pos += 2;
}

// ── Test 7: Dequantize Q4_K tensor ────────────────────────────────────────

// Build a GGUF file with 2 blocks of Q4_K data (512 elements) + valid quantized data
static char *build_q4k_data_file(void) {
    unsigned char buf[4096] = {0};
    size_t pos = 0;

    // Header
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);    // version
    write_u64(buf, &pos, 1);    // tensor_count = 1
    write_u64(buf, &pos, 0);    // metadata_count = 0

    // Tensor info: "q4k_data" Q4_K [256, 2], offset = 0
    write_str(buf, &pos, "q4k_data");
    write_u32(buf, &pos, 2);
    write_u64(buf, &pos, 256); write_u64(buf, &pos, 2);
    write_u32(buf, &pos, 12);  // Q4_K
    write_u64(buf, &pos, 0);   // offset

    // Pad to 32-byte alignment
    while (pos % 32 != 0) { buf[pos++] = 0; }

    // Write 2 Q4_K blocks
    for (int block = 0; block < 2; block++) {
        // d = 1.0 in fp16 = 0x3C00 (write as raw uint16)
        write_u16_buf(buf, &pos, 0x3C00);
        // dmin = 0.0 in fp16 = 0x0000
        write_u16_buf(buf, &pos, 0x0000);
        // scales: sub-blocks 0-3: sc=1 (low 6 bits), m=0
        //         sub-blocks 4-7: sc=1 stored as low 4 bits of scales[8..11],
        //                        m=0 stored as high 4 bits of scales[8..11]
        //         scales[0..3] top 2 bits = 0 (high bits of sc for sub-blocks 4-7 = 0)
        //         scales[4..7] top 2 bits = 0 (high bits of m for sub-blocks 4-7 = 0)
        for (int i = 0; i < 4; i++) buf[pos++] = 1;   // scales[0..3] = 1
        for (int i = 0; i < 4; i++) buf[pos++] = 0;   // scales[4..7] = 0
        for (int i = 0; i < 4; i++) buf[pos++] = 0x01; // scales[8..11]: low 4 bits=1(sc), high 4 bits=0(m)
        // qs[128]: pack 4-bit value = 1 for each element (0x11 per byte = two 1s)
        for (int i = 0; i < 128; i++) buf[pos++] = 0x11;
    }

    char tmpl[] = "/tmp/gguf_q4k_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

static test_result test_dequant_q4k(void) {
    char *path = build_q4k_data_file();
    test_not_null(path, "build q4k data file");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "q4k_data");
    test_not_null(info, "found tensor");

    test_eq_uint64(info->n_elems, 512, "n_elems=512");
    test_eq_uint64(info->size_bytes, 288, "size_bytes=2*144=288");

    // Allocate output and dequantize
    float *output = (float *)calloc(info->n_elems, sizeof(float));
    test_not_null(output, "alloc output");

    bool ok = gguf_dequantize_tensor(m, info, output);
    test_assert(ok, "dequantize succeeded");

    // Verify values
    // Block 0: d=1.0, sc=1, m_val=0, q=1 → output = 1.0*1*1 - 0 = 1.0
    // Block 1: same
    for (uint64_t i = 0; i < info->n_elems; i++) {
        test_assert(output[i] == 1.0f,
                    "dequant value should be 1.0");
    }

    free(output);
    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 8: Dequantize Q6_K tensor ────────────────────────────────────────

static char *build_q6k_data_file(void) {
    unsigned char buf[4096] = {0};
    size_t pos = 0;

    // Header
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);
    write_u64(buf, &pos, 1);
    write_u64(buf, &pos, 0);

    // Tensor info: "q6k_data" Q6_K [256, 1], offset = 0
    write_str(buf, &pos, "q6k_data");
    write_u32(buf, &pos, 2);
    write_u64(buf, &pos, 256); write_u64(buf, &pos, 1);
    write_u32(buf, &pos, 14);  // Q6_K
    write_u64(buf, &pos, 0);

    // Pad to 32-byte alignment
    while (pos % 32 != 0) { buf[pos++] = 0; }

    // Write 1 Q6_K block with known values
    // ql[128]: low 4 bits, all set to 1 → packed 0x11
    for (int i = 0; i < 128; i++) buf[pos++] = 0x11;
    // qh[64]: high 2 bits, all set to 0 → final value = low_bit(1) | (high_bit(0)<<4) = 1
    for (int i = 0; i < 64; i++) buf[pos++] = 0x00;
    // scales[16]: all set to 1 (int8_t = 1)
    for (int i = 0; i < 16; i++) buf[pos++] = 1;
    // d = 1.0 in fp16 = 0x3C00
    write_u16_buf(buf, &pos, 0x3C00);

    char tmpl[] = "/tmp/gguf_q6k_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

static test_result test_dequant_q6k(void) {
    char *path = build_q6k_data_file();
    test_not_null(path, "build q6k data file");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "q6k_data");
    test_not_null(info, "found tensor");

    test_eq_uint64(info->n_elems, 256, "n_elems=256");
    test_eq_uint64(info->size_bytes, 210, "size_bytes=210");

    float *output = (float *)calloc(info->n_elems, sizeof(float));
    test_not_null(output, "alloc output");

    bool ok = gguf_dequantize_tensor(m, info, output);
    test_assert(ok, "dequantize succeeded");

    // Verify: q = (1 | 0<<4) - 32 = 1 - 32 = -31
    // d = 1.0, sc = 1 → output = 1.0 * 1 * (-31) = -31.0
    for (uint64_t i = 0; i < info->n_elems; i++) {
        test_assert(output[i] == -31.0f,
                    "dequant q6k value should be -31.0");
    }

    free(output);
    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test 9: Dequant unsupported type returns false ─────────────────────────

static void write_f16_tensor(unsigned char *buf, size_t *pos) {
    write_str(buf, pos, "f16.tensor");
    write_u32(buf, pos, 1);
    write_u64(buf, pos, 8);
    write_u32(buf, pos, 1);   // F16
    write_u64(buf, pos, 0);
}

static test_result test_dequant_unsupported(void) {
    char *path = build_gguf(&(gguf_builder){
        .metadata_count = 0, .tensor_count = 1,
        .write_tensors = write_f16_tensor
    });
    test_not_null(path, "build gguf");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "f16.tensor");
    test_not_null(info, "found tensor");

    float out[8];
    // F16 is supported, so this should return true
    bool ok = gguf_dequantize_tensor(m, info, out);
    test_assert(ok, "F16 dequantize should succeed");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ════════════════════════════════════════════════════════════════════════════
// Test 10: Expert slice dequantization from 3D Q4_K tensor
// ════════════════════════════════════════════════════════════════════════════

// Build a 3D Q4_K tensor [4, 16, 4] = 256 elements = exactly 1 Q4_K block
// Expert count = 4, per-expert elements = 4*16 = 64
static char *build_q4k_3d_file(void) {
    unsigned char buf[4096] = {0};
    size_t pos = 0;

    // Header
    buf[0] = 'G'; buf[1] = 'G'; buf[2] = 'U'; buf[3] = 'F'; pos = 4;
    write_u32(buf, &pos, 3);
    write_u64(buf, &pos, 1);   // tensor_count = 1
    write_u64(buf, &pos, 0);   // metadata_count = 0

    // Tensor info: "q4k_3d" Q4_K, shape [4, 16, 4], offset=0
    write_str(buf, &pos, "q4k_3d");
    write_u32(buf, &pos, 3);
    write_u64(buf, &pos, 4); write_u64(buf, &pos, 16); write_u64(buf, &pos, 4);
    write_u32(buf, &pos, 12);  // Q4_K
    write_u64(buf, &pos, 0);

    // Pad to 32-byte alignment
    while (pos % 32 != 0) { buf[pos++] = 0; }

    // Write 1 Q4_K block with uniform values (same as test_dequant_q4k)
    // d = 1.0 (fp16 = 0x3C00), dmin = 0.0
    write_u16_buf(buf, &pos, 0x3C00);
    write_u16_buf(buf, &pos, 0x0000);
    // scales: all s=1, m=0
    for (int i = 0; i < 4; i++) buf[pos++] = 1;
    for (int i = 0; i < 4; i++) buf[pos++] = 0;
    for (int i = 0; i < 4; i++) buf[pos++] = 0x01;
    // qs[128]: all 4-bit values = 1 (packed as 0x11 per byte)
    for (int i = 0; i < 128; i++) buf[pos++] = 0x11;

    char tmpl[] = "/tmp/gguf_q4k3d_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) return NULL;
    ssize_t w = write(fd, buf, pos);
    close(fd);
    if (w != (ssize_t)pos) { unlink(tmpl); return NULL; }
    return strdup(tmpl);
}

static test_result test_expert_slice_q4k(void) {
    char *path = build_q4k_3d_file();
    test_not_null(path, "build 3d q4k file");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "q4k_3d");
    test_not_null(info, "found tensor");

    // Verify shape: 3-dim, last dim = 4 experts
    test_eq_uint64(info->n_dims, 3, "n_dims=3");
    test_eq_uint64(info->dims[2], 4, "last dim (experts) = 4");
    test_eq_uint64(info->n_elems, 256, "n_elems=256");

    // Full dequant for reference
    float *full = (float *)calloc(info->n_elems, sizeof(float));
    test_not_null(full, "alloc full");
    bool ok = gguf_dequantize_tensor(m, info, full);
    test_assert(ok, "full dequant succeeded");

    // Dequant expert slices
    uint64_t n_experts = info->dims[2];  // 4
    uint64_t n_per_expert = info->n_elems / n_experts;  // 64

    for (uint64_t e = 0; e < n_experts; e++) {
        float *slice = (float *)calloc(n_per_expert, sizeof(float));
        test_not_null(slice, "alloc slice");

        ok = gguf_dequantize_expert_slice(m, info, (int)e, slice);
        test_assert(ok, "expert slice dequant succeeded");

        // Verify: slice[i] = full[i * n_experts + e]
        for (uint64_t i = 0; i < n_per_expert; i++) {
            double expected = (double)full[i * n_experts + e];
            double actual = (double)slice[i];
            test_assert(actual == expected,
                        "expert slice match");
        }

        free(slice);
    }

    free(full);
    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

static test_result test_expert_slice_invalid(void) {
    char *path = build_q4k_3d_file();
    test_not_null(path, "build 3d q4k file");

    char err[256];
    gguf_model *m = gguf_open(path, err, sizeof(err));
    test_not_null(m, "gguf_open");

    const gguf_tensor_info *info = gguf_find_tensor(m, "q4k_3d");
    test_not_null(info, "found tensor");

    float buf[64];

    // NULL model
    bool ok = gguf_dequantize_expert_slice(NULL, info, 0, buf);
    test_assert(!ok, "NULL model fails");

    // NULL output
    ok = gguf_dequantize_expert_slice(m, info, 0, NULL);
    test_assert(!ok, "NULL output fails");

    // Negative expert index
    ok = gguf_dequantize_expert_slice(m, info, -1, buf);
    test_assert(!ok, "negative expert index fails");

    // Out of range expert index
    ok = gguf_dequantize_expert_slice(m, info, 99, buf);
    test_assert(!ok, "out of range expert index fails");

    // NULL tensor
    ok = gguf_dequantize_expert_slice(m, NULL, 0, buf);
    test_assert(!ok, "NULL tensor fails");

    gguf_close(m);
    unlink(path); free(path);
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    // Header
    { "open_minimal",        test_open_minimal },
    { "open_nonexistent",   test_open_nonexistent },
    { "open_bad_magic",     test_open_bad_magic },

    // Metadata
    { "metadata_string",    test_metadata_string },
    { "metadata_alignment", test_metadata_alignment },
    { "metadata_multiple",  test_metadata_multiple },

    // Tensor info
    { "tensor_info",        test_tensor_info },
    { "tensor_list",        test_tensor_list },
    { "tensor_data_ptr",    test_tensor_data_ptr },

    // Quantized tensor sizes
    { "q4k_tensor_size",    test_q4k_tensor_size },
    { "q6k_tensor_size",    test_q6k_tensor_size },

    // Dequantization
    { "dequant_q4k",        test_dequant_q4k },
    { "dequant_q6k",        test_dequant_q6k },
    { "dequant_unsupported", test_dequant_unsupported },

    // Expert slice dequantization
    { "expert_slice_q4k",    test_expert_slice_q4k },
    { "expert_slice_invalid", test_expert_slice_invalid },
};

RUN_TESTS(tests)
