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
    // GGUF string metadata value: type(STRING)=11 + string
    write_u32(buf, pos, 11);  // GGUF_META_TYPE_STRING
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
};

RUN_TESTS(tests)
