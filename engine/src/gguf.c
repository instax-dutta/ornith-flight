// SPDX-License-Identifier: MIT
// GGUF parser — header + metadata + tensor info. TDD-built.

#include "gguf.h"
#include "third_party/gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

// ── fp16 conversion ─────────────────────────────────────────────────────────

static inline float fp16_to_fp32(uint16_t h) {
    // IEEE 754 half-precision to float
    if (h == 0) return 0.0f;  // handle zero
    // For normal-range fp16: sign, exp+112 bias, mant<<13
    uint32_t u = ((uint32_t)(h & 0x8000) << 16)
               | ((((uint32_t)(h & 0x7C00) >> 10) + 112) << 23)
               | ((uint32_t)(h & 0x03FF) << 13);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// ── Q4_K dequantization helper ──────────────────────────────────────────────

// Extract scale (d) and min (m) for sub-block j from packed scales[12] array.
static inline void get_scale_min_k4(int j, const uint8_t *scales,
                                     uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = scales[j] & 63;
        *m = scales[j + 4] & 63;
    } else {
        *d = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4);
        *m = (scales[j + 4] >> 4) | ((scales[j - 0] >> 6) << 4);
    }
}

// Dequantize one Q4_K super-block (256 elements → 256 floats)
static void dequantize_block_q4_K(const block_q4_K *x, float *y) {
    const uint8_t *q = x->qs;
    const float d = fp16_to_fp32(x->d);
    const float min = fp16_to_fp32(x->dmin);
    int is = 0;
    for (int j = 0; j < QK_K; j += 64) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x->scales, &sc, &m);
        const float d1 = d * (float)sc;
        const float m1 = min * (float)m;
        get_scale_min_k4(is + 1, x->scales, &sc, &m);
        const float d2 = d * (float)sc;
        const float m2 = min * (float)m;
        for (int l = 0; l < 32; ++l) *y++ = d1 * (float)(q[l] & 0x0F) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
        is += 2;
    }
}

// Dequantize one Q6_K super-block (256 elements → 256 floats)
static void dequantize_block_q6_K(const block_q6_K *x, float *y) {
    const float d = fp16_to_fp32(x->d);
    const uint8_t *ql = x->ql;
    const uint8_t *qh = x->qh;
    const int8_t *sc = x->scales;
    for (int n = 0; n < QK_K; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int8_t q1 = (int8_t)((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int8_t q2 = (int8_t)((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l + 0]  = d * (float)sc[is + 0] * (float)q1;
            y[l + 32] = d * (float)sc[is + 2] * (float)q2;
            y[l + 64] = d * (float)sc[is + 4] * (float)q3;
            y[l + 96] = d * (float)sc[is + 6] * (float)q4;
        }
        y  += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ── Internal helpers ─────────────────────────────────────────────────────────

typedef struct {
    uint64_t len;
    char    *data;  // pointer into mmap (NOT NUL-terminated)
} gguf_str;

typedef struct {
    char                  *key;  // NUL-terminated copy
    gguf_metadata_value    val;
    uint32_t               raw_type;  // original GGUF value type
} metadata_entry;

struct gguf_model {
    int       fd;
    void     *mapped;
    size_t    mapped_size;
    uint32_t  version;
    uint64_t  tensor_count;
    uint64_t  metadata_count;
    size_t    alignment;
    char      architecture[64];
    bool      arch_set;

    metadata_entry     *metadata;
    gguf_tensor_info   *tensors;
    uint64_t            tensors_alloc;
    size_t              tensor_data_offset;  // file offset where tensor data starts
};

// ── Read helpers (little-endian, all data in mmap) ───────────────────────────

static bool read_at(const gguf_model *m, size_t pos, void *buf, size_t n) {
    if (pos + n > m->mapped_size) return false;
    memcpy(buf, (const uint8_t *)m->mapped + pos, n);
    return true;
}

static bool read_str(const gguf_model *m, size_t *pos, gguf_str *out) {
    if (!read_at(m, *pos, &out->len, 8)) return false;
    *pos += 8;
    if (out->len > 0) {
        if (*pos + out->len > m->mapped_size) return false;
        out->data = (char *)((const uint8_t *)m->mapped + *pos);
        *pos += out->len;
    } else {
        out->data = "";
    }
    return true;
}

static bool read_metadata_val(const gguf_model *m, size_t *pos,
                              uint32_t type, gguf_metadata_value *out) {
    memset(out, 0, sizeof(*out));

    switch (type) {
    case 0: { // UINT8
        uint8_t v;
        if (!read_at(m, *pos, &v, 1)) return false;
        *pos += 1;
        out->type = GGUF_VALUE_UINT8; out->value.uint8 = v;
        return true;
    }
    case 1: { // INT8
        int8_t v;
        if (!read_at(m, *pos, &v, 1)) return false;
        *pos += 1;
        out->type = GGUF_VALUE_INT8; out->value.int8 = v;
        return true;
    }
    case 2: { // UINT16
        uint16_t v;
        if (!read_at(m, *pos, &v, 2)) return false;
        *pos += 2;
        out->type = GGUF_VALUE_UINT16; out->value.uint16 = v;
        return true;
    }
    case 3: { // INT16
        int16_t v;
        if (!read_at(m, *pos, &v, 2)) return false;
        *pos += 2;
        out->type = GGUF_VALUE_INT16; out->value.int16 = v;
        return true;
    }
    case 4: { // UINT32
        uint32_t v;
        if (!read_at(m, *pos, &v, 4)) return false;
        *pos += 4;
        out->type = GGUF_VALUE_UINT32; out->value.uint32 = v;
        return true;
    }
    case 5: { // INT32
        int32_t v;
        if (!read_at(m, *pos, &v, 4)) return false;
        *pos += 4;
        out->type = GGUF_VALUE_INT32; out->value.int32 = v;
        return true;
    }
    case 6: { // FLOAT32
        float v;
        if (!read_at(m, *pos, &v, 4)) return false;
        *pos += 4;
        out->type = GGUF_VALUE_FLOAT32; out->value.float32 = v;
        return true;
    }
    case 7: { // BOOL
        uint8_t v;
        if (!read_at(m, *pos, &v, 1)) return false;
        *pos += 1;
        out->type = GGUF_VALUE_BOOL; out->value.boolean = (v != 0);
        return true;
    }
    case 8: { // STRING
        gguf_str s;
        if (!read_str(m, pos, &s)) return false;
        out->type = GGUF_VALUE_STRING;
        out->value.string.data = s.data;
        out->value.string.len = s.len;
        return true;
    }
    case 9: { // ARRAY
        uint32_t elem_type;
        if (!read_at(m, *pos, &elem_type, 4)) return false;
        *pos += 4;

        uint64_t count;
        if (!read_at(m, *pos, &count, 8)) return false;
        *pos += 8;

        // Store the raw data start and count
        size_t data_start = *pos;

        // Skip over the array elements to advance pos
        bool ok = true;
        for (uint64_t i = 0; i < count && ok; i++) {
            switch (elem_type) {
            case 0: case 1: *pos += 1; break;  // UINT8, INT8
            case 2: case 3: *pos += 2; break;  // UINT16, INT16
            case 4: case 5: *pos += 4; break;  // UINT32, INT32
            case 6: *pos += 4; break;  // FLOAT32
            case 7: *pos += 1; break;  // BOOL
            case 8: { // STRING element
                gguf_str s;
                ok = read_str(m, pos, &s);
                break;
            }
            case 9: // ARRAY element — skip 4+8 bytes (type + count)
                *pos += 4 + 8; break;
            case 10: *pos += 8; break;  // UINT64
            case 11: *pos += 8; break;  // INT64
            case 12: *pos += 8; break;  // FLOAT64
            default: ok = false; break;
            }
        }

        if (!ok) return false;

        out->type = GGUF_VALUE_ARRAY;
        out->value.array.elem_type = (gguf_value_type)elem_type;
        out->value.array.count = (size_t)count;
        out->value.array.data = (const uint8_t *)m->mapped + data_start;
        return true;
    }
    case 10: { // UINT64
        uint64_t v;
        if (!read_at(m, *pos, &v, 8)) return false;
        *pos += 8;
        out->type = GGUF_VALUE_UINT64; out->value.uint64 = v;
        return true;
    }
    case 11: { // INT64
        int64_t v;
        if (!read_at(m, *pos, &v, 8)) return false;
        *pos += 8;
        out->type = GGUF_VALUE_INT64; out->value.int64 = v;
        return true;
    }
    case 12: { // FLOAT64
        double v;
        if (!read_at(m, *pos, &v, 8)) return false;
        *pos += 8;
        out->type = GGUF_VALUE_FLOAT64; out->value.float64 = v;
        return true;
    }
    default:
        return false;
    }
}

static int tensor_cmp(const void *a, const void *b) {
    const gguf_tensor_info *ta = (const gguf_tensor_info *)a;
    const gguf_tensor_info *tb = (const gguf_tensor_info *)b;
    return strcmp(ta->name, tb->name);
}

static inline size_t align_up(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

// ── Parsing ──────────────────────────────────────────────────────────────────

static bool parse_all(gguf_model *m, char *err, size_t err_sz) {
    // Parse header: magic(4) + version(4) + tensor_count(8) + metadata_count(8)
    size_t pos = 0;
    uint32_t magic;

    if (!read_at(m, pos, &magic, 4)) { snprintf(err, err_sz, "too small"); return false; }
    pos += 4;
    if (magic != GGUF_MAGIC_U32) { snprintf(err, err_sz, "bad magic"); return false; }

    if (!read_at(m, pos, &m->version, 4)) { snprintf(err, err_sz, "no version"); return false; }
    pos += 4;
    if (m->version > 3) { snprintf(err, err_sz, "bad version"); return false; }

    if (!read_at(m, pos, &m->tensor_count, 8)) { snprintf(err, err_sz, "no tensor_count"); return false; }
    pos += 8;
    if (!read_at(m, pos, &m->metadata_count, 8)) { snprintf(err, err_sz, "no metadata_count"); return false; }
    pos += 8;

    m->alignment = 32;  // default

    // Parse metadata
    if (m->metadata_count > 0) {
        m->metadata = (metadata_entry *)calloc(m->metadata_count, sizeof(metadata_entry));
        if (!m->metadata) { snprintf(err, err_sz, "OOM metadata"); return false; }

        for (uint64_t i = 0; i < m->metadata_count; i++) {
            // Key
            gguf_str key;
            if (!read_str(m, &pos, &key)) {
                snprintf(err, err_sz, "bad meta key at entry %llu, pos=%zu, mapped=%zu",
                         (unsigned long long)i, pos, m->mapped_size);
                return false;
            }
            m->metadata[i].key = (char *)malloc(key.len + 1);
            memcpy(m->metadata[i].key, key.data, key.len);
            m->metadata[i].key[key.len] = '\0';

            // Value type
            uint32_t vt;
            if (!read_at(m, pos, &vt, 4)) { snprintf(err, err_sz, "bad meta type"); return false; }
            pos += 4;
            m->metadata[i].raw_type = vt;

            // Value
            if (!read_metadata_val(m, &pos, vt, &m->metadata[i].val)) {
                snprintf(err, err_sz, "bad meta val for '%s'", m->metadata[i].key);
                return false;
            }

            // Capture architecture
            if (strcmp(m->metadata[i].key, "general.architecture") == 0 &&
                vt == 8) {
                size_t alen = m->metadata[i].val.value.string.len;
                if (alen >= sizeof(m->architecture)) alen = sizeof(m->architecture) - 1;
                memcpy(m->architecture, m->metadata[i].val.value.string.data, alen);
                m->architecture[alen] = '\0';
                m->arch_set = true;
            }

            // Capture alignment
            if (strcmp(m->metadata[i].key, "general.alignment") == 0 && vt == 4) {
                m->alignment = m->metadata[i].val.value.uint32;
            }
        }
    }

    // Parse tensor info
    if (m->tensor_count > 0) {
        m->tensors = (gguf_tensor_info *)calloc(m->tensor_count, sizeof(gguf_tensor_info));
        if (!m->tensors) { snprintf(err, err_sz, "OOM tensors"); return false; }

        for (uint64_t i = 0; i < m->tensor_count; i++) {
            gguf_tensor_info *t = &m->tensors[i];

            gguf_str name;
            if (!read_str(m, &pos, &name)) { snprintf(err, err_sz, "bad tensor name"); return false; }
            char *nc = (char *)malloc(name.len + 1);
            memcpy(nc, name.data, name.len);
            nc[name.len] = '\0';
            t->name = nc;

            uint32_t nd;
            if (!read_at(m, pos, &nd, 4)) return false; pos += 4;
            t->n_dims = nd;
            if (nd > 4) nd = 4;

            for (uint32_t d = 0; d < nd; d++) {
                uint64_t dim;
                if (!read_at(m, pos, &dim, 8)) return false; pos += 8;
                t->dims[d] = dim;
            }
            for (uint32_t d = nd; d < 4; d++) t->dims[d] = 1;

            uint32_t ty;
            if (!read_at(m, pos, &ty, 4)) return false; pos += 4;
            t->type = (ggml_type)ty;

            if (!read_at(m, pos, &t->offset, 8)) return false; pos += 8;

            t->n_elems = 1;
            for (uint32_t d = 0; d < 4; d++) t->n_elems *= t->dims[d];
            t->size_bytes = ggml_row_size((ggml_type)t->type, t->n_elems);
        }

        // Compute tensor data start (aligned)
        m->tensor_data_offset = align_up(pos, m->alignment);

        // Sort tensors for binary search
        qsort(m->tensors, m->tensor_count, sizeof(gguf_tensor_info), tensor_cmp);
        m->tensors_alloc = m->tensor_count;
    } else {
        m->tensor_data_offset = pos;
    }

    return true;
}

// ── Public API ───────────────────────────────────────────────────────────────

gguf_model *gguf_open(const char *filepath, char *err_buf, size_t err_buf_size) {
    if (!filepath) { snprintf(err_buf, err_buf_size, "NULL path"); return NULL; }

    int fd = open(filepath, O_RDONLY);
    if (fd < 0) { snprintf(err_buf, err_buf_size, "cannot open"); return NULL; }

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); snprintf(err_buf, err_buf_size, "fstat"); return NULL; }

    // ── Map the full file, then advise kernel not to cache tensor data ──
    void *mapped = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) { close(fd); snprintf(err_buf, err_buf_size, "mmap"); return NULL; }

    gguf_model *m = (gguf_model *)calloc(1, sizeof(gguf_model));
    m->fd = fd;
    m->mapped = mapped;
    m->mapped_size = (size_t)st.st_size;

    if (!parse_all(m, err_buf, err_buf_size)) {
        gguf_close(m);
        return NULL;
    }

    // Phase D: advise kernel not to cache tensor data pages.
    // The mmap stays valid but we read tensor data via pread() to avoid
    // paging in the full 20 GB model file on 8 GB machines.
    // MADV_DONTNEED frees any pages already faulted in for the tensor data.
    size_t tensor_start = m->tensor_data_offset;
    if (tensor_start < m->mapped_size) {
        size_t tensor_len = m->mapped_size - tensor_start;
        // Only advise for files large enough to benefit
        if (tensor_len > 1024 * 1024) {
            madvise((uint8_t*)m->mapped + tensor_start, tensor_len, MADV_DONTNEED);
        }
    }

    return m;
}

void gguf_close(gguf_model *model) {
    if (!model) return;
    if (model->mapped) munmap(model->mapped, model->mapped_size);
    if (model->fd >= 0) close(model->fd);
    if (model->metadata) {
        for (uint64_t i = 0; i < model->metadata_count; i++) free(model->metadata[i].key);
        free(model->metadata);
    }
    if (model->tensors) {
        for (uint64_t i = 0; i < model->tensors_alloc; i++)
            free((void *)model->tensors[i].name);
        free(model->tensors);
    }
    memset(model, 0, sizeof(*model));
    free(model);
}

uint32_t gguf_version(const gguf_model *m) { return m ? m->version : 0; }
uint64_t gguf_tensor_count(const gguf_model *m) { return m ? m->tensor_count : 0; }
uint64_t gguf_metadata_count(const gguf_model *m) { return m ? m->metadata_count : 0; }
const char *gguf_architecture(const gguf_model *m) { return (m && m->arch_set) ? m->architecture : NULL; }
size_t gguf_alignment(const gguf_model *m) { return m ? m->alignment : 32; }

bool gguf_find_metadata(const gguf_model *m, const char *key, gguf_metadata_value *out) {
    if (!m || !key || !out) return false;
    for (uint64_t i = 0; i < m->metadata_count; i++) {
        if (strcmp(m->metadata[i].key, key) == 0) {
            *out = m->metadata[i].val;
            return true;
        }
    }
    return false;
}

const gguf_tensor_info *gguf_find_tensor(const gguf_model *m, const char *name) {
    if (!m || !name) return NULL;
    gguf_tensor_info key;
    key.name = name;
    return (const gguf_tensor_info *)bsearch(
        &key, m->tensors, m->tensors_alloc, sizeof(gguf_tensor_info), tensor_cmp);
}

uint64_t gguf_list_tensors(const gguf_model *m, const gguf_tensor_info **tensors, uint64_t max) {
    if (!m) return 0;
    if (!tensors) return m->tensors_alloc;
    uint64_t n = m->tensors_alloc < max ? m->tensors_alloc : max;
    for (uint64_t i = 0; i < n; i++) tensors[i] = &m->tensors[i];
    return n;
}

const void *gguf_tensor_data(const gguf_model *m, const char *name) {
    const gguf_tensor_info *info = gguf_find_tensor(m, name);
    return info ? gguf_tensor_data_from_info(m, info) : NULL;
}

const void *gguf_tensor_data_from_info(const gguf_model *m, const gguf_tensor_info *info) {
    if (!m || !info) return NULL;
    if (!m->mapped) return NULL;
    return (const uint8_t *)m->mapped + m->tensor_data_offset + info->offset;
}

// ── Single-element Q4_K and Q6_K extraction helpers ────────────────────────

// Extract one element at position 'pos' (0-255) from a Q4_K super-block.
static float dequantize_q4_K_one(const block_q4_K *block, int pos) {
    int sub = pos / 32;          // which of 8 sub-blocks (0-7)
    int chunk = pos / 64;        // which 64-element chunk (0-3)
    int p = pos % 64;            // position within chunk (0-63)
    int s = p / 32;              // which sub-block within chunk (0 or 1)
    int sub_pos = p % 32;        // position within sub-block (0-31)

    uint8_t sc, m;
    get_scale_min_k4(sub, block->scales, &sc, &m);

    float d_val = fp16_to_fp32(block->d) * (float)sc;
    float min_val = fp16_to_fp32(block->dmin) * (float)m;

    int qs_idx = chunk * 32 + sub_pos;
    uint8_t q = (s == 0) ? (block->qs[qs_idx] & 0x0F) : (block->qs[qs_idx] >> 4);

    return d_val * (float)q - min_val;
}

// Extract one element at position 'pos' (0-255) from a Q6_K super-block.
// Uses the same unpacking pattern as dequantize_block_q6_K.
static float dequantize_q6_K_one(const block_q6_K *block, int pos) {
    // Q6_K block: 256 elements, processed in two 128-element passes.
    // Each pass: ql advances 64, qh advances 32, sc advances 8.
    // Within pass, 32 positions (l=0..31) with 4 lanes each:
    //   lane0 (y[l+0]):  ql[l+0] low nibble  | qh[l] bits 0-1, sc[is+0]
    //   lane1 (y[l+32]): ql[l+32] low nibble | qh[l] bits 2-3, sc[is+2]
    //   lane2 (y[l+64]): ql[l+0] high nibble | qh[l] bits 4-5, sc[is+4]
    //   lane3 (y[l+96]): ql[l+32] high nibble| qh[l] bits 6-7, sc[is+6]
    //   where is = l/16 (0 for first 16 positions, 1 for second 16)

    float d_out = fp16_to_fp32(block->d);

    int pass   = pos / 128;             // 0 or 1
    int lane   = (pos % 128) / 32;      // 0,1,2,3
    int l      = pos % 32;              // 0-31
    int is     = l / 16;                // 0 or 1

    // ql index: pass * 64 + lane_offset(0 for lanes 0,2; 32 for lanes 1,3) + l
    int ql_off = (lane % 2 == 0) ? l : (32 + l);
    int ql_idx = pass * 64 + ql_off;
    int ql_byte = (int)block->ql[ql_idx];

    // qh index: pass * 32 + l
    int qh_byte = (int)block->qh[pass * 32 + l];

    // Extract 4-bit low part, 2-bit high part
    int q_val;
    if (lane < 2) {  // lanes 0,1: use low nibble of ql
        q_val = (ql_byte & 0x0F) | (((qh_byte >> (lane * 2)) & 3) << 4);
    } else {  // lanes 2,3: use high nibble of ql
        q_val = (ql_byte >> 4) | (((qh_byte >> (lane * 2)) & 3) << 4);
    }
    q_val -= 32;

    // Scale index: pass * 8 + lane * 2 + is
    int sc_idx = pass * 8 + lane * 2 + is;
    int8_t sc_val = block->scales[sc_idx];

    return d_out * (float)sc_val * (float)q_val;
}

// ── Streaming read: pread raw tensor data from file ───────────────────────

size_t gguf_read_tensor_data(const gguf_model *m,
                             const gguf_tensor_info *tensor,
                             void *buffer, size_t buffer_size) {
    if (!m || !tensor || !buffer) return 0;
    if (buffer_size < tensor->size_bytes) return 0;
    if (m->fd < 0) return 0;

    size_t file_offset = m->tensor_data_offset + tensor->offset;
    ssize_t n = pread(m->fd, buffer, tensor->size_bytes, file_offset);
    if ((size_t)n != tensor->size_bytes) return 0;
    return (size_t)n;
}

size_t gguf_read_tensor_bytes(const gguf_model *m,
                              const gguf_tensor_info *tensor,
                              size_t byte_offset,
                              void *buffer,
                              size_t buffer_size) {
    if (!m || !tensor || !buffer) return 0;
    if (byte_offset + buffer_size > tensor->size_bytes) return 0;
    if (m->fd < 0) return 0;

    size_t file_offset = m->tensor_data_offset + tensor->offset + byte_offset;
    ssize_t n = pread(m->fd, buffer, buffer_size, file_offset);
    if ((size_t)n != buffer_size) return 0;
    return (size_t)n;
}

// ── Public dequantization API (from buffer) ──────────────────────────────

bool gguf_dequantize_tensor_from_buf(const void *data,
                                     const gguf_tensor_info *tensor,
                                     float *output) {
    if (!data || !tensor || !output) return false;

    uint64_t n_blocks = tensor->n_elems / QK_K;

    switch (tensor->type) {
    case GGML_TYPE_F32:
        memcpy(output, data, tensor->size_bytes);
        return true;
    case GGML_TYPE_F16: {
        const uint16_t *src = (const uint16_t *)data;
        for (uint64_t i = 0; i < tensor->n_elems; i++) {
            output[i] = fp16_to_fp32(src[i]);
        }
        return true;
    }
    case GGML_TYPE_Q4_K:
        for (uint64_t b = 0; b < n_blocks; b++) {
            const block_q4_K *block = (const block_q4_K *)data + b;
            dequantize_block_q4_K(block, output + b * QK_K);
        }
        return true;
    case GGML_TYPE_Q6_K:
        for (uint64_t b = 0; b < n_blocks; b++) {
            const block_q6_K *block = (const block_q6_K *)data + b;
            dequantize_block_q6_K(block, output + b * QK_K);
        }
        return true;
    default:
        return false;
    }
}

// Wrapper — reads from mmap if tensor data is in metadata mapping
bool gguf_dequantize_tensor(const gguf_model *m,
                            const gguf_tensor_info *tensor,
                            float *output) {
    if (!m || !tensor || !output) return false;
    const void *data = gguf_tensor_data_from_info(m, tensor);
    if (!data) return false;
    return gguf_dequantize_tensor_from_buf(data, tensor, output);
}

// ── Expert slice dequantization (from buffer) ──────────────────────────────

bool gguf_dequantize_expert_slice_from_buf(const void *data,
                                           const gguf_tensor_info *tensor,
                                           int expert_idx,
                                           float *output) {
    if (!data || !tensor || !output) return false;
    if (expert_idx < 0) return false;

    // The last (fastest-varying) dimension is the expert dimension
    uint64_t n_experts = tensor->dims[tensor->n_dims - 1];
    if ((uint64_t)expert_idx >= n_experts) return false;
    if (n_experts == 0) return false;

    uint64_t n_elems_per_expert = tensor->n_elems / n_experts;

    switch (tensor->type) {
    case GGML_TYPE_F32: {
        // Strided: every n_experts-th element belongs to expert_idx
        const float *src = (const float *)data;
        for (uint64_t i = 0; i < n_elems_per_expert; i++) {
            output[i] = src[i * n_experts + expert_idx];
        }
        return true;
    }
    case GGML_TYPE_F16: {
        // Strided fp16: every n_experts-th half belongs to expert_idx
        const uint16_t *src = (const uint16_t *)data;
        for (uint64_t i = 0; i < n_elems_per_expert; i++) {
            output[i] = fp16_to_fp32(src[i * n_experts + expert_idx]);
        }
        return true;
    }
    case GGML_TYPE_Q4_K: {
        // Q4_K: each 256-element super-block has exactly 1 element per expert
        // (since n_experts = 256 = QK_K in standard MoE models)
        for (uint64_t i = 0; i < n_elems_per_expert; i++) {
            // Linear position in the fused tensor for the i-th element of expert_idx
            uint64_t linear = i * n_experts + (uint64_t)expert_idx;
            uint64_t block_idx = linear / QK_K;
            int pos_in_block = (int)(linear % QK_K);

            const block_q4_K *block = (const block_q4_K *)data + block_idx;
            output[i] = dequantize_q4_K_one(block, pos_in_block);
        }
        return true;
    }
    case GGML_TYPE_Q6_K: {
        for (uint64_t i = 0; i < n_elems_per_expert; i++) {
            uint64_t linear = i * n_experts + (uint64_t)expert_idx;
            uint64_t block_idx = linear / QK_K;
            int pos_in_block = (int)(linear % QK_K);

            const block_q6_K *block = (const block_q6_K *)data + block_idx;
            output[i] = dequantize_q6_K_one(block, pos_in_block);
        }
        return true;
    }
    default:
        return false;
    }
}

// Wrapper — fallback to _from_buf if tensor data is accessible via mmap
bool gguf_dequantize_expert_slice(const gguf_model *m,
                                  const gguf_tensor_info *tensor,
                                  int expert_idx,
                                  float *output) {
    if (!m || !tensor || !output) return false;
    const void *data = gguf_tensor_data_from_info(m, tensor);
    if (!data) return false;
    return gguf_dequantize_expert_slice_from_buf(data, tensor, expert_idx, output);
}

// ── Architecture-prefixed parameter lookup ───────────────────────────────────

static void make_key(const gguf_model *m, const char *param, char *key, size_t key_sz) {
    const char *arch = gguf_architecture(m);
    if (arch) {
        snprintf(key, key_sz, "%s.%s", arch, param);
    } else {
        snprintf(key, key_sz, "unknown.%s", param);
    }
}

bool gguf_get_param_u32(const gguf_model *m, const char *param, uint32_t *out) {
    if (!m || !param || !out) return false;
    char key[128];
    make_key(m, param, key, sizeof(key));
    gguf_metadata_value val;
    if (!gguf_find_metadata(m, key, &val)) return false;
    if (val.type == GGUF_VALUE_UINT32) { *out = val.value.uint32; return true; }
    if (val.type == GGUF_VALUE_INT32)  { *out = (uint32_t)val.value.int32; return true; }
    return false;
}

bool gguf_get_param_f32(const gguf_model *m, const char *param, float *out) {
    if (!m || !param || !out) return false;
    char key[128];
    make_key(m, param, key, sizeof(key));
    gguf_metadata_value val;
    if (!gguf_find_metadata(m, key, &val)) return false;
    if (val.type == GGUF_VALUE_FLOAT32) { *out = val.value.float32; return true; }
    if (val.type == GGUF_VALUE_FLOAT64) { *out = (float)val.value.float64; return true; }
    return false;
}
