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
    case 4: { // UINT32
        uint32_t v;
        if (!read_at(m, *pos, &v, 4)) return false;
        *pos += 4;
        out->type = GGUF_VALUE_UINT32; out->value.uint32 = v;
        return true;
    }
    case 8: { // FLOAT32
        float v;
        if (!read_at(m, *pos, &v, 4)) return false;
        *pos += 4;
        out->type = GGUF_VALUE_FLOAT32; out->value.float32 = v;
        return true;
    }
    case 11: { // STRING
        gguf_str s;
        if (!read_str(m, pos, &s)) return false;
        out->type = GGUF_VALUE_STRING;
        out->value.string.data = s.data;
        out->value.string.len = s.len;
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
            if (!read_str(m, &pos, &key)) { snprintf(err, err_sz, "bad meta key"); return false; }
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
                vt == 11) {
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
            t->size_bytes = t->n_elems * 4;  // assume F32 for now
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
    return (const uint8_t *)m->mapped + m->tensor_data_offset + info->offset;
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
