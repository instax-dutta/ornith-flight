// SPDX-License-Identifier: MIT
// BPE tokenizer — loads vocabulary + merges from GGUF metadata.
// Supports GPT-2 style BPE (byte-level, merge-ranks).

#include "tokenizer.h"
#include "gguf.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

// ── Internal types ──────────────────────────────────────────────────────────

#define HASH_CAPACITY 524288  // ~2x expected merge count (248K)

typedef struct {
    uint32_t a;
    uint32_t b;
    int      rank;  // -1 = empty
} merge_entry;

typedef struct {
    uint32_t id;
    char    *token;
    bool     is_special;
} vocab_entry;

struct tokenizer {
    vocab_entry *vocab;
    int          vocab_size;
    int          vocab_cap;

    // BPE merge table (hash map)
    merge_entry *merge_hash;
    int          merge_count;

    // Special token IDs
    uint32_t bos_id;
    uint32_t eos_id;
    uint32_t unk_id;
    uint32_t pad_id;
};

// ── Hash helpers ────────────────────────────────────────────────────────────

static uint32_t hash_pair(uint32_t a, uint32_t b) {
    return (a * 2654435761U) ^ (b * 2246822519U);
}

static void merge_table_init(tokenizer *tok) {
    tok->merge_hash = (merge_entry *)calloc(HASH_CAPACITY, sizeof(merge_entry));
    tok->merge_count = 0;
    if (tok->merge_hash) {
        for (size_t i = 0; i < HASH_CAPACITY; i++)
            tok->merge_hash[i].rank = -1;
    }
}

static void merge_table_set(tokenizer *tok, uint32_t a, uint32_t b, int rank) {
    if (!tok->merge_hash) return;
    uint32_t h = hash_pair(a, b) % HASH_CAPACITY;
    for (size_t i = 0; i < HASH_CAPACITY; i++) {
        size_t idx = (h + i) % HASH_CAPACITY;
        if (tok->merge_hash[idx].rank == -1) {
            tok->merge_hash[idx].a = a;
            tok->merge_hash[idx].b = b;
            tok->merge_hash[idx].rank = rank;
            tok->merge_count++;
            return;
        }
        if (tok->merge_hash[idx].a == a && tok->merge_hash[idx].b == b) {
            tok->merge_hash[idx].rank = rank;  // update
            return;
        }
    }
}

static int merge_table_get(const tokenizer *tok, uint32_t a, uint32_t b) {
    if (!tok->merge_hash) return -1;
    uint32_t h = hash_pair(a, b) % HASH_CAPACITY;
    for (size_t i = 0; i < HASH_CAPACITY; i++) {
        size_t idx = (h + i) % HASH_CAPACITY;
        if (tok->merge_hash[idx].rank == -1) return -1;
        if (tok->merge_hash[idx].a == a && tok->merge_hash[idx].b == b)
            return tok->merge_hash[idx].rank;
    }
    return -1;
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

tokenizer *tokenizer_create(void) {
    tokenizer *tok = (tokenizer *)calloc(1, sizeof(tokenizer));
    if (!tok) return NULL;

    tok->vocab_cap = 256;
    tok->vocab = (vocab_entry *)calloc((size_t)tok->vocab_cap, sizeof(vocab_entry));
    if (!tok->vocab) { free(tok); return NULL; }

    merge_table_init(tok);

    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->unk_id = 0;
    tok->pad_id = 3;
    return tok;
}

tokenizer *tokenizer_create_from_gguf(const gguf_model *gguf) {
    if (!gguf) return NULL;

    // Look up tokenizer metadata
    gguf_metadata_value val;
    bool has_tokens = gguf_find_metadata(gguf, "tokenizer.ggml.tokens", &val);
    bool has_merges = gguf_find_metadata(gguf, "tokenizer.ggml.merges", &val);

    if (!has_tokens || !has_merges) {
        fprintf(stderr, "Tokenizer: GGUF missing tokenizer.ggml.tokens or merges\n");
        return NULL;
    }

    // Read tokens array
    size_t   token_count = 0;
    const uint8_t *token_data = NULL;

    if (gguf_find_metadata(gguf, "tokenizer.ggml.tokens", &val) &&
        val.type == GGUF_VALUE_ARRAY) {
        uint32_t elem_type = (uint32_t)val.value.array.elem_type;
        if (elem_type != 8) {  // 8 = GGUF_VALUE_STRING
            fprintf(stderr, "Tokenizer: expected string array for tokens, got type %u\n", elem_type);
            return NULL;
        }
        token_count = val.value.array.count;
        token_data = (const uint8_t *)val.value.array.data;
    }
    if (!token_data || token_count == 0) return NULL;

    // Read merges array
    size_t   merge_count = 0;
    const uint8_t *merge_data = NULL;

    if (gguf_find_metadata(gguf, "tokenizer.ggml.merges", &val) &&
        val.type == GGUF_VALUE_ARRAY) {
        merge_count = val.value.array.count;
        merge_data = (const uint8_t *)val.value.array.data;
    }

    // Read special token IDs
    uint32_t bos_id = 1, eos_id = 2, pad_id = 3, unk_id = 0;

    if (gguf_find_metadata(gguf, "tokenizer.ggml.bos_token_id", &val) &&
        val.type == GGUF_VALUE_UINT32) bos_id = val.value.uint32;
    if (gguf_find_metadata(gguf, "tokenizer.ggml.eos_token_id", &val) &&
        val.type == GGUF_VALUE_UINT32) eos_id = val.value.uint32;
    if (gguf_find_metadata(gguf, "tokenizer.ggml.padding_token_id", &val) &&
        val.type == GGUF_VALUE_UINT32) pad_id = val.value.uint32;
    if (gguf_find_metadata(gguf, "tokenizer.ggml.unknown_token_id", &val) &&
        val.type == GGUF_VALUE_UINT32) unk_id = val.value.uint32;
    if (gguf_find_metadata(gguf, "tokenizer.ggml.unk_token_id", &val) &&
        val.type == GGUF_VALUE_UINT32) unk_id = val.value.uint32;

    // Also check for token_type array to detect special tokens
    const uint8_t *type_data = NULL;
    size_t type_count = 0;
    if (gguf_find_metadata(gguf, "tokenizer.ggml.token_type", &val) &&
        val.type == GGUF_VALUE_ARRAY) {
        // Validate element type is INT32 (3) or UINT32 (4) before casting
        // (GGUF stores token_type arrays as INT32 in most llama.cpp exports)
        if (val.value.array.elem_type == 3 || val.value.array.elem_type == 4) {
            type_data = (const uint8_t *)val.value.array.data;
            type_count = val.value.array.count;
        }
    }

    // Allocate tokenizer
    tokenizer *tok = (tokenizer *)calloc(1, sizeof(tokenizer));
    if (!tok) return NULL;

    tok->vocab_cap = (int)token_count + 256;
    tok->vocab = (vocab_entry *)calloc((size_t)tok->vocab_cap, sizeof(vocab_entry));
    if (!tok->vocab) { free(tok); return NULL; }

    merge_table_init(tok);

    tok->bos_id = bos_id;
    tok->eos_id = eos_id;
    tok->pad_id = pad_id;
    tok->unk_id = unk_id;

    // ── Load vocabulary (single sequential pass) ────────────────────────
    char buf[4096];
    size_t data_off = 0;
    for (size_t i = 0; i < token_count; i++) {
        // Read string length
        uint64_t slen;
        if (data_off + 8 > 1000000000ULL) break;
        memcpy(&slen, token_data + data_off, 8);
        data_off += 8;
        if (slen >= sizeof(buf)) slen = sizeof(buf) - 1;
        if (data_off + slen > 1000000000ULL) break;
        memcpy(buf, token_data + data_off, (size_t)slen);
        buf[slen] = '\0';
        data_off += (size_t)slen;

        tok->vocab[i].id = (uint32_t)i;
        tok->vocab[i].token = strdup(buf);

        // Detect special tokens (marked in token_type as type 3)
        bool is_special = false;
        if (type_data && i < type_count) {
            const uint32_t *tt = (const uint32_t *)type_data;
            if (tt[i] == 3) is_special = true;
        }
        // Fallback heuristic
        if (!is_special && buf[0] == '<' && buf[strlen(buf)-1] == '>')
            is_special = true;

        tok->vocab[i].is_special = is_special;
    }
    tok->vocab_size = (int)token_count;

    // ── Load BPE merges (single sequential pass) ────────────────────────
    // Each merge is a string like "token1 token2"
    // The position in the array IS the merge rank (lower = higher priority)
    char pair_buf[4096];
    size_t m_off = 0;
    for (size_t i = 0; i < merge_count; i++) {
        // Check capacity
        if (tok->merge_count >= HASH_CAPACITY - 100) break;

        // Read merge string length
        uint64_t mlen;
        if (m_off + 8 > 1000000000ULL) break;
        memcpy(&mlen, merge_data + m_off, 8);
        m_off += 8;
        if (mlen >= sizeof(pair_buf)) mlen = sizeof(pair_buf) - 1;
        if (m_off + mlen > 1000000000ULL) break;
        memcpy(pair_buf, merge_data + m_off, (size_t)mlen);
        pair_buf[mlen] = '\0';
        m_off += (size_t)mlen;

        // Parse "token1 token2" — find the space separator
        char *space = strchr(pair_buf, ' ');
        if (!space) continue;

        *space = '\0';
        const char *a_str = pair_buf;
        const char *b_str = space + 1;

        // Find token IDs by looking up in vocab
        uint32_t a_id = tokenizer_token_to_id(tok, a_str);
        uint32_t b_id = tokenizer_token_to_id(tok, b_str);
        if (a_id >= (uint32_t)tok->vocab_size || b_id >= (uint32_t)tok->vocab_size)
            continue;

        merge_table_set(tok, a_id, b_id, (int)i);
    }

    return tok;
}

void tokenizer_destroy(tokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++)
        free(tok->vocab[i].token);
    free(tok->vocab);
    free(tok->merge_hash);
    memset(tok, 0, sizeof(*tok));
    free(tok);
}

// ── Vocabulary management ────────────────────────────────────────────────────

static int find_id(const tokenizer *tok, const char *token) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i].token && strcmp(tok->vocab[i].token, token) == 0)
            return i;
    }
    return -1;
}

bool tokenizer_add_token(tokenizer *tok, const char *token, uint32_t id) {
    if (!tok || !token) return false;
    if (find_id(tok, token) >= 0) return true;

    if ((int)id >= tok->vocab_size) {
        if ((int)id >= tok->vocab_cap) {
            int new_cap = tok->vocab_cap * 2;
            if ((int)id >= new_cap) new_cap = (int)id + 1024;
            vocab_entry *nv = (vocab_entry *)realloc(
                tok->vocab, (size_t)new_cap * sizeof(vocab_entry));
            if (!nv) return false;
            memset(&nv[tok->vocab_cap], 0,
                   (size_t)(new_cap - tok->vocab_cap) * sizeof(vocab_entry));
            tok->vocab = nv;
            tok->vocab_cap = new_cap;
        }
        tok->vocab_size = (int)id + 1;
    }

    free(tok->vocab[id].token);
    tok->vocab[id].id = id;
    tok->vocab[id].token = strdup(token);
    tok->vocab[id].is_special = (token[0] == '<');
    return true;
}

uint32_t tokenizer_token_to_id(tokenizer *tok, const char *token) {
    if (!tok || !token) return tok ? tok->unk_id : 0;
    int idx = find_id(tok, token);
    return idx >= 0 ? tok->vocab[idx].id : tok->unk_id;
}

const char *tokenizer_id_to_token(tokenizer *tok, uint32_t id) {
    if (!tok || (int)id >= tok->vocab_size) return NULL;
    return tok->vocab[id].token;
}

// ── Special tokens ───────────────────────────────────────────────────────────

uint32_t tokenizer_bos_id(const tokenizer *tok) { return tok ? tok->bos_id : 0; }
uint32_t tokenizer_eos_id(const tokenizer *tok) { return tok ? tok->eos_id : 0; }
uint32_t tokenizer_unk_id(const tokenizer *tok) { return tok ? tok->unk_id : 0; }
uint32_t tokenizer_pad_id(const tokenizer *tok) { return tok ? tok->pad_id : 0; }
int tokenizer_vocab_size(const tokenizer *tok) { return tok ? tok->vocab_size : 0; }

bool tokenizer_is_special(const tokenizer *tok, uint32_t id) {
    if (!tok || (int)id >= tok->vocab_size) return false;
    return tok->vocab[id].is_special;
}

// ═════════════════════════════════════════════════════════════════════════════
// BPE Encoding
// ═════════════════════════════════════════════════════════════════════════════

int tokenizer_encode(tokenizer *tok, const char *text,
                     uint32_t *tokens, int max_tokens) {
    if (!tok || !text) return 0;

    int text_len = (int)strlen(text);
    if (text_len == 0) return 0;

    // ── Byte-level encoding ────────────────────────────────────────────
    // Step 1: Convert text to bytes with GPT-2 style prepended space
    // Each byte becomes a token. A space is prepended to the text.
    // Max BPE tokens: text_len + 1 (for prepended space)

    // Allocate working buffer for byte tokens
    int max_subwords = text_len * 2 + 64;  // generous estimate
    uint32_t *subwords = (uint32_t *)malloc((size_t)max_subwords * sizeof(uint32_t));
    if (!subwords) return 0;
    int n_sub = 0;

    // Prepend a space (GPT-2 convention)
    if (n_sub < max_subwords) subwords[n_sub++] = ' ';  // token 32 = space byte

    // Convert each byte to its byte-level token
    for (int i = 0; i < text_len && n_sub < max_subwords; i++) {
        unsigned char c = (unsigned char)text[i];
        subwords[n_sub++] = (uint32_t)c;  // byte tokens are IDs 0-255
    }

    // ── BPE merge loop ─────────────────────────────────────────────────
    // Repeatedly find the pair with the lowest merge rank
    // Use a greedy approach: find the minimum-rank pair among consecutive tokens

    // For efficiency with long sequences, we track the best merge per position.
    // Simple O(N^2) approach: for each iteration, scan all pairs for min rank.
    // With N typically < 256 (prompt tokens), this is fine.

    bool merged = true;
    while (merged && n_sub > 1) {
        merged = false;
        int best_rank = INT_MAX;
        int best_pos = -1;

        // Find the pair with the lowest merge rank
        for (int i = 0; i < n_sub - 1; i++) {
            int rank = merge_table_get(tok, subwords[i], subwords[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos = i;
                if (rank == 0) break;  // can't get better than 0
            }
        }

        if (best_pos >= 0) {
            // Merge: replace pair with the merged token
            // The merged token's ID is the vocab entry for the concatenation
            // of the two original token strings.
            uint32_t a = subwords[best_pos];
            uint32_t b = subwords[best_pos + 1];

            // Build the merged string to look up in vocab
            const char *a_str = (a < (uint32_t)tok->vocab_size) ? tok->vocab[a].token : "";
            const char *b_str = (b < (uint32_t)tok->vocab_size) ? tok->vocab[b].token : "";

            // Look up the merged token ID in the vocabulary
            // The merged string is a_str + b_str
            char merged_str[256];
            snprintf(merged_str, sizeof(merged_str), "%s%s", a_str, b_str);

            uint32_t merged_id = tokenizer_token_to_id(tok, merged_str);

            // If not found, it might be a byte-type merge that isn't in vocab
            // (e.g., a through byte token that doesn't have a separate vocab entry)
            if ((int)merged_id == (int)tok->unk_id && merged_id != tok->unk_id) {
                // check again
            }
            if (merged_id >= (uint32_t)tok->vocab_size || merged_id == tok->unk_id) {
                // Try building the merged string differently
                // The merged token's string representation could be the
                // concatenation of the raw byte strings
                merged_id = tok->unk_id;
            }

            if (merged_id != tok->unk_id || a < 256 || b < 256) {
                // Shift remaining subwords left
                subwords[best_pos] = (merged_id != tok->unk_id) ? merged_id : tok->unk_id;
                for (int j = best_pos + 1; j < n_sub - 1; j++)
                    subwords[j] = subwords[j + 1];
                n_sub--;
                merged = true;
            } else {
                // Mark this pair as unmergeable and continue
                // (set to sentinel value)
                break;
            }
        }
    }

    // ── Output ─────────────────────────────────────────────────────────
    int result_count = n_sub;
    if (tokens && result_count > max_tokens)
        result_count = max_tokens;

    if (tokens) {
        for (int i = 0; i < result_count; i++)
            tokens[i] = subwords[i];
    }

    free(subwords);
    return result_count;
}

// ── Decoding ─────────────────────────────────────────────────────────────────

char *tokenizer_decode(tokenizer *tok, const uint32_t *tokens, int n_tokens) {
    if (!tok || !tokens || n_tokens <= 0) return NULL;

    size_t buf_size = (size_t)n_tokens * 32 + 1;
    char *result = (char *)calloc(buf_size, 1);
    if (!result) return NULL;

    size_t pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        const char *token_str = tokenizer_id_to_token(tok, tokens[i]);
        if (!token_str) continue;

        // Skip special tokens
        if (tok->vocab[tokens[i]].is_special) continue;

        size_t len = strlen(token_str);
        if (pos + len < buf_size) {
            memcpy(result + pos, token_str, len);
            pos += len;
        }
    }
    result[pos] = '\0';

    return result;
}
