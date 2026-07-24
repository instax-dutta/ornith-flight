// SPDX-License-Identifier: MIT
// BPE tokenizer — in-memory vocabulary, encode/decode, special tokens.

#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_VOCAB 1024

typedef struct {
    uint32_t id;
    char    *token;
    bool     is_special;
} vocab_entry;

struct tokenizer {
    vocab_entry *vocab;
    int          vocab_size;
    int          vocab_cap;
    uint32_t     bos_id;
    uint32_t     eos_id;
    uint32_t     unk_id;
    uint32_t     pad_id;
};

// ── Lifecycle ────────────────────────────────────────────────────────────────

tokenizer *tokenizer_create(void) {
    tokenizer *tok = (tokenizer *)calloc(1, sizeof(tokenizer));
    if (!tok) return NULL;

    tok->vocab_cap = INITIAL_VOCAB;
    tok->vocab = (vocab_entry *)calloc((size_t)tok->vocab_cap, sizeof(vocab_entry));
    if (!tok->vocab) { free(tok); return NULL; }

    tok->vocab_size = 0;
    tok->bos_id = 1;
    tok->eos_id = 2;
    tok->unk_id = 0;
    tok->pad_id = 3;

    return tok;
}

tokenizer *tokenizer_create_from_json(const char *json_path) {
    (void)json_path;
    // For now, create empty tokenizer (real impl would parse tokenizer.json)
    return tokenizer_create();
}

void tokenizer_destroy(tokenizer *tok) {
    if (!tok) return;
    for (int i = 0; i < tok->vocab_size; i++) {
        free(tok->vocab[i].token);
    }
    free(tok->vocab);
    memset(tok, 0, sizeof(*tok));
    free(tok);
}

// ── Vocabulary management ────────────────────────────────────────────────────

static int find_id(const tokenizer *tok, const char *token) {
    for (int i = 0; i < tok->vocab_size; i++) {
        if (tok->vocab[i].token && strcmp(tok->vocab[i].token, token) == 0) {
            return i;
        }
    }
    return -1;
}

bool tokenizer_add_token(tokenizer *tok, const char *token, uint32_t id) {
    if (!tok || !token) return false;

    // Check if already exists
    if (find_id(tok, token) >= 0) return true;

    // Find slot by ID
    if ((int)id >= tok->vocab_size) {
        // Expand if needed
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

    // If slot was occupied, free old token
    free(tok->vocab[id].token);

    tok->vocab[id].id = id;
    tok->vocab[id].token = strdup(token);
    tok->vocab[id].is_special = (token[0] == '<');  // heuristic: <...> tokens are special

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

// ── Encoding ─────────────────────────────────────────────────────────────────

int tokenizer_encode(tokenizer *tok, const char *text,
                     uint32_t *tokens, int max_tokens) {
    if (!tok || !text) return 0;

    // Simple word-boundary encoding: split by space, look up each word
    // In production: full BPE merge logic

    int count = 0;
    const char *p = text;

    while (*p) {
        // Skip whitespace
        while (*p == ' ' || *p == '\t' || *p == '\n') p++;
        if (!*p) break;

        // Find word boundary
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;

        // Extract word
        int len = (int)(p - start);
        char *word = (char *)malloc((size_t)len + 1);
        memcpy(word, start, (size_t)len);
        word[len] = '\0';

        // Look up in vocab
        uint32_t id = tokenizer_token_to_id(tok, word);

        // If tokens buffer is provided and within max, store
        if (tokens && count < max_tokens) {
            tokens[count] = id;
        }
        count++;

        free(word);
    }

    return count;
}

// ── Decoding ─────────────────────────────────────────────────────────────────

char *tokenizer_decode(tokenizer *tok, const uint32_t *tokens, int n_tokens) {
    if (!tok || !tokens || n_tokens <= 0) return NULL;

    // Estimate: average token ~8 bytes
    size_t buf_size = (size_t)n_tokens * 16 + 1;
    char *result = (char *)calloc(buf_size, 1);
    if (!result) return NULL;

    size_t pos = 0;
    for (int i = 0; i < n_tokens; i++) {
        const char *token_str = tokenizer_id_to_token(tok, tokens[i]);
        if (!token_str) continue;

        // Skip special tokens in output
        if (tok->vocab[tokens[i]].is_special) continue;

        size_t len = strlen(token_str);
        if (pos + len + 1 < buf_size) {
            if (pos > 0) {
                result[pos++] = ' ';  // space between tokens
            }
            memcpy(result + pos, token_str, len);
            pos += len;
        }
    }
    result[pos] = '\0';

    return result;
}
