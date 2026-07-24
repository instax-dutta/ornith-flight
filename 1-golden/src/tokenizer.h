// SPDX-License-Identifier: MIT
// BPE tokenizer — token lookup, encode/decode, special tokens.

#ifndef ORNITH_TOKENIZER_H
#define ORNITH_TOKENIZER_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct tokenizer tokenizer;

tokenizer *tokenizer_create(void);
tokenizer *tokenizer_create_from_json(const char *json_path);
void tokenizer_destroy(tokenizer *tok);

int tokenizer_encode(tokenizer *tok, const char *text,
                     uint32_t *tokens, int max_tokens);
char *tokenizer_decode(tokenizer *tok, const uint32_t *tokens, int n_tokens);

uint32_t tokenizer_token_to_id(tokenizer *tok, const char *token);
const char *tokenizer_id_to_token(tokenizer *tok, uint32_t id);

uint32_t tokenizer_bos_id(const tokenizer *tok);
uint32_t tokenizer_eos_id(const tokenizer *tok);
uint32_t tokenizer_unk_id(const tokenizer *tok);
uint32_t tokenizer_pad_id(const tokenizer *tok);

int tokenizer_vocab_size(const tokenizer *tok);
bool tokenizer_is_special(const tokenizer *tok, uint32_t id);

bool tokenizer_add_token(tokenizer *tok, const char *token, uint32_t id);

#endif
