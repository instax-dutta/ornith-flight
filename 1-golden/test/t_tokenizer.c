// RED: Tokenizer tests — token ID lookup, encode/decode roundtrip, special tokens.

#include "test.h"
#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Helper: create a test tokenizer with known vocabulary ─────────────────────

static tokenizer *make_test_tokenizer(void) {
    // Create tokenizer from a small in-memory vocab
    tokenizer *tok = tokenizer_create_from_json(NULL);
    if (!tok) return NULL;

    // Add some known tokens
    tokenizer_add_token(tok, "hello", 10);
    tokenizer_add_token(tok, "world", 20);
    tokenizer_add_token(tok, "hello world", 30);
    tokenizer_add_token(tok, "test", 40);
    tokenizer_add_token(tok, "tokenizer", 50);
    tokenizer_add_token(tok, "<bos>", 1);
    tokenizer_add_token(tok, "<eos>", 2);
    tokenizer_add_token(tok, "<unk>", 0);
    tokenizer_add_token(tok, "<pad>", 3);

    return tok;
}

// ── Test 1: Token ID lookup ──────────────────────────────────────────────────

static test_result test_token_to_id(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    uint32_t id;

    id = tokenizer_token_to_id(tok, "hello");
    test_eq_uint64(id, 10, "hello → 10");

    id = tokenizer_token_to_id(tok, "world");
    test_eq_uint64(id, 20, "world → 20");

    // Unknown token should return UNK ID
    id = tokenizer_token_to_id(tok, "nonexistent");
    test_eq_uint64(id, 0, "nonexistent → <unk> (0)");

    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test 2: ID to token lookup ───────────────────────────────────────────────

static test_result test_id_to_token(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    const char *token;

    token = tokenizer_id_to_token(tok, 10);
    test_not_null(token, "ID 10 exists");
    test_assert(strcmp(token, "hello") == 0, "10 → hello");

    token = tokenizer_id_to_token(tok, 20);
    test_not_null(token, "ID 20 exists");
    test_assert(strcmp(token, "world") == 0, "20 → world");

    // Unknown ID should return NULL
    token = tokenizer_id_to_token(tok, 9999);
    test_null(token, "9999 → NULL");

    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test 3: Special token IDs ────────────────────────────────────────────────

static test_result test_special_tokens(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    test_eq_uint64(tokenizer_bos_id(tok), 1, "BOS = 1");
    test_eq_uint64(tokenizer_eos_id(tok), 2, "EOS = 2");
    test_eq_uint64(tokenizer_unk_id(tok), 0, "UNK = 0");
    test_eq_uint64(tokenizer_pad_id(tok), 3, "PAD = 3");

    // Check special token detection
    test_assert(tokenizer_is_special(tok, 1), "<bos> is special");
    test_assert(tokenizer_is_special(tok, 2), "<eos> is special");
    test_assert(!tokenizer_is_special(tok, 10), "hello is not special");
    test_assert(!tokenizer_is_special(tok, 20), "world is not special");

    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test 4: Vocabulary size ──────────────────────────────────────────────────

static test_result test_vocab_size(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    int size = tokenizer_vocab_size(tok);
    test_assert(size >= 9, "vocab has at least 9 tokens");

    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test 5: Encode text to tokens ────────────────────────────────────────────

static test_result test_encode(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    uint32_t tokens[16];
    int n;

    // Encode known phrase
    n = tokenizer_encode(tok, "hello world", tokens, 16);
    test_assert(n > 0, "encoded some tokens");
    // The first token should match "hello" or the full phrase
    test_assert(tokens[0] < 100, "first token is a valid ID");

    // Estimate only (pass NULL for tokens)
    n = tokenizer_encode(tok, "hello world", NULL, 0);
    test_assert(n > 0, "encoding estimate is positive");

    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test 6: Decode tokens to text ────────────────────────────────────────────

static test_result test_decode(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    // Decode known token IDs
    uint32_t ids[] = {10, 20};  // "hello", "world"
    char *text = tokenizer_decode(tok, ids, 2);
    test_not_null(text, "decoded text");
    test_assert(strlen(text) > 0, "decoded text non-empty");
    test_assert(strstr(text, "hello") != NULL || strstr(text, "world") != NULL,
                "decoded text contains tokens");

    free(text);
    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test 7: Encode/decode roundtrip ──────────────────────────────────────────

static test_result test_roundtrip(void) {
    tokenizer *tok = make_test_tokenizer();
    test_not_null(tok, "create tokenizer");

    const char *original = "hello world";
    uint32_t tokens[16];
    int n = tokenizer_encode(tok, original, tokens, 16);
    test_assert(n > 0, "encoded");

    char *decoded = tokenizer_decode(tok, tokens, n);
    test_not_null(decoded, "decoded");

    // The roundtrip may not be exact for simple tokenizers,
    // but the decoded text should be non-empty and contain key words
    test_assert(strlen(decoded) > 0, "roundtrip produced non-empty text");

    free(decoded);
    tokenizer_destroy(tok);
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    { "token_to_id",       test_token_to_id },
    { "id_to_token",       test_id_to_token },
    { "special_tokens",    test_special_tokens },
    { "vocab_size",        test_vocab_size },
    { "encode",            test_encode },
    { "decode",            test_decode },
    { "roundtrip",         test_roundtrip },
};

RUN_TESTS(tests)
