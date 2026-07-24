// SPDX-License-Identifier: MIT
// Minimalist C test framework — explicit test list, no magic.

#ifndef ORNITH_TEST_H
#define ORNITH_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

typedef enum { TEST_PASS, TEST_FAIL, TEST_ERROR } test_result;
typedef test_result (*test_func)(void);

#define test_assert(cond, msg) do {                                         \
    if (!(cond)) {                                                          \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);   \
        return TEST_FAIL;                                                   \
    }                                                                       \
} while (0)

#define test_eq_uint64(a, b, msg) do {                                      \
    unsigned long long _a = (unsigned long long)(a);                        \
    unsigned long long _b = (unsigned long long)(b);                        \
    if (_a != _b) {                                                         \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);   \
        fprintf(stderr, "    Expected: %llu  Got: %llu\n", _a, _b);        \
        return TEST_FAIL;                                                   \
    }                                                                       \
} while (0)

#define test_not_null(ptr, msg) do {                                        \
    if ((ptr) == NULL) {                                                    \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);   \
        return TEST_FAIL;                                                   \
    }                                                                       \
} while (0)

#define test_null(ptr, msg) do {                                            \
    if ((ptr) != NULL) {                                                    \
        fprintf(stderr, "  FAIL: %s:%d: %s\n", __FILE__, __LINE__, msg);   \
        return TEST_FAIL;                                                   \
    }                                                                       \
} while (0)

// Simple test list (each test file defines its own array)
typedef struct {
    const char *name;
    test_func   func;
} test_entry;

#define RUN_TESTS(tests)                                                    \
    int main(void) {                                                        \
        int n = sizeof(tests) / sizeof(tests[0]);                           \
        int passed = 0, failed = 0;                                         \
        printf("\\n=== Running %d tests ===\\n\\n", n);                    \
        for (int i = 0; i < n; i++) {                                       \
            printf("  [%d/%d] %s ... ", i+1, n, tests[i].name);            \
            fflush(stdout);                                                 \
            test_result r = tests[i].func();                                \
            if (r == TEST_PASS) { printf("PASS\\n"); passed++; }            \
            else if (r == TEST_FAIL) { printf("FAIL\\n"); failed++; }       \
            else { printf("ERROR\\n"); failed++; }                          \
        }                                                                   \
        printf("\\n=== %d/%d passed, %d failed ===\\n",                    \
               passed, n, failed);                                          \
        return failed > 0 ? 1 : 0;                                          \
    }

#endif
