// SPDX-License-Identifier: MIT
//
// Test runner — executes all registered tests, prints results.

#include "test.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

// Global test suite
test_suite __test_suite = { .count = 0, .passed = 0, .failed = 0, .errors = 0 };

void test_run_all(void) {
    if (__test_suite.count == 0) {
        printf("No tests registered.\n");
        return;
    }

    printf("\n=== Running %d tests ===\n\n", __test_suite.count);

    clock_t start = clock();

    for (int i = 0; i < __test_suite.count; i++) {
        test_case *t = &__test_suite.tests[i];
        printf("  [%d/%d] %s ... ", i + 1, __test_suite.count, t->name);
        fflush(stdout);

        test_result result = t->func();

        switch (result) {
        case TEST_PASS:
            printf("PASS\n");
            __test_suite.passed++;
            break;
        case TEST_FAIL:
            printf("FAIL\n");
            __test_suite.failed++;
            break;
        case TEST_ERROR:
            printf("ERROR\n");
            __test_suite.errors++;
            break;
        }
    }

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    printf("\n=== Results ===\n");
    printf("  Total:  %d\n", __test_suite.count);
    printf("  Passed: %d\n", __test_suite.passed);
    printf("  Failed: %d\n", __test_suite.failed);
    printf("  Errors: %d\n", __test_suite.errors);
    printf("  Time:   %.3fs\n", elapsed);
}

int test_run_single(const char *name) {
    for (int i = 0; i < __test_suite.count; i++) {
        if (strcmp(__test_suite.tests[i].name, name) == 0) {
            printf("  %s ... ", name);
            fflush(stdout);
            test_result result = __test_suite.tests[i].func();
            printf("%s\n", result == TEST_PASS ? "PASS" :
                          result == TEST_FAIL ? "FAIL" : "ERROR");
            return result == TEST_PASS ? 0 : 1;
        }
    }
    printf("  Test '%s' not found.\n", name);
    return 1;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        int failed = 0;
        for (int i = 1; i < argc; i++) {
            if (test_run_single(argv[i]) != 0) failed = 1;
        }
        return failed;
    }
    test_run_all();
    return __test_suite.failed > 0 || __test_suite.errors > 0 ? 1 : 0;
}
