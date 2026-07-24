// RED: GPU abstraction tests — buffer alloc/free, copy roundtrip, matmul.
// Uses a CPU fallback backend so tests work on any platform.

#include "test.h"
#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Test 1: Context lifecycle ────────────────────────────────────────────────

static test_result test_context_create_destroy(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init should succeed");

    gpu_print_info(ctx);

    size_t mem = gpu_memory_used(ctx);
    test_assert(mem == 0, "initial memory usage is 0");

    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test 2: Buffer allocation and free ───────────────────────────────────────

static test_result test_buffer_alloc_free(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    // Allocate a small buffer
    gpu_buffer *buf = gpu_alloc(ctx, 1024, GPU_BUF_DEFAULT);
    test_not_null(buf, "alloc 1024 bytes");
    test_assert(gpu_memory_used(ctx) == 1024, "memory usage = 1024 after alloc");

    // Free and verify memory decreases
    gpu_free(ctx, buf);
    test_assert(gpu_memory_used(ctx) == 0, "memory usage = 0 after free");

    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test 3: Shared buffer CPU pointer ────────────────────────────────────────

static test_result test_buffer_shared_cpu_ptr(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    gpu_buffer *buf = gpu_alloc(ctx, 64, GPU_BUF_SHARED);
    test_not_null(buf, "alloc shared buffer");

    // Write to CPU pointer
    void *cpu_ptr = gpu_get_cpu_ptr(ctx, buf);
    test_not_null(cpu_ptr, "CPU pointer is accessible");
    memset(cpu_ptr, 0xAB, 64);

    // Read back and verify
    unsigned char *data = (unsigned char *)cpu_ptr;
    test_assert(data[0] == 0xAB, "first byte = 0xAB");
    test_assert(data[63] == 0xAB, "last byte = 0xAB");

    gpu_free(ctx, buf);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test 4: Copy to device and to host ───────────────────────────────────────

static test_result test_copy_roundtrip(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    // Create host data
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Copy to device
    gpu_buffer *buf = gpu_alloc(ctx, sizeof(src), GPU_BUF_SHARED);
    test_not_null(buf, "alloc buffer");

    gpu_copy_to_device(ctx, buf, src, sizeof(src));

    // Copy back to host
    float dst[4] = {0};
    gpu_copy_to_host(ctx, dst, buf, sizeof(dst));

    // Verify roundtrip
    test_assert(dst[0] == 1.0f, "dst[0] = 1.0");
    test_assert(dst[1] == 2.0f, "dst[1] = 2.0");
    test_assert(dst[2] == 3.0f, "dst[2] = 3.0");
    test_assert(dst[3] == 4.0f, "dst[3] = 4.0");

    gpu_free(ctx, buf);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test 5: Copy between GPU buffers ─────────────────────────────────────────

static test_result test_buffer_copy(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    float data_a[4] = {10, 20, 30, 40};
    gpu_buffer *a = gpu_alloc(ctx, sizeof(data_a), GPU_BUF_SHARED);
    gpu_buffer *b = gpu_alloc(ctx, sizeof(data_a), GPU_BUF_SHARED);
    test_not_null(a, "alloc A");
    test_not_null(b, "alloc B");

    gpu_copy_to_device(ctx, a, data_a, sizeof(data_a));
    gpu_copy_buffer(ctx, b, a, sizeof(data_a));

    float result[4] = {0};
    gpu_copy_to_host(ctx, result, b, sizeof(result));

    test_assert(result[0] == 10, "result[0] = 10");
    test_assert(result[3] == 40, "result[3] = 40");

    gpu_free(ctx, a);
    gpu_free(ctx, b);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test 6: Matmul API call ──────────────────────────────────────────────────

static test_result test_matmul_dispatch(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    // A = 2x3, B = 3x4, C = 2x4
    float A_data[6] = {1, 2, 3, 4, 5, 6};
    float B_data[12] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    gpu_buffer *A = gpu_alloc(ctx, sizeof(A_data), GPU_BUF_SHARED);
    gpu_buffer *B = gpu_alloc(ctx, sizeof(B_data), GPU_BUF_SHARED);
    gpu_buffer *C = gpu_alloc(ctx, 2 * 4 * sizeof(float), GPU_BUF_SHARED);
    test_not_null(A, "alloc A");
    test_not_null(B, "alloc B");
    test_not_null(C, "alloc C");

    gpu_copy_to_device(ctx, A, A_data, sizeof(A_data));
    gpu_copy_to_device(ctx, B, B_data, sizeof(B_data));

    // Dispatch matmul: C = A * B (2x3 * 3x4 = 2x4)
    gpu_matmul(ctx, C, A, B, 2, 4, 3, QUANT_NONE, QUANT_NONE);
    gpu_sync(ctx);

    // Read back result
    float result[8] = {0};
    gpu_copy_to_host(ctx, result, C, sizeof(result));

    // Manual check: C[0][0] = 1*7 + 2*11 + 3*15 = 7 + 22 + 45 = 74
    test_assert(result[0] == 74.0f, "C[0][0] = 74");
    // C[1][3] = 4*10 + 5*14 + 6*18 = 40 + 70 + 108 = 218
    test_assert(result[7] == 218.0f, "C[1][3] = 218");

    gpu_free(ctx, A);
    gpu_free(ctx, B);
    gpu_free(ctx, C);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test 7: Matmul int4 dispatch ─────────────────────────────────────────────

static test_result test_matmul_int4_dispatch(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    // A_quant is int4 (simulated), A_scale is float, A is 2x3
    // For CPU fallback, we just verify the API doesn't crash
    float fake_quant[12] = {0};  // 3*4 bytes for 24 int4 values
    float fake_scale[2] = {1.0f, 1.0f};  // 2 blocks

    gpu_buffer *Aq = gpu_alloc(ctx, sizeof(fake_quant), GPU_BUF_SHARED);
    gpu_buffer *As = gpu_alloc(ctx, sizeof(fake_scale), GPU_BUF_SHARED);
    gpu_buffer *C  = gpu_alloc(ctx, 2 * 4 * sizeof(float), GPU_BUF_SHARED);
    test_not_null(Aq, "alloc A_quant");
    test_not_null(As, "alloc A_scale");
    test_not_null(C, "alloc C");

    gpu_copy_to_device(ctx, Aq, fake_quant, sizeof(fake_quant));
    gpu_copy_to_device(ctx, As, fake_scale, sizeof(fake_scale));

    // Dispatch int4 matmul (just verify it doesn't crash)
    gpu_matmul_int4(ctx, C, Aq, As, 2, 4, 3);
    gpu_sync(ctx);

    gpu_free(ctx, Aq);
    gpu_free(ctx, As);
    gpu_free(ctx, C);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    { "context_create_destroy",  test_context_create_destroy },
    { "buffer_alloc_free",       test_buffer_alloc_free },
    { "buffer_shared_cpu_ptr",   test_buffer_shared_cpu_ptr },
    { "copy_roundtrip",          test_copy_roundtrip },
    { "buffer_copy",             test_buffer_copy },
    { "matmul_dispatch",         test_matmul_dispatch },
    { "matmul_int4_dispatch",    test_matmul_int4_dispatch },
};

RUN_TESTS(tests)
