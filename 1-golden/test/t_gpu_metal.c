// SPDX-License-Identifier: MIT
// Metal GPU shader tests — exercises matmul_vec_kernel and rmsnorm_kernel
// (via gpu_metal.m) with known float values and compares against expected
// CPU-computed results.
//
// Build & run with:  make test-metal
#include "test.h"
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ═════════════════════════════════════════════════════════════════════════════
// Helper: compare two float buffers with tolerance
// ═════════════════════════════════════════════════════════════════════════════

static bool approx_eq(const float *a, const float *b, int n, float tol) {
    for (int i = 0; i < n; i++) {
        if (fabsf(a[i] - b[i]) > tol) {
            fprintf(stderr, "    MISMATCH at [%d]: expected %.8f  got %.8f  (diff %.2e)\n",
                    i, b[i], a[i], fabsf(a[i] - b[i]));
            return false;
        }
    }
    return true;
}

#define TOLERANCE 1e-5f

// ═════════════════════════════════════════════════════════════════════════════
// Test 1: vec = [1,2,3,4], mat = identity 4×4  →  out = [1,2,3,4]
// matmul_vec_kernel: out[M] = vec[K] @ mat[K][M]
// K=4, M=4, vec=[1,2,3,4], mat=I = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_matmul_identity(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    int K = 4, M = 4;
    float vec[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    // Identity matrix in row-major: mat[K][M]
    float mat[16] = {1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1};
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    gpu_buffer *gpu_vec = gpu_alloc(ctx, K * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_mat = gpu_alloc(ctx, K * M * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_out = gpu_alloc(ctx, M * sizeof(float), GPU_BUF_SHARED);
    test_not_null(gpu_vec, "alloc vec");
    test_not_null(gpu_mat, "alloc mat");
    test_not_null(gpu_out, "alloc out");

    gpu_copy_to_device(ctx, gpu_vec, vec, K * sizeof(float));
    gpu_copy_to_device(ctx, gpu_mat, mat, K * M * sizeof(float));

    // gpu_matmul with M=1 batch, N=M=4 output dims, K=4 input dims
    gpu_matmul(ctx, gpu_out, gpu_vec, gpu_mat, 1, M, K, QUANT_NONE, QUANT_NONE);
    gpu_sync(ctx);

    float result[4];
    gpu_copy_to_host(ctx, result, gpu_out, M * sizeof(float));

    test_assert(approx_eq(result, expected, M, TOLERANCE),
                "matmul_identity: out = vec");

    gpu_free(ctx, gpu_vec);
    gpu_free(ctx, gpu_mat);
    gpu_free(ctx, gpu_out);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 2: vec=[1,2,3], mat=3×2 with known values
// K=3, M=2
// mat = [4,5,  6,7,  8,9]  (row-major: mat[0]=[4,5], mat[1]=[6,7], mat[2]=[8,9])
// out[0] = 1*4 + 2*6 + 3*8 =  4+12+24 = 40
// out[1] = 1*5 + 2*7 + 3*9 =  5+14+27 = 46
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_matmul_small(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    int K = 3, M = 2;
    float vec[3] = {1.0f, 2.0f, 3.0f};
    float mat[6] = {4,5,  6,7,  8,9};
    float expected[2] = {40.0f, 46.0f};

    gpu_buffer *gpu_vec = gpu_alloc(ctx, K * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_mat = gpu_alloc(ctx, K * M * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_out = gpu_alloc(ctx, M * sizeof(float), GPU_BUF_SHARED);
    test_not_null(gpu_vec, "alloc vec");
    test_not_null(gpu_mat, "alloc mat");
    test_not_null(gpu_out, "alloc out");

    gpu_copy_to_device(ctx, gpu_vec, vec, K * sizeof(float));
    gpu_copy_to_device(ctx, gpu_mat, mat, K * M * sizeof(float));
    gpu_matmul(ctx, gpu_out, gpu_vec, gpu_mat, 1, M, K, QUANT_NONE, QUANT_NONE);
    gpu_sync(ctx);

    float result[2];
    gpu_copy_to_host(ctx, result, gpu_out, M * sizeof(float));

    test_assert(approx_eq(result, expected, M, TOLERANCE),
                "matmul_small: 3×2 multiply correct");

    gpu_free(ctx, gpu_vec);
    gpu_free(ctx, gpu_mat);
    gpu_free(ctx, gpu_out);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 3: vec full-range, mat large — verifies all M threads work correctly.
// K=8, M=16  (model-like: K=hidden_dim, M=d_model)
// vec[i] = (float)i * 0.5
// mat[i*M + j] = (float)(i * M + j) * 0.01
// Expected computed on CPU inside the test.
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_matmul_model_like(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    int K = 8, M = 16;
    int n_floats = K * M;
    float *vec = (float *)malloc(K * sizeof(float));
    float *mat = (float *)malloc(n_floats * sizeof(float));
    float *expected = (float *)malloc(M * sizeof(float));
    test_not_null(vec, "malloc vec");
    test_not_null(mat, "malloc mat");
    test_not_null(expected, "malloc expected");

    // Fill with predictable data
    for (int i = 0; i < K; i++) vec[i] = (float)i * 0.5f;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < M; j++)
            mat[i * M + j] = (float)(i * M + j) * 0.01f;

    // Compute expected on CPU: out[j] = sum_i vec[i] * mat[i][j]
    for (int j = 0; j < M; j++) {
        float sum = 0.0f;
        for (int i = 0; i < K; i++)
            sum += vec[i] * mat[i * M + j];
        expected[j] = sum;
    }

    gpu_buffer *gpu_vec = gpu_alloc(ctx, K * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_mat = gpu_alloc(ctx, n_floats * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_out = gpu_alloc(ctx, M * sizeof(float), GPU_BUF_SHARED);
    test_not_null(gpu_vec, "alloc vec");
    test_not_null(gpu_mat, "alloc mat");
    test_not_null(gpu_out, "alloc out");

    gpu_copy_to_device(ctx, gpu_vec, vec, K * sizeof(float));
    gpu_copy_to_device(ctx, gpu_mat, mat, n_floats * sizeof(float));

    // Dispatch: batch=1, N=M=16 output dims, K=8 input dims
    gpu_matmul(ctx, gpu_out, gpu_vec, gpu_mat, 1, M, K, QUANT_NONE, QUANT_NONE);
    gpu_sync(ctx);

    float *result = (float *)malloc(M * sizeof(float));
    test_not_null(result, "malloc result");
    gpu_copy_to_host(ctx, result, gpu_out, M * sizeof(float));

    bool match = approx_eq(result, expected, M, TOLERANCE);
    test_assert(match, "matmul_model_like: all 16 outputs match CPU");

    free(vec); free(mat); free(expected); free(result);
    gpu_free(ctx, gpu_vec);
    gpu_free(ctx, gpu_mat);
    gpu_free(ctx, gpu_out);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 4: RMSNorm on a tiny vector (dim=4) with uniform weight.
// x = [1, 2, 3, 4], weight = [0.5, 0.5, 0.5, 0.5]
// sum_sq = 1+4+9+16 = 30
// inv_rms = 1/sqrt(30/4 + 1e-6) = 1/sqrt(7.5) ≈ 0.365148
// out[i] = weight[i] * x[i] * inv_rms
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_rmsnorm_tiny(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    int dim = 4;
    float x[4]      = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[4] = {0.5f, 0.5f, 0.5f, 0.5f};

    // Compute expected on CPU
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
    float expected[4];
    for (int i = 0; i < dim; i++) expected[i] = weight[i] * x[i] * inv_rms;

    gpu_buffer *gpu_x  = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_w  = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_out = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    test_not_null(gpu_x, "alloc x");
    test_not_null(gpu_w, "alloc weight");
    test_not_null(gpu_out, "alloc out");

    gpu_copy_to_device(ctx, gpu_x, x, dim * sizeof(float));
    gpu_copy_to_device(ctx, gpu_w, weight, dim * sizeof(float));

    gpu_rmsnorm(ctx, gpu_out, gpu_x, gpu_w, 1, dim);
    gpu_sync(ctx);

    float result[4];
    gpu_copy_to_host(ctx, result, gpu_out, dim * sizeof(float));

    test_assert(approx_eq(result, expected, dim, TOLERANCE),
                "rmsnorm_tiny: 4-element norm matches CPU");

    gpu_free(ctx, gpu_x);
    gpu_free(ctx, gpu_w);
    gpu_free(ctx, gpu_out);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 5: RMSNorm on a model-like vector (dim=64) with varying weight.
// Verifies the threadgroup reduction across multiple threadgroups.
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_rmsnorm_model_like(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    int dim = 64;
    float *x      = (float *)malloc(dim * sizeof(float));
    float *weight = (float *)malloc(dim * sizeof(float));
    float *expected = (float *)malloc(dim * sizeof(float));
    test_not_null(x, "malloc x");
    test_not_null(weight, "malloc weight");
    test_not_null(expected, "malloc expected");

    // Fill with realistic data
    for (int i = 0; i < dim; i++) {
        x[i] = sinf((float)i * 0.3f);
        weight[i] = 0.5f + 0.5f * cosf((float)i * 0.2f);
    }

    // CPU expected
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
    for (int i = 0; i < dim; i++) expected[i] = weight[i] * x[i] * inv_rms;

    gpu_buffer *gpu_x  = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_w  = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_out = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    test_not_null(gpu_x, "alloc x");
    test_not_null(gpu_w, "alloc weight");
    test_not_null(gpu_out, "alloc out");

    gpu_copy_to_device(ctx, gpu_x, x, dim * sizeof(float));
    gpu_copy_to_device(ctx, gpu_w, weight, dim * sizeof(float));

    gpu_rmsnorm(ctx, gpu_out, gpu_x, gpu_w, 1, dim);
    gpu_sync(ctx);

    float *result = (float *)malloc(dim * sizeof(float));
    test_not_null(result, "malloc result");
    gpu_copy_to_host(ctx, result, gpu_out, dim * sizeof(float));

    bool match = approx_eq(result, expected, dim, TOLERANCE);
    test_assert(match, "rmsnorm_model_like: 64-element norm matches CPU");

    free(x); free(weight); free(expected); free(result);
    gpu_free(ctx, gpu_x);
    gpu_free(ctx, gpu_w);
    gpu_free(ctx, gpu_out);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test 6: RMSNorm on dim=2048 (the actual Ornith d_model).
// Stress test: verifies the multi-group threadgroup reduction works at scale.
// ═════════════════════════════════════════════════════════════════════════════

static test_result test_rmsnorm_d2048(void) {
    gpu_context *ctx = gpu_init();
    test_not_null(ctx, "gpu_init");

    int dim = 2048;
    float *x      = (float *)malloc(dim * sizeof(float));
    float *weight = (float *)malloc(dim * sizeof(float));
    float *expected = (float *)malloc(dim * sizeof(float));
    test_not_null(x, "malloc x");
    test_not_null(weight, "malloc weight");
    test_not_null(expected, "malloc expected");

    // Fill with realistic activation-like data
    for (int i = 0; i < dim; i++) {
        x[i] = ((float)i / (float)dim - 0.5f) * 2.0f;
        weight[i] = 0.9f + 0.1f * sinf((float)i * 0.01f);
    }

    // CPU expected
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);
    for (int i = 0; i < dim; i++) expected[i] = weight[i] * x[i] * inv_rms;

    gpu_buffer *gpu_x  = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_w  = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    gpu_buffer *gpu_out = gpu_alloc(ctx, dim * sizeof(float), GPU_BUF_SHARED);
    test_not_null(gpu_x, "alloc x (2048)");
    test_not_null(gpu_w, "alloc weight (2048)");
    test_not_null(gpu_out, "alloc out (2048)");

    gpu_copy_to_device(ctx, gpu_x, x, dim * sizeof(float));
    gpu_copy_to_device(ctx, gpu_w, weight, dim * sizeof(float));

    gpu_rmsnorm(ctx, gpu_out, gpu_x, gpu_w, 1, dim);
    gpu_sync(ctx);

    float *result = (float *)malloc(dim * sizeof(float));
    test_not_null(result, "malloc result");
    gpu_copy_to_host(ctx, result, gpu_out, dim * sizeof(float));

    // Use looser tolerance for large reduction
    bool match = approx_eq(result, expected, dim, 1e-4f);
    test_assert(match, "rmsnorm_d2048: full-scale 2048-elem norm matches CPU");

    free(x); free(weight); free(expected); free(result);
    gpu_free(ctx, gpu_x);
    gpu_free(ctx, gpu_w);
    gpu_free(ctx, gpu_out);
    gpu_destroy(ctx);
    return TEST_PASS;
}

// ═════════════════════════════════════════════════════════════════════════════
// Test list
// ═════════════════════════════════════════════════════════════════════════════

static test_entry tests[] = {
    { "matmul_identity",       test_matmul_identity },
    { "matmul_small",          test_matmul_small },
    { "matmul_model_like",     test_matmul_model_like },
    { "rmsnorm_tiny",          test_rmsnorm_tiny },
    { "rmsnorm_model_like",    test_rmsnorm_model_like },
    { "rmsnorm_d2048",         test_rmsnorm_d2048 },
};

RUN_TESTS(tests)
