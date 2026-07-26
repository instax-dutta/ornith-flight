// SPDX-License-Identifier: MIT
// Metal GPU backend — drop-in replacement for gpu.c on Apple Silicon.
// Uses runtime-compiled Metal shaders from src/metal/ops.metal.
// Compile with: make metal

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ── Forward declarations ────────────────────────────────────────────────────

static id<MTLComputePipelineState> compile_kernel(id<MTLDevice> dev,
                                                   id<MTLLibrary> lib,
                                                   NSString *name, NSError **err);

// ── Struct definitions ──────────────────────────────────────────────────────

struct gpu_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;

    // Compute pipeline states
    id<MTLComputePipelineState> matmul_vec_ps;
    id<MTLComputePipelineState> rmsnorm_partial_ps;
    id<MTLComputePipelineState> rmsnorm_apply_ps;
    id<MTLComputePipelineState> silu_ps;
    id<MTLComputePipelineState> silu_mul_ps;
    id<MTLComputePipelineState> rope_ps;
    id<MTLComputePipelineState> add_ps;
    id<MTLComputePipelineState> softmax_ps;

    // Reusable intermediate buffers (partial sums for RMSNorm)
    id<MTLBuffer> partial_sums_buf;
    int partial_sums_len;

    size_t used_memory;
    bool   initialized;
};

struct gpu_buffer {
    id<MTLBuffer> buffer;
    size_t         size;
    gpu_buffer_flags flags;
};

// ── Metal shader source (loaded from ops.metal) ─────────────────────────────
// We embed the shader source as a string for runtime compilation.
// This avoids requiring Xcode's metal toolchain for .metallib generation.

static NSString *load_metal_shader_source(void) {
    // First try to load from file (for development)
    FILE *fp = fopen("src/metal/ops.metal", "r");
    if (!fp) {
        // Fall back to compiled-in string
        return @R"(
#include <metal_stdlib>
using namespace metal;

kernel void matmul_vec_kernel(device const float *vec [[buffer(0)]], device const float *mat [[buffer(1)]], device float *out [[buffer(2)]], constant int &K [[buffer(3)]], constant int &M [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)M) return;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) sum += vec[i] * mat[i * M + gid];
    out[gid] = sum;
}

kernel void rmsnorm_partial_kernel(device const float *x [[buffer(0)]], device float *psums [[buffer(1)]], constant int &dim [[buffer(2)]], threadgroup float *shared [[threadgroup(0)]], uint gid [[thread_position_in_grid]], uint lid [[thread_position_in_threadgroup]], uint gidx [[threadgroup_position_in_grid]], uint tps [[threads_per_threadgroup]]) {
    float val = (gid < (uint)dim) ? x[gid] : 0.0f;
    shared[lid] = val * val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = tps / 2; s > 0; s >>= 1) {
        if (lid < s) shared[lid] += shared[lid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) psums[gidx] = shared[0];
}

kernel void rmsnorm_apply_kernel(device const float *x [[buffer(0)]], device const float *weight [[buffer(1)]], device float *out [[buffer(2)]], constant float &inv_rms [[buffer(3)]], constant int &dim [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint)dim) return;
    out[gid] = weight[gid] * (x[gid] * inv_rms);
}

kernel void silu_kernel(device float *data [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
    float x = data[gid];
    data[gid] = x / (1.0f + exp(-x));
}

kernel void silu_mul_kernel(device const float *gate [[buffer(0)]], device const float *up [[buffer(1)]], device float *out [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    float g = gate[gid];
    float u = up[gid];
    out[gid] = (g / (1.0f + exp(-g))) * u;
}

kernel void rope_kernel(device float *qk [[buffer(0)]], constant int &hdim [[buffer(1)]], constant int &nheads [[buffer(2)]], constant int &rd [[buffer(3)]], constant int &pos [[buffer(4)]], constant float &theta [[buffer(5)]], uint gid [[thread_position_in_grid]]) {
    int head = gid / (rd / 2);
    int pair = gid % (rd / 2);
    if (head >= nheads) return;
    int i = pair * 2; if (i >= rd) return;
    float inv_f = 1.0f / pow(theta, (float)i / (float)rd);
    float cv = cos((float)pos * inv_f), sv = sin((float)pos * inv_f);
    device float *hv = qk + head * hdim;
    float a = hv[i], b = hv[i + 1];
    hv[i] = a * cv - b * sv; hv[i+1] = a * sv + b * cv;
}

kernel void add_kernel(device float *a [[buffer(0)]], device const float *b [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    a[gid] += b[gid];
}
)";
    }

    // Read from file
    fseek(fp, 0, SEEK_END);
    long len = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)len + 1);
    if (!buf) { fclose(fp); return @""; }
    fread(buf, 1, (size_t)len, fp);
    buf[len] = '\0';
    fclose(fp);
    NSString *src = [NSString stringWithUTF8String:buf];
    free(buf);
    return src;
}

// ── Internal helpers ────────────────────────────────────────────────────────

static void dispatch_matmul_vec(gpu_context *ctx, id<MTLCommandBuffer> cb,
                                 id<MTLBuffer> vec, id<MTLBuffer> mat,
                                 id<MTLBuffer> out, int K, int M) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:ctx->matmul_vec_ps];
    [enc setBuffer:vec offset:0 atIndex:0];
    [enc setBuffer:mat offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&K length:sizeof(int) atIndex:3];
    [enc setBytes:&M length:sizeof(int) atIndex:4];

    MTLSize grid = MTLSizeMake(M, 1, 1);
    MTLSize tg  = MTLSizeMake(256, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tg];
    [enc endEncoding];
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

gpu_context *gpu_init(void) {
    if (@available(macOS 10.15, *)) {
        // proceed
    } else {
        fprintf(stderr, "Metal requires macOS 10.15+\n");
        return NULL;
    }

    gpu_context *ctx = (gpu_context *)calloc(1, sizeof(gpu_context));
    if (!ctx) return NULL;

    @autoreleasepool {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "Metal: no device found\n");
            free(ctx); return NULL;
        }

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "Metal: failed to create command queue\n");
            free(ctx); return NULL;
        }

        // Load shader source
        NSString *src = load_metal_shader_source();
        if ([src length] == 0) {
            fprintf(stderr, "Metal: empty shader source\n");
            free(ctx); return NULL;
        }

        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion2_4;
        ctx->library = [ctx->device newLibraryWithSource:src
                                                  options:opts
                                                    error:&err];
        if (!ctx->library) {
            fprintf(stderr, "Metal: shader compile error: %s\n",
                    [[err localizedDescription] UTF8String]);
            free(ctx); return NULL;
        }

        // Compile all kernels
        ctx->matmul_vec_ps     = compile_kernel(ctx->device, ctx->library, @"matmul_vec_kernel", &err);
        ctx->rmsnorm_partial_ps = compile_kernel(ctx->device, ctx->library, @"rmsnorm_partial_kernel", &err);
        ctx->rmsnorm_apply_ps  = compile_kernel(ctx->device, ctx->library, @"rmsnorm_apply_kernel", &err);
        ctx->silu_ps           = compile_kernel(ctx->device, ctx->library, @"silu_kernel", &err);
        ctx->silu_mul_ps       = compile_kernel(ctx->device, ctx->library, @"silu_mul_kernel", &err);
        ctx->rope_ps           = compile_kernel(ctx->device, ctx->library, @"rope_kernel", &err);
        ctx->add_ps            = compile_kernel(ctx->device, ctx->library, @"add_kernel", &err);
        ctx->softmax_ps        = compile_kernel(ctx->device, ctx->library, @"softmax_kernel", &err);

        if (!ctx->matmul_vec_ps) {
            fprintf(stderr, "Metal: failed to compile matmul_vec_kernel\n");
            free(ctx); return NULL;
        }

        // Allocate intermediate buffer for RMSNorm partial sums (max 256 groups)
        ctx->partial_sums_len = 256;
        ctx->partial_sums_buf = [ctx->device newBufferWithLength:ctx->partial_sums_len * sizeof(float)
                                                         options:MTLResourceStorageModeShared];
    }

    ctx->initialized = true;
    return ctx;
}

void gpu_destroy(gpu_context *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        // ARC releases all ObjC objects — zero non-ObjC fields before free
        ctx->device = nil;
        ctx->queue = nil;
        ctx->library = nil;
        ctx->matmul_vec_ps = nil;
        ctx->rmsnorm_partial_ps = nil;
        ctx->rmsnorm_apply_ps = nil;
        ctx->silu_ps = nil;
        ctx->silu_mul_ps = nil;
        ctx->rope_ps = nil;
        ctx->add_ps = nil;
        ctx->softmax_ps = nil;
        ctx->partial_sums_buf = nil;
    }
    ctx->used_memory = 0;
    ctx->initialized = false;
    ctx->partial_sums_len = 0;
    free(ctx);
}

// ── Buffer management ───────────────────────────────────────────────────────

gpu_buffer *gpu_alloc(gpu_context *ctx, size_t size, gpu_buffer_flags flags) {
    if (!ctx || size == 0) return NULL;

    gpu_buffer *buf = (gpu_buffer *)calloc(1, sizeof(gpu_buffer));
    if (!buf) return NULL;

    MTLResourceOptions opts = MTLResourceStorageModeShared;
    if (flags & GPU_BUF_CONSTANT) opts = MTLResourceStorageModeManaged;

    @autoreleasepool {
        buf->buffer = [ctx->device newBufferWithLength:size options:opts];
        if (!buf->buffer) { free(buf); return NULL; }
    }

    buf->size = size;
    buf->flags = flags;
    ctx->used_memory += size;
    return buf;
}

void gpu_free(gpu_context *ctx, gpu_buffer *buf) {
    if (!ctx || !buf) return;
    ctx->used_memory -= buf->size;
    @autoreleasepool { buf->buffer = nil; }
    buf->size = 0;
    buf->flags = 0;
    free(buf);
}

void *gpu_get_cpu_ptr(gpu_context *ctx, gpu_buffer *buf) {
    (void)ctx;
    return buf ? [buf->buffer contents] : NULL;
}

size_t gpu_buffer_size(gpu_context *ctx, const gpu_buffer *buf) {
    (void)ctx;
    return buf ? buf->size : 0;
}

void gpu_copy_to_device(gpu_context *ctx, gpu_buffer *dst,
                        const void *src, size_t size) {
    if (!ctx || !dst || !src) return;
    size_t copy = size < dst->size ? size : dst->size;
    memcpy([dst->buffer contents], src, copy);
}

void gpu_copy_to_host(gpu_context *ctx, void *dst,
                      const gpu_buffer *src, size_t size) {
    if (!ctx || !src || !dst) return;
    size_t copy = size < src->size ? size : src->size;
    memcpy(dst, [src->buffer contents], copy);
}

void gpu_copy_buffer(gpu_context *ctx, gpu_buffer *dst,
                     const gpu_buffer *src, size_t size) {
    if (!ctx || !dst || !src) return;
    size_t copy = size;
    if (copy > dst->size) copy = dst->size;
    if (copy > src->size) copy = src->size;
    memcpy([dst->buffer contents], [src->buffer contents], copy);
}

// ── Compute: MatMul ─────────────────────────────────────────────────────────

void gpu_matmul(gpu_context *ctx, gpu_buffer *C,
                const gpu_buffer *A, const gpu_buffer *B,
                int M, int N, int K,
                quant_mode a_quant, quant_mode b_quant) {
    (void)M; (void)a_quant; (void)b_quant;
    if (!ctx || !C || !A || !B) return;

    id<MTLBuffer> mA = A->buffer;
    id<MTLBuffer> mB = B->buffer;
    id<MTLBuffer> mC = C->buffer;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        dispatch_matmul_vec(ctx, cb, mA, mB, mC, K, N);
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void gpu_matmul_int4(gpu_context *ctx, gpu_buffer *C,
                     const gpu_buffer *A_quant, const gpu_buffer *A_scale,
                     int M, int N, int K) {
    // Fallback: zero output (int4 dequant not accelerated yet)
    (void)A_quant; (void)A_scale; (void)K;
    if (!ctx || !C) return;
    memset([C->buffer contents], 0, (size_t)M * N * sizeof(float));
}

// ── Compute: RMSNorm ────────────────────────────────────────────────────────

void gpu_rmsnorm(gpu_context *ctx, gpu_buffer *out,
                 const gpu_buffer *x, const gpu_buffer *weight,
                 int rows, int dim) {
    if (!ctx || !out || !x || !weight || rows <= 0 || dim <= 0) return;

    id<MTLBuffer> mX = x->buffer;
    id<MTLBuffer> mW = weight->buffer;
    id<MTLBuffer> mO = out->buffer;

    int tg_size = 256;
    int n_groups = (dim + tg_size - 1) / tg_size;
    if (n_groups > ctx->partial_sums_len) n_groups = ctx->partial_sums_len;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];

        // Dispatch 1: compute partial sums
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->rmsnorm_partial_ps];
            [enc setBuffer:mX offset:0 atIndex:0];
            [enc setBuffer:ctx->partial_sums_buf offset:0 atIndex:1];
            [enc setBytes:&dim length:sizeof(int) atIndex:2];
            [enc setThreadgroupMemoryLength:tg_size * sizeof(float) atIndex:0];
            MTLSize grid = MTLSizeMake((NSUInteger)dim, 1, 1);
            MTLSize tg  = MTLSizeMake((NSUInteger)tg_size, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        // Read partial sums and compute inv_rms on CPU
        float *psums = (float *)[ctx->partial_sums_buf contents];
        float ss = 0.0f;
        for (int i = 0; i < n_groups; i++) ss += psums[i];
        float inv_rms = 1.0f / sqrtf(ss / (float)dim + 1e-6f);

        // Dispatch 2: apply normalization
        id<MTLCommandBuffer> cb2 = [ctx->queue commandBuffer];
        {
            id<MTLComputeCommandEncoder> enc = [cb2 computeCommandEncoder];
            [enc setComputePipelineState:ctx->rmsnorm_apply_ps];
            [enc setBuffer:mX offset:0 atIndex:0];
            [enc setBuffer:mW offset:0 atIndex:1];
            [enc setBuffer:mO offset:0 atIndex:2];
            [enc setBytes:&inv_rms length:sizeof(float) atIndex:3];
            [enc setBytes:&dim length:sizeof(int) atIndex:4];
            MTLSize grid = MTLSizeMake((NSUInteger)dim, 1, 1);
            MTLSize tg  = MTLSizeMake(256, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }
        [cb2 commit];
        [cb2 waitUntilCompleted];
    }
}

// ── Compute: SiLU ───────────────────────────────────────────────────────────

void gpu_silu(gpu_context *ctx, gpu_buffer *out,
              const gpu_buffer *x, int n) {
    if (!ctx || !out || !x || n <= 0) return;

    id<MTLBuffer> mX = x->buffer;
    id<MTLBuffer> mO = out->buffer;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        // If in-place (out == x), just apply silu_kernel on the shared buffer
        if (mO == mX) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->silu_ps];
            [enc setBuffer:mO offset:0 atIndex:0];
            MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
            MTLSize tg  = MTLSizeMake(256, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        } else {
            // Copy then apply
            // For simplicity, do CPU SiLU on out-of-place
            float *xp = (float *)[mX contents];
            float *op = (float *)[mO contents];
            for (int i = 0; i < n; i++) {
                float xv = xp[i];
                op[i] = xv / (1.0f + expf(-xv));
            }
        }
        [cb commit];
        [cb waitUntilCompleted];
    }
}

void gpu_silu_mul(gpu_context *ctx, gpu_buffer *out,
                  const gpu_buffer *gate, const gpu_buffer *up, int n) {
    if (!ctx || !out || !gate || !up || n <= 0) return;

    id<MTLBuffer> mG = gate->buffer;
    id<MTLBuffer> mU = up->buffer;
    id<MTLBuffer> mO = out->buffer;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->silu_mul_ps];
        [enc setBuffer:mG offset:0 atIndex:0];
        [enc setBuffer:mU offset:0 atIndex:1];
        [enc setBuffer:mO offset:0 atIndex:2];
        MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize tg  = MTLSizeMake(256, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// ── Compute: RoPE ───────────────────────────────────────────────────────────

void gpu_rope(gpu_context *ctx, gpu_buffer *Q, gpu_buffer *K,
              int seq_len, int head_dim, int n_heads, int n_kv_heads,
              float base, int pos_start) {
    (void)seq_len;
    if (!ctx || !Q || !K) return;

    id<MTLBuffer> mQ = Q->buffer;
    id<MTLBuffer> mK = K->buffer;

    int rot_dims = head_dim < 64 ? head_dim : 64; // RoPE on first 64 dims

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];

        // Apply RoPE to Q
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->rope_ps];
            [enc setBuffer:mQ offset:0 atIndex:0];
            [enc setBytes:&head_dim length:sizeof(int) atIndex:1];
            [enc setBytes:&n_heads length:sizeof(int) atIndex:2];
            [enc setBytes:&rot_dims length:sizeof(int) atIndex:3];
            [enc setBytes:&pos_start length:sizeof(int) atIndex:4];
            [enc setBytes:&base length:sizeof(float) atIndex:5];
            int n_pairs = n_heads * (rot_dims / 2);
            MTLSize grid = MTLSizeMake((NSUInteger)n_pairs, 1, 1);
            MTLSize tg  = MTLSizeMake(256, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        // Apply RoPE to K
        {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ctx->rope_ps];
            [enc setBuffer:mK offset:0 atIndex:0];
            [enc setBytes:&head_dim length:sizeof(int) atIndex:1];
            [enc setBytes:&n_kv_heads length:sizeof(int) atIndex:2];
            [enc setBytes:&rot_dims length:sizeof(int) atIndex:3];
            [enc setBytes:&pos_start length:sizeof(int) atIndex:4];
            [enc setBytes:&base length:sizeof(float) atIndex:5];
            int n_pairs = n_kv_heads * (rot_dims / 2);
            MTLSize grid = MTLSizeMake((NSUInteger)n_pairs, 1, 1);
            MTLSize tg  = MTLSizeMake(256, 1, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];
    }
}

// ── Compute: Attention (CPU-only stub for now) ──────────────────────────────

void gpu_attention(gpu_context *ctx, gpu_buffer *out,
                   const gpu_buffer *Q, const gpu_buffer *K_cache,
                   const gpu_buffer *V_cache, int seq_len, int head_dim) {
    (void)ctx; (void)out; (void)Q; (void)K_cache; (void)V_cache;
    (void)seq_len; (void)head_dim;
    // Full attention on GPU is handled via the CPU path in model.c
    // which calls gpu_matmul for QKV projections and does attention on CPU
}

// ── Compute: DeltaNet (stub) ────────────────────────────────────────────────

void gpu_deltanet_step(gpu_context *ctx, gpu_buffer *state,
                       const gpu_buffer *Q, const gpu_buffer *K,
                       const gpu_buffer *V, int dim) {
    (void)ctx; (void)state; (void)Q; (void)K; (void)V; (void)dim;
}

// ── Compute: Expert MLP (stub) ──────────────────────────────────────────────

void gpu_expert_mlp(gpu_context *ctx, gpu_buffer *out,
                    const gpu_buffer *x, const gpu_buffer *gate_w,
                    const gpu_buffer *up_w, const gpu_buffer *down_w,
                    int d_model, int hidden_dim, quant_mode quant) {
    // The CPU path handles expert MLP directly via matmul + silu_mul
    (void)ctx; (void)out; (void)x; (void)gate_w; (void)up_w;
    (void)down_w; (void)d_model; (void)hidden_dim; (void)quant;
}

// ── Compute: Router (stub) ──────────────────────────────────────────────────

void gpu_router(gpu_context *ctx, gpu_buffer *out,
                const gpu_buffer *x, const gpu_buffer *router_weight,
                int d_model, int n_experts) {
    (void)ctx; (void)out; (void)x; (void)router_weight;
    (void)d_model; (void)n_experts;
}

void gpu_topk_softmax(gpu_context *ctx, uint32_t *indices, float *weights,
                      const gpu_buffer *logits, int n_experts, int top_k) {
    (void)ctx; (void)indices; (void)weights; (void)logits;
    (void)n_experts; (void)top_k;
}

// ── Compute: Element-wise add ───────────────────────────────────────────────

void gpu_add(gpu_context *ctx, gpu_buffer *out,
             const gpu_buffer *a, const gpu_buffer *b, int n) {
    if (!ctx || !out || !a || !b || n <= 0) return;

    id<MTLBuffer> mA = a->buffer;
    id<MTLBuffer> mB = b->buffer;
    id<MTLBuffer> mO = out->buffer;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:ctx->add_ps];
        [enc setBuffer:mO offset:0 atIndex:0];

        // For add kernel, read from b, write to out (which may be a)
        // The kernel does: out[i] += b[i], so if out == a, it's in-place
        if (mO == mA) {
            [enc setBuffer:mO offset:0 atIndex:0]; // a
            [enc setBuffer:mB offset:0 atIndex:1]; // b
        } else {
            [enc setBuffer:mO offset:0 atIndex:0]; // a (will be overwritten)
            [enc setBuffer:mB offset:0 atIndex:1]; // b
            // Copy a to out first
            memcpy([mO contents], [mA contents], (size_t)n * sizeof(float));
        }

        MTLSize grid = MTLSizeMake((NSUInteger)n, 1, 1);
        MTLSize tg  = MTLSizeMake(256, 1, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// ── Sync / Info ─────────────────────────────────────────────────────────────

void gpu_sync(gpu_context *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

size_t gpu_memory_used(gpu_context *ctx) {
    return ctx ? ctx->used_memory : 0;
}

void gpu_print_info(gpu_context *ctx) {
    if (!ctx) return;
    printf("=== GPU Info (Metal backend) ===\n");
    printf("Device:   %s\n", [[ctx->device name] UTF8String]);
    printf("Memory:   %zu bytes\n", ctx->used_memory);
    if ([ctx->device hasUnifiedMemory])
        printf("Unified:  yes\n");
    printf("Kerels:   %s\n", ctx->matmul_vec_ps ? "matmul_vec" : "");
}

// ── Helper: compile a single kernel function ────────────────────────────────

static id<MTLComputePipelineState> compile_kernel(id<MTLDevice> dev,
                                                   id<MTLLibrary> lib,
                                                   NSString *name,
                                                   NSError **err) {
    id<MTLFunction> func = [lib newFunctionWithName:name];
    if (!func) return nil;
    return [dev newComputePipelineStateWithFunction:func error:err];
}
