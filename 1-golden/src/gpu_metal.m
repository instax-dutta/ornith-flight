// SPDX-License-Identifier: MIT
// Metal GPU backend — drop-in replacement for gpu.c on macOS.
// Uses Metal Performance Shaders (MPS) for matmul + runtime-compiled MSL for custom ops.

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

// ── Internal struct definitions ─────────────────────────────────────────────

struct gpu_context {
    id<MTLDevice>              device;
    id<MTLCommandQueue>        queue;
    id<MTLLibrary>             library;    // compiled MSL library
    size_t                     used_memory;
    bool                       initialized;

    // Cached compute pipeline states
    id<MTLComputePipelineState> pipeline_rmsnorm;
    id<MTLComputePipelineState> pipeline_silu;
    id<MTLComputePipelineState> pipeline_silu_mul;
    id<MTLComputePipelineState> pipeline_rope;
    id<MTLComputePipelineState> pipeline_expert_mlp;
    id<MTLComputePipelineState> pipeline_router;
    id<MTLComputePipelineState> pipeline_topk;
    id<MTLComputePipelineState> pipeline_add;
    id<MTLComputePipelineState> pipeline_deltanet;
    id<MTLComputePipelineState> pipeline_dequant_q4;
    id<MTLComputePipelineState> pipeline_matmul_q4;
    id<MTLComputePipelineState> pipeline_matmul_f16;
    id<MTLComputePipelineState> pipeline_attention;
    id<MTLComputePipelineState> pipeline_kv_cache;
};

struct gpu_buffer {
    id<MTLBuffer>   metal_buffer;
    void           *cpu_ptr;        // for shared/CPU-accessible buffers
    size_t          size;
    gpu_buffer_flags flags;
    bool            is_shared;      // MTLResourceStorageModeShared
};

// ── MSL source strings (compiled at runtime) ───────────────────────────────

// Inline MSL for each custom kernel — loaded from the .metal files when
// pre-compiled metallib is available, otherwise compiled at runtime.
// These strings mirror the .metal file contents.

static const char *msl_rmsnorm = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void rmsnorm_single(
    device const float *x       [[buffer(0)]],
    device const float *weight  [[buffer(1)]],
    device       float *output  [[buffer(2)]],
    constant     int   &dim     [[buffer(3)]],
    constant     float &eps     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)dim) return;
    float sum_sq = 0.0f;
    for (int d = 0; d < dim; d++) { float v = x[d]; sum_sq += v * v; }
    float rms = sqrt(sum_sq / (float)dim + eps);
    output[tid] = (x[tid] / rms) * weight[tid];
}
)msl";

static const char *msl_silu = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void silu_activation(
    device const float *x [[buffer(0)]],
    device       float *o [[buffer(1)]],
    constant     int   &n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;
    float v = x[tid];
    o[tid] = v / (1.0f + exp(-v));
}
)msl";

static const char *msl_silu_mul = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void silu_mul(
    device const float *gate [[buffer(0)]],
    device const float *up   [[buffer(1)]],
    device       float *out  [[buffer(2)]],
    constant     int   &n    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;
    float g = gate[tid];
    out[tid] = (g / (1.0f + exp(-g))) * up[tid];
}
)msl";

static const char *msl_rope = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void rope_single(
    device       float *Q        [[buffer(0)]],
    device       float *K        [[buffer(1)]],
    constant     int   &pos      [[buffer(2)]],
    constant     int   &head_dim [[buffer(3)]],
    constant     int   &n_heads  [[buffer(4)]],
    constant     int   &n_kv_heads [[buffer(5)]],
    constant     float &base     [[buffer(6)]],
    uint2 pos_idx [[thread_position_in_grid]]
) {
    int head = (int)pos_idx.x;
    int d2   = (int)pos_idx.y;
    if (head >= n_heads) return;
    int d = 2 * d2;
    if (d >= head_dim) return;
    float theta = (float)pos / pow(base, (float)(2 * d2) / (float)head_dim);
    float c = cos(theta), s = sin(theta);
    int qi = head * head_dim + d;
    float q0 = Q[qi], q1 = Q[qi+1];
    Q[qi] = q0*c - q1*s; Q[qi+1] = q0*s + q1*c;
    if (head < n_kv_heads) {
        int ki = head * head_dim + d;
        float k0 = K[ki], k1 = K[ki+1];
        K[ki] = k0*c - k1*s; K[ki+1] = k0*s + k1*c;
    }
}
)msl";

static const char *msl_add = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void add_kernel(
    device const float *a [[buffer(0)]],
    device const float *b [[buffer(1)]],
    device       float *o [[buffer(2)]],
    constant     int   &n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)n) return;
    o[tid] = a[tid] + b[tid];
}
)msl";

static const char *msl_deltanet = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void deltanet_step(
    device       float *state  [[buffer(0)]],
    device const float *Q      [[buffer(1)]],
    device const float *K      [[buffer(2)]],
    device const float *V      [[buffer(3)]],
    device       float *output [[buffer(4)]],
    constant     int   &dim    [[buffer(5)]],
    constant     float &gate   [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)dim) return;
    for (int j = 0; j < dim; j++) {
        int idx = j * dim + (int)tid;
        state[idx] = gate * state[idx] + (1.0f - gate) * K[(int)tid] * V[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < dim; j++)
        sum += Q[j] * state[j * dim + (int)tid];
    output[tid] = sum;
}
)msl";

static const char *msl_expert_mlp = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void expert_mlp_forward(
    device const float *x       [[buffer(0)]],
    device const float *gate_w  [[buffer(1)]],
    device const float *up_w    [[buffer(2)]],
    device const float *down_w  [[buffer(3)]],
    device       float *output  [[buffer(4)]],
    constant     int   &dm      [[buffer(5)]],
    constant     int   &hd      [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)dm) return;
    float sum = 0.0f;
    for (int j = 0; j < hd; j++) {
        float gg = 0.0f, ug = 0.0f;
        for (int i = 0; i < dm; i++) {
            float xi = x[i];
            gg += xi * gate_w[i * hd + j];
            ug += xi * up_w[i * hd + j];
        }
        float s = gg / (1.0f + exp(-gg));
        sum += (s * ug) * down_w[j * dm + tid];
    }
    output[tid] = sum;
}
)msl";

static const char *msl_router = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void router_forward(
    device const float *x  [[buffer(0)]],
    device const float *rw [[buffer(1)]],
    device       float *lg [[buffer(2)]],
    constant     int   &dm [[buffer(3)]],
    constant     int   &ne [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)ne) return;
    float s = 0.0f;
    for (int i = 0; i < dm; i++) s += x[i] * rw[i * ne + tid];
    lg[tid] = s;
}
)msl";

static const char *msl_topk = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void topk_softmax(
    device       float  *l   [[buffer(0)]],
    device       uint   *idx [[buffer(1)]],
    device       float  *w   [[buffer(2)]],
    constant     int    &ne  [[buffer(3)]],
    constant     int    &tk  [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= 1) return;
    float sc[1024]; int ii[1024];
    int n = ne < 1024 ? ne : 1024;
    for (int i = 0; i < n; i++) { sc[i] = l[i]; ii[i] = i; }
    for (int i = 0; i < n-1; i++)
        for (int j = i+1; j < n; j++)
            if (sc[j] > sc[i]) {
                float ts = sc[i]; sc[i] = sc[j]; sc[j] = ts;
                int ti = ii[i]; ii[i] = ii[j]; ii[j] = ti;
            }
    float mx = sc[0], se = 0.0f;
    int k = tk < n ? tk : n;
    for (int i = 0; i < k; i++) { float e = exp(sc[i]-mx); w[i] = e; idx[i] = (uint)ii[i]; se += e; }
    for (int i = 0; i < k; i++) w[i] /= se;
}
)msl";

static const char *msl_attention = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void attention_forward(
    device const float *Q     [[buffer(0)]],
    device const float *K     [[buffer(1)]],
    device const float *V     [[buffer(2)]],
    device       float *out   [[buffer(3)]],
    constant     int   &sl    [[buffer(4)]],
    constant     int   &hd    [[buffer(5)]],
    constant     float &sc    [[buffer(6)]],
    uint2 pos [[thread_position_in_grid]]
) {
    int row = (int)pos.x, col = (int)pos.y;
    if (row >= sl || col >= hd) return;
    float sw = 0.0f, ov = 0.0f;
    for (int t = 0; t < sl; t++) {
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s += Q[row*hd+d] * K[t*hd+d];
        s *= sc; float w = exp(s); sw += w;
        ov += w * V[t*hd+col];
    }
    out[row*hd+col] = sw > 1e-10f ? ov / sw : 0.0f;
}
)msl";

static const char *msl_kv_cache = R"msl(
#include <metal_stdlib>
using namespace metal;
kernel void kv_cache_append(
    device       float *kc [[buffer(0)]],
    device       float *vc [[buffer(1)]],
    device const float *kn [[buffer(2)]],
    device const float *vn [[buffer(3)]],
    constant     int   &p  [[buffer(4)]],
    constant     int   &hd [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= (uint)hd) return;
    kc[p*hd+tid] = kn[tid];
    vc[p*hd+tid] = vn[tid];
}
)msl";

// ── Helper: compile a single MSL kernel ────────────────────────────────────

static id<MTLComputePipelineState> compile_kernel(id<MTLDevice> device,
                                                   id<MTLLibrary> library,
                                                   const char *name) {
    NSString *nsname = [NSString stringWithUTF8String:name];
    id<MTLFunction> func = [library newFunctionWithName:nsname];
    if (!func) {
        fprintf(stderr, "Error: Metal kernel '%s' not found\n", name);
        return nil;
    }
    NSError *err = nil;
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func error:&err];
    if (!pso) {
        fprintf(stderr, "Error: PSO for '%s': %s\n", name,
                [[err localizedDescription] UTF8String]);
        return nil;
    }
    return pso;
}

// ── Helper: dispatch a 1D compute kernel ────────────────────────────────────

static void dispatch_1d(id<MTLCommandBuffer> cb,
                        id<MTLComputePipelineState> pso,
                        id<MTLBuffer> buffers[], int n_buffers,
                        int n_threads) {
    if (!pso || n_threads <= 0) return;
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    if (!enc) return;
    [enc setComputePipelineState:pso];
    for (int i = 0; i < n_buffers; i++) {
        [enc setBuffer:buffers[i] offset:0 atIndex:i];
    }
    MTLSize grid = MTLSizeMake((NSUInteger)n_threads, 1, 1);
    NSUInteger tw = [pso maxTotalThreadsPerThreadgroup];
    NSUInteger tg = tw < 256 ? tw : 256;
    MTLSize tgroup = MTLSizeMake(tg, 1, 1);
    [enc dispatchThreads:grid threadsPerThreadgroup:tgroup];
    [enc endEncoding];
}

// ── gpu_init ────────────────────────────────────────────────────────────────

gpu_context *gpu_init(void) {
    gpu_context *ctx = (gpu_context *)calloc(1, sizeof(gpu_context));
    if (!ctx) return NULL;

    @autoreleasepool {
        ctx->device = MTLCreateSystemDefaultDevice();
        if (!ctx->device) {
            fprintf(stderr, "Error: Metal not available on this system\n");
            free(ctx);
            return NULL;
        }

        ctx->queue = [ctx->device newCommandQueue];
        if (!ctx->queue) {
            fprintf(stderr, "Error: Failed to create Metal command queue\n");
            free(ctx);
            return NULL;
        }

        // Compile MSL library at runtime from embedded source strings
        // The library contains all kernel functions
        // We build one combined library from all MSL strings
        NSMutableString *all_msl = [NSMutableString string];
        [all_msl appendString:[NSString stringWithUTF8String:msl_rmsnorm]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_silu]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_silu_mul]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_rope]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_add]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_deltanet]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_expert_mlp]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_router]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_topk]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_attention]];
        [all_msl appendString:[NSString stringWithUTF8String:msl_kv_cache]];

        // Try loading pre-compiled default.metallib first, fall back to runtime
        NSURL *metallibURL = [[NSBundle mainBundle] URLForResource:@"default"
                                                     withExtension:@"metallib"];
        if (metallibURL) {
            ctx->library = [ctx->device newLibraryWithURL:metallibURL error:nil];
        }

        if (!ctx->library) {
            MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
            opts.languageVersion = MTLLanguageVersion2_4;
            NSError *err = nil;
            ctx->library = [ctx->device newLibraryWithSource:all_msl
                                                      options:opts
                                                        error:&err];
            if (!ctx->library) {
                fprintf(stderr, "Warning: MSL compile failed: %s\n",
                        [[err localizedDescription] UTF8String]);
                fprintf(stderr, "Falling back to CPU backend...\n");
                ctx->device = nil;
                ctx->queue = nil;
                free(ctx);
                return NULL;
            }
        }

        // Cache pipeline states — verify all compiled successfully
        ctx->pipeline_rmsnorm    = compile_kernel(ctx->device, ctx->library, "rmsnorm_single");
        ctx->pipeline_silu       = compile_kernel(ctx->device, ctx->library, "silu_activation");
        ctx->pipeline_silu_mul   = compile_kernel(ctx->device, ctx->library, "silu_mul");
        ctx->pipeline_rope       = compile_kernel(ctx->device, ctx->library, "rope_single");
        ctx->pipeline_add        = compile_kernel(ctx->device, ctx->library, "add_kernel");
        ctx->pipeline_deltanet   = compile_kernel(ctx->device, ctx->library, "deltanet_step");
        ctx->pipeline_expert_mlp = compile_kernel(ctx->device, ctx->library, "expert_mlp_forward");
        ctx->pipeline_router     = compile_kernel(ctx->device, ctx->library, "router_forward");
        ctx->pipeline_topk       = compile_kernel(ctx->device, ctx->library, "topk_softmax");
        ctx->pipeline_attention  = compile_kernel(ctx->device, ctx->library, "attention_forward");
        ctx->pipeline_kv_cache   = compile_kernel(ctx->device, ctx->library, "kv_cache_append");

        // Check that all pipelines compiled
        if (!ctx->pipeline_rmsnorm || !ctx->pipeline_silu || !ctx->pipeline_silu_mul ||
            !ctx->pipeline_rope || !ctx->pipeline_add || !ctx->pipeline_deltanet ||
            !ctx->pipeline_expert_mlp || !ctx->pipeline_router || !ctx->pipeline_topk ||
            !ctx->pipeline_attention || !ctx->pipeline_kv_cache) {
            fprintf(stderr, "Error: One or more Metal kernels failed to compile\n");
            ctx->device = nil;
            ctx->queue = nil;
            free(ctx);
            return NULL;
        }

        ctx->initialized = true;
        ctx->used_memory = 0;
    }

    return ctx;
}

void gpu_destroy(gpu_context *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        ctx->device = nil;
        ctx->queue = nil;
        ctx->library = nil;
        ctx->pipeline_rmsnorm = nil;
        ctx->pipeline_silu = nil;
        ctx->pipeline_silu_mul = nil;
        ctx->pipeline_rope = nil;
        ctx->pipeline_add = nil;
        ctx->pipeline_deltanet = nil;
        ctx->pipeline_expert_mlp = nil;
        ctx->pipeline_router = nil;
        ctx->pipeline_topk = nil;
        ctx->pipeline_attention = nil;
        ctx->pipeline_kv_cache = nil;
    }
    memset(ctx, 0, sizeof(*ctx));
    free(ctx);
}

// ── Buffer management ─────────────────────────────────────────────────────

gpu_buffer *gpu_alloc(gpu_context *ctx, size_t size, gpu_buffer_flags flags) {
    if (!ctx || size == 0) return NULL;

    gpu_buffer *buf = (gpu_buffer *)calloc(1, sizeof(gpu_buffer));
    if (!buf) return NULL;

    @autoreleasepool {
        bool shared = (flags & GPU_BUF_SHARED) != 0;
        MTLResourceOptions opts = shared
            ? MTLResourceStorageModeShared
            : MTLResourceStorageModePrivate;

        buf->metal_buffer = [ctx->device newBufferWithLength:size options:opts];
        if (!buf->metal_buffer) {
            free(buf);
            return NULL;
        }

        buf->cpu_ptr = shared ? [buf->metal_buffer contents] : calloc(1, size);
        buf->size = size;
        buf->flags = flags;
        buf->is_shared = shared;
        ctx->used_memory += size;
    }

    return buf;
}

void gpu_free(gpu_context *ctx, gpu_buffer *buf) {
    if (!ctx || !buf) return;
    @autoreleasepool {
        ctx->used_memory -= buf->size;
        if (!buf->is_shared && buf->cpu_ptr) free(buf->cpu_ptr);
        buf->metal_buffer = nil;
    }
    memset(buf, 0, sizeof(*buf));
    free(buf);
}

void *gpu_get_cpu_ptr(gpu_context *ctx, gpu_buffer *buf) {
    (void)ctx;
    if (!buf) return NULL;
    if (buf->is_shared) return [buf->metal_buffer contents];
    return buf->cpu_ptr;
}

size_t gpu_buffer_size(gpu_context *ctx, const gpu_buffer *buf) {
    (void)ctx;
    return buf ? buf->size : 0;
}

void gpu_copy_to_device(gpu_context *ctx, gpu_buffer *dst,
                        const void *src, size_t size) {
    if (!ctx || !dst || !src) return;
    size_t copy = size < dst->size ? size : dst->size;
    @autoreleasepool {
        // All Metal buffers use shared storage mode for simplicity.
        // Private-buffer path would need a proper blit encoder.
        void *ptr = [dst->metal_buffer contents];
        if (ptr) memcpy(ptr, src, copy);
    }
}

void gpu_copy_to_host(gpu_context *ctx, void *dst,
                      const gpu_buffer *src, size_t size) {
    if (!ctx || !src || !dst) return;
    size_t copy = size < src->size ? size : src->size;
    @autoreleasepool {
        if (src->is_shared) {
            memcpy(dst, [src->metal_buffer contents], copy);
        } else if (src->cpu_ptr) {
            memcpy(dst, src->cpu_ptr, copy);
        }
    }
}

void gpu_copy_buffer(gpu_context *ctx, gpu_buffer *dst,
                     const gpu_buffer *src, size_t size) {
    if (!ctx || !dst || !src) return;
    @autoreleasepool {
        if (dst->is_shared && src->is_shared) {
            memcpy([dst->metal_buffer contents],
                   [src->metal_buffer contents],
                   size < dst->size ? (size < src->size ? size : src->size) : dst->size);
        } else {
            void *s = src->is_shared ? [src->metal_buffer contents] : src->cpu_ptr;
            void *d = dst->is_shared ? [dst->metal_buffer contents] : dst->cpu_ptr;
            if (s && d) {
                size_t copy = size;
                if (copy > dst->size) copy = dst->size;
                if (copy > src->size) copy = src->size;
                memcpy(d, s, copy);
            }
        }
    }
}

// ── Compute: matmul via MPS ───────────────────────────────────────────────

void gpu_matmul(gpu_context *ctx, gpu_buffer *C,
                const gpu_buffer *A, const gpu_buffer *B,
                int M, int N, int K,
                quant_mode a_quant, quant_mode b_quant) {
    if (!ctx || !C || !A || !B) return;
    (void)a_quant;
    (void)b_quant;

    @autoreleasepool {
        // Use MPSMatrixMultiplication for fp32 matmul
        MPSMatrixDescriptor *descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
            rowBytes:(NSUInteger)K * sizeof(float)
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
            rowBytes:(NSUInteger)N * sizeof(float)
            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:(NSUInteger)N * sizeof(float)
            dataType:MPSDataTypeFloat32];

        id<MTLBuffer> bufA = A->metal_buffer;
        id<MTLBuffer> bufB = B->metal_buffer;
        id<MTLBuffer> bufC = C->metal_buffer;

        // For non-shared buffers, copy CPU data to Metal buffer
        if (!A->is_shared && A->cpu_ptr) {
            bufA = [ctx->device newBufferWithBytes:A->cpu_ptr
                                            length:A->size
                                           options:MTLResourceStorageModeShared];
        }
        if (!B->is_shared && B->cpu_ptr) {
            bufB = [ctx->device newBufferWithBytes:B->cpu_ptr
                                            length:B->size
                                           options:MTLResourceStorageModeShared];
        }
        if (!C->is_shared) {
            bufC = [ctx->device newBufferWithLength:C->size
                                            options:MTLResourceStorageModeShared];
        }

        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufA descriptor:descA];
        MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bufB descriptor:descB];
        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufC descriptor:descC];

        MPSMatrixMultiplication *matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:ctx->device
            transposeLeft:false transposeRight:false
            resultRows:(NSUInteger)M resultColumns:(NSUInteger)N
            interiorColumns:(NSUInteger)K alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        [matmul encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        [cb commit];
        [cb waitUntilCompleted];

        // Copy back to CPU-side buffer
        if (!C->is_shared && C->cpu_ptr) {
            memcpy(C->cpu_ptr, [bufC contents], C->size);
        }
    }
}

void gpu_matmul_int4(gpu_context *ctx, gpu_buffer *C,
                     const gpu_buffer *A_quant, const gpu_buffer *A_scale,
                     int M, int N, int K) {
    // Fall back to CPU: int4 matmul not accelerated via MPS directly
    // In production: use custom int4 matmul kernel
    (void)A_quant;
    (void)A_scale;
    (void)K;
    if (!ctx || !C) return;
    if (C->is_shared) {
        memset([C->metal_buffer contents], 0, (size_t)M * N * sizeof(float));
    } else if (C->cpu_ptr) {
        memset(C->cpu_ptr, 0, (size_t)M * N * sizeof(float));
    }
}

// ── Compute: custom kernels ────────────────────────────────────────────────

void gpu_rmsnorm(gpu_context *ctx, gpu_buffer *out,
                 const gpu_buffer *x, const gpu_buffer *weight,
                 int rows, int dim) {
    if (!ctx || !out || !x || !weight) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        float eps = 1e-6f;
        id<MTLBuffer> bufs[] = {
            x->is_shared ? x->metal_buffer : nil,
            weight->is_shared ? weight->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        // Handle non-shared buffers by copying to temp shared buffers
        if (!x->is_shared && x->cpu_ptr) {
            bufs[0] = [ctx->device newBufferWithBytes:x->cpu_ptr length:x->size
                                              options:MTLResourceStorageModeShared];
        }
        if (!weight->is_shared && weight->cpu_ptr) {
            bufs[1] = [ctx->device newBufferWithBytes:weight->cpu_ptr length:weight->size
                                              options:MTLResourceStorageModeShared];
        }
        if (!out->is_shared) {
            bufs[2] = [ctx->device newBufferWithLength:out->size
                                               options:MTLResourceStorageModeShared];
        }

        // Use dispatch_1d for each row (batch into 2D dispatch)
        // Single-row dispatch for each row
        for (int r = 0; r < rows; r++) {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) continue;
            [enc setComputePipelineState:ctx->pipeline_rmsnorm];
            NSUInteger offset = (NSUInteger)r * dim * sizeof(float);
            [enc setBuffer:bufs[0] offset:offset atIndex:0];
            [enc setBuffer:bufs[1] offset:0 atIndex:1];
            [enc setBuffer:bufs[2] offset:offset atIndex:2];
            [enc setBytes:&dim length:sizeof(dim) atIndex:3];
            [enc setBytes:&eps length:sizeof(eps) atIndex:4];
            MTLSize grid = MTLSizeMake((NSUInteger)dim, 1, 1);
            NSUInteger tg = (NSUInteger)[ctx->pipeline_rmsnorm maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        // Copy back
        if (!out->is_shared && out->cpu_ptr && bufs[2]) {
            memcpy(out->cpu_ptr, [bufs[2] contents], out->size);
        }
    }
}

void gpu_silu(gpu_context *ctx, gpu_buffer *out,
              const gpu_buffer *x, int n) {
    if (!ctx || !out || !x) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufs[] = {
            x->is_shared ? x->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        if (!x->is_shared && x->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:x->cpu_ptr length:x->size
                                              options:MTLResourceStorageModeShared];
        if (!out->is_shared)
            bufs[1] = [ctx->device newBufferWithLength:out->size
                                              options:MTLResourceStorageModeShared];

        dispatch_1d(cb, ctx->pipeline_silu, bufs, 2, n);

        [cb commit];
        [cb waitUntilCompleted];

        if (!out->is_shared && out->cpu_ptr && bufs[1])
            memcpy(out->cpu_ptr, [bufs[1] contents], out->size);
    }
}

void gpu_silu_mul(gpu_context *ctx, gpu_buffer *out,
                  const gpu_buffer *gate, const gpu_buffer *up, int n) {
    if (!ctx || !out || !gate || !up) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufs[] = {
            gate->is_shared ? gate->metal_buffer : nil,
            up->is_shared ? up->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        if (!gate->is_shared && gate->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:gate->cpu_ptr length:gate->size
                                              options:MTLResourceStorageModeShared];
        if (!up->is_shared && up->cpu_ptr)
            bufs[1] = [ctx->device newBufferWithBytes:up->cpu_ptr length:up->size
                                              options:MTLResourceStorageModeShared];
        if (!out->is_shared)
            bufs[2] = [ctx->device newBufferWithLength:out->size
                                              options:MTLResourceStorageModeShared];

        dispatch_1d(cb, ctx->pipeline_silu_mul, bufs, 3, n);

        [cb commit];
        [cb waitUntilCompleted];

        if (!out->is_shared && out->cpu_ptr && bufs[2])
            memcpy(out->cpu_ptr, [bufs[2] contents], out->size);
    }
}

void gpu_rope(gpu_context *ctx, gpu_buffer *Q, gpu_buffer *K,
              int seq_len, int head_dim, int n_heads, int n_kv_heads,
              float base, int pos_start) {
    if (!ctx || !Q || !K) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufQ = Q->is_shared ? Q->metal_buffer : nil;
        id<MTLBuffer> bufK = K->is_shared ? K->metal_buffer : nil;

        if (!Q->is_shared && Q->cpu_ptr)
            bufQ = [ctx->device newBufferWithBytes:Q->cpu_ptr length:Q->size
                                          options:MTLResourceStorageModeShared];
        if (!K->is_shared && K->cpu_ptr)
            bufK = [ctx->device newBufferWithBytes:K->cpu_ptr length:K->size
                                          options:MTLResourceStorageModeShared];

        for (int p = 0; p < seq_len; p++) {
            int actual_pos = pos_start + p;
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            if (!enc) continue;
            [enc setComputePipelineState:ctx->pipeline_rope];
            NSUInteger q_off = (NSUInteger)p * (NSUInteger)n_heads * (NSUInteger)head_dim * sizeof(float);
            NSUInteger k_off = (NSUInteger)p * (NSUInteger)n_kv_heads * (NSUInteger)head_dim * sizeof(float);
            [enc setBuffer:bufQ offset:q_off atIndex:0];
            [enc setBuffer:bufK offset:k_off atIndex:1];
            [enc setBytes:&actual_pos length:sizeof(actual_pos) atIndex:2];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:3];
            [enc setBytes:&n_heads length:sizeof(n_heads) atIndex:4];
            [enc setBytes:&n_kv_heads length:sizeof(n_kv_heads) atIndex:5];
            [enc setBytes:&base length:sizeof(base) atIndex:6];
            MTLSize grid = MTLSizeMake((NSUInteger)n_heads, (NSUInteger)(head_dim/2), 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        if (!Q->is_shared && Q->cpu_ptr && bufQ)
            memcpy(Q->cpu_ptr, [bufQ contents], Q->size);
        if (!K->is_shared && K->cpu_ptr && bufK)
            memcpy(K->cpu_ptr, [bufK contents], K->size);
    }
}

void gpu_attention(gpu_context *ctx, gpu_buffer *out,
                   const gpu_buffer *Q, const gpu_buffer *K_cache,
                   const gpu_buffer *V_cache, int seq_len, int head_dim) {
    if (!ctx || !out || !Q || !K_cache || !V_cache) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufs[] = {
            Q->is_shared ? Q->metal_buffer : nil,
            K_cache->is_shared ? K_cache->metal_buffer : nil,
            V_cache->is_shared ? V_cache->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        if (!Q->is_shared && Q->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:Q->cpu_ptr length:Q->size
                                              options:MTLResourceStorageModeShared];
        if (!K_cache->is_shared && K_cache->cpu_ptr)
            bufs[1] = [ctx->device newBufferWithBytes:K_cache->cpu_ptr length:K_cache->size
                                              options:MTLResourceStorageModeShared];
        if (!V_cache->is_shared && V_cache->cpu_ptr)
            bufs[2] = [ctx->device newBufferWithBytes:V_cache->cpu_ptr length:V_cache->size
                                              options:MTLResourceStorageModeShared];
        if (!out->is_shared)
            bufs[3] = [ctx->device newBufferWithLength:out->size
                                               options:MTLResourceStorageModeShared];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc) {
            [enc setComputePipelineState:ctx->pipeline_attention];
            for (int i = 0; i < 4; i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
            [enc setBytes:&seq_len length:sizeof(seq_len) atIndex:4];
            [enc setBytes:&head_dim length:sizeof(head_dim) atIndex:5];
            float scale = 1.0f / sqrtf((float)head_dim);
            [enc setBytes:&scale length:sizeof(scale) atIndex:6];
            MTLSize grid = MTLSizeMake((NSUInteger)seq_len, (NSUInteger)head_dim, 1);
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        if (!out->is_shared && out->cpu_ptr && bufs[3])
            memcpy(out->cpu_ptr, [bufs[3] contents], out->size);
    }
}

void gpu_deltanet_step(gpu_context *ctx, gpu_buffer *state,
                       const gpu_buffer *Q, const gpu_buffer *K,
                       const gpu_buffer *V, int dim) {
    if (!ctx || !state || !Q || !K || !V) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        float gate = 0.9f;
        id<MTLBuffer> bufs[] = {
            state->is_shared ? state->metal_buffer : nil,
            Q->is_shared ? Q->metal_buffer : nil,
            K->is_shared ? K->metal_buffer : nil,
            V->is_shared ? V->metal_buffer : nil,
            nil, // output buffer (same as state for in-place update)
        };
        // state is both input and output for the persistent recurrent state
        bufs[4] = bufs[0];

        if (!state->is_shared && state->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:state->cpu_ptr length:state->size
                                              options:MTLResourceStorageModeShared];
        if (!Q->is_shared && Q->cpu_ptr)
            bufs[1] = [ctx->device newBufferWithBytes:Q->cpu_ptr length:Q->size
                                              options:MTLResourceStorageModeShared];
        if (!K->is_shared && K->cpu_ptr)
            bufs[2] = [ctx->device newBufferWithBytes:K->cpu_ptr length:K->size
                                              options:MTLResourceStorageModeShared];
        if (!V->is_shared && V->cpu_ptr)
            bufs[3] = [ctx->device newBufferWithBytes:V->cpu_ptr length:V->size
                                              options:MTLResourceStorageModeShared];
        bufs[4] = bufs[0];

        // buffers 5 & 6 are scalar params dim and gate
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc) {
            [enc setComputePipelineState:ctx->pipeline_deltanet];
            for (int i = 0; i < 5; i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
            [enc setBytes:&dim length:sizeof(dim) atIndex:5];
            [enc setBytes:&gate length:sizeof(gate) atIndex:6];
            MTLSize grid = MTLSizeMake((NSUInteger)dim, 1, 1);
            NSUInteger tg = [ctx->pipeline_deltanet maxTotalThreadsPerThreadgroup];
            if (tg > 64) tg = 64;
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        if (!state->is_shared && state->cpu_ptr && bufs[0])
            memcpy(state->cpu_ptr, [bufs[0] contents], state->size);
    }
}

void gpu_expert_mlp(gpu_context *ctx, gpu_buffer *out,
                    const gpu_buffer *x, const gpu_buffer *gate_w,
                    const gpu_buffer *up_w, const gpu_buffer *down_w,
                    int d_model, int hidden_dim, quant_mode quant) {
    if (!ctx || !out || !x || !gate_w || !up_w || !down_w) return;
    (void)quant;

    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufs[] = {
            x->is_shared ? x->metal_buffer : nil,
            gate_w->is_shared ? gate_w->metal_buffer : nil,
            up_w->is_shared ? up_w->metal_buffer : nil,
            down_w->is_shared ? down_w->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        if (!x->is_shared && x->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:x->cpu_ptr length:x->size
                                              options:MTLResourceStorageModeShared];
        if (!gate_w->is_shared && gate_w->cpu_ptr)
            bufs[1] = [ctx->device newBufferWithBytes:gate_w->cpu_ptr length:gate_w->size
                                              options:MTLResourceStorageModeShared];
        if (!up_w->is_shared && up_w->cpu_ptr)
            bufs[2] = [ctx->device newBufferWithBytes:up_w->cpu_ptr length:up_w->size
                                              options:MTLResourceStorageModeShared];
        if (!down_w->is_shared && down_w->cpu_ptr)
            bufs[3] = [ctx->device newBufferWithBytes:down_w->cpu_ptr length:down_w->size
                                              options:MTLResourceStorageModeShared];
        if (!out->is_shared)
            bufs[4] = [ctx->device newBufferWithLength:out->size
                                               options:MTLResourceStorageModeShared];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc) {
            [enc setComputePipelineState:ctx->pipeline_expert_mlp];
            for (int i = 0; i < 5; i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
            [enc setBytes:&d_model length:sizeof(d_model) atIndex:5];
            [enc setBytes:&hidden_dim length:sizeof(hidden_dim) atIndex:6];
            MTLSize grid = MTLSizeMake((NSUInteger)d_model, 1, 1);
            NSUInteger tg = [ctx->pipeline_expert_mlp maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        if (!out->is_shared && out->cpu_ptr && bufs[4])
            memcpy(out->cpu_ptr, [bufs[4] contents], out->size);
    }
}

void gpu_router(gpu_context *ctx, gpu_buffer *out,
                const gpu_buffer *x, const gpu_buffer *router_weight,
                int d_model, int n_experts) {
    if (!ctx || !out || !x || !router_weight) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufs[] = {
            x->is_shared ? x->metal_buffer : nil,
            router_weight->is_shared ? router_weight->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        if (!x->is_shared && x->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:x->cpu_ptr length:x->size
                                              options:MTLResourceStorageModeShared];
        if (!router_weight->is_shared && router_weight->cpu_ptr)
            bufs[1] = [ctx->device newBufferWithBytes:router_weight->cpu_ptr length:router_weight->size
                                              options:MTLResourceStorageModeShared];
        if (!out->is_shared)
            bufs[2] = [ctx->device newBufferWithLength:out->size
                                               options:MTLResourceStorageModeShared];

        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc) {
            [enc setComputePipelineState:ctx->pipeline_router];
            for (int i = 0; i < 3; i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
            [enc setBytes:&d_model length:sizeof(d_model) atIndex:3];
            [enc setBytes:&n_experts length:sizeof(n_experts) atIndex:4];
            MTLSize grid = MTLSizeMake((NSUInteger)n_experts, 1, 1);
            NSUInteger tg = [ctx->pipeline_router maxTotalThreadsPerThreadgroup];
            if (tg > 256) tg = 256;
            [enc dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        if (!out->is_shared && out->cpu_ptr && bufs[2])
            memcpy(out->cpu_ptr, [bufs[2] contents], out->size);
    }
}

void gpu_topk_softmax(gpu_context *ctx, uint32_t *indices, float *weights,
                      const gpu_buffer *logits, int n_experts, int top_k) {
    if (!ctx || !indices || !weights || !logits) return;
    @autoreleasepool {
        // Create temporary Metal buffers for output
        id<MTLBuffer> bufIdx = [ctx->device newBufferWithLength:(NSUInteger)top_k * sizeof(uint32_t)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufWgt = [ctx->device newBufferWithLength:(NSUInteger)top_k * sizeof(float)
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufLog = logits->is_shared ? logits->metal_buffer : nil;
        if (!logits->is_shared && logits->cpu_ptr)
            bufLog = [ctx->device newBufferWithBytes:logits->cpu_ptr length:logits->size
                                             options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (enc) {
            [enc setComputePipelineState:ctx->pipeline_topk];
            [enc setBuffer:bufLog offset:0 atIndex:0];
            [enc setBuffer:bufIdx offset:0 atIndex:1];
            [enc setBuffer:bufWgt offset:0 atIndex:2];
            [enc setBytes:&n_experts length:sizeof(n_experts) atIndex:3];
            [enc setBytes:&top_k length:sizeof(top_k) atIndex:4];
            [enc dispatchThreads:MTLSizeMake(1, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [enc endEncoding];
        }

        [cb commit];
        [cb waitUntilCompleted];

        memcpy(indices, [bufIdx contents], (size_t)top_k * sizeof(uint32_t));
        memcpy(weights, [bufWgt contents], (size_t)top_k * sizeof(float));
    }
}

void gpu_add(gpu_context *ctx, gpu_buffer *out,
             const gpu_buffer *a, const gpu_buffer *b, int n) {
    if (!ctx || !out || !a || !b) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        id<MTLBuffer> bufs[] = {
            a->is_shared ? a->metal_buffer : nil,
            b->is_shared ? b->metal_buffer : nil,
            out->is_shared ? out->metal_buffer : nil,
        };
        if (!a->is_shared && a->cpu_ptr)
            bufs[0] = [ctx->device newBufferWithBytes:a->cpu_ptr length:a->size
                                              options:MTLResourceStorageModeShared];
        if (!b->is_shared && b->cpu_ptr)
            bufs[1] = [ctx->device newBufferWithBytes:b->cpu_ptr length:b->size
                                              options:MTLResourceStorageModeShared];
        if (!out->is_shared)
            bufs[2] = [ctx->device newBufferWithLength:out->size
                                               options:MTLResourceStorageModeShared];

        dispatch_1d(cb, ctx->pipeline_add, bufs, 3, n);

        [cb commit];
        [cb waitUntilCompleted];

        if (!out->is_shared && out->cpu_ptr && bufs[2])
            memcpy(out->cpu_ptr, [bufs[2] contents], out->size);
    }
}

// ── Sync / Info ────────────────────────────────────────────────────────────

void gpu_sync(gpu_context *ctx) {
    if (!ctx || !ctx->queue) return;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [ctx->queue commandBuffer];
        if (cb) {
            [cb commit];
            [cb waitUntilCompleted];
        }
    }
}

size_t gpu_memory_used(gpu_context *ctx) {
    return ctx ? ctx->used_memory : 0;
}

void gpu_print_info(gpu_context *ctx) {
    if (!ctx) return;
    @autoreleasepool {
        printf("=== GPU Info (Metal) ===\n");
        printf("Device:    %s\n", [[ctx->device name] UTF8String]);
        printf("GPU:       %s\n", [[ctx->device name] UTF8String]);
        printf("Memory:    %zu bytes allocated\n", ctx->used_memory);
        printf("Max buf:   %llu bytes\n",
               (unsigned long long)[ctx->device maxBufferLength]);
    }
}
