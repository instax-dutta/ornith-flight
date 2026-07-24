// SPDX-License-Identifier: MIT
// Inference loop — token generation with KV cache management.

#ifndef ORNITH_INFERENCE_H
#define ORNITH_INFERENCE_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "model.h"
#include "memory.h"

typedef struct inference_engine inference_engine;

typedef struct {
    int      max_tokens;
    int      top_k;
    float    temperature;
    float    repeat_penalty;
    int      n_threads;
    bool     stream;
    char     stop_token[64];
    int      seed;
} generation_params;

static inline generation_params generation_params_default(void) {
    generation_params p;
    p.max_tokens     = 512;
    p.top_k          = 40;
    p.temperature    = 0.7f;
    p.repeat_penalty = 1.0f;
    p.n_threads      = 1;
    p.stream         = true;
    p.stop_token[0]  = '\0';
    p.seed           = -1;
    return p;
}

typedef struct {
    uint32_t *tokens;
    int       n_tokens;
    int       n_prompt_tokens;
    double    t_prefill_ms;
    double    t_decode_ms;
    double    tokens_per_sec;
    double    ttft_ms;
    bool      truncated;
    char     *text;
} generation_result;

inference_engine *inference_init(ornith_model *model);
void inference_destroy(inference_engine *engine);

void inference_reset(inference_engine *engine);

generation_result *inference_generate(inference_engine *engine,
                                      const char *prompt,
                                      const generation_params *params);

generation_result *inference_generate_tokens(inference_engine *engine,
                                             const uint32_t *prompt_tokens,
                                             int n_prompt_tokens,
                                             const generation_params *params);

void inference_result_free(generation_result *result);
void inference_print_stats(const generation_result *result);

uint32_t inference_eos_token(inference_engine *engine);

#endif
