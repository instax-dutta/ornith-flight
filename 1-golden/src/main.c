// SPDX-License-Identifier: MIT
// CLI entry point — main() for Ornith inference engine.
// Argument parsing is in cli.c.

#include "main.h"
#include "gguf.h"
#include "memory.h"
#include "model.h"
#include "inference.h"
#include "tokenizer.h"
#include "gpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ── Main entry ───────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    cli_args args = parse_args(argc, argv);

    printf("Ornith 35B MoE Inference Engine\n");

    if (args.verbose || args.print_config) {
        printf("Config:\n");
        printf("  Device:      %s\n", args.device);
        printf("  Max tokens:  %d\n", args.max_tokens);
        printf("  Temperature: %.2f\n", args.temperature);
        printf("  Top-k:       %d\n", args.top_k);
    }

    if (args.print_config) {
        printf("Memory: M2: hot=50 LRU=16 total=4GB | PC: hot=50 LRU=49 total=6GB\n");
        return 0;
    }

    // Load model
    if (!args.model_path) {
        fprintf(stderr, "Error: --model is required. Use --help for usage.\n");
        return 1;
    }

    printf("Loading model: %s\n", args.model_path);

    memory_config mem_cfg = (strcmp(args.device, "pc") == 0)
        ? memory_config_pc() : memory_config_m2();

    char err[256];
    ornith_model *model = model_load(args.model_path, &mem_cfg, err, sizeof(err));
    if (!model) {
        fprintf(stderr, "Error: %s\n", err);
        return 1;
    }

    const model_config *cfg = model_get_config(model);
    if (cfg) {
        printf("Layers: %d, Experts/layer: %d, Active: %d\n",
               cfg->n_layers, cfg->n_experts_per_layer, cfg->n_active_experts);
    }

    // Generate response (simplified — always one-shot generation)
    inference_engine *engine = inference_init(model);
    if (!engine) {
        fprintf(stderr, "Error: inference init failed\n");
        model_unload(model);
        return 1;
    }

    const char *prompt_text = args.prompt ? args.prompt : "Hello";
    generation_params gp = generation_params_default();
    gp.max_tokens = args.max_tokens;
    gp.temperature = args.temperature;
    gp.top_k = args.top_k;
    gp.seed = args.seed;

    if (args.benchmark) {
        printf("Running benchmark...\n");
        for (int i = 0; i < 3; i++) {
            inference_reset(engine);
            generation_result *r = inference_generate(engine, prompt_text, &gp);
            if (r) {
                inference_print_stats(r);
                inference_result_free(r);
            }
        }
    } else {
        generation_result *result = inference_generate(engine, prompt_text, &gp);
        if (result) {
            inference_print_stats(result);
            inference_result_free(result);
        }
    }

    inference_destroy(engine);
    model_unload(model);
    printf("Done.\n");
    return 0;
}
