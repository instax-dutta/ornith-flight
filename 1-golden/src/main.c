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
    fflush(stdout);

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
    fflush(stdout);

    memory_config mem_cfg = (strcmp(args.device, "pc") == 0)
        ? memory_config_pc() : memory_config_m2();

    char err[256];
    ornith_model *model = model_load(args.model_path, &mem_cfg, err, sizeof(err), args.verbose, args.dry_run);
    if (!model) {
        fprintf(stderr, "Error: %s\n", err);
        return 1;
    }

    if (args.dry_run) {
        printf("Dry run complete. No weights allocated.\n");
        model_unload(model);
        return 0;
    }

    // Load tokenizer from the model's GGUF
    // The model's gguf handle is accessible via the ornith_model internals.
    // We pass the model_get_config to get the gguf... actually pass model directly.
    // For now, create a tokenizer from the GGUF path directly
    tokenizer *tok = NULL;
    {
        // Open a separate GGUF handle just for tokenizer metadata
        // (the model's gguf is a private member, but we can reuse the path)
        gguf_model *gguf_tok = gguf_open(args.model_path, err, sizeof(err));
        if (gguf_tok) {
            tok = tokenizer_create_from_gguf(gguf_tok);
            if (tok && args.verbose) {
                printf("Tokenizer: %d tokens, %s\n",
                       tokenizer_vocab_size(tok),
                       tok ? "BPE loaded" : "fallback");
            }
            gguf_close(gguf_tok);  // tokenizer keeps its own copies
        }
    }

    inference_engine *engine = inference_init(model, tok);
    if (!engine) {
        fprintf(stderr, "Error: inference init failed\n");
        tokenizer_destroy(tok);
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
                printf("\n=== Generation %d ===\n", i+1);
                inference_print_stats(r);
                inference_result_free(r);
            }
        }
    } else {
        generation_result *result = inference_generate(engine, prompt_text, &gp);
        if (result) {
            printf("\n=== Output ===\n%s\n", result->text ? result->text : "(no output)");
            printf("\n");
            inference_print_stats(result);
            inference_result_free(result);
        }
    }

    inference_destroy(engine);
    tokenizer_destroy(tok);
    model_unload(model);
    printf("Done.\n");
    return 0;
}
