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

// ── Interactive chat helpers ─────────────────────────────────────────────────

#define CHAT_BUF_INIT 4096
#define CHAT_BUF_GROW 2048

typedef struct {
    char  *text;
    size_t len;
    size_t cap;
} chat_buffer;

static chat_buffer *chat_buffer_create(void) {
    chat_buffer *cb = (chat_buffer *)calloc(1, sizeof(chat_buffer));
    if (!cb) return NULL;
    cb->cap = CHAT_BUF_INIT;
    cb->text = (char *)malloc(cb->cap);
    if (!cb->text) { free(cb); return NULL; }
    cb->text[0] = '\0';
    cb->len = 0;
    return cb;
}

static void chat_buffer_free(chat_buffer *cb) {
    if (!cb) return;
    free(cb->text);
    memset(cb, 0, sizeof(*cb));
    free(cb);
}

static void chat_buffer_append(chat_buffer *cb, const char *s) {
    if (!cb || !s) return;
    size_t slen = strlen(s);
    size_t needed = cb->len + slen + 1;
    if (needed > cb->cap) {
        size_t new_cap = cb->cap + (slen > CHAT_BUF_GROW ? slen + CHAT_BUF_GROW : CHAT_BUF_GROW);
        char *tmp = (char *)realloc(cb->text, new_cap);
        if (!tmp) return;
        cb->text = tmp;
        cb->cap = new_cap;
    }
    memcpy(cb->text + cb->len, s, slen);
    cb->len += slen;
    cb->text[cb->len] = '\0';
}

// Read a single line from stdin (dynamically allocated).
// Returns NULL on EOF or error. Caller must free.
static char *read_line(void) {
    size_t cap = 256;
    size_t len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    int c;
    while ((c = getchar()) != EOF && c != '\n') {
        if (len + 1 >= cap) {
            cap *= 2;
            char *tmp = (char *)realloc(buf, cap);
            if (!tmp) { free(buf); return NULL; }
            buf = tmp;
        }
        buf[len++] = (char)c;
    }
    buf[len] = '\0';

    if (len == 0 && c == EOF) {
        free(buf);
        return NULL;
    }
    return buf;
}

// Format the chat history + user input into a single prompt string for the model.
// Uses a simple "User: ...\nAssistant: " format (chat template).
static char *format_chat_prompt(const char *system_prompt,
                                 chat_buffer *history,
                                 const char *user_input) {
    // Build: [system]\n\n[history]\nUser: [input]\nAssistant:
    size_t sys_len = system_prompt ? strlen(system_prompt) : 0;
    size_t hist_len = history ? history->len : 0;
    size_t inp_len = strlen(user_input);

    // "User: " = 6, "\n\nAssistant: " = 14
    size_t total = sys_len + (sys_len > 0 ? 2 : 0)  // system + "\n\n"
                 + hist_len + (hist_len > 0 ? 1 : 0)  // history + "\n"
                 + 6 + inp_len + 12 + 1;              // "User: " + input + "\nAssistant: " + null

    char *prompt = (char *)malloc(total);
    if (!prompt) return NULL;
    prompt[0] = '\0';

    if (system_prompt) {
        strcat(prompt, system_prompt);
        strcat(prompt, "\n\n");
    }
    if (history && history->len > 0) {
        strcat(prompt, history->text);
        strcat(prompt, "\n");
    }
    strcat(prompt, "User: ");
    strcat(prompt, user_input);
    strcat(prompt, "\nAssistant:");

    return prompt;
}

// Interactive chat REPL
static int run_interactive(inference_engine *engine, tokenizer *tok,
                            const char *system_prompt,
                            generation_params *gp) {
    (void)tok;
    printf("\n");
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Ornith Interactive Chat                                ║\n");
    printf("║  Type /exit to quit, /reset to clear history, /help    ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n");
    printf("\n");

    chat_buffer *history = chat_buffer_create();
    if (!history) {
        fprintf(stderr, "Error: failed to allocate chat buffer\n");
        return 1;
    }

    // Save a copy of the system prompt for template formatting
    const char *sys = system_prompt && system_prompt[0]
                      ? system_prompt
                      : "You are Ornith, a helpful AI assistant.";

    int ret = 0;
    while (1) {
        // Print prompt
        printf(">>> ");
        fflush(stdout);

        char *input = read_line();
        if (!input) {
            // EOF (Ctrl+D)
            printf("\n");
            break;
        }

        // Trim trailing whitespace
        {
            size_t ilen = strlen(input);
            while (ilen > 0 && (input[ilen - 1] == ' ' || input[ilen - 1] == '\t'))
                input[--ilen] = '\0';
        }

        if (input[0] == '\0') {
            free(input);
            continue;  // empty line, re-prompt
        }

        // Handle commands
        if (input[0] == '/') {
            if (strcmp(input, "/exit") == 0 || strcmp(input, "/quit") == 0) {
                free(input);
                break;
            } else if (strcmp(input, "/reset") == 0) {
                chat_buffer_free(history);
                history = chat_buffer_create();
                if (!history) { free(input); ret = 1; break; }
                printf("  → Conversation history cleared.\n\n");
                fflush(stdout);
                free(input);
                continue;
            } else if (strcmp(input, "/help") == 0) {
                printf("  Commands:\n");
                printf("    /exit, /quit   Exit interactive mode\n");
                printf("    /reset         Clear conversation history\n");
                printf("    /help          Show this help\n");
                printf("\n");
                fflush(stdout);
                free(input);
                continue;
            }
            // Unknown command — treat as normal input
        }

        // Save user input for history before freeing
        char *input_copy = strdup(input);

        // Format prompt: system + history + "User: input\nAssistant:"
        char *formatted = format_chat_prompt(sys, history, input);
        free(input);
        if (!formatted) { free(input_copy); ret = 1; break; }

        // Show "thinking" indicator
        printf("\033[90m  Assistant is typing...\033[0m\n");
        fflush(stdout);

        // Generate response
        generation_result *result = inference_generate(engine, formatted, gp);
        free(formatted);

        if (!result) {
            printf("\033[31mError: generation failed\033[0m\n");
            fflush(stdout);
            free(input_copy);
            continue;
        }

        const char *output = result->text ? result->text : "";

        // Print response
        printf("\033[1;34m%s\033[0m", output);
        printf("\n");

        // Print stats in dim text
        printf("\033[90m  [%d tokens, %.1fs, %.2f tok/s]\033[0m\n",
               result->n_tokens - result->n_prompt_tokens,
               result->t_decode_ms / 1000.0,
               result->tokens_per_sec);
        fflush(stdout);

        // Update history with this turn
        chat_buffer_append(history, "User: ");
        chat_buffer_append(history, input_copy);
        chat_buffer_append(history, "\n");
        chat_buffer_append(history, "Assistant: ");
        chat_buffer_append(history, output);

        free(input_copy);
        inference_result_free(result);
    }

    chat_buffer_free(history);
    return ret;
}

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
    if (model && (args.trace || args.verbose)) {
        model_set_trace(model, true);
    }
    if (!model) {
        fprintf(stderr, "Error: %s\n", err);
        return 1;
    }

    if (args.dry_run) {
        printf("Dry run complete. No weights allocated.\n");
        model_unload(model);
        return 0;
    }

    // Load tokenizer from the model's already-open GGUF handle
    // (no need to open a second mmap of the 20 GB file)
    tokenizer *tok = NULL;
    {
        gguf_model *gguf_tok = model_get_gguf(model);
        if (gguf_tok) {
            tok = tokenizer_create_from_gguf(gguf_tok);
        }
        if (tok && args.verbose) {
            printf("Tokenizer: %d tokens, BPE loaded\n",
                   tokenizer_vocab_size(tok));
        } else if (!tok && args.verbose) {
            printf("Tokenizer: not available (GGUF has no merges), using byte fallback\n");
        }
    }

    inference_engine *engine = inference_init(model, tok);
    if (!engine) {
        fprintf(stderr, "Error: inference init failed\n");
        tokenizer_destroy(tok);
        model_unload(model);
        return 1;
    }

    generation_params gp = generation_params_default();
    gp.max_tokens = args.max_tokens;
    gp.temperature = args.temperature;
    gp.top_k = args.top_k;
    gp.seed = args.seed;

    if (args.interactive) {
        int ret = run_interactive(engine, tok, args.system_prompt, &gp);
        inference_destroy(engine);
        tokenizer_destroy(tok);
        model_unload(model);
        return ret;
    }

    const char *prompt_text = args.prompt ? args.prompt : "Hello";

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
