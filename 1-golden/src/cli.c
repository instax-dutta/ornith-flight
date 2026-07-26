// SPDX-License-Identifier: MIT
// CLI argument parsing — separated from main() so tests can link against it.

#include "main.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cli_args parse_args(int argc, char **argv) {
    cli_args args;
    memset(&args, 0, sizeof(args));
    args.max_tokens    = 512;
    args.temperature   = 0.7f;
    args.top_k         = 40;
    args.seed          = -1;
    args.n_threads     = 1;
    strncpy(args.device, "m2", sizeof(args.device));

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 || strcmp(argv[i], "-m") == 0) {
            if (i + 1 < argc) args.model_path = argv[++i];
        } else if (strcmp(argv[i], "--prompt") == 0 || strcmp(argv[i], "-p") == 0) {
            if (i + 1 < argc) args.prompt = argv[++i];
        } else if (strcmp(argv[i], "--system") == 0 || strcmp(argv[i], "-s") == 0) {
            if (i + 1 < argc) args.system_prompt = argv[++i];
        } else if (strcmp(argv[i], "--max-tokens") == 0 || strcmp(argv[i], "-n") == 0) {
            if (i + 1 < argc) args.max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temperature") == 0 || strcmp(argv[i], "-t") == 0) {
            if (i + 1 < argc) args.temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "-k") == 0) {
            if (i + 1 < argc) args.top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0) {
            if (i + 1 < argc) args.seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) args.n_threads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--interactive") == 0 || strcmp(argv[i], "-i") == 0) {
            args.interactive = true;
        } else if (strcmp(argv[i], "--benchmark") == 0 || strcmp(argv[i], "-b") == 0) {
            args.benchmark = true;
        } else if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            args.verbose = true;
        } else if (strcmp(argv[i], "--config") == 0 || strcmp(argv[i], "-c") == 0) {
            args.print_config = true;
        } else if (strcmp(argv[i], "--dry-run") == 0) {
            args.dry_run = true;
        } else if (strcmp(argv[i], "--device") == 0 || strcmp(argv[i], "-d") == 0) {
            if (i + 1 < argc) {
                strncpy(args.device, argv[++i], sizeof(args.device) - 1);
            }
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: ornith [options]\n");
            printf("\n");
            printf("Options:\n");
            printf("  -m, --model PATH     GGUF model file path\n");
            printf("  -p, --prompt TEXT    Prompt text\n");
            printf("  -s, --system TEXT    System prompt\n");
            printf("  -n, --max-tokens N   Max tokens to generate (default: 512)\n");
            printf("  -t, --temperature F  Sampling temperature (default: 0.7)\n");
            printf("  -k, --top-k N        Top-k sampling (default: 40)\n");
            printf("      --seed N         Random seed (-1 = time-based)\n");
            printf("      --threads N      CPU threads\n");
            printf("  -i, --interactive    Interactive chat mode\n");
            printf("  -b, --benchmark      Run benchmark\n");
            printf("  -v, --verbose        Verbose output\n");
            printf("  -c, --config         Print configuration\n");
            printf("      --dry-run        Load config and exit without allocating weights\n");
            printf("  -d, --device DEVICE  Target device (m2 or pc)\n");
            printf("  -h, --help           Show this help\n");
            exit(0);
        }
    }

    return args;
}
