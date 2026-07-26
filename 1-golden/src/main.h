// SPDX-License-Identifier: MIT
// CLI interface — argument parsing, testable separately from main().

#ifndef ORNITH_MAIN_H
#define ORNITH_MAIN_H

#include <stdbool.h>

typedef struct {
    char    *model_path;
    char    *prompt;
    char    *system_prompt;
    int      max_tokens;
    float    temperature;
    int      top_k;
    int      seed;
    int      n_threads;
    bool     interactive;
    bool     benchmark;
    bool     verbose;
    bool     print_config;
    bool     dry_run;
    char     device[16];
} cli_args;

cli_args parse_args(int argc, char **argv);

#endif
