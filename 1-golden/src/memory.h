// SPDX-License-Identifier: MIT
// Memory/cache system — minimal tiered LRU + hot-store for MoE expert streaming.

#ifndef ORNITH_MEMORY_H
#define ORNITH_MEMORY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef struct memory_manager memory_manager;

typedef struct {
    int    hot_store_experts;  // 0 = no hot-store
    int    lru_capacity;       // max LRU entries
    size_t expert_size_bytes;
    size_t non_routed_size;
    int    async_io_threads;
    int    prefetch_queue_depth;
    bool   double_buffering;
    int    lookahead_layers;
} memory_config;

typedef struct {
    uint64_t hot_hits;
    uint64_t hot_misses;
    uint64_t lru_hits;
    uint64_t lru_misses;
    uint64_t evictions;
    uint64_t prefetches_issued;
    uint64_t prefetches_completed;
    uint64_t io_bytes_read;
    uint64_t io_stall_ns;
    double   hot_hit_rate;
    double   lru_hit_rate;
    double   total_hit_rate;
    size_t   hot_usage;
    size_t   lru_usage;
} cache_stats;

memory_manager *memory_init(const memory_config *cfg);
void memory_destroy(memory_manager *mm);

bool memory_hot_promote(memory_manager *mm, uint32_t expert_id);

const char *memory_get(memory_manager *mm, uint32_t expert_id, void **data);
cache_stats memory_get_stats(memory_manager *mm);
void memory_reset_stats(memory_manager *mm);

memory_config memory_config_m2(void);
memory_config memory_config_pc(void);

#endif
