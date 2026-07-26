// SPDX-License-Identifier: MIT
// Memory/cache system — minimal tiered LRU + hot-store for MoE expert streaming.

#ifndef ORNITH_MEMORY_H
#define ORNITH_MEMORY_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Maximum number of concurrent prefetch jobs
#define ORNITH_MAX_PREFETCH_JOBS 256

typedef struct memory_manager memory_manager;

// ── Prefetch job type ────────────────────────────────────────────────────────
// A background worker thread calls func(arg). The caller polls or waits
// for completion via memory_prefetch_poll() / memory_prefetch_wait().
typedef void (*prefetch_func_t)(void *arg);

typedef struct {
    prefetch_func_t func;
    void           *arg;
    volatile bool   completed;
    volatile bool   active;   // whether this slot is occupied
} prefetch_job;

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

// Store data for a cached expert (must already exist in cache via memory_get miss or memory_hot_promote).
// Returns true on success, false if the expert_id is not in any cache slot.
bool memory_set(memory_manager *mm, uint32_t expert_id, const void *data, size_t size);

cache_stats memory_get_stats(memory_manager *mm);
void memory_reset_stats(memory_manager *mm);

memory_config memory_config_m2(void);
memory_config memory_config_pc(void);

// ── Async prefetch API ───────────────────────────────────────────────────────
// Submit a job to be executed by a background I/O thread.
// Returns a job ID (>= 0) on success, -1 on failure (queue full).
// Storage for the prefetch_job struct is managed internally — the caller
// must NOT free the arg until the job completes.
int memory_prefetch_submit(memory_manager *mm, prefetch_func_t func, void *arg);

// Check if a previously submitted prefetch job has completed.
// Returns true if completed, false if still running or invalid ID.
bool memory_prefetch_poll(memory_manager *mm, int job_id);

// Block until a prefetch job completes. Returns immediately if already done.
void memory_prefetch_wait(memory_manager *mm, int job_id);

// Number of active (submitted but not completed) prefetch jobs.
int memory_prefetch_pending(memory_manager *mm);

// Check if async I/O is enabled (has worker threads).
bool memory_async_enabled(const memory_manager *mm);

// Number of lookahead layers configured for prefetch.
int memory_lookahead_layers(const memory_manager *mm);

#endif
