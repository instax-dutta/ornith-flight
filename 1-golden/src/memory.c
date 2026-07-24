// SPDX-License-Identifier: MIT
// Memory/cache system — LRU cache + hot-store pinning. Minimal for TDD tests.

#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

#define MAX_HOT 256
#define MAX_LRU 512

typedef struct {
    uint32_t expert_id;
    bool     occupied;
    uint64_t last_access;
} slot;

struct memory_manager {
    memory_config cfg;
    slot    hot_store[MAX_HOT];
    int     hot_count;
    slot    lru_cache[MAX_LRU];
    int     lru_count;
    uint64_t tick;

    // Stats
    cache_stats stats;
};

// ── Helpers ──────────────────────────────────────────────────────────────────

static int find_slot(slot *arr, int count, uint32_t id) {
    for (int i = 0; i < count; i++) {
        if (arr[i].occupied && arr[i].expert_id == id) return i;
    }
    return -1;
}

static int find_lru(slot *arr, int count) {
    int oldest = -1;
    uint64_t oldest_time = UINT64_MAX;
    for (int i = 0; i < count; i++) {
        if (arr[i].occupied && arr[i].last_access < oldest_time) {
            oldest_time = arr[i].last_access;
            oldest = i;
        }
    }
    return oldest;
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

memory_manager *memory_init(const memory_config *cfg) {
    if (!cfg) return NULL;
    memory_manager *mm = (memory_manager *)calloc(1, sizeof(memory_manager));
    if (!mm) return NULL;
    mm->cfg = *cfg;
    mm->tick = 0;
    return mm;
}

void memory_destroy(memory_manager *mm) {
    if (!mm) return;
    memset(mm, 0, sizeof(*mm));
    free(mm);
}

// ── Hot-store ────────────────────────────────────────────────────────────────

bool memory_hot_promote(memory_manager *mm, uint32_t expert_id) {
    if (!mm) return false;

    // Already in hot-store
    if (find_slot(mm->hot_store, mm->hot_count, expert_id) >= 0)
        return true;

    if (mm->hot_count >= mm->cfg.hot_store_experts)
        return false;

    slot *s = &mm->hot_store[mm->hot_count++];
    s->expert_id = expert_id;
    s->occupied = true;
    s->last_access = mm->tick++;
    return true;
}

// ── Cache access ─────────────────────────────────────────────────────────────

const char *memory_get(memory_manager *mm, uint32_t expert_id, void **data) {
    if (!mm || !data) return "miss";
    *data = NULL;
    uint64_t now = mm->tick++;

    // 1. Check hot-store
    int idx = find_slot(mm->hot_store, mm->hot_count, expert_id);
    if (idx >= 0) {
        mm->hot_store[idx].last_access = now;
        mm->stats.hot_hits++;
        *data = (void *)(uintptr_t)expert_id;  // dummy pointer for test
        return "hot";
    }

    // 2. Check LRU cache
    idx = find_slot(mm->lru_cache, mm->lru_count, expert_id);
    if (idx >= 0) {
        mm->lru_cache[idx].last_access = now;
        mm->stats.lru_hits++;
        *data = (void *)(uintptr_t)expert_id;
        return "lru";
    }

    // 3. Miss — count hot miss (not found in hot-store)
    mm->stats.hot_misses++;
    mm->stats.lru_misses++;

    if (mm->lru_count < mm->cfg.lru_capacity) {
        // Add new slot
        slot *s = &mm->lru_cache[mm->lru_count++];
        s->expert_id = expert_id;
        s->occupied = true;
        s->last_access = now;
    } else {
        // Evict LRU
        int victim = find_lru(mm->lru_cache, mm->lru_count);
        if (victim >= 0) {
            mm->stats.evictions++;
            mm->lru_cache[victim].expert_id = expert_id;
            mm->lru_cache[victim].last_access = now;
        }
    }

    return "miss";
}

cache_stats memory_get_stats(memory_manager *mm) {
    cache_stats s;
    memset(&s, 0, sizeof(s));
    if (!mm) return s;

    s = mm->stats;

    // Each memory_get call = one unique access, so:
    // total = hot_hits(unique) + lru_hits(unique) + lru_misses(unique misses)
    uint64_t total_accesses = s.hot_hits + s.lru_hits + s.lru_misses;

    s.hot_hit_rate = 0.0;  // not computed from aggregate counters
    s.lru_hit_rate = s.lru_hits + s.lru_misses > 0
        ? (double)s.lru_hits / (s.lru_hits + s.lru_misses) : 0.0;
    s.total_hit_rate = total_accesses > 0
        ? (double)(s.hot_hits + s.lru_hits) / total_accesses : 0.0;
    s.hot_usage = (size_t)mm->hot_count;
    s.lru_usage = (size_t)mm->lru_count;

    return s;
}

void memory_reset_stats(memory_manager *mm) {
    if (!mm) return;
    memset(&mm->stats, 0, sizeof(mm->stats));
}

memory_config memory_config_m2(void) {
    memory_config c;
    c.hot_store_experts = 50;
    c.lru_capacity = 16;
    c.expert_size_bytes = (size_t)(62.0 * 1024 * 1024);
    c.non_routed_size = (size_t)(1.5 * 1024 * 1024 * 1024);
    c.async_io_threads = 2;
    c.prefetch_queue_depth = 8;
    c.double_buffering = true;
    c.lookahead_layers = 1;
    return c;
}

memory_config memory_config_pc(void) {
    memory_config c;
    c.hot_store_experts = 50;
    c.lru_capacity = 49;
    c.expert_size_bytes = (size_t)(62.0 * 1024 * 1024);
    c.non_routed_size = (size_t)(1.5 * 1024 * 1024 * 1024);
    c.async_io_threads = 4;
    c.prefetch_queue_depth = 16;
    c.double_buffering = true;
    c.lookahead_layers = 1;
    return c;
}
