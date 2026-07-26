// SPDX-License-Identifier: MIT
// Memory/cache system — LRU cache + hot-store pinning with real data storage.
// TDD: RED — write failing test first, then implement.

#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>  // usleep

#define MAX_HOT 256
#define MAX_LRU 512

typedef struct {
    uint32_t expert_id;
    bool     occupied;
    uint64_t last_access;
    void    *data;          // pointer to cached expert weight data (NULL until memory_set)
    size_t   data_size;     // actual size of stored data
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

    // ── Async prefetch engine ──────────────────────────────────────────
    prefetch_job  jobs[ORNITH_MAX_PREFETCH_JOBS];
    int           job_write;       // next slot to write
    int           job_read;        // next slot to read by workers
    int           job_count;       // number of active jobs
    pthread_t    *workers;
    int           n_workers;
    pthread_mutex_t job_mutex;
    pthread_cond_t  job_cond;
    bool         shutdown;
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

// Free data buffer in a slot, leaving the slot metadata intact.
// After calling, slot->data = NULL and slot->data_size = 0.
static void slot_free_data(slot *s) {
    if (s && s->data) {
        free(s->data);
        s->data = NULL;
        s->data_size = 0;
    }
}

// ── Async prefetch worker ───────────────────────────────────────────────────

static void *prefetch_worker(void *arg) {
    memory_manager *mm = (memory_manager *)arg;
    if (!mm) return NULL;

    while (1) {
        int my_idx;
        prefetch_func_t local_func;
        void *local_arg;

        pthread_mutex_lock(&mm->job_mutex);

        // Wait for a job or shutdown signal
        while (mm->job_count == 0 && !mm->shutdown) {
            pthread_cond_wait(&mm->job_cond, &mm->job_mutex);
        }

        if (mm->shutdown) {
            pthread_mutex_unlock(&mm->job_mutex);
            return NULL;
        }

        // Dequeue: copy func/arg to locals while under lock, but keep slot
        // reserved by NOT decrementing job_count until after execution.
        my_idx = mm->job_read;
        prefetch_job *job = &mm->jobs[my_idx];
        local_func = job->func;
        local_arg = job->arg;
        job->active = true;
        job->completed = false;
        mm->job_read = (mm->job_read + 1) % ORNITH_MAX_PREFETCH_JOBS;
        // job_count NOT decremented yet — slot remains reserved
        pthread_mutex_unlock(&mm->job_mutex);

        // Execute the job using local copies (safe even if slot gets reused)
        if (local_func) {
            local_func(local_arg);
        }

        // Release slot atomically: mark completed, then decrement count
        pthread_mutex_lock(&mm->job_mutex);
        mm->jobs[my_idx].completed = true;
        mm->jobs[my_idx].active = false;
        mm->job_count--;
        mm->stats.prefetches_completed++;
        pthread_mutex_unlock(&mm->job_mutex);
    }

    return NULL;
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

memory_manager *memory_init(const memory_config *cfg) {
    if (!cfg) return NULL;
    memory_manager *mm = (memory_manager *)calloc(1, sizeof(memory_manager));
    if (!mm) return NULL;
    mm->cfg = *cfg;
    mm->tick = 0;

    // Initialize prefetch engine
    mm->job_write = 0;
    mm->job_read = 0;
    mm->job_count = 0;
    mm->shutdown = false;
    pthread_mutex_init(&mm->job_mutex, NULL);
    pthread_cond_init(&mm->job_cond, NULL);

    int n_threads = cfg->async_io_threads;
    if (n_threads > 0) {
        mm->workers = (pthread_t *)calloc((size_t)n_threads, sizeof(pthread_t));
        if (mm->workers) {
            mm->n_workers = n_threads;
            for (int i = 0; i < n_threads; i++) {
                pthread_create(&mm->workers[i], NULL, prefetch_worker, mm);
            }
        }
    }

    return mm;
}

void memory_destroy(memory_manager *mm) {
    if (!mm) return;

    // Shutdown prefetch workers
    if (mm->n_workers > 0 && mm->workers) {
        pthread_mutex_lock(&mm->job_mutex);
        mm->shutdown = true;
        pthread_cond_broadcast(&mm->job_cond);  // wake all workers
        pthread_mutex_unlock(&mm->job_mutex);

        for (int i = 0; i < mm->n_workers; i++) {
            pthread_join(mm->workers[i], NULL);
        }
        free(mm->workers);
        mm->workers = NULL;
        mm->n_workers = 0;
    }

    pthread_mutex_destroy(&mm->job_mutex);
    pthread_cond_destroy(&mm->job_cond);

    // Free all data buffers in hot-store
    for (int i = 0; i < mm->hot_count; i++) {
        slot_free_data(&mm->hot_store[i]);
    }
    // Free all data buffers in LRU
    for (int i = 0; i < mm->lru_count; i++) {
        slot_free_data(&mm->lru_cache[i]);
    }

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
    s->data = NULL;    // data filled later via memory_set
    s->data_size = 0;
    return true;
}

// ── Async prefetch API ──────────────────────────────────────────────────────

int memory_prefetch_submit(memory_manager *mm, prefetch_func_t func, void *arg) {
    if (!mm || !func) return -1;
    if (mm->n_workers == 0) return -1;  // no workers to execute the job

    pthread_mutex_lock(&mm->job_mutex);

    if (mm->job_count >= ORNITH_MAX_PREFETCH_JOBS) {
        pthread_mutex_unlock(&mm->job_mutex);
        return -1;  // queue full
    }

    int id = mm->job_write;
    mm->jobs[id].func = func;
    mm->jobs[id].arg = arg;
    mm->jobs[id].completed = false;
    mm->jobs[id].active = false;
    mm->job_write = (id + 1) % ORNITH_MAX_PREFETCH_JOBS;
    mm->job_count++;
    mm->stats.prefetches_issued++;

    pthread_cond_signal(&mm->job_cond);
    pthread_mutex_unlock(&mm->job_mutex);

    return id;
}

bool memory_prefetch_poll(memory_manager *mm, int job_id) {
    if (!mm || job_id < 0 || job_id >= ORNITH_MAX_PREFETCH_JOBS) return false;
    // Volatile read — safe without mutex for a completion flag
    return mm->jobs[job_id].completed;
}

void memory_prefetch_wait(memory_manager *mm, int job_id) {
    if (!mm || job_id < 0 || job_id >= ORNITH_MAX_PREFETCH_JOBS) return;
    // Spin-wait (the job is doing I/O, will complete quickly)
    while (!mm->jobs[job_id].completed) {
        // Brief sleep to let the worker thread make progress
        usleep(100);
    }
}

int memory_prefetch_pending(memory_manager *mm) {
    if (!mm) return 0;
    pthread_mutex_lock(&mm->job_mutex);
    int count = mm->job_count;
    pthread_mutex_unlock(&mm->job_mutex);
    return count;
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
        *data = mm->hot_store[idx].data;
        return "hot";
    }

    // 2. Check LRU cache
    idx = find_slot(mm->lru_cache, mm->lru_count, expert_id);
    if (idx >= 0) {
        mm->lru_cache[idx].last_access = now;
        mm->stats.lru_hits++;
        *data = mm->lru_cache[idx].data;
        return "lru";
    }

    // 3. Miss — count
    mm->stats.hot_misses++;
    mm->stats.lru_misses++;

    if (mm->lru_count < mm->cfg.lru_capacity) {
        // Add new slot (data starts NULL — caller calls memory_set to fill it)
        slot *s = &mm->lru_cache[mm->lru_count++];
        s->expert_id = expert_id;
        s->occupied = true;
        s->last_access = now;
        s->data = NULL;
        s->data_size = 0;
    } else {
        // Evict LRU
        int victim = find_lru(mm->lru_cache, mm->lru_count);
        if (victim >= 0) {
            mm->stats.evictions++;
            slot_free_data(&mm->lru_cache[victim]);  // free old data
            mm->lru_cache[victim].expert_id = expert_id;
            mm->lru_cache[victim].last_access = now;
            mm->lru_cache[victim].data = NULL;
            mm->lru_cache[victim].data_size = 0;
        }
    }

    *data = NULL;
    return "miss";
}

// ── Cache data storage ───────────────────────────────────────────────────────

bool memory_set(memory_manager *mm, uint32_t expert_id, const void *data, size_t size) {
    if (!mm || !data || size == 0) return false;

    // Check hot-store first
    int idx = find_slot(mm->hot_store, mm->hot_count, expert_id);
    if (idx >= 0) {
        slot *s = &mm->hot_store[idx];
        // Reallocate if current buffer is too small or NULL
        if (s->data_size < size) {
            void *new_data = realloc(s->data, size);
            if (!new_data) return false;
            s->data = new_data;
        }
        memcpy(s->data, data, size);
        s->data_size = size;
        return true;
    }

    // Check LRU
    idx = find_slot(mm->lru_cache, mm->lru_count, expert_id);
    if (idx >= 0) {
        slot *s = &mm->lru_cache[idx];
        if (s->data_size < size) {
            void *new_data = realloc(s->data, size);
            if (!new_data) return false;
            s->data = new_data;
        }
        memcpy(s->data, data, size);
        s->data_size = size;
        return true;
    }

    // Expert not found in any cache slot — caller must call memory_get first
    return false;
}

// ── Prefetch config queries ─────────────────────────────────────────────────

bool memory_async_enabled(const memory_manager *mm) {
    return mm && mm->n_workers > 0;
}

int memory_lookahead_layers(const memory_manager *mm) {
    return mm ? mm->cfg.lookahead_layers : 0;
}

// ── Stats ────────────────────────────────────────────────────────────────────

cache_stats memory_get_stats(memory_manager *mm) {
    cache_stats s;
    memset(&s, 0, sizeof(s));
    if (!mm) return s;

    s = mm->stats;

    uint64_t total_accesses = s.hot_hits + s.lru_hits + s.lru_misses;

    s.hot_hit_rate = 0.0;
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

// ── Default configs ──────────────────────────────────────────────────────────

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
