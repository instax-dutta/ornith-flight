// RED: Memory/cache system tests.
// Tests: LRU eviction, hot-store pinning, tiered lookup (hot before LRU), stats.

#include "test.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>  // usleep

// ── Helper: create a memory manager with small capacity for testing ───────────

static memory_manager *make_test_mm(int hot_slots, int lru_slots) {
    memory_config cfg;
    cfg.hot_store_experts   = hot_slots;
    cfg.lru_capacity        = lru_slots;
    cfg.expert_size_bytes   = 64;       // tiny size for testing
    cfg.non_routed_size     = 1024;
    cfg.async_io_threads    = 0;        // no background I/O for tests
    cfg.prefetch_queue_depth = 4;
    cfg.double_buffering    = false;
    cfg.lookahead_layers    = 1;
    return memory_init(&cfg);
}

// ── Test 1: LRU eviction order ───────────────────────────────────────────────

static test_result test_lru_eviction_order(void) {
    // LRU capacity = 2. Add 3 experts. First-added should be evicted.
    memory_manager *mm = make_test_mm(0, 2);   // 0 hot, 2 LRU
    test_not_null(mm, "memory_init");

    void *data;
    const char *r;

    // Add expert 0, 1, 2
    r = memory_get(mm, 0, &data);    // miss, added
    r = memory_get(mm, 1, &data);    // miss, added
    r = memory_get(mm, 2, &data);    // miss, evicts 0, adds 2

    // Verify 1 and 2 are still in cache (check these first — checking 0 would evict)
    r = memory_get(mm, 1, &data);
    test_assert(strcmp(r, "lru") == 0, "expert 1 still in LRU");

    r = memory_get(mm, 2, &data);
    test_assert(strcmp(r, "lru") == 0, "expert 2 still in LRU");

    // Now verify 0 was evicted (this miss-triggers another eviction, that's fine)
    r = memory_get(mm, 0, &data);
    test_assert(strcmp(r, "miss") == 0, "expert 0 was evicted (LRU)");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 2: Hot-store pinning (never evicted) ────────────────────────────────

static test_result test_hotstore_pin(void) {
    // Hot-store capacity = 1, LRU = 1.
    // Promote expert 99 to hot-store.
    // Add experts 0,1,2 to LRU (which will evict). Expert 99 should remain.
    memory_manager *mm = make_test_mm(1, 1);
    test_not_null(mm, "memory_init");

    bool ok = memory_hot_promote(mm, 99);
    test_assert(ok, "promote expert 99 to hot-store");

    void *data;
    const char *r;

    // Fill LRU and cause evictions
    r = memory_get(mm, 0, &data);  test_assert(strcmp(r, "miss") == 0, "miss 0");
    r = memory_get(mm, 1, &data);  test_assert(strcmp(r, "miss") == 0, "miss 1");

    // Expert 99 should still be in hot-store
    r = memory_get(mm, 99, &data);
    test_assert(strcmp(r, "hot") == 0, "expert 99 still in hot-store after LRU churn");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 3: Tiered lookup — hot before LRU ───────────────────────────────────

static test_result test_tiered_hot_first(void) {
    // Promote 0 to hot-store. Then get 0 via LRU (should return "hot", not "lru")
    memory_manager *mm = make_test_mm(1, 2);
    test_not_null(mm, "memory_init");

    memory_hot_promote(mm, 0);

    void *data;
    const char *r = memory_get(mm, 0, &data);
    test_assert(strcmp(r, "hot") == 0, "expert 0 returned from hot-store, not LRU");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 4: Stats tracking ───────────────────────────────────────────────────

static test_result test_stats_tracking(void) {
    memory_manager *mm = make_test_mm(0, 2);
    test_not_null(mm, "memory_init");

    void *data;

    // 2 hits, 2 misses
    memory_get(mm, 0, &data);  // miss
    memory_get(mm, 1, &data);  // miss
    memory_get(mm, 0, &data);  // hit
    memory_get(mm, 1, &data);  // hit

    cache_stats stats = memory_get_stats(mm);

    test_assert(stats.lru_hits == 2, "2 LRU hits");
    test_assert(stats.lru_misses == 2, "2 LRU misses");
    test_assert(stats.hot_hits == 0, "0 hot hits");
    test_assert(stats.hot_misses == 2, "2 hot misses (two initial misses)");
    test_assert(stats.total_hit_rate > 0.49 && stats.total_hit_rate < 0.51,
                "hit rate ~50%");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 5: LRU access refreshes (recently accessed not evicted) ─────────────

static test_result test_lru_access_refresh(void) {
    // LRU = 2. Add 0, 1. Access 0 again (makes it recently used).
    // Add 2. Expert 1 should be evicted (oldest), 0 should remain.
    memory_manager *mm = make_test_mm(0, 2);
    test_not_null(mm, "memory_init");

    void *data;
    memory_get(mm, 0, &data);  // miss
    memory_get(mm, 1, &data);  // miss
    memory_get(mm, 0, &data);  // hit — makes 0 recently used

    memory_get(mm, 2, &data);  // miss — should evict 1 (oldest)

    const char *r0 = memory_get(mm, 0, &data);
    const char *r1 = memory_get(mm, 1, &data);

    test_assert(strcmp(r0, "lru") == 0, "expert 0 still in cache (refreshed)");
    test_assert(strcmp(r1, "miss") == 0, "expert 1 evicted (oldest)");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 7: memory_set stores data after miss ─────────────────────────────

static test_result test_memory_set_put(void) {
    // Create cache, get a miss (creates slot), then set data, then verify get returns it.
    memory_manager *mm = make_test_mm(0, 2);
    test_not_null(mm, "memory_init");

    // Miss creates the slot
    void *data;
    const char *r = memory_get(mm, 42, &data);
    test_assert(strcmp(r, "miss") == 0, "first get is miss");
    test_null(data, "data is NULL on miss");

    // Now store data into the slot
    float expected[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    bool ok = memory_set(mm, 42, expected, sizeof(expected));
    test_assert(ok, "memory_set succeeds");

    // Next get should return "lru" with the correct data
    r = memory_get(mm, 42, &data);
    test_assert(strcmp(r, "lru") == 0, "second get is lru hit");
    test_not_null(data, "data is not NULL after set");
    float *got = (float *)data;
    test_assert(got[0] == 1.0f, "data[0] == 1.0");
    test_assert(got[1] == 2.0f, "data[1] == 2.0");
    test_assert(got[2] == 3.0f, "data[2] == 3.0");
    test_assert(got[3] == 4.0f, "data[3] == 4.0");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 8: memory_set on hot-store entry ────────────────────────────────────

static test_result test_memory_set_hotstore(void) {
    // Promote to hot-store, then set data, verify get returns "hot" with data.
    memory_manager *mm = make_test_mm(1, 2);
    test_not_null(mm, "memory_init");

    bool ok = memory_hot_promote(mm, 99);
    test_assert(ok, "promote expert 99");

    // Set data on hot entry
    float expected[2] = {3.14f, 2.71f};
    ok = memory_set(mm, 99, expected, sizeof(expected));
    test_assert(ok, "memory_set on hot entry");

    // Get should return "hot" with the data
    void *data;
    const char *r = memory_get(mm, 99, &data);
    test_assert(strcmp(r, "hot") == 0, "get returns hot");
    test_not_null(data, "data is not NULL");
    float *got = (float *)data;
    test_assert(got[0] == 3.14f, "data[0] == 3.14");
    test_assert(got[1] == 2.71f, "data[1] == 2.71");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 9: Eviction frees old data ─────────────────────────────────────────

static test_result test_memory_eviction_frees_data(void) {
    // LRU=1 with data. Adding new expert should evict old one and free its data.
    memory_manager *mm = make_test_mm(0, 1);
    test_not_null(mm, "memory_init");

    void *data;
    float val[1] = {42.0f};
    memory_get(mm, 0, &data);                // miss → LRU gets expert 0
    memory_set(mm, 0, val, sizeof(val));      // store data for slot 0

    // Expert 0 is in LRU with data
    const char *r = memory_get(mm, 0, &data);
    test_assert(strcmp(r, "lru") == 0, "expert 0 is in LRU before eviction");
    test_assert(*(float *)data == 42.0f, "expert 0 data intact before eviction");

    // Add expert 1 → evicts expert 0, frees its data
    memory_get(mm, 1, &data);                // miss → evicts 0, adds 1

    // Expert 0 should be evicted
    r = memory_get(mm, 0, &data);
    test_assert(strcmp(r, "miss") == 0, "evicted expert returns miss");
    test_null(data, "data is NULL for evicted expert");

    memory_destroy(mm);
    return TEST_PASS;
}

// ════════════════════════════════════════════════════════════════════════════
// Async Prefetch Tests
// ════════════════════════════════════════════════════════════════════════════

// Shared state for prefetch job testing
typedef struct {
    int  input;
    int  result;
    bool done;
} prefetch_test_state;

static void test_prefetch_job_func(void *arg) {
    prefetch_test_state *s = (prefetch_test_state *)arg;
    if (s) {
        // Simulate I/O work: compute something
        s->result = s->input * 2;
        s->done = true;
    }
}

// ── Test 10: Submit and poll prefetch job ────────────────────────────────────

static test_result test_prefetch_submit_poll(void) {
    // Create memory manager with 1 async I/O thread
    memory_config cfg;
    cfg.hot_store_experts    = 0;
    cfg.lru_capacity         = 4;
    cfg.expert_size_bytes    = 64;
    cfg.non_routed_size      = 1024;
    cfg.async_io_threads     = 1;   // enable background worker
    cfg.prefetch_queue_depth = 8;
    cfg.double_buffering     = false;
    cfg.lookahead_layers     = 1;
    memory_manager *mm = memory_init(&cfg);
    test_not_null(mm, "memory_init with 1 worker");

    // Submit a prefetch job
    prefetch_test_state state;
    state.input = 21;
    state.result = 0;
    state.done = false;

    int job_id = memory_prefetch_submit(mm, test_prefetch_job_func, &state);
    test_assert(job_id >= 0, "prefetch submit returns valid job ID");

    // Poll until complete (should complete quickly)
    int polls = 0;
    while (!memory_prefetch_poll(mm, job_id) && polls < 1000) {
        usleep(100);
        polls++;
    }
    test_assert(polls < 1000, "prefetch completed within poll limit");
    test_assert(state.done, "job func was called (done flag set)");
    test_assert(state.result == 42, "job computed correct result (21 * 2 = 42)");

    // Stats should show 1 prefetch issued and completed
    cache_stats stats = memory_get_stats(mm);
    test_assert(stats.prefetches_issued == 1, "stats: 1 prefetch issued");
    test_assert(stats.prefetches_completed == 1, "stats: 1 prefetch completed");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 11: Multiple prefetch jobs ──────────────────────────────────────────

static void test_multi_job_func(void *arg) {
    int *counter = (int *)arg;
    if (counter) (*counter)++;
}

static test_result test_prefetch_multiple_jobs(void) {
    memory_config cfg;
    cfg.hot_store_experts    = 0;
    cfg.lru_capacity         = 4;
    cfg.expert_size_bytes    = 64;
    cfg.non_routed_size      = 1024;
    cfg.async_io_threads     = 2;   // 2 workers
    cfg.prefetch_queue_depth = 16;
    cfg.double_buffering     = false;
    cfg.lookahead_layers     = 1;
    memory_manager *mm = memory_init(&cfg);
    test_not_null(mm, "memory_init with 2 workers");

    int counter = 0;
    int job_ids[10];
    int n_jobs = 10;

    // Submit 10 jobs
    for (int i = 0; i < n_jobs; i++) {
        job_ids[i] = memory_prefetch_submit(mm, test_multi_job_func, &counter);
        test_assert(job_ids[i] >= 0, "prefetch submit valid");
    }

    // Wait for all to complete
    for (int i = 0; i < n_jobs; i++) {
        memory_prefetch_wait(mm, job_ids[i]);
    }

    test_assert(counter == n_jobs, "all 10 jobs executed (counter = 10)");

    // No more pending jobs
    int pending = memory_prefetch_pending(mm);
    test_assert(pending == 0, "no pending jobs after all complete");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 12: Null/invalid prefetch args ──────────────────────────────────────

static test_result test_prefetch_invalid_args(void) {
    memory_manager *mm = make_test_mm(0, 4);  // 0 threads
    test_not_null(mm, "memory_init");

    // Submit with NULL manager
    int id = memory_prefetch_submit(NULL, test_prefetch_job_func, NULL);
    test_assert(id == -1, "NULL manager returns -1");

    // Submit with NULL function
    int x = 0;
    id = memory_prefetch_submit(mm, NULL, &x);
    test_assert(id == -1, "NULL func returns -1");

    // Poll with invalid ID
    bool ok = memory_prefetch_poll(mm, -1);
    test_assert(!ok, "poll invalid ID returns false");

    // Pending count
    int pending = memory_prefetch_pending(mm);
    test_assert(pending == 0, "no pending jobs");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test 6: Empty cache returns miss ─────────────────────────────────────────

static test_result test_empty_cache_miss(void) {
    memory_manager *mm = make_test_mm(0, 2);
    test_not_null(mm, "memory_init");

    void *data;
    const char *r = memory_get(mm, 999, &data);
    test_assert(strcmp(r, "miss") == 0, "expert 999 not in empty cache");

    cache_stats stats = memory_get_stats(mm);
    test_assert(stats.total_hit_rate == 0.0, "hit rate is 0 for empty cache");

    memory_destroy(mm);
    return TEST_PASS;
}

// ── Test list ────────────────────────────────────────────────────────────────

static test_entry tests[] = {
    { "lru_eviction_order",     test_lru_eviction_order },
    { "hotstore_pin",           test_hotstore_pin },
    { "tiered_hot_first",       test_tiered_hot_first },
    { "stats_tracking",         test_stats_tracking },
    { "lru_access_refresh",     test_lru_access_refresh },
    { "empty_cache_miss",       test_empty_cache_miss },
    { "memory_set_put",         test_memory_set_put },
    { "memory_set_hotstore",    test_memory_set_hotstore },
    { "memory_eviction_frees_data", test_memory_eviction_frees_data },

    { "prefetch_submit_poll",   test_prefetch_submit_poll },
    { "prefetch_multiple_jobs", test_prefetch_multiple_jobs },
    { "prefetch_invalid_args",  test_prefetch_invalid_args },
};

RUN_TESTS(tests)
