// RED: Memory/cache system tests.
// Tests: LRU eviction, hot-store pinning, tiered lookup (hot before LRU), stats.

#include "test.h"
#include "memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    { "lru_eviction_order",    test_lru_eviction_order },
    { "hotstore_pin",          test_hotstore_pin },
    { "tiered_hot_first",      test_tiered_hot_first },
    { "stats_tracking",        test_stats_tracking },
    { "lru_access_refresh",    test_lru_access_refresh },
    { "empty_cache_miss",      test_empty_cache_miss },
};

RUN_TESTS(tests)
