"""LRU cache simulation for routed experts."""

from collections import OrderedDict


class ExpertLRUCache:
    """
    Least-recently-used cache for MoE expert weights.

    Experts are loaded from disk when requested (cache miss) and evicted
    by LRU policy when capacity is exceeded.
    """

    def __init__(self, capacity_bytes, expert_size_bytes, name="default"):
        self.capacity = capacity_bytes
        self.expert_size = expert_size_bytes
        self.max_experts = capacity_bytes // expert_size_bytes
        self.name = name
        self._cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, expert_id):
        if expert_id in self._cache:
            self._cache.move_to_end(expert_id)
            self.hits += 1
            return True
        self.misses += 1
        self._load(expert_id)
        return False

    def _load(self, expert_id):
        if len(self._cache) >= self.max_experts:
            self._cache.popitem(last=False)
            self.evictions += 1
        self._cache[expert_id] = True

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    @property
    def usage(self):
        return len(self._cache) / self.max_experts if self.max_experts > 0 else 0.0

    def stats(self):
        return {
            "name": self.name,
            "capacity_experts": self.max_experts,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": self.hit_rate,
            "usage": self.usage,
        }


class TieredCache:
    """
    Two-tier cache: small pinned hot-store + larger LRU.

    Hot-store experts are never evicted. LRU evicts normally.
    """

    def __init__(self, hot_capacity, lru_capacity, expert_size):
        self.hot = ExpertLRUCache(hot_capacity, expert_size, "hot-store")
        self.lru = ExpertLRUCache(lru_capacity, expert_size, "lru")
        self.expert_size = expert_size

    def promote(self, expert_id):
        if expert_id not in self.hot._cache and len(self.hot._cache) < self.hot.max_experts:
            self.hot._cache[expert_id] = True
            self.lru._cache.pop(expert_id, None)

    def get(self, expert_id):
        # Check hot-store first (without recording miss)
        if expert_id in self.hot._cache:
            self.hot._cache.move_to_end(expert_id)
            self.hot.hits += 1
            return "hot"

        # Check LRU cache
        if expert_id in self.lru._cache:
            self.lru._cache.move_to_end(expert_id)
            self.lru.hits += 1
            return "lru"

        # Miss - load into LRU
        self.lru.misses += 1
        if len(self.lru._cache) >= self.lru.max_experts:
            self.lru._cache.popitem(last=False)
            self.lru.evictions += 1
        self.lru._cache[expert_id] = True
        return "miss"

    def stats(self):
        return {"hot": self.hot.stats(), "lru": self.lru.stats()}


def simulate_trace(trace, cache):
    """
    Run a routing trace through a cache and return hit rates.

    trace: list of (token_step, layer, [expert_ids]) events
    """
    for _, _, expert_ids in trace:
        for eid in expert_ids:
            cache.get(eid)
    return cache.stats()
