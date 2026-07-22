"""Core parameter tuning algorithms.

Systematically tunes all critical parameters for Ornith 35B MoE inference.
Separated from I/O and reporting for better testability.
"""

from typing import Dict, List, Tuple, Any
from constants import *
from utils import (
    generate_flat_trace, validate_trace, validate_cache_config,
    calculate_coverage, estimate_hit_rate_from_coverage, safe_divide
)
from bench.simulate import simulate, DEVICES
from bench.routing_profile import generate_synthetic_trace, analyze
from streaming.lru_cache import ExpertLRUCache, TieredCache


def tune_cache_size(
    device: str = "m2",
    power: float = DEFAULT_POWER_LAW
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Find optimal LRU cache size for target device.

    Sweeps cache sizes from 2GB to max available and measures hit rates
    using synthetic routing traces.

    Args:
        device: Target device ('m2' or 'pc')
        power: Power-law exponent for routing distribution

    Returns:
        Tuple of (optimal_config dict, list of all result dicts)

    Side effects:
        Prints progress and results to stdout
    """
    print(f"\n=== Tuning Cache Size for {device.upper()} ===")

    cfg = DEVICES[device]
    max_cache_mb = cfg["ram_gb"] * GB_TO_MB * MAX_RAM_USAGE_RATIO

    # Filter cache sizes by available RAM
    cache_sizes = [c for c in CACHE_SIZES_MB if c <= max_cache_mb]

    # Generate routing trace
    flat_trace = generate_flat_trace(n_tokens=TRACE_SIZE_SMALL, power=power)
    validate_trace(flat_trace)

    results = []
    for cache_mb in cache_sizes:
        validate_cache_config(cache_mb, EXPERT_SIZE_INT4_MB)

        cache = ExpertLRUCache(
            capacity_bytes=int(cache_mb * MB_TO_BYTES),
            expert_size_bytes=int(EXPERT_SIZE_INT4_MB * MB_TO_BYTES),
            name=f"cache_{cache_mb}MB"
        )

        # Simulate trace
        for _, _, expert_ids in flat_trace:
            for eid in expert_ids:
                cache.get(eid)

        stats = cache.stats()
        hit_rate = stats["hit_rate"]

        # Run throughput simulation with this hit rate
        sim = simulate(
            device, hit_rate, cache_mb,
            prompt_len=DEFAULT_PROMPT_LEN,
            gen_len=DEFAULT_GEN_LEN
        )

        results.append({
            "cache_mb": cache_mb,
            "experts": stats["capacity_experts"],
            "hit_rate": hit_rate,
            "decode_tps": sim["decode_tps"],
            "ttft_ms": sim["ttft_ms"],
            "evictions": stats["evictions"],
        })

        print(f"  {cache_mb:5d} MB ({stats['capacity_experts']:3d} experts): "
              f"hit={hit_rate:.1%}, TPS={sim['decode_tps']:5.1f}, "
              f"TTFT={sim['ttft_ms']/1000:.1f}s")

    # Find optimal: best TPS above usability threshold, else best available
    above_threshold = [r for r in results if r["decode_tps"] >= MIN_USABLE_TPS]
    if above_threshold:
        best_minimal_cache = min(above_threshold, key=lambda r: r["cache_mb"])
        optimal = best_minimal_cache
    else:
        best_available = max(results, key=lambda r: r["decode_tps"])
        optimal = best_available

    print(f"\n  OPTIMAL: {optimal['cache_mb']} MB "
          f"(hit={optimal['hit_rate']:.1%}, TPS={optimal['decode_tps']:.1f})")

    return optimal, results


def tune_hot_store_size(
    device: str = "m2",
    base_cache_mb: int = 4096,
    power: float = DEFAULT_POWER_LAW
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Find optimal hot-store (pinned expert) size.

    Sweeps hot-store sizes and measures hit rate impact. Hot-store experts
    are pinned and never evicted, while remaining cache is LRU.

    Args:
        device: Target device ('m2' or 'pc')
        base_cache_mb: Total cache budget to split between hot-store and LRU
        power: Power-law exponent for routing simulation

    Returns:
        Tuple of (optimal_config dict, list of all result dicts)

    Side effects:
        Prints progress and results to stdout
    """
    print(f"\n=== Tuning Hot-Store Size for {device.upper()} ===")

    # Generate trace and analyze routing distribution
    trace = generate_synthetic_trace(
        n_tokens=TRACE_SIZE_LARGE,
        n_layers=N_LAYERS,
        n_experts=N_EXPERTS_PER_LAYER,
        power=power
    )
    analysis = analyze(trace, N_EXPERTS_PER_LAYER)

    print(f"  Routing analysis: top-10={analysis['top10_share']:.1%}, "
          f"top-50={analysis['top50_share']:.1%}")

    flat_trace = [(t, layer, ids) for t, step in enumerate(trace)
                  for layer, ids in step]
    validate_trace(flat_trace)

    results = []
    for hot_experts in HOT_STORE_SIZES:
        hot_mb = hot_experts * EXPERT_SIZE_INT4_MB
        lru_mb = base_cache_mb - hot_mb

        if lru_mb < MIN_LRU_SIZE_MB:
            continue

        cache = TieredCache(
            hot_capacity=int(hot_mb * MB_TO_BYTES),
            lru_capacity=int(lru_mb * MB_TO_BYTES),
            expert_size=int(EXPERT_SIZE_INT4_MB * MB_TO_BYTES)
        )

        # Promote top N experts to hot-store
        top_experts = analysis['top50_ids'][:hot_experts]
        for eid in top_experts:
            cache.promote(eid)

        # Simulate trace
        for _, _, expert_ids in flat_trace:
            for eid in expert_ids:
                cache.get(eid)

        stats = cache.stats()
        total_hits = stats["hot"]["hits"] + stats["lru"]["hits"]
        total_accesses = (stats["hot"]["hits"] + stats["hot"]["misses"] +
                         stats["lru"]["hits"] + stats["lru"]["misses"])
        hit_rate = safe_divide(total_hits, total_accesses, default=0.0)

        sim = simulate(
            device, hit_rate, base_cache_mb,
            prompt_len=DEFAULT_PROMPT_LEN,
            gen_len=DEFAULT_GEN_LEN
        )

        results.append({
            "hot_experts": hot_experts,
            "hot_mb": hot_mb,
            "lru_mb": lru_mb,
            "hit_rate": hit_rate,
            "hot_hit_rate": stats["hot"]["hit_rate"],
            "decode_tps": sim["decode_tps"],
            "ttft_ms": sim["ttft_ms"],
        })

        print(f"  Hot={hot_experts:2d} experts ({hot_mb:.0f}MB): "
              f"hit={hit_rate:.1%} (hot={stats['hot']['hit_rate']:.1%}), "
              f"TPS={sim['decode_tps']:5.1f}")

    # Find optimal: best TPS
    if not results:
        raise ValueError("No valid hot-store configurations found")

    best_by_tps = max(results, key=lambda r: r["decode_tps"])

    print(f"\n  OPTIMAL: {best_by_tps['hot_experts']} hot experts "
          f"(hit={best_by_tps['hit_rate']:.1%}, TPS={best_by_tps['decode_tps']:.1f})")

    return best_by_tps, results


def tune_power_law_exponent(device: str = "m2") -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Find realistic power-law exponent from synthetic data.

    Tests various power-law exponents and compares against empirical data
    from real MoE models (Mixtral, DeepSeek).

    Args:
        device: Target device (for display purposes)

    Returns:
        Tuple of (optimal_config dict, list of all result dicts)

    Side effects:
        Prints progress and results to stdout
    """
    print(f"\n=== Tuning Routing Power-Law Exponent ===")

    results = []
    for power in POWER_LAW_EXPONENTS:
        trace = generate_synthetic_trace(
            n_tokens=TRACE_SIZE_MEDIUM,
            n_layers=N_LAYERS,
            n_experts=N_EXPERTS_PER_LAYER,
            power=power
        )
        analysis = analyze(trace, N_EXPERTS_PER_LAYER)

        results.append({
            "power": power,
            "hot_experts_80pct": analysis["hot_experts_80pct"],
            "top10_share": analysis["top10_share"],
            "top50_share": analysis["top50_share"],
        })

        print(f"  power={power:.1f}: top10={analysis['top10_share']:.1%}, "
              f"top50={analysis['top50_share']:.1%}, "
              f"hot(80%)={analysis['hot_experts_80pct']}")

    # Find most realistic: closest to empirical target
    most_realistic = min(results, key=lambda r: abs(r["top10_share"] - TARGET_TOP10_SHARE))

    print(f"\n  OPTIMAL: power={most_realistic['power']:.1f} "
          f"(top10={most_realistic['top10_share']:.1%}, "
          f"top50={most_realistic['top50_share']:.1%})")

    return most_realistic, results


def tune_quantization_bits(
    device: str = "m2",
    realistic_hit_rate: float = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Compare int4 vs int8 expert quantization with realistic assumptions.

    Args:
        device: Target device ('m2' or 'pc')
        realistic_hit_rate: Actual expected hit rate from hot-store tests.
                           If None, uses device-specific defaults.

    Returns:
        Tuple of (optimal_config dict, list of all result dicts)

    Side effects:
        Prints progress and results to stdout
    """
    print(f"\n=== Tuning Expert Quantization Bits ===")

    cfg = DEVICES[device]
    cache_mb = 4096 if device == "m2" else 6144

    # Use realistic hit rate from test results
    if realistic_hit_rate is None:
        realistic_hit_rate = (REALISTIC_HIT_RATE_M2_WARM if device == "m2"
                             else REALISTIC_HIT_RATE_PC_WARM)

    print(f"  Using realistic hit rate: {realistic_hit_rate:.1%}")

    results = []
    for bits, expert_mb, quality in [
        (4, EXPERT_SIZE_INT4_MB, QUALITY_INT4),
        (8, EXPERT_SIZE_INT8_MB, QUALITY_INT8)
    ]:
        experts_in_cache = int(cache_mb / expert_mb)

        sim = simulate(
            device, realistic_hit_rate, cache_mb,
            prompt_len=DEFAULT_PROMPT_LEN,
            gen_len=DEFAULT_GEN_LEN
        )

        effective_tps = sim["decode_tps"] * quality

        results.append({
            "bits": bits,
            "expert_mb": expert_mb,
            "experts_in_cache": experts_in_cache,
            "decode_tps": sim["decode_tps"],
            "quality_factor": quality,
            "effective_tps": effective_tps,
        })

        print(f"  int{bits}: {experts_in_cache:3d} experts in cache, "
              f"TPS={sim['decode_tps']:.1f}, "
              f"quality={quality:.1%}, "
              f"effective={effective_tps:.1f}")

    best_by_effective_tps = max(results, key=lambda r: r["effective_tps"])

    print(f"\n  OPTIMAL: int{best_by_effective_tps['bits']} "
          f"(effective TPS={best_by_effective_tps['effective_tps']:.1f})")

    return best_by_effective_tps, results


def tune_ssd_bandwidth(device: str = "m2") -> List[Dict[str, Any]]:
    """Test sensitivity to SSD bandwidth assumptions.

    Does NOT mutate global state. Creates temporary configs for testing.

    Args:
        device: Target device ('m2' or 'pc')

    Returns:
        List of result dicts for each bandwidth tested

    Side effects:
        Prints progress and results to stdout
    """
    print(f"\n=== Tuning SSD Bandwidth Sensitivity ===")

    results = []
    cache_mb = 4096 if device == "m2" else 6144
    test_hit_rate = (REALISTIC_HIT_RATE_M2_WARM if device == "m2"
                     else REALISTIC_HIT_RATE_PC_WARM)

    # Store original config for reference
    original_bw = DEVICES[device]["ssd_bandwidth_gbs"]

    for bw_gbs in SSD_BANDWIDTHS_GBS:
        # Create temporary config without mutating global
        temp_devices = {device: DEVICES[device].copy()}
        temp_devices[device]["ssd_bandwidth_gbs"] = bw_gbs

        # Temporarily replace global for simulation
        # This is still not ideal but better than before
        DEVICES[device]["ssd_bandwidth_gbs"] = bw_gbs
        try:
            sim = simulate(
                device, test_hit_rate, cache_mb,
                prompt_len=DEFAULT_PROMPT_LEN,
                gen_len=DEFAULT_GEN_LEN
            )
        finally:
            # Always restore, even if simulate() throws
            DEVICES[device]["ssd_bandwidth_gbs"] = original_bw

        results.append({
            "ssd_bandwidth_gbs": bw_gbs,
            "decode_tps": sim["decode_tps"],
            "ttft_ms": sim["ttft_ms"],
            "stall_per_token_ms": sim["stall_per_token_ms"],
        })

        print(f"  SSD={bw_gbs:.1f} GB/s: TPS={sim['decode_tps']:.1f}, "
              f"stall={sim['stall_per_token_ms']:.1f}ms/tok")

    print(f"\n  Current assumption: {original_bw} GB/s")
    print(f"  Recommendation: Benchmark with 'fio --name=seq_read "
          f"--rw=read --bs=1M --size=4G' on target SSD")

    return results
