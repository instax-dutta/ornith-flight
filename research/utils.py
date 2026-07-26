"""Helper utilities for parameter optimization.

Extracted common patterns to reduce duplication and improve testability.
"""

from typing import List, Tuple, Dict, Any
from constants import (
    N_LAYERS, N_EXPERTS_PER_LAYER, EXPERT_SIZE_INT4_MB,
    DEFAULT_POWER_LAW, TRACE_SIZE_LARGE
)
from bench.routing_profile import generate_synthetic_trace


def generate_flat_trace(
    n_tokens: int = TRACE_SIZE_LARGE,
    power: float = DEFAULT_POWER_LAW
) -> List[Tuple[int, int, List[int]]]:
    """Generate and flatten a synthetic routing trace for testing.

    Args:
        n_tokens: Number of tokens to generate
        power: Power-law exponent for routing distribution

    Returns:
        Flattened trace as list of (token_step, layer, expert_ids) tuples
    """
    trace = generate_synthetic_trace(n_tokens, N_LAYERS, N_EXPERTS_PER_LAYER, power)
    return [(t, layer, ids) for t, step in enumerate(trace) for layer, ids in step]


def validate_trace(trace: List[Any]) -> None:
    """Validate that a trace is non-empty and well-formed.

    Args:
        trace: Trace to validate

    Raises:
        ValueError: If trace is empty or malformed
    """
    if not trace or len(trace) == 0:
        raise ValueError("Trace must contain at least one token")


def validate_cache_config(cache_mb: float, expert_size_mb: float) -> None:
    """Validate that cache configuration is viable.

    Args:
        cache_mb: Total cache size in MB
        expert_size_mb: Size of one expert in MB

    Raises:
        ValueError: If cache is too small to hold even one expert
    """
    if cache_mb < expert_size_mb:
        raise ValueError(
            f"Cache size ({cache_mb} MB) is smaller than one expert "
            f"({expert_size_mb} MB)"
        )


def calculate_coverage(experts_cached: int, total_experts: int) -> float:
    """Calculate what percentage of experts fit in cache.

    Args:
        experts_cached: Number of experts that fit in cache
        total_experts: Total number of experts in model

    Returns:
        Coverage as a fraction (0.0 to 1.0)
    """
    return experts_cached / total_experts if total_experts > 0 else 0.0


def estimate_hit_rate_from_coverage(coverage: float) -> float:
    """Estimate cache hit rate from coverage using empirical model.

    Based on power-law routing distribution with exponent ~1.5.
    Top 1% of experts handle ~20% of traffic.

    Args:
        coverage: Fraction of experts cached (0.0 to 1.0)

    Returns:
        Estimated hit rate (bounded between 0.1% and 30%)
    """
    if coverage < 0.01:
        base_rate = coverage * 20
    else:
        base_rate = 0.20 + (coverage - 0.01) * 5

    # Bound between realistic limits
    return max(0.001, min(0.30, base_rate))


def format_config_summary(config: Dict[str, Any]) -> str:
    """Format configuration dictionary as human-readable string.

    Args:
        config: Configuration dictionary

    Returns:
        Formatted multi-line string
    """
    lines = []
    lines.append(f"Device: {config.get('device', 'unknown')}")

    if 'memory' in config:
        mem = config['memory']
        lines.append(f"Cache: {mem.get('total_cache_mb', 0)} MB")
        lines.append(f"  Hot-store: {mem.get('hot_store_experts', 0)} experts "
                    f"({mem.get('hot_store_mb', 0):.0f} MB)")
        lines.append(f"  LRU: {mem.get('lru_cache_mb', 0):.0f} MB")

    if 'performance' in config:
        perf = config['performance']
        lines.append(f"Expected: {perf.get('expected_hit_rate', 0):.1%} hit rate, "
                    f"{perf.get('expected_decode_tps', 0):.2f} TPS")

    return "\n".join(lines)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is zero
    """
    return numerator / denominator if denominator != 0 else default


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp a value between min and max.

    Args:
        value: Value to clamp
        min_val: Minimum bound
        max_val: Maximum bound

    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))
