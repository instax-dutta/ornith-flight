"""Output formatting and reporting for parameter tuning results.

Handles JSON serialization, file I/O, and formatted output.
"""

import json
from typing import Dict, List, Any
from constants import EXPERT_SIZE_INT4_MB, N_ACTIVE_EXPERTS


def generate_optimized_config(
    device: str,
    optimal_power: float,
    optimal_cache_mb: int,
    optimal_hot_experts: int,
    optimal_bits: int,
    hot_result: Dict[str, Any],
    ssd_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate final optimized configuration dictionary.

    Args:
        device: Target device ('m2' or 'pc')
        optimal_power: Optimal power-law exponent
        optimal_cache_mb: Optimal total cache size in MB
        optimal_hot_experts: Optimal number of hot-store experts
        optimal_bits: Optimal quantization bits (4 or 8)
        hot_result: Results from hot-store tuning
        ssd_results: Results from SSD bandwidth testing

    Returns:
        Configuration dictionary ready for JSON serialization
    """
    from bench.simulate import DEVICES

    config = {
        "device": device,
        "model": "Ornith-35B-MoE",
        "memory": {
            "lru_cache_mb": optimal_cache_mb - (optimal_hot_experts * EXPERT_SIZE_INT4_MB),
            "hot_store_experts": optimal_hot_experts,
            "hot_store_mb": optimal_hot_experts * EXPERT_SIZE_INT4_MB,
            "total_cache_mb": optimal_cache_mb,
        },
        "quantization": {
            "routed_experts": f"int{optimal_bits}",
            "non_routed": "int8",
            "router": "fp16",
            "kv_cache": "fp16",
        },
        "routing": {
            "power_law_exponent": optimal_power,
            "n_active_experts": N_ACTIVE_EXPERTS,
        },
        "performance": {
            "expected_hit_rate": hot_result["hit_rate"],
            "expected_decode_tps": hot_result["decode_tps"],
            "expected_ttft_ms": hot_result["ttft_ms"],
        },
        "hardware": {
            "gpu_bandwidth_gbs": DEVICES[device]["gpu_bandwidth_gbs"],
            "ssd_bandwidth_gbs": DEVICES[device]["ssd_bandwidth_gbs"],
            "ram_gb": DEVICES[device]["ram_gb"],
        },
    }

    return config


def save_config_to_file(config: Dict[str, Any], output_path: str) -> None:
    """Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        output_path: Path to output file

    Side effects:
        Writes JSON file to disk
    """
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)


def print_config(config: Dict[str, Any]) -> None:
    """Pretty-print configuration to stdout.

    Args:
        config: Configuration dictionary

    Side effects:
        Prints to stdout
    """
    print(json.dumps(config, indent=2))


def print_section_header(title: str, width: int = 70) -> None:
    """Print a formatted section header.

    Args:
        title: Section title
        width: Total width of header

    Side effects:
        Prints to stdout
    """
    print(f"\n{'='*width}")
    print(title)
    print('='*width)
