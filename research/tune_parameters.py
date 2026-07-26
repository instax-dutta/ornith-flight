"""Comprehensive parameter tuning with test-backed optimization.

Systematically tunes all critical parameters for Ornith 35B MoE inference.
Refactored for better separation of concerns and testability.

Usage:
    python3 tune_parameters.py --device m2 --component all
    python3 tune_parameters.py --device pc --component cache
"""

import argparse
from typing import Dict, Any
from tuner import (
    tune_cache_size,
    tune_hot_store_size,
    tune_power_law_exponent,
    tune_quantization_bits,
    tune_ssd_bandwidth
)
from reporters import (
    generate_optimized_config,
    save_config_to_file,
    print_config,
    print_section_header
)


def run_comprehensive_tuning(device: str = "m2") -> Dict[str, Any]:
    """Run all tuning passes and generate optimized config.

    Args:
        device: Target device ('m2' or 'pc')

    Returns:
        Final optimized configuration dictionary

    Side effects:
        Prints progress to stdout
        Writes config file to disk
    """
    print_section_header(f"COMPREHENSIVE PARAMETER TUNING FOR {device.upper()}")

    # 1. Tune power-law exponent first (establishes routing distribution)
    power_result, _ = tune_power_law_exponent(device)
    optimal_power = power_result["power"]

    # 2. Tune cache size with optimal power-law
    cache_result, _ = tune_cache_size(device, power=optimal_power)
    optimal_cache_mb = cache_result["cache_mb"]

    # 3. Tune hot-store with optimal cache size
    # Pass realistic hit rate from cache tuning to quantization test
    hot_result, _ = tune_hot_store_size(device, optimal_cache_mb, power=optimal_power)
    optimal_hot_experts = hot_result["hot_experts"]

    # 4. Tune quantization with realistic hit rate
    quant_result, _ = tune_quantization_bits(device, realistic_hit_rate=hot_result["hit_rate"])
    optimal_bits = quant_result["bits"]

    # 5. Check SSD sensitivity
    ssd_results = tune_ssd_bandwidth(device)

    # Generate final config
    config = generate_optimized_config(
        device=device,
        optimal_power=optimal_power,
        optimal_cache_mb=optimal_cache_mb,
        optimal_hot_experts=optimal_hot_experts,
        optimal_bits=optimal_bits,
        hot_result=hot_result,
        ssd_results=ssd_results
    )

    print_section_header(f"OPTIMIZED CONFIGURATION FOR {device.upper()}")
    print_config(config)

    # Save to file
    output_path = f"research/config_{device}_optimized.json"
    save_config_to_file(config, output_path)
    print(f"\n  Saved to: {output_path}")

    return config


def main():
    """Command-line interface for parameter tuning."""
    parser = argparse.ArgumentParser(
        description="Tune Ornith 35B MoE parameters with test-backed optimization"
    )
    parser.add_argument(
        "--device",
        choices=["m2", "pc"],
        default="m2",
        help="Target device (m2 for MacBook Air M2, pc for RTX 4060)"
    )
    parser.add_argument(
        "--component",
        choices=["cache", "hotstore", "power", "quant", "ssd", "all"],
        default="all",
        help="Component to tune (default: all)"
    )
    args = parser.parse_args()

    # Run individual component or full suite
    if args.component == "all":
        run_comprehensive_tuning(args.device)
    elif args.component == "cache":
        tune_cache_size(args.device)
    elif args.component == "hotstore":
        tune_hot_store_size(args.device)
    elif args.component == "power":
        tune_power_law_exponent(args.device)
    elif args.component == "quant":
        tune_quantization_bits(args.device)
    elif args.component == "ssd":
        tune_ssd_bandwidth(args.device)


if __name__ == "__main__":
    main()
