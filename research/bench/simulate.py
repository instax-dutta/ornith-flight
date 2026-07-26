"""Throughput simulation for Ornith 35B MoE on target hardware.

Models the full inference pipeline with lookahead prefetch overlap.

Usage:
    python bench/simulate.py --device m2 --hit-rate 0.6 --cache 4096
    python bench/simulate.py --device pc  --hit-rate 0.8 --cache 6144
"""

import argparse

DEVICES = {
    "m2": {
        "gpu_bandwidth_gbs": 100,
        "ssd_bandwidth_gbs": 3.0,
        "ram_gb": 8,
    },
    "pc": {
        "gpu_bandwidth_gbs": 250,
        "ssd_bandwidth_gbs": 3.0,  # same SSD bottleneck as M2
        "ram_gb": 16,
    },
}

ACTIVE_EXPERTS = 8
N_LAYERS = 28
ACTIVE_PARAMS = 3e9
EXPERT_SIZE_BYTES = 62.0 * 1024 * 1024


def simulate(device, hit_rate, cache_mb, prompt_len=128, gen_len=50):
    cfg = DEVICES[device]
    gpu_bw = cfg["gpu_bandwidth_gbs"] * 1e9
    ssd_bw = cfg["ssd_bandwidth_gbs"] * 1e9
    cache_experts = int(cache_mb * 1024 * 1024 / EXPERT_SIZE_BYTES)

    compute_per_token = ACTIVE_PARAMS / gpu_bw
    compute_per_layer = compute_per_token / N_LAYERS
    miss_experts_per_layer = ACTIVE_EXPERTS * (1 - hit_rate)
    io_per_miss = EXPERT_SIZE_BYTES / ssd_bw
    io_per_layer = miss_experts_per_layer * io_per_miss

    # With async lookahead, I/O of layer N+1 overlaps compute of layer N
    # Stall occurs only when I/O per layer > compute per layer
    stall_per_layer = max(0, io_per_layer - compute_per_layer)
    stall_per_token = stall_per_layer * N_LAYERS

    # Prefill: first layer has no overlap; rest overlap
    prefill_stall = N_LAYERS * stall_per_layer + (io_per_layer - stall_per_layer)
    prefill_overlap = prompt_len * max(stall_per_layer, 0)
    prefill_time = prompt_len * (N_LAYERS * compute_per_layer) + prefill_stall

    decode_stall = max(0, stall_per_token)
    decode_per_token = compute_per_token + decode_stall

    ttft = prefill_time / 1  # no overlap during cold-start prefill
    decode_tps = 1.0 / decode_per_token if decode_per_token > 0 else 999
    total_time = ttft + gen_len * decode_per_token

    return {
        "device": device,
        "hit_rate": hit_rate,
        "cache_experts": cache_experts,
        "compute_per_token_ms": compute_per_token * 1000,
        "io_per_token_ms": (io_per_layer * N_LAYERS) * 1000,
        "overlapped_io_per_token_ms": max(0, io_per_layer * N_LAYERS - compute_per_token) * 1000,
        "stall_per_token_ms": decode_stall * 1000,
        "ttft_ms": ttft * 1000,
        "decode_tps": decode_tps,
        "total_time_s": total_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["m2", "pc"], default="m2")
    parser.add_argument("--hit-rate", type=float, default=0.6)
    parser.add_argument("--cache", type=float, default=4096)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--gen", type=int, default=50)
    args = parser.parse_args()

    result = simulate(args.device, args.hit_rate, args.cache,
                      args.prompt, args.gen)

    print(f"=== Ornith 35B MoE Simulation ===")
    print(f"Device:          {result['device']}")
    print(f"Cache:           {args.cache} MB ({result['cache_experts']} experts)")
    print(f"Hit rate:        {result['hit_rate']:.0%}")
    print(f"Compute/token:   {result['compute_per_token_ms']:.1f} ms")
    print(f"I/O/token:       {result['io_per_token_ms']:.1f} ms")
    print(f"Overlapped I/O:  {result['overlapped_io_per_token_ms']:.1f} ms hidden")
    print(f"Stall/token:     {result['stall_per_token_ms']:.1f} ms")
    print(f"TTFT:            {result['ttft_ms']:.0f} ms ({result['ttft_ms']/1000:.1f}s)")
    print(f"Decode TPS:      {result['decode_tps']:.1f} tok/s")
    print(f"Total time:      {result['total_time_s']:.1f}s for {args.prompt}+{args.gen} tokens")
