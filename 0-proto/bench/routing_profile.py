"""Analyze routing patterns to identify hot experts for hot-store pinning.

Generates synthetic routing traces and identifies frequently-routed experts.
In production, this would consume real Ornith attention/router logits.

Typical MoE routing is power-law: ~20% of experts handle ~80% of tokens.
Hot-store should pin that top 20%.
"""

import argparse
import random


def generate_synthetic_trace(n_tokens, n_layers, n_experts, power=2.0):
    """
    Generate a synthetic routing trace with power-law expert distribution.
    Experts 0-15 are hot (~20% of tokens), rest are cold.

    Note: expert_id is the GLOBAL expert ID across all layers.
    For a model with 256 experts per layer and 28 layers, total experts = 256 * 28 = 7168
    """
    trace = []
    # Power-law weights for experts
    weights = [1.0 / ((i + 1) ** (1.0 / power)) for i in range(n_experts)]
    total = sum(weights)
    probs = [w / total for w in weights]

    for t in range(n_tokens):
        step = []
        for layer in range(n_layers):
            # Each layer has its own set of experts
            # Global expert ID = layer * n_experts + local_expert_id
            selected_local = random.choices(range(n_experts), weights=probs, k=8)
            selected_global = [layer * n_experts + local_id for local_id in selected_local]
            step.append((layer, selected_global))
        trace.append(step)
    return trace


def analyze(trace, n_experts):
    """Analyze routing trace and return statistics.

    Note: Counts are across ALL layers, so total unique experts = n_experts * n_layers
    """
    # Find total unique experts from the trace
    all_expert_ids = set()
    for step in trace:
        for layer, ids in step:
            all_expert_ids.update(ids)

    total_experts = len(all_expert_ids)
    counts = {eid: 0 for eid in all_expert_ids}
    total = 0

    for step in trace:
        for layer, ids in step:
            for eid in ids:
                counts[eid] += 1
                total += 1

    ranked = sorted(counts.items(), key=lambda x: -x[1])
    cumulative = 0
    hot_count = 0
    hot_threshold = int(total * 0.8)
    for rank, (eid, cnt) in enumerate(ranked):
        cumulative += cnt
        if cumulative >= hot_threshold:
            hot_count = rank + 1
            break

    return {
        "total_routing_events": total,
        "total_unique_experts": total_experts,
        "hot_experts_80pct": hot_count,
        "hot_share_pct": hot_count / total_experts * 100,
        "top10_share": sum(c for _, c in ranked[:10]) / total,
        "top50_share": sum(c for _, c in ranked[:50]) / total,
        "top10_ids": [eid for eid, _ in ranked[:10]],
        "top50_ids": [eid for eid, _ in ranked[:50]],
        "histogram": [c for _, c in ranked[:50]],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--power", type=float, default=2.0)
    args = parser.parse_args()

    trace = generate_synthetic_trace(args.tokens, args.layers, args.experts, args.power)
    result = analyze(trace, args.experts)

    print(f"=== Routing Profile ===")
    print(f"Total routing events:   {result['total_routing_events']}")
    print(f"Total unique experts:   {result['total_unique_experts']}")
    print(f"Hot experts (80% rule): {result['hot_experts_80pct']} of {result['total_unique_experts']}")
    print(f"Hot share:              {result['hot_share_pct']:.1f}%")
    print(f"Top 10 expert share:    {result['top10_share']:.1%}")
    print(f"Top 50 expert share:    {result['top50_share']:.1%}")
    print(f"Top 10 IDs:             {result['top10_ids']}")
    print(f"Top 50 IDs:             {result['top50_ids']}")
