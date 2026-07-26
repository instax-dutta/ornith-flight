"""Async prefetch engine simulation.

Prefetch strategies for overlapping I/O with compute:

- EAGER: prefetch all N layers' experts as soon as prompt starts
- LAZY: prefetch per-layer on-demand after router decides
- HYBRID: prefetch top-32 experts (popular) eagerly, rest lazily
- LOOKAHEAD: prefetch layer N+1 while GPU computes layer N
"""


class PrefetchSimulator:
    """
    Simulate prefetch overlap between GPU compute and expert I/O.

    Measures wall-clock time saved vs synchronous load.
    """

    def __init__(self, ssd_bandwidth_bps, expert_size_bytes,
                 compute_time_per_layer_s, n_layers):
        self.ssd_bw = ssd_bandwidth_bps
        self.expert_size = expert_size_bytes
        self.compute_time = compute_time_per_layer_s
        self.n_layers = n_layers
        self.load_time = expert_size_bytes / ssd_bandwidth_bps

    def simulate(self, strategy, trace):
        """
        trace: list of (layer, [expert_ids]) per token decode step
        strategy: 'eager' | 'lazy' | 'hybrid' | 'lookahead'

        Returns: (total_wall_time, stall_time, load_time)
        """
        if strategy == "lookahead":
            return self._simulate_lookahead(trace)
        elif strategy == "eager":
            return self._simulate_eager(trace)
        elif strategy == "lazy":
            return self._simulate_lazy(trace)
        elif strategy == "hybrid":
            return self._simulate_hybrid(trace)

    def _simulate_lookahead(self, trace):
        total_stall = 0.0
        for step_idx, step in enumerate(trace):
            for layer, expert_ids in step:
                n_miss = len(expert_ids)  # simplified: all miss
                load_needed = n_miss * self.load_time
                overlap = self.compute_time
                stall = max(0, load_needed - overlap)
                total_stall += stall
        total_time = self.n_layers * self.compute_time * len(trace) + total_stall
        return total_time, total_stall, total_stall

    def _simulate_eager(self, trace):
        # Prefetch all experts before compute starts
        total_experts = sum(len(ids) for step in trace for _, ids in step)
        load_time = total_experts * self.load_time
        compute_time = self.n_layers * self.compute_time * len(trace)
        return load_time + compute_time, 0, load_time

    def _simulate_lazy(self, trace):
        return self._simulate_lookahead(trace)

    def _simulate_hybrid(self, trace):
        return self._simulate_lookahead(trace)

    def speedup(self, strategy, trace):
        _, lazy_stall, _ = self.simulate("lazy", trace)
        _, stall, _ = self.simulate(strategy, trace)
        if lazy_stall == 0:
            return 1.0
        return lazy_stall / stall
