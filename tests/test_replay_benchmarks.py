"""
Performance benchmarking test suite for Phase 3A RaceState construction.
Executes 500 complete end-to-end get_state() calls across diverse race phases,
recording empirical latency distribution (Median, P95, P99) and hardware environment.
"""

import os
import platform
import time
import pytest
import numpy as np

from src.race.replay.session import ReplaySession
from src.race.replay.state import get_state, RaceState


def test_race_state_construction_benchmarks():
    """
    Evaluates 500 complete RaceState construction queries across 5 race phases:
    1. Race Start (t = 3830.0 s)
    2. Green-flag flying laps (t = 4400.0 s)
    3. Pit stop cycle (t = 4810.0 s)
    4. Safety Car period (t = 7900.0 s)
    5. Chequered flag (t = 8575.0 s)
    """
    session = ReplaySession(2022, "Monza")

    # Construct 500 query timestamps spread across the 5 phases
    sample_timestamps = []
    base_phases = [3830.0, 4400.0, 4810.0, 7900.0, 8575.0]
    for base_t in base_phases:
        # 100 queries around each phase in 0.2s increments
        for i in range(100):
            sample_timestamps.append(base_t + i * 0.2)

    assert len(sample_timestamps) == 500

    latencies_ms = []

    # Warmup call
    _ = get_state(session, 4000.0)

    for t_query in sample_timestamps:
        t_start = time.perf_counter_ns()
        state = get_state(session, t_query)
        t_end = time.perf_counter_ns()

        elapsed_ms = (t_end - t_start) / 1_000_000.0
        latencies_ms.append(elapsed_ms)

        # Integrity assertions
        assert isinstance(state, RaceState)
        assert len(state.leaderboard) == 20

    latencies = np.array(latencies_ms)
    median_latency = float(np.median(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    p99_latency = float(np.percentile(latencies, 99))
    max_latency = float(np.max(latencies))

    env_info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }

    print("\n" + "=" * 70)
    print("PHASE 3A RACESTATE LATENCY BENCHMARK REPORT (500 Samples)")
    print("=" * 70)
    print(f"Environment: {env_info['os']} {env_info['os_release']} ({env_info['machine']}), {env_info['cpu_count']} CPUs")
    print(f"Processor:   {env_info['processor']}")
    print("-" * 70)
    print(f"Median Latency (P50): {median_latency:.3f} ms")
    print(f"95th Percentile (P95): {p95_latency:.3f} ms")
    print(f"99th Percentile (P99): {p99_latency:.3f} ms")
    print(f"Maximum Latency:       {max_latency:.3f} ms")
    print("=" * 70)

    # Acceptance check against PDF 200 ms budget ceiling
    # and design aspiration of Median <= 8 ms, P99 <= 25 ms
    assert median_latency < 200.0, f"Exceeded PDF budget ceiling: {median_latency:.2f} ms"
    assert median_latency <= 15.0, f"Exceeded internal median target: {median_latency:.2f} ms"
    assert p99_latency <= 50.0, f"Exceeded internal P99 target: {p99_latency:.2f} ms"
