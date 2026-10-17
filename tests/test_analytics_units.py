"""
Synthetic unit tests for Stage 2: Historical Race Analysis (M2 Analytics).
Tests mathematical properties, invariants, edge cases, and temporal barriers
without depending on network or live APIs.
"""

import numpy as np
import pandas as pd
import pytest

from src.race.config import FUEL_CORRECTION_S_PER_LAP
from src.race.analytics.mini_sectors import compute_mini_sectors
from src.race.analytics.telemetry import compute_delta_trace, compute_section_stats
from src.race.analytics.tyres import compute_tyre_degradation
from src.race.analytics.gaps import compute_gap_history, detect_battles, classify_overtakes
from src.race.analytics.strategy import analyze_pit_stops, compute_time_loss_breakdown
from src.race.analytics.comparison import compare_laps


# ============================================================================
# 1. MINI-SECTORS SYNTHETIC UNIT TESTS
# ============================================================================

def test_mini_sectors_synthetic_uniform():
    """Verify that a synthetic constant-speed lap produces 20 identical mini-sectors summing to lap time."""
    # 5000m lap at 50 m/s = 100.0s lap
    distances = np.linspace(0, 5000, 501)
    times = distances / 50.0  # exactly 0.0 to 100.0s
    speeds = np.full_like(distances, 180.0)  # 50 m/s = 180 km/h

    tel = pd.DataFrame({"distance_m": distances, "time_s": times, "speed_kmh": speeds})
    res = compute_mini_sectors(tel, lap_time_s=100.0, num_mini_sectors=20)

    assert res["status"] == "VALID"
    assert res["is_accurate"] is True
    assert len(res["mini_sectors"]) == 20
    assert abs(res["sum_duration_s"] - 100.0) < 1e-4
    assert res["discrepancy_s"] < 1e-4

    for ms in res["mini_sectors"]:
        assert abs(ms["duration_s"] - 5.0) < 1e-3
        assert abs(ms["avg_speed_kmh"] - 180.0) < 1e-3


def test_mini_sectors_gap_detection():
    """Verify that a telemetry gap > 2.0s triggers UNRELIABLE_DATA_GAP and is_accurate=False."""
    distances = [0, 500, 1000, 2000, 3000, 4000, 5000]
    # Inject a 3.0s jump between samples
    times = [0.0, 10.0, 20.0, 25.0, 28.5, 30.0, 40.0]  # gap of 3.5s between 25.0 and 28.5
    tel = pd.DataFrame({"distance_m": distances, "time_s": times, "speed_kmh": [150]*7})

    res = compute_mini_sectors(tel, lap_time_s=40.0, num_mini_sectors=20)
    assert res["status"] == "UNRELIABLE_DATA_GAP"
    assert res["is_accurate"] is False


def test_mini_sectors_insufficient_data():
    """Verify empty or short telemetry returns INSUFFICIENT_TELEMETRY."""
    empty_df = pd.DataFrame()
    res = compute_mini_sectors(empty_df, lap_time_s=80.0)
    assert res["status"] == "INSUFFICIENT_TELEMETRY"
    assert res["is_accurate"] is False


# ============================================================================
# 2. DELTA TRACE & SECTION STATS SYNTHETIC UNIT TESTS
# ============================================================================

def test_delta_trace_antisymmetry():
    """Verify that delta trace is strictly antisymmetric: delta(A, B) = -delta(B, A)."""
    dist = np.linspace(0, 5000, 101)
    t1 = dist / 50.0  # 100s
    t2 = dist / 48.0  # ~104.16s

    tel1 = pd.DataFrame({"distance_m": dist, "time_s": t1, "speed_kmh": np.full_like(dist, 180.0)})
    tel2 = pd.DataFrame({"distance_m": dist, "time_s": t2, "speed_kmh": np.full_like(dist, 172.8)})

    df_12 = compute_delta_trace(tel1, tel2, step_m=50.0)
    df_21 = compute_delta_trace(tel2, tel1, step_m=50.0)

    assert not df_12.empty
    assert not df_21.empty
    np.testing.assert_allclose(df_12["delta_s"].values, -df_21["delta_s"].values, atol=1e-3)
    # Self-comparison must be exactly zero everywhere
    df_self = compute_delta_trace(tel1, tel1, step_m=50.0)
    np.testing.assert_allclose(df_self["delta_s"].values, 0.0, atol=1e-4)


def test_section_stats_synthetic():
    """Verify section apex minimum speed and braking point extraction."""
    # Lap with a corner between 1000m and 1500m
    d = np.linspace(0, 3000, 301)
    t = d / 50.0
    speed = np.full_like(d, 300.0)
    # Slow down to 100 km/h in corner
    corner_mask = (d >= 1000) & (d <= 1500)
    speed[corner_mask] = 120.0
    brake = np.zeros_like(d)
    brake[(d >= 950) & (d <= 1100)] = 100.0

    tel1 = pd.DataFrame({"distance_m": d, "time_s": t, "speed_kmh": speed, "brake_pct": brake})
    tel_dict = {"DRIVER_A": tel1}
    track_outline = {"sectors": [{"sector": 1, "start_m": 800.0, "end_m": 1600.0}]}

    stats_res = compute_section_stats("1", tel_dict, track_outline)
    assert stats_res["fastest_driver"] == "DRIVER_A"
    drv_stats = stats_res["drivers"]["DRIVER_A"]
    assert drv_stats["min_speed_kmh"] == 120.0
    assert drv_stats["braking_point_m"] == 950.0


# ============================================================================
# 3. TYRE DEGRADATION SYNTHETIC UNIT TESTS
# ============================================================================

def test_tyre_degradation_synthetic_recovery():
    """Verify that linear regression recovers injected slope with fuel correction."""
    # Stint with 10 clean laps
    # Injected true tyre deg: 0.050 s/lap
    # Fuel benefit: 0.033 s/lap
    # Net observed slope: 0.050 - 0.033 = 0.017 s/lap
    laps = np.arange(1, 11)
    base_time = 85.0
    raw_times = base_time + 0.017 * (laps - 1)

    laps_df = pd.DataFrame({
        "driver": "TEST",
        "stint": 1,
        "lap_number": laps,
        "lap_time_s": raw_times,
        "is_clean": True,
        "compound": "MEDIUM",
        "tyre_age_laps": laps,
    })

    res = compute_tyre_degradation(laps_df, "TEST", 1, fuel_correction_s_per_lap=0.033)
    assert res["clean_laps_count"] == 10
    assert res["is_estimate"] is True
    # Recovered raw slope should be ~0.017
    assert abs(res["raw_slope_s_per_lap"] - 0.017) < 1e-3
    # Estimated degradation rate (fuel corrected) should be ~0.050 s/lap
    assert abs(res["estimated_deg_rate_s_per_lap"] - 0.050) < 1e-3
    assert res["confidence"] == "HIGH"


def test_tyre_degradation_insufficient_data():
    """Stint with fewer than 4 clean laps must return INSUFFICIENT_DATA and None slope."""
    laps_df = pd.DataFrame({
        "driver": "TEST",
        "stint": 1,
        "lap_number": [1, 2, 3],
        "lap_time_s": [85.0, 85.1, 85.2],
        "is_clean": True,
        "compound": "SOFT",
    })
    res = compute_tyre_degradation(laps_df, "TEST", 1)
    assert res["clean_laps_count"] == 3
    assert res["estimated_deg_rate_s_per_lap"] is None
    assert res["confidence"] == "INSUFFICIENT_DATA"


# ============================================================================
# 4. GAP HISTORY & BATTLE DETECTION SYNTHETIC TESTS
# ============================================================================

def test_gap_history_antisymmetry():
    """Verify gap history is antisymmetric between drivers."""
    laps_df = pd.DataFrame({
        "driver": ["D1", "D2", "D1", "D2"],
        "lap_number": [1, 1, 2, 2],
        "session_time_s": [100.0, 101.5, 185.0, 186.0],
        "is_clean": [True, True, True, True],
    })

    g12 = compute_gap_history(laps_df, "D1", "D2")
    g21 = compute_gap_history(laps_df, "D2", "D1")

    assert not g12.empty
    assert not g21.empty
    np.testing.assert_allclose(g12["gap_s"].values, -g21["gap_s"].values, atol=1e-3)
    assert g12.iloc[0]["leader"] == "D1"
    assert g12.iloc[0]["gap_s"] == 1.5


def test_battle_detection_criteria():
    """Verify active battle criteria and ensure proximity is never claimed as an overtake."""
    # Lap 1: D1 at 100.0, D2 at 101.5 (gap 1.5s -> not a battle)
    # Lap 2: D1 at 185.0, D2 at 185.8 (gap 0.8s, closing rate 0.7s/lap -> active battle!)
    laps_df = pd.DataFrame({
        "driver": ["D1", "D2", "D1", "D2"],
        "lap_number": [1, 1, 2, 2],
        "session_time_s": [100.0, 101.5, 185.0, 185.8],
        "is_clean": [True, True, True, True],
        "track_status": ["1", "1", "1", "1"],
        "is_pit_in": [False, False, False, False],
        "is_pit_out": [False, False, False, False],
    })

    battles_l1 = detect_battles(laps_df, as_of_lap=1)
    assert len(battles_l1) == 0  # gap 1.5s > 1.0s

    battles_l2 = detect_battles(laps_df, as_of_lap=2)
    assert len(battles_l2) == 1
    b = battles_l2[0]
    assert b["leader"] == "D1"
    assert b["chaser"] == "D2"
    assert b["gap_s"] == 0.8
    assert b["closing_rate_s_per_lap"] == 0.7
    assert b["is_closing"] is True
    # Crucial check: proximity must NEVER claim an overtake has occurred
    assert b["is_overtake_confirmed"] is False


# ============================================================================
# 5. STRATEGY & TIME LOSS DECOMPOSITION SYNTHETIC TESTS
# ============================================================================

def test_pit_stop_analysis_no_double_counting():
    """Verify pit transit and stationary times are non-overlapping and not double-counted."""
    # Case A: Measured stationary duration (2.8s < 15s)
    stints_df_stat = pd.DataFrame({
        "driver": ["VER", "VER"],
        "stint": [1, 2],
        "start_lap": [1, 16],
        "end_lap": [15, 30],
        "pit_in_lap": [15, None],
        "pit_stop_duration_s": [2.8, None],
    })

    laps_df = pd.DataFrame({
        "driver": ["VER"] * 4,
        "lap_number": [14, 15, 16, 17],
        "lap_time_s": [85.0, 105.0, 100.0, 85.0],  # in-lap 105s (+20s), out-lap 100s (+15s)
        "track_status": ["1"] * 4,
        "is_clean": [True, False, False, True],
    })

    res_stat = analyze_pit_stops(stints_df_stat, laps_df, "VER", reference_stationary_s=2.5)
    assert res_stat["total_stops"] == 1
    stop = res_stat["stops"][0]
    assert stop["measured_stationary_s"] == 2.8
    assert stop["stationary_excess_s"] == 0.3  # 2.8 - 2.5
    # Total pit loss = (105 + 100) - 2 * 85 = 35.0s
    assert stop["estimated_total_pit_loss_s"] == 35.0

    # Case B: Measured pit lane duration (24.5s >= 15s) where stationary is unavailable
    stints_df_lane = pd.DataFrame({
        "driver": ["VER", "VER"],
        "stint": [1, 2],
        "start_lap": [1, 16],
        "end_lap": [15, 30],
        "pit_in_lap": [15, None],
        "pit_stop_duration_s": [24.5, None],
    })
    res_lane = analyze_pit_stops(stints_df_lane, laps_df, "VER", reference_stationary_s=2.5)
    stop_lane = res_lane["stops"][0]
    assert stop_lane["measured_pit_lane_duration_s"] == 24.5
    assert stop_lane["measured_stationary_s"] is None
    assert stop_lane["stationary_data_status"] == "UNAVAILABLE_IN_PROVIDER_FEED"
    # Transit loss is estimated: 24.5 - 2.5 = 22.0s
    assert stop_lane["estimated_pit_lane_transit_s"] == 22.0
    assert stop_lane["is_transit_loss_estimated"] is True
    # Without explicit parallel straight baseline: track entry/exit loss is explicitly UNAVAILABLE
    assert stop_lane["estimated_track_in_out_loss_s"] is None
    assert stop_lane["track_loss_status"] == "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"

    # With explicit parallel straight baseline (e.g. 4.0s):
    # Net lane loss = 24.5 - 4.0 = 20.5s; Track entry/exit loss = 35.0 - 20.5 = 14.5s
    res_lane_str = analyze_pit_stops(stints_df_lane, laps_df, "VER", reference_stationary_s=2.5, parallel_straight_time_s=4.0)
    stop_str = res_lane_str["stops"][0]
    assert stop_str["estimated_track_in_out_loss_s"] == 14.5
    assert stop_str["track_loss_status"] == "ESTIMATED_WITH_STRAIGHT_BASELINE"
    # Total pit loss already encompasses transit and stationary (never add stationary to it!)
    assert stop_lane["estimated_total_pit_loss_s"] == 35.0


def test_time_loss_breakdown_waterfall():
    """Verify non-overlapping waterfall decomposition and exact accounting identity."""
    laps_df = pd.DataFrame({
        "driver": ["VER"] * 5,
        "lap_number": [1, 2, 3, 4, 5],
        "lap_time_s": [85.0, 84.0, 110.0, 105.0, 86.0],  # lap 2 is faster (-1.0s surplus)
        "track_status": ["1", "1", "4", "1", "1"],  # lap 3 is safety car (+25s)
        "is_clean": [True, True, False, False, True],
        "is_pit_in": [False, False, False, True, False],  # lap 4 is pit in (+20s)
        "is_pit_out": [False, False, False, False, False],
        "tyre_age_laps": [1, 2, 3, 4, 1],
        "session_time_s": [85.0, 169.0, 279.0, 384.0, 470.0],
    })
    stints_df = pd.DataFrame({"driver": ["VER"], "stint": [1], "start_lap": [1], "end_lap": [5], "pit_in_lap": [4], "pit_stop_duration_s": [24.5]})

    res = compute_time_loss_breakdown(laps_df, stints_df, "VER")
    assert res["baseline_scope"] == "COMPLETED_RACE_FIELD_MEDIAN"
    factors = res["loss_factors_s"]
    # Lap 3 under SC: 110 - 85 = 25s caution loss
    assert factors["caution_and_incidents_s"] == 25.0
    # Lap 4 pit in (105-85=20s) and Lap 5 pit out (86-85=1s): total pit loss = 21s
    assert factors["pit_stop_and_transit_s"] == 21.0

    # Total measured deficit = total_driver_time - total_baseline_time
    # driver_total = 85 + 84 + 110 + 105 + 86 = 470.0
    # baseline = 5 * 85.0 = 425.0 -> total deficit = 45.0s
    assert res["total_deficit_vs_baseline_s"] == 45.0
    # Exact accounting identity must hold: Total == Caution + Pit + Traffic + Tyre + Residual
    sum_factors = (
        factors["caution_and_incidents_s"]
        + factors["pit_stop_and_transit_s"]
        + factors["traffic_s"]
        + factors["tyre_degradation_s"]
        + factors["unexplained_residual_s"]
    )
    assert abs(res["total_deficit_vs_baseline_s"] - sum_factors) < 0.02
    assert res["accounting_identity_verified"] is True


def test_overtake_classification_confirmed_and_inferred():
    """Verify on-track overtake classification with physical distance-aligned crossover detection."""
    laps_df = pd.DataFrame({
        "driver": ["VER", "LEC", "VER", "LEC"],
        "lap_number": [9, 9, 10, 10],
        "session_time_s": [800.0, 798.5, 885.0, 886.2],  # LEC was ahead on lap 9 (798.5 < 800.0), VER passes on lap 10 (885.0 < 886.2)
        "lap_time_s": [85.0, 86.5, 85.0, 87.7],
        "track_status": ["1", "1", "1", "1"],
        "is_pit_in": [False, False, False, False],
        "is_pit_out": [False, False, False, False],
    })

    # Case 1: Without telemetry loader -> INFERRED_ON_TRACK_OVERTAKE
    overtakes_inf = classify_overtakes(laps_df)
    assert len(overtakes_inf) == 1
    ev_inf = overtakes_inf[0]
    assert ev_inf["overtaking_driver"] == "VER"
    assert ev_inf["overtaken_driver"] == "LEC"
    assert ev_inf["classification"] == "INFERRED_ON_TRACK_OVERTAKE"
    assert ev_inf["confidence"] == "INFERRED"
    assert ev_inf["is_on_track_overtake"] is True
    assert ev_inf["crossover_distance_m"] is None

    # Case 2: Genuine physical crossing in telemetry
    # LEC started at 798.5s, VER at 800.0s (LEC was ahead by 1.5s).
    # VER drives lap in 85.0s, LEC in 87.7s.
    # Crossover occurs at: 800.0 + (d/5793)*85.0 = 798.5 + (d/5793)*87.7 => d* = 5793 * (1.5 / 2.7) = 3218.3m
    def mock_tel_crossing(drv, lap):
        d = np.linspace(0.0, 5793.0, 100)
        t_lap = 85.0 if drv == "VER" else 87.7
        t = (d / 5793.0) * t_lap
        return pd.DataFrame({"distance_m": d, "time_s": t})

    overtakes_conf = classify_overtakes(laps_df, telemetry_loader_func=mock_tel_crossing)
    assert len(overtakes_conf) == 1
    ev_conf = overtakes_conf[0]
    assert ev_conf["classification"] == "CONFIRMED_ON_TRACK_OVERTAKE"
    assert ev_conf["confidence"] == "CONFIRMED"
    assert ev_conf["is_on_track_overtake"] is True
    assert ev_conf["crossover_distance_m"] is not None
    assert abs(ev_conf["crossover_distance_m"] - 3218.3) < 30.0

    # Case 3: Ambiguous case — Continuous telemetry but NO crossing demonstrated
    # e.g., both drivers have same pace profile, LEC stays ahead throughout the telemetry trace
    def mock_tel_no_crossing(drv, lap):
        d = np.linspace(0.0, 5793.0, 100)
        t = (d / 5793.0) * 85.0  # Identical elapsed lap time: VER is always 1.5s behind in clock time
        return pd.DataFrame({"distance_m": d, "time_s": t})

    overtakes_ambig = classify_overtakes(laps_df, telemetry_loader_func=mock_tel_no_crossing)
    assert len(overtakes_ambig) == 1
    ev_ambig = overtakes_ambig[0]
    # Without physical crossover in telemetry, event MUST NOT be falsely confirmed; retained as inferred
    assert ev_ambig["classification"] == "INFERRED_ON_TRACK_OVERTAKE"
    assert ev_ambig["confidence"] == "INFERRED"
    assert ev_ambig["is_on_track_overtake"] is True
    assert ev_ambig["crossover_distance_m"] is None
    assert "did not demonstrate a distinct on-track crossing" in ev_ambig["evidence"]


def test_overtake_classification_pit_strategy_swap():
    """A lap-end position swap caused by a pit stop must NOT count as an on-track overtake."""
    laps_df = pd.DataFrame({
        "driver": ["VER", "LEC", "VER", "LEC"],
        "lap_number": [14, 14, 15, 15],
        "session_time_s": [1200.0, 1195.0, 1285.0, 1315.0],  # LEC pits on lap 15
        "lap_time_s": [85.0, 85.0, 85.0, 120.0],
        "track_status": ["1", "1", "1", "1"],
        "is_pit_in": [False, False, False, True],  # LEC in pit lane
        "is_pit_out": [False, False, False, False],
    })

    overtakes = classify_overtakes(laps_df)
    assert len(overtakes) == 1
    ev = overtakes[0]
    assert ev["classification"] == "PIT_STRATEGY_SWAP"
    assert ev["is_on_track_overtake"] is False
    assert ev["confidence"] == "CONFIRMED"


def test_overtake_classification_retirement_and_caution():
    """Retirement passes and safety car position changes must be classified correctly."""
    # Retirement case
    laps_df_ret = pd.DataFrame({
        "driver": ["VER", "LEC", "VER"],
        "lap_number": [20, 20, 21],  # LEC does not complete lap 21 (retired)
        "session_time_s": [1700.0, 1698.0, 1785.0],
        "lap_time_s": [85.0, 85.0, 85.0],
        "track_status": ["1", "1", "1"],
    })
    # LEC stopped after lap 20
    overtakes_ret = classify_overtakes(laps_df_ret)
    # LEC is absent on lap 21, so VER passing LEC is not an on-track battle
    assert all(o["is_on_track_overtake"] is False for o in overtakes_ret)

    # Caution case: passing under Safety Car (track_status != "1")
    laps_df_sc = pd.DataFrame({
        "driver": ["VER", "LEC", "VER", "LEC"],
        "lap_number": [25, 25, 26, 26],
        "session_time_s": [2100.0, 2095.0, 2210.0, 2220.0],
        "lap_time_s": [85.0, 85.0, 110.0, 125.0],
        "track_status": ["1", "1", "4", "4"],  # Safety car on lap 26
        "is_pit_in": [False, False, False, False],
        "is_pit_out": [False, False, False, False],
    })
    overtakes_sc = classify_overtakes(laps_df_sc)
    assert len(overtakes_sc) == 1
    assert overtakes_sc[0]["classification"] == "CAUTION_PERIOD_CHANGE"
    assert overtakes_sc[0]["is_on_track_overtake"] is False


def test_overtake_classification_telemetry_gap():
    """Telemetry containing gap > 2s must return UNKNOWN_DATA_GAP."""
    laps_df = pd.DataFrame({
        "driver": ["VER", "LEC", "VER", "LEC"],
        "lap_number": [9, 9, 10, 10],
        "session_time_s": [800.0, 798.5, 885.0, 886.2],
        "lap_time_s": [85.0, 86.5, 85.0, 87.7],
        "track_status": ["1", "1", "1", "1"],
        "is_pit_in": [False, False, False, False],
        "is_pit_out": [False, False, False, False],
    })

    # Mock telemetry with 3.5s gap (> 2.0s threshold)
    def mock_tel_gap(drv, lap):
        return pd.DataFrame({"distance_m": [0.0, 1000.0, 5000.0], "time_s": [0.0, 15.0, 18.5 + (0.0 if drv == 'VER' else 3.5)]})

    overtakes = classify_overtakes(laps_df, telemetry_loader_func=mock_tel_gap)
    assert len(overtakes) == 1
    assert overtakes[0]["classification"] == "UNKNOWN_DATA_GAP"
    assert overtakes[0]["confidence"] == "UNKNOWN"
    assert overtakes[0]["is_on_track_overtake"] is False


def test_openf1_precision_tiering_and_empirical_benchmark():
    """Verify compute_mini_sectors returns provider tiering and empirical benchmarks."""
    tel_openf1 = pd.DataFrame({
        "distance_m": np.linspace(0, 5793, 100),
        "time_s": np.linspace(0, 85.2, 100),
        "speed_kmh": np.full(100, 250.0),
    })
    tel_openf1.attrs["provider"] = "OpenF1"

    res = compute_mini_sectors(tel_openf1, 85.0)
    assert res["provider"] == "OpenF1"
    assert res["precision_tier"] == "ESTIMATED_COORDINATE_DERIVED"
    assert res["empirical_benchmark"]["monza_2023_clean_median_residual_s"] == 0.1764
    assert res["empirical_benchmark"]["monza_2023_clean_max_residual_s"] == 0.3852
    assert "never rescaled" in res["accuracy_notes"]
    assert res["is_within_provider_benchmark"] is True


# ============================================================================
# 6. AS_OF REPLAY BARRIER IN ANALYTICS
# ============================================================================

def test_as_of_barrier_in_analytics():
    """Verify that all analytics functions strictly exclude future data beyond as_of_lap."""
    laps = list(range(1, 31))
    laps_df = pd.DataFrame({
        "driver": ["VER"] * 30 + ["LEC"] * 30,
        "lap_number": laps * 2,
        "lap_time_s": [85.0 + 0.05 * l for l in laps] * 2,
        "session_time_s": [float(l * 85.0) for l in laps] * 2,
        "is_clean": [True] * 60,
        "track_status": ["1"] * 60,
        "stint": [1] * 30 + [1] * 30,
        "tyre_age_laps": laps * 2,
        "compound": ["MEDIUM"] * 60,
    })

    # Test compare_laps with as_of_lap=15
    comp = compare_laps({"laps": laps_df}, ["VER", "LEC"], lap_selection="last_clean", as_of_lap=15)
    max_lap_returned = comp["summary_table"]["lap_number"].max()
    assert max_lap_returned <= 15

    # Test gaps with as_of_lap=15
    gaps = compute_gap_history(laps_df, "VER", "LEC", as_of_lap=15)
    assert gaps["lap_number"].max() == 15

    # Test tyres with as_of_lap=15
    deg = compute_tyre_degradation(laps_df, "VER", 1, as_of_lap=15)
    assert deg["clean_laps_count"] == 15
