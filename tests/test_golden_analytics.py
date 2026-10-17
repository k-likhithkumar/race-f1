"""
Real-data golden validation tests for Stage 2: Historical Race Analysis (M2 Analytics).
Validates analytics functions against cached 2022 Monza (FastF1) and 2023 Monza (OpenF1)
datasets completely offline.
"""

import numpy as np
import pandas as pd
import pytest

from src.race.data.cache import load_cached_race
from src.race.analytics.telemetry_loader import load_lap_telemetry
from src.race.analytics.mini_sectors import compute_mini_sectors
from src.race.analytics.telemetry import compute_delta_trace, compute_section_stats
from src.race.analytics.tyres import compute_tyre_degradation
from src.race.analytics.gaps import compute_gap_history, detect_battles, classify_overtakes
from src.race.analytics.strategy import analyze_pit_stops, compute_time_loss_breakdown
from src.race.analytics.comparison import compare_laps


# ============================================================================
# 1. FASTF1 REAL DATA TESTS (2022 MONZA)
# ============================================================================

def test_2022_monza_fastf1_telemetry_schema():
    """Verify FastF1 telemetry loader provides all 7 channels with valid physics."""
    tel = load_lap_telemetry(2022, "Monza", "VER", 10)
    assert not tel.empty
    assert tel.attrs.get("provider") == "FastF1"

    # All 7 channels must be present
    required_channels = ["distance_m", "time_s", "speed_kmh", "throttle_pct", "brake_pct", "gear", "drs"]
    for ch in required_channels:
        assert ch in tel.columns
        assert tel[ch].notna().any(), f"Channel {ch} is empty in FastF1 telemetry"

    # Verify physical sanity
    assert tel["distance_m"].is_monotonic_increasing
    assert 5500 < tel["distance_m"].max() < 6000  # Monza is ~5793m
    assert tel["speed_kmh"].max() > 300.0        # F1 car at Monza tops 330+ km/h
    assert tel["speed_kmh"].min() > 40.0         # Prima Variante apex


def test_2022_monza_mini_sector_golden_accuracy():
    """
    Validate PDF Topic 18 (Page 21) acceptance criterion on real FastF1 data:
    - Mini-sector times sum to official lap time within +/- 0.05s on >= 98% of clean laps
      (all others explained - specifically Lap 1 standing start grid displacement).
    """
    race_data = load_cached_race(2022, "Monza")
    laps_df = race_data["laps"]
    track_outline = race_data["track_outline"]

    # Test across clean laps of the top 3 drivers (VER, LEC, RUS)
    test_drivers = ["VER", "LEC", "RUS"]
    clean_laps = laps_df[laps_df["driver"].isin(test_drivers) & (laps_df["is_clean"] == True)].copy()

    total_tested = 0
    passed_tolerance = 0
    discrepancies = []
    explained_outliers = []

    for _, row in clean_laps.iterrows():
        drv = row["driver"]
        lap_num = int(row["lap_number"])
        lap_t = float(row["lap_time_s"])

        tel = load_lap_telemetry(2022, "Monza", drv, lap_num)
        if tel.empty:
            continue

        res = compute_mini_sectors(tel, lap_t, track_outline=track_outline)
        if res["status"] == "VALID":
            total_tested += 1
            disc = res["discrepancy_s"]
            discrepancies.append(disc)
            if res["is_accurate"]:
                passed_tolerance += 1
            else:
                # Per PDF Page 21: "(all others explained)"
                assert lap_num == 1, f"Unexpected outlier on flying lap {lap_num} for {drv}: {disc:.4f}s"
                explained_outliers.append((drv, lap_num, disc))

    assert total_tested >= 50, f"Expected at least 50 clean laps to test, got {total_tested}"
    
    # 1. On flying clean laps (lap > 1), pass rate is 100% (exceeding >= 98% target)
    flying_tested = total_tested - len(explained_outliers)
    assert flying_tested >= 50
    flying_pass_rate = passed_tolerance / flying_tested
    assert flying_pass_rate >= 0.98, f"Flying clean pass rate {flying_pass_rate:.1%} is below 98%"

    # 2. Every single outlier is strictly accounted for by Lap 1 standing start
    assert len(explained_outliers) == 3
    for drv, lap, disc in explained_outliers:
        assert lap == 1
        assert disc < 0.35  # Max discrepancy on standing start is 0.31s (RUS)


def test_2022_monza_real_delta_trace():
    """Verify distance-aligned delta trace between Verstappen and Leclerc on Lap 10."""
    tel_ver = load_lap_telemetry(2022, "Monza", "VER", 10)
    tel_lec = load_lap_telemetry(2022, "Monza", "LEC", 10)

    delta_df = compute_delta_trace(tel_ver, tel_lec, step_m=20.0)
    assert not delta_df.empty
    assert "delta_s" in delta_df.columns
    # Distance should span full lap
    assert delta_df["distance_m"].max() > 5700
    # Boundary check: start delta should be close to 0
    assert abs(delta_df.iloc[0]["delta_s"]) < 0.1


def test_2022_monza_real_tyre_degradation():
    """Verify fuel-corrected tyre degradation estimation on Verstappen Stint 1."""
    race_data = load_cached_race(2022, "Monza")
    laps_df = race_data["laps"]

    deg = compute_tyre_degradation(laps_df, "VER", stint=1)
    assert deg["clean_laps_count"] >= 10
    assert deg["is_estimate"] is True
    assert deg["compound"] == "SOFT"
    # Degradation rate should be a physically sensible estimate (typically 0.0 to 0.15 s/lap)
    assert deg["estimated_deg_rate_s_per_lap"] is not None
    assert -0.05 < deg["estimated_deg_rate_s_per_lap"] < 0.20
    assert deg["confidence"] in ["HIGH", "MEDIUM"]


def test_2022_monza_real_pit_stop_analysis():
    """Verify pit stop duration, transit estimation, and total loss for Leclerc in 2022 Monza."""
    race_data = load_cached_race(2022, "Monza")
    res = analyze_pit_stops(race_data["stints"], race_data["laps"], "LEC", reference_stationary_s=2.5)
    assert res["total_stops"] >= 1
    stop1 = res["stops"][0]
    # Measured pit lane duration (at Monza pit lane is ~23-25s from entry to exit transponder)
    assert stop1["measured_pit_lane_duration_s"] is not None
    assert 20.0 < stop1["measured_pit_lane_duration_s"] < 30.0
    # Stationary duration is explicitly marked unavailable (not recorded in raw transponder stream)
    assert stop1["measured_stationary_s"] is None
    assert stop1["stationary_data_status"] == "UNAVAILABLE_IN_PROVIDER_FEED"
    # Estimated pit lane transit duration: pit lane duration - reference stationary (2.5s)
    assert stop1["estimated_pit_lane_transit_s"] is not None
    assert 18.0 < stop1["estimated_pit_lane_transit_s"] < 28.0
    assert stop1["is_transit_loss_estimated"] is True
    # Total pit loss on track across in-lap and out-lap relative to clean pace (~20s to 45s)
    assert stop1["estimated_total_pit_loss_s"] is not None
    assert 20.0 < stop1["estimated_total_pit_loss_s"] < 45.0
    # Track in/out deceleration and acceleration loss without straight baseline:
    # Transponders do not measure parallel straight transit, so track in/out loss is explicitly UNAVAILABLE
    assert stop1["estimated_track_in_out_loss_s"] is None
    assert stop1["track_loss_status"] == "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"

    # When an explicit straight baseline is provided (e.g. 4.2s flying straight transit at Monza):
    res_straight = analyze_pit_stops(race_data["stints"], race_data["laps"], "LEC", reference_stationary_s=2.5, parallel_straight_time_s=4.2)
    stop_str = res_straight["stops"][0]
    assert stop_str["estimated_track_in_out_loss_s"] is not None
    assert stop_str["estimated_track_in_out_loss_s"] >= 0.0
    assert stop_str["track_loss_status"] == "ESTIMATED_WITH_STRAIGHT_BASELINE"
    # No double counting: pit lane transit is strictly less than total pit loss
    assert stop1["estimated_pit_lane_transit_s"] < stop1["estimated_total_pit_loss_s"]


# ============================================================================
# 2. OPENF1 REAL DATA TESTS (2023 MONZA)
# ============================================================================

def test_2023_monza_openf1_telemetry_schema():
    """Verify provider-aware OpenF1 telemetry preserves unavailable channels as NaN."""
    tel = load_lap_telemetry(2023, "Monza", "VER", 10)
    assert not tel.empty
    assert tel.attrs.get("provider") == "OpenF1"

    # Distance, time, and speed must be present and valid
    assert "distance_m" in tel.columns
    assert "time_s" in tel.columns
    assert "speed_kmh" in tel.columns
    assert tel["distance_m"].notna().all()
    assert tel["time_s"].notna().all()

    # Unavailable channels must be explicitly NaN
    unavailable_channels = ["throttle_pct", "brake_pct", "gear", "drs"]
    for ch in unavailable_channels:
        assert ch in tel.columns
        assert tel[ch].isna().all(), f"Expected channel {ch} to be NaN in OpenF1 location data"

    # Distance should cover Monza track length (~5793m within coordinate resolution)
    assert 5500 < tel["distance_m"].max() < 6000


def test_2023_monza_openf1_mini_sector_empirical_benchmark():
    """
    Empirical benchmark of OpenF1 coordinate integration on clean laps.
    Reports actual discrepancy distribution without artificial rescaling or warping.
    """
    race_data = load_cached_race(2023, "Monza")
    laps_df = race_data["laps"]
    track_outline = race_data["track_outline"]

    ver_clean = laps_df[(laps_df["driver"] == "VER") & (laps_df["is_clean"] == True)]
    discrepancies = []

    for _, row in ver_clean.head(10).iterrows():
        lap_num = int(row["lap_number"])
        lap_t = float(row["lap_time_s"])
        tel = load_lap_telemetry(2023, "Monza", "VER", lap_num)
        if tel.empty:
            continue

        res = compute_mini_sectors(tel, lap_t, track_outline=track_outline)
        if res["status"] == "VALID":
            discrepancies.append(res["discrepancy_s"])
            assert res["provider"] == "OpenF1"
            assert res["precision_tier"] == "ESTIMATED_COORDINATE_DERIVED"
            assert res["empirical_benchmark"]["monza_2023_clean_median_residual_s"] == 0.1764

    assert len(discrepancies) >= 5
    median_disc = float(np.median(discrepancies))
    max_disc = float(np.max(discrepancies))
    # Log empirical benchmark transparently
    print(f"\n[OpenF1 2023 Monza Empirical Benchmark] Median: {median_disc:.4f}s, Max: {max_disc:.4f}s")
    # Residuals must be finite and within plausible GPS path integration error
    assert median_disc < 0.50, f"OpenF1 median discrepancy {median_disc:.4f}s exceeds plausible threshold"


# ============================================================================
# 3. MULTI-DRIVER COMPARISON ENGINE & OVERTAKE REAL DATA TESTS
# ============================================================================

def test_compare_laps_real_data_sync_modes():
    """Verify compare_laps works across sync_mode='lap' and sync_mode='tyre_age'."""
    race_data = load_cached_race(2022, "Monza")

    # Lap sync mode
    comp_lap = compare_laps(race_data, ["VER", "LEC"], lap_selection=[10, 11, 12], sync_mode="lap")
    assert comp_lap["reference_driver"] == "VER"
    assert comp_lap["sync_mode"] == "lap"
    assert not comp_lap["summary_table"].empty
    assert "lap_number" in comp_lap["summary_table"].columns

    # Tyre-age sync mode
    comp_tyre = compare_laps(race_data, ["VER", "LEC"], lap_selection="fastest", sync_mode="tyre_age")
    assert comp_tyre["sync_mode"] == "tyre_age"
    assert comp_tyre["align_key"] == "tyre_age_laps"


def test_2022_monza_real_overtake_classification():
    """Verify overtake classification distinguishes pit-lane swaps from on-track passes in real race data."""
    race_data = load_cached_race(2022, "Monza")
    laps_df = race_data["laps"]
    stints_df = race_data["stints"]

    overtakes = classify_overtakes(laps_df, stints_df=stints_df)
    assert len(overtakes) > 0

    # Categorization counts
    classifications = [o["classification"] for o in overtakes]
    assert "PIT_STRATEGY_SWAP" in classifications
    assert any("ON_TRACK" in c for c in classifications)

    # Pit swaps must have is_on_track_overtake == False
    for o in overtakes:
        if o["classification"] == "PIT_STRATEGY_SWAP":
            assert o["is_on_track_overtake"] is False
            assert "pit lane" in o["evidence"].lower()
        elif "ON_TRACK" in o["classification"]:
            assert o["is_on_track_overtake"] is True

    # Real telemetry verification: Lap 5 Verstappen passes Russell into Turn 1
    def real_tel_loader(drv, lap):
        return load_lap_telemetry(2022, "Monza", drv, lap)

    overtakes_lap5 = classify_overtakes(
        laps_df[laps_df["lap_number"] <= 5],
        stints_df=stints_df,
        telemetry_loader_func=real_tel_loader,
    )
    ver_rus = [o for o in overtakes_lap5 if o["overtaking_driver"] == "VER" and o["overtaken_driver"] == "RUS" and o["lap_number"] == 5]
    assert len(ver_rus) == 1
    v_r = ver_rus[0]
    assert v_r["classification"] == "CONFIRMED_ON_TRACK_OVERTAKE"
    assert v_r["confidence"] == "CONFIRMED"
    assert v_r["is_on_track_overtake"] is True
    assert v_r["crossover_distance_m"] is not None
    # Verified on Monza straight into Variante del Rettifilo (~850m)
    assert 800.0 < v_r["crossover_distance_m"] < 900.0
