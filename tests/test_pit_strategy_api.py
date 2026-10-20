"""
Focused tests for Checkpoint 4: Pit Strategy API & Analytics.
Verifies:
1. Timestamp-level lifecycle boundaries (IN_PROGRESS, OUT_LAP_IN_PROGRESS, COMPLETED).
2. Illustrative benchmark assumption for reference_stationary_s (never measured).
3. Metric semantics, anti-double-counting, and exact accounting identity (+-0.01s).
4. Explicit UNKNOWN track status handling (never defaulted to green or caution).
5. Grounded heuristic tactical classifications with supporting evidence and fallbacks.
6. HTTP API endpoint /api/races/{race_id}/pits with FastAPI TestClient.
"""

import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.race.api.app import app
from src.race.analytics.strategy import analyze_pit_stops, compute_time_loss_breakdown
from src.race.data.processor import is_track_status_strictly_green
from src.race.api.session_manager import session_manager


client = TestClient(app)


# ============================================================================
# 1. PIT-STOP LIFECYCLE BOUNDARY TESTS
# ============================================================================

def test_pit_lifecycle_immediate_before_at_after_boundaries():
    """
    Test timestamps immediately before, exactly at, and after each lifecycle boundary:
    T_pit_in = 2717.948 (in-lap crossing)
    pit_lane_duration = 24.5s
    T_pit_out = 2742.448 (pit exit)
    T_out_crossing = 2825.579 (out-lap S/F crossing)
    """
    t_pit_in = 2717.948
    pit_duration = 24.5
    t_pit_out = t_pit_in + pit_duration  # 2742.448
    t_out = 2825.579

    stints_df = pd.DataFrame({
        "driver": ["ALB", "ALB"],
        "stint": [1, 2],
        "start_lap": [1, 16],
        "end_lap": [15, 30],
        "pit_in_lap": [15, None],
        "pit_stop_duration_s": [pit_duration, None],
    })

    laps_df = pd.DataFrame({
        "driver": ["ALB"] * 4,
        "lap_number": [14, 15, 16, 17],
        "lap_time_s": [87.723, 92.301, 107.605, 87.087],
        "session_time_s": [2625.808, t_pit_in, t_out, 2912.638],
        "track_status": ["1", "1", "1", "1"],
        "is_clean": [True, False, False, True],
        "position": [7, 7, 10, 10],
    })

    # --- Boundary 1: Pit Entry (T_pit_in = 2717.948) ---

    # 1a. Immediately before pit entry: stop has NOT started
    res_before_entry = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_pit_in - 0.001)
    assert res_before_entry["total_stops"] == 0
    assert len(res_before_entry["stops"]) == 0
    assert res_before_entry["total_completed_stops"] == 0
    assert res_before_entry["active_stop"] is None

    # 1b. Exactly at pit entry: IN_PROGRESS
    res_at_entry = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_pit_in)
    assert res_at_entry["total_stops"] == 1
    assert res_at_entry["total_completed_stops"] == 0  # not completed!
    assert res_at_entry["active_stop"] is not None
    stop_entry = res_at_entry["stops"][0]
    assert stop_entry["status"] == "IN_PROGRESS"
    assert stop_entry["measured_pit_lane_duration_s"] is None  # not exited yet
    assert stop_entry["out_lap_time_s"] is None
    assert stop_entry["estimated_total_pit_loss_s"] is None
    assert stop_entry["tactical_evidence"]["classification"] == "UNKNOWN"

    # 1c. Midway through pit lane: IN_PROGRESS
    res_mid_lane = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_pit_in + 10.0)
    assert res_mid_lane["total_completed_stops"] == 0
    assert res_mid_lane["stops"][0]["status"] == "IN_PROGRESS"
    assert res_mid_lane["stops"][0]["measured_pit_lane_duration_s"] is None

    # --- Boundary 2: Pit Exit (T_pit_out = 2742.448) ---

    # 2a. Immediately before pit exit: still IN_PROGRESS
    res_before_exit = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_pit_out - 0.001)
    assert res_before_exit["stops"][0]["status"] == "IN_PROGRESS"
    assert res_before_exit["stops"][0]["measured_pit_lane_duration_s"] is None

    # 2b. Exactly at pit exit: OUT_LAP_IN_PROGRESS
    res_at_exit = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_pit_out)
    assert res_at_exit["total_stops"] == 1
    assert res_at_exit["total_completed_stops"] == 0  # out-lap not finished!
    stop_exit = res_at_exit["stops"][0]
    assert stop_exit["status"] == "OUT_LAP_IN_PROGRESS"
    assert stop_exit["measured_pit_lane_duration_s"] == 24.5  # lane traversal measured
    assert stop_exit["out_lap_time_s"] is None  # out-lap NOT finished
    assert stop_exit["estimated_total_pit_loss_s"] is None  # cannot evaluate net loss yet
    assert stop_exit["tactical_evidence"]["classification"] == "UNKNOWN"

    # 2c. Midway through out-lap: OUT_LAP_IN_PROGRESS
    res_mid_out = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_pit_out + 30.0)
    assert res_mid_out["total_completed_stops"] == 0
    assert res_mid_out["stops"][0]["status"] == "OUT_LAP_IN_PROGRESS"
    assert res_mid_out["stops"][0]["estimated_total_pit_loss_s"] is None

    # --- Boundary 3: Out-Lap S/F Crossing (T_out_crossing = 2825.579) ---

    # 3a. Immediately before out-lap crossing: OUT_LAP_IN_PROGRESS
    res_before_out = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_out - 0.001)
    assert res_before_out["total_completed_stops"] == 0
    assert res_before_out["stops"][0]["status"] == "OUT_LAP_IN_PROGRESS"
    assert res_before_out["stops"][0]["estimated_total_pit_loss_s"] is None

    # 3b. Exactly at out-lap crossing: COMPLETED
    res_at_out = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_out)
    assert res_at_out["total_stops"] == 1
    assert res_at_out["total_completed_stops"] == 1  # now completed!
    assert res_at_out["active_stop"] is None
    stop_out = res_at_out["stops"][0]
    assert stop_out["status"] == "COMPLETED"
    assert stop_out["in_lap_time_s"] == 92.301
    assert stop_out["out_lap_time_s"] == 107.605
    assert stop_out["measured_pit_lane_duration_s"] == 24.5
    assert stop_out["estimated_total_pit_loss_s"] is not None

    # 3c. After out-lap crossing: stays COMPLETED
    res_after = analyze_pit_stops(stints_df, laps_df, "ALB", as_of_time_s=t_out + 50.0)
    assert res_after["total_completed_stops"] == 1
    assert res_after["stops"][0]["status"] == "COMPLETED"


# ============================================================================
# 2. ILLUSTRATIVE STATIONARY ASSUMPTION (2.5s)
# ============================================================================

def test_stationary_assumption_labeling_and_never_measured():
    """Verify reference_stationary_s (2.5s) is strictly labeled an assumption and measured_stationary_s is None."""
    stints_df = pd.DataFrame({
        "driver": ["VER", "VER"],
        "stint": [1, 2],
        "start_lap": [1, 16],
        "end_lap": [15, 30],
        "pit_in_lap": [15, None],
        "pit_stop_duration_s": [24.5, None],
    })
    laps_df = pd.DataFrame({
        "driver": ["VER"] * 4,
        "lap_number": [14, 15, 16, 17],
        "lap_time_s": [85.0, 95.0, 105.0, 85.0],
        "session_time_s": [1000.0, 1095.0, 1200.0, 1285.0],
        "track_status": ["1"] * 4,
        "is_clean": [True, False, False, True],
        "position": [1, 1, 1, 1],
    })

    res = analyze_pit_stops(stints_df, laps_df, "VER", as_of_time_s=1200.0, reference_stationary_s=2.5)
    stop = res["stops"][0]

    # Measured stationary time is explicitly None (not provided in transponder feeds)
    assert stop["measured_stationary_s"] is None
    assert stop["is_stationary_measured"] is False
    assert stop["stationary_data_status"] == "UNAVAILABLE_IN_PROVIDER_FEED"

    # Illustrative assumption is explicitly disclosed
    assert stop["reference_stationary_s"] == 2.5
    assert stop["stationary_metric_type"] == "ILLUSTRATIVE_ASSUMPTION"
    assert "Illustrative benchmark assumption" in stop["stationary_assumption_label"]
    assert "illustrative 2.5s assumption" in stop["stationary_disclaimer"]

    # Transit estimate uses assumption without claiming measurement
    assert stop["estimated_pit_lane_transit_s"] == 22.0  # 24.5 - 2.5
    assert stop["is_transit_loss_estimated"] is True
    assert "REFERENCE_STATIONARY" in stop["transit_estimation_basis"]


# ============================================================================
# 3. TIME-LOSS ACCOUNTING & ANTI-DOUBLE-COUNTING
# ============================================================================

def test_time_loss_accounting_identity_and_anti_double_counting():
    """Verify exact accounting identity (+-0.01s) and prevention of double-counting."""
    laps_df = pd.DataFrame({
        "driver": ["VER"] * 5,
        "lap_number": [1, 2, 3, 4, 5],
        "lap_time_s": [85.0, 84.0, 110.0, 105.0, 86.0],  # lap 2: -1.0s surplus, lap 3: +25s caution, lap 4: +20s pit
        "track_status": ["1", "1", "4", "1", "1"],
        "is_clean": [True, True, False, False, True],
        "is_pit_in": [False, False, False, True, False],
        "is_pit_out": [False, False, False, False, False],
        "tyre_age_laps": [1, 2, 3, 4, 1],
        "session_time_s": [85.0, 169.0, 279.0, 384.0, 470.0],
    })
    stints_df = pd.DataFrame({
        "driver": ["VER"],
        "stint": [1],
        "start_lap": [1],
        "end_lap": [5],
        "pit_in_lap": [4],
        "pit_stop_duration_s": [24.5],
    })

    res = compute_time_loss_breakdown(laps_df, stints_df, "VER")
    factors = res["loss_factors_s"]

    # Total measured deficit = 470.0 - 5 * 85.0 = 45.0s
    total_deficit = res["total_deficit_vs_baseline_s"]
    assert total_deficit == 45.0

    sum_components = (
        factors["caution_and_incidents_s"]
        + factors["pit_stop_and_transit_s"]
        + factors["traffic_s"]
        + factors["tyre_degradation_s"]
        + factors["unexplained_residual_s"]
    )

    # Must balance within +-0.01s
    assert abs(total_deficit - sum_components) <= 0.01
    assert res["accounting_identity_verified"] is True

    # Pit loss encompasses transit & stop; no double counting
    assert factors["pit_stop_and_transit_s"] == 21.0

    # Ensure prohibited direct subtraction of lane duration is guarded
    pit_res = analyze_pit_stops(stints_df, laps_df, "VER")
    assert pit_res["stops"][0]["estimated_track_in_out_loss_s"] is None
    assert pit_res["stops"][0]["track_loss_status"] == "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"


# ============================================================================
# 4. EXPLICIT UNKNOWN TRACK STATUS (NO GREEN DEFAULTING)
# ============================================================================

def test_missing_track_status_not_defaulted_to_green_or_caution():
    """Verify missing/unknown track status is never treated as green or caution."""
    # 1. is_track_status_strictly_green check
    assert is_track_status_strictly_green("UNKNOWN") is False
    assert is_track_status_strictly_green("") is False
    assert is_track_status_strictly_green(None) is False
    assert is_track_status_strictly_green(np.nan) is False

    # 2. compute_time_loss_breakdown check with UNKNOWN track status
    laps_df = pd.DataFrame({
        "driver": ["NOR"] * 3,
        "lap_number": [1, 2, 3],
        "lap_time_s": [85.0, 95.0, 85.0],  # Lap 2 is +10s slower, but track status is UNKNOWN
        "track_status": ["1", "UNKNOWN", "1"],
        "is_clean": [True, False, True],
        "is_pit_in": [False, False, False],
        "is_pit_out": [False, False, False],
        "tyre_age_laps": [1, 2, 3],
        "session_time_s": [85.0, 180.0, 265.0],
    })
    stints_df = pd.DataFrame({"driver": ["NOR"], "stint": [1], "start_lap": [1], "end_lap": [3], "pit_in_lap": [None], "pit_stop_duration_s": [None]})

    res = compute_time_loss_breakdown(laps_df, stints_df, "NOR")
    factors = res["loss_factors_s"]

    # Lap 2 deficit (+10s) must NOT be attributed to caution because status is UNKNOWN!
    assert factors["caution_and_incidents_s"] == 0.0
    # It must balance into unexplained residual
    assert factors["unexplained_residual_s"] == 10.0
    assert res["accounting_identity_verified"] is True


# ============================================================================
# 5. GROUNDED HEURISTIC TACTICAL CLASSIFICATIONS & EVIDENCE
# ============================================================================

def test_heuristic_undercut_and_isolated_stop():
    """Verify undercut attempt requires competitive proximity (<=3.0s) and identifies rival evidence."""
    # Scenario: LEC trailing VER by 1.5s on lap 19, pits on lap 20 (VER pits on lap 21)
    stints_df = pd.DataFrame({
        "driver": ["LEC", "LEC", "VER", "VER"],
        "stint": [1, 2, 1, 2],
        "start_lap": [1, 21, 1, 22],
        "end_lap": [20, 40, 21, 40],
        "pit_in_lap": [20, None, 21, None],
        "pit_stop_duration_s": [24.0, None, 24.0, None],
    })

    laps_df = pd.DataFrame({
        "driver": ["VER", "LEC", "VER", "LEC", "VER", "LEC", "VER", "LEC", "VER", "LEC"],
        "lap_number": [19, 19, 20, 20, 21, 21, 22, 22, 23, 23],
        "position": [1, 2, 1, 2, 1, 2, 2, 1, 2, 1],  # LEC jumps VER
        "session_time_s": [
            1000.0, 1001.5,  # Lap 19: LEC is 1.5s behind VER
            1085.0, 1098.0,  # Lap 20: LEC in pits
            1175.0, 1195.0,  # Lap 21: VER in pits, LEC on out-lap
            1260.0, 1255.0,  # Lap 22: S/F crossing
            1345.0, 1340.0,  # Lap 23
        ],
        "lap_time_s": [85.0, 85.0, 85.0, 96.5, 90.0, 97.0, 85.0, 60.0, 85.0, 85.0],
        "track_status": ["1"] * 10,
        "is_clean": [True] * 10,
    })

    # Evaluate LEC pit stop after out-lap completion
    res_undercut = analyze_pit_stops(stints_df, laps_df, "LEC", as_of_time_s=1300.0)
    stop = res_undercut["stops"][0]
    tac = stop["tactical_evidence"]

    assert tac["is_heuristic"] is True
    assert tac["classification"] == "UNDERCUT_ATTEMPT"
    assert tac["rival_driver"] == "VER"
    assert tac["pre_stop_gap_s"] == 1.5
    assert tac["lap_delta"] == -1
    assert "Heuristic: LEC pitted Lap 20 trailing VER by 1.5s" in tac["evidence_summary"]

    # Scenario B: Isolated car with no rival within 3.0s window
    laps_isolated = laps_df.copy()
    # Change lap 19 gap to 15.0s
    laps_isolated.loc[(laps_isolated["driver"] == "LEC") & (laps_isolated["lap_number"] == 19), "session_time_s"] = 1015.0

    res_isolated = analyze_pit_stops(stints_df, laps_isolated, "LEC", as_of_time_s=1300.0)
    tac_iso = res_isolated["stops"][0]["tactical_evidence"]
    assert tac_iso["classification"] == "REGULAR_STOP"
    assert "Isolated stop" in tac_iso["evidence_summary"]


# ============================================================================
# 6. HTTP API ENDPOINT INTEGRATION TESTS
# ============================================================================

def test_api_pit_strategy_endpoint():
    """Verify GET /api/races/{race_id}/pits returns valid schema and point-in-time responses."""
    # Test valid 2023 Monza ALB pit query
    resp = client.get("/api/races/2023_monza/pits?driver=ALB&as_of=2717.948")
    assert resp.status_code == 200
    data = resp.json()
    assert data["race_id"] == "2023_monza"
    assert data["driver"] == "ALB"
    assert data["status"] == "SUCCESS"
    assert data["as_of_time_s"] == 2717.948
    # ALB is at pit entry -> IN_PROGRESS
    assert len(data["stops"]) == 1
    assert data["stops"][0]["status"] == "IN_PROGRESS"
    assert data["total_completed_stops"] == 0

    # Test query after out-lap completion
    resp_done = client.get("/api/races/2023_monza/pits?driver=ALB&as_of=2830.0")
    assert resp_done.status_code == 200
    data_done = resp_done.json()
    assert data_done["total_completed_stops"] >= 1
    assert data_done["stops"][0]["status"] == "COMPLETED"
    assert data_done["stops"][0]["estimated_total_pit_loss_s"] is not None

    # Test pre-race response before T_0
    resp_pre = client.get("/api/races/2023_monza/pits?driver=ALB&as_of=0.0")
    assert resp_pre.status_code == 200
    data_pre = resp_pre.json()
    assert data_pre["status"] == "PRE_RACE"
    assert data_pre["total_completed_stops"] == 0

    # Test error on invalid driver
    resp_err = client.get("/api/races/2023_monza/pits?driver=XYZ")
    assert resp_err.status_code == 422
    assert "Driver 'XYZ' not found" in resp_err.json()["detail"]

    # Test error on nonexistent race
    resp_no_race = client.get("/api/races/1990_spacex/pits?driver=VER")
    assert resp_no_race.status_code == 404
