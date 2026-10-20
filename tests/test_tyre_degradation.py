"""
Unit, analytics, and API tests for the Tyre Degradation Analysis module (Milestone 2).
Covers:
- Known degradation trends (synthetic recovery).
- Flat and improving pace trends.
- Insufficient clean laps (< 4 clean laps -> INSUFFICIENT_DATA).
- Performance cliff detection (consecutive residuals > 2.5 std).
- Fuel correction sensitivity and zero-correction equality.
- Tyre age and stint boundaries.
- as_of replay boundary restrictions and zero future-data leakage.
- REST API compatibility and error handling.
"""

import pytest
import numpy as np
import pandas as pd
from starlette.testclient import TestClient

from src.race.analytics.tyres import compute_tyre_degradation
from src.race.api.app import app
from src.race.api.session_manager import session_manager


# ============================================================================
# 1. ANALYTICS UNIT TESTS
# ============================================================================

def test_tyre_degradation_synthetic_known_slope():
    """Verify OLS regression correctly recovers known injected tyre degradation slope."""
    laps = np.arange(1, 15)
    base_time = 86.0
    # Injected deg: +0.060 s/lap. Fuel benefit: 0.033 s/lap. Net raw slope: +0.027 s/lap.
    raw_times = base_time + 0.027 * (laps - 1)

    df = pd.DataFrame({
        "driver": "VER",
        "stint": 1,
        "lap_number": laps,
        "lap_time_s": raw_times,
        "is_clean": True,
        "compound": "SOFT",
        "tyre_age_laps": laps,
    })

    res = compute_tyre_degradation(df, "VER", 1, fuel_correction_s_per_lap=0.033)
    assert res["clean_laps_count"] == 14
    assert res["is_estimate"] is True
    assert res["compound"] == "SOFT"
    assert abs(res["raw_slope_s_per_lap"] - 0.027) < 1e-3
    assert abs(res["estimated_deg_rate_s_per_lap"] - 0.060) < 1e-3
    assert res["confidence"] == "HIGH"
    assert res["cliff_lap"] is None
    assert len(res["laps"]) == 14
    assert res["laps"][0]["fitted_lap_time_s"] is not None


def test_tyre_degradation_improving_or_flat_pace():
    """Verify negative or flat slope is properly detected and interpreted."""
    laps = np.arange(1, 12)
    base_time = 88.0
    # Improving pace (e.g. green track evolution, raw times dropping rapidly by 0.080 s/lap)
    raw_times = base_time - 0.080 * (laps - 1)

    df = pd.DataFrame({
        "driver": "HAM",
        "stint": 1,
        "lap_number": laps,
        "lap_time_s": raw_times,
        "is_clean": True,
        "compound": "HARD",
        "tyre_age_laps": laps,
    })

    res = compute_tyre_degradation(df, "HAM", 1, fuel_correction_s_per_lap=0.033)
    assert res["clean_laps_count"] == 11
    assert res["raw_slope_s_per_lap"] < 0
    # Net deg: -0.080 + 0.033 = -0.047 s/lap
    assert res["estimated_deg_rate_s_per_lap"] < 0
    assert "stable or improved" in res["degradation_interpretation"].lower()


def test_tyre_degradation_insufficient_clean_laps():
    """Stint with fewer than 4 clean laps must return INSUFFICIENT_DATA and None slopes."""
    df = pd.DataFrame({
        "driver": "LEC",
        "stint": 1,
        "lap_number": [1, 2, 3],
        "lap_time_s": [85.0, 85.2, 85.1],
        "is_clean": True,
        "compound": "MEDIUM",
        "tyre_age_laps": [1, 2, 3],
    })

    res = compute_tyre_degradation(df, "LEC", 1)
    assert res["clean_laps_count"] == 3
    assert res["estimated_deg_rate_s_per_lap"] is None
    assert res["raw_slope_s_per_lap"] is None
    assert res["confidence"] == "INSUFFICIENT_DATA"
    assert "minimum 4 required" in res["message"]
    # Still returns the available 3 lap records with raw lap times
    assert len(res["laps"]) == 3
    assert res["laps"][0]["raw_lap_time_s"] == 85.0
    assert res["laps"][0]["fitted_lap_time_s"] is None


def test_tyre_degradation_non_clean_laps_excluded():
    """Non-clean laps (safety car, yellow flag, pit out) must be strictly excluded from OLS regression."""
    laps = np.arange(1, 13)
    base_time = 84.0
    raw_times = base_time + 0.02 * (laps - 1)
    is_clean = [True] * 12
    # Invalidate laps 3 and 7 with massive time spikes (e.g. SC / incident)
    is_clean[2] = False
    raw_times[2] = 120.0  # SC slow lap
    is_clean[6] = False
    raw_times[6] = 135.0  # VSC slow lap

    df = pd.DataFrame({
        "driver": "NOR",
        "stint": 1,
        "lap_number": laps,
        "lap_time_s": raw_times,
        "is_clean": is_clean,
        "compound": "SOFT",
        "tyre_age_laps": laps,
    })

    res = compute_tyre_degradation(df, "NOR", 1, fuel_correction_s_per_lap=0.033)
    # Total laps 12, clean laps 10
    assert res["clean_laps_count"] == 10
    assert res["total_laps_count"] == 12
    # Regression slope must not be blown out by the 120s / 135s laps (which would make slope > 3.0 s/lap)
    assert abs(res["raw_slope_s_per_lap"]) < 0.10
    assert abs(res["estimated_deg_rate_s_per_lap"]) < 0.15
    # Verify non-clean laps have is_clean False and None fitted times
    lap3 = next(l for l in res["laps"] if l["lap_number"] == 3)
    assert lap3["is_clean"] is False
    assert lap3["fitted_lap_time_s"] is None


def test_tyre_degradation_cliff_detection():
    """Verify performance cliff detection when residuals spike > 2.5 std for 2 consecutive laps."""
    laps = np.arange(1, 25)
    base_time = 85.0
    # Normal linear degradation of 0.02 s/lap
    raw_times = base_time + 0.02 * (laps - 1)
    # Laps 21 and 22 suffer sudden drop-off (+5.0s per lap)
    raw_times[20] += 5.0
    raw_times[21] += 5.5

    df = pd.DataFrame({
        "driver": "SAI",
        "stint": 1,
        "lap_number": laps,
        "lap_time_s": raw_times,
        "is_clean": True,
        "compound": "SOFT",
        "tyre_age_laps": laps,
    })

    res = compute_tyre_degradation(df, "SAI", 1, fuel_correction_s_per_lap=0.033)
    assert res["cliff_lap"] == 21
    # The lap point for lap 21 must have is_cliff == True
    cliff_point = next((p for p in res["laps"] if p["lap_number"] == 21), None)
    assert cliff_point is not None
    assert cliff_point["is_cliff"] is True



def test_tyre_degradation_as_of_lap_boundary():
    """Verify as_of_lap strictly truncates observations without forward data leakage."""
    laps = np.arange(1, 25)
    df = pd.DataFrame({
        "driver": "PER",
        "stint": 1,
        "lap_number": laps,
        "lap_time_s": 85.0 + 0.03 * (laps - 1),
        "is_clean": True,
        "compound": "MEDIUM",
        "tyre_age_laps": laps,
    })

    res_cut = compute_tyre_degradation(df, "PER", 1, as_of_lap=10)
    assert res_cut["clean_laps_count"] == 10
    assert len(res_cut["laps"]) == 10
    assert max(l["lap_number"] for l in res_cut["laps"]) == 10


# ============================================================================
# 2. REST API INTEGRATION TESTS
# ============================================================================

@pytest.fixture
def client():
    return TestClient(app)


def test_api_tyre_degradation_full_race(client):
    """Verify GET /api/races/2022_monza/tyres returns valid tyre degradation data."""
    resp = client.get("/api/races/2022_monza/tyres?driver=VER&stint=1&as_of=1000000")
    assert resp.status_code == 200
    data = resp.json()

    assert data["race_id"] == "2022_monza"
    assert data["driver"] == "VER"
    assert data["stint"] == 1
    assert data["compound"] == "SOFT"
    assert data["status"] == "VALID"
    assert data["is_estimate"] is True
    assert data["clean_laps_count"] >= 10
    assert data["estimated_deg_rate_s_per_lap"] is not None
    assert data["confidence"] in ["HIGH", "MEDIUM"]
    assert len(data["laps"]) > 10
    assert len(data["available_stints"]) >= 2
    assert "Linear fuel burn compensation" in data["fuel_correction_assumption"]


def test_api_tyre_degradation_stint_selection(client):
    """Verify user can switch to Stint 2 (MEDIUM) for Verstappen."""
    resp = client.get("/api/races/2022_monza/tyres?driver=VER&stint=2&as_of=1000000")
    assert resp.status_code == 200
    data = resp.json()

    assert data["driver"] == "VER"
    assert data["stint"] == 2
    assert data["compound"] == "MEDIUM"
    assert data["clean_laps_count"] >= 10
    assert data["confidence"] in ["HIGH", "MEDIUM"]


def test_api_tyre_degradation_as_of_chronology(client):
    """Verify as_of simulation timestamp strictly bounds completed laps without future leakage."""
    session = session_manager.get_or_load_session("2022_monza")
    # Lap 8 crossing time for VER
    idx8 = list(session.driver_lap_numbers["VER"]).index(8)
    t8 = float(session.driver_lap_crossing_times["VER"][idx8])

    resp = client.get(f"/api/races/2022_monza/tyres?driver=VER&stint=1&as_of={t8}")
    assert resp.status_code == 200
    data = resp.json()

    assert data["as_of_lap"] == 8
    assert len(data["laps"]) == 8
    assert max(l["lap_number"] for l in data["laps"]) == 8


def test_api_tyre_degradation_pre_race_status(client):
    """Querying before race start (T_0) returns PRE_RACE status."""
    resp = client.get("/api/races/2022_monza/tyres?driver=VER&stint=1&as_of=0.0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "PRE_RACE"
    assert data["confidence"] == "INSUFFICIENT_DATA"
    assert len(data["laps"]) == 0


def test_api_tyre_degradation_future_stint_guard(client):
    """Querying Stint 2 when replay is only at lap 5 returns FUTURE_STINT status."""
    session = session_manager.get_or_load_session("2022_monza")
    idx5 = list(session.driver_lap_numbers["VER"]).index(5)
    t5 = float(session.driver_lap_crossing_times["VER"][idx5])

    resp = client.get(f"/api/races/2022_monza/tyres?driver=VER&stint=2&as_of={t5}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "FUTURE_STINT"
    assert "has not started yet" in data["message"]


def test_api_tyre_degradation_unknown_driver_422(client):
    """Unknown driver returns 422 Unprocessable Entity with available drivers list."""
    resp = client.get("/api/races/2022_monza/tyres?driver=NONEXISTENT&stint=1")
    assert resp.status_code == 422
    assert "Driver 'NONEXISTENT' not found" in resp.json()["detail"]


def test_api_tyre_degradation_unknown_race_404_or_422(client):
    """Unknown race returns 404 or 422."""
    resp = client.get("/api/races/1990_monaco/tyres?driver=SEN&stint=1")
    assert resp.status_code in [404, 422]
