"""
Comprehensive API integration tests for Checkpoint 2A: Driver Comparison API.
Tests point-in-time as_of line-crossing chronology, lap selection modes,
tyre-age synchronization, fuel correction, missing reference handling, and input validation.
"""

import pytest
from starlette.testclient import TestClient
from src.race.api.app import app
from src.race.replay.session import ReplaySession
from src.race.config import FUEL_CORRECTION_S_PER_LAP

client = TestClient(app)


@pytest.fixture(scope="module")
def session_2022():
    return ReplaySession(2022, "monza")


def test_compare_single_lap_basic(session_2022):
    """Verify single lap comparison for 2 drivers with reference delta."""
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "reference_driver": "VER",
        "sync_mode": "lap",
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["reference_driver"] == "VER"
    assert data["align_key"] == "lap_number"
    assert len(data["laps"]) == 2

    ver_item = next(item for item in data["laps"] if item["driver"] == "VER")
    lec_item = next(item for item in data["laps"] if item["driver"] == "LEC")

    assert ver_item["lap_number"] == 10
    assert lec_item["lap_number"] == 10
    assert ver_item["delta_to_reference_s"] == 0.0  # Reference vs self is 0.0

    # LEC delta must equal lec.lap_time_s - ver.lap_time_s
    expected_delta = round(lec_item["lap_time_s"] - ver_item["lap_time_s"], 3)
    assert abs(lec_item["delta_to_reference_s"] - expected_delta) < 1e-3


def test_compare_range_lap_mode(session_2022):
    """Verify lap range comparison across multiple laps."""
    payload = {
        "drivers": ["VER", "RUS"],
        "lap_mode": "range",
        "lap_range": [5, 8],
        "reference_driver": "VER",
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    laps = data["laps"]
    assert len(laps) == 8  # 4 laps each for 2 drivers
    assert all(5 <= item["lap_number"] <= 8 for item in laps)


def test_compare_fastest_lap_mode(session_2022):
    """Verify fastest lap mode selects the minimum clean lap time up to as_of."""
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "fastest",
        "clean_only": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert len(data["laps"]) == 2
    for item in data["laps"]:
        assert item["is_clean"] is True
        assert item["lap_time_s"] is not None


def test_compare_stint_lap_mode(session_2022):
    """Verify stint mode returns all completed laps in that stint."""
    payload = {
        "drivers": ["VER"],
        "lap_mode": "stint",
        "stint_number": 1,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert len(data["laps"]) > 0
    assert all(item["stint"] == 1 for item in data["laps"])


def test_compare_last_clean_lap_mode(session_2022):
    """Verify last_clean returns at most N clean laps up to as_of."""
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "last_clean",
        "last_n_clean": 3,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    ver_laps = [item for item in data["laps"] if item["driver"] == "VER"]
    assert len(ver_laps) == 3
    assert all(item["is_clean"] for item in ver_laps)


def test_compare_pre_race_as_of(session_2022):
    """Verify as_of before T_0 returns PRE_RACE with 0 completed laps."""
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 1,
        "as_of": session_2022.T_0 - 50.0,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "PRE_RACE"
    assert data["total_completed_laps"] == 0
    assert len(data["laps"]) == 0


def test_compare_per_driver_as_of_chronology_barrier(session_2022):
    """
    CRITICAL CAUSALITY TEST:
    Verify that query at t_as_of strictly excludes future laps.
    """
    t_lap15 = session_2022.get_leader_crossing_time(15)

    # At exact moment leader crosses Lap 15, no driver can have lap > 15
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "range",
        "lap_range": [1, 30],
        "as_of": t_lap15,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Max lap returned for any driver must be <= 15
    assert all(item["lap_number"] <= 15 for item in data["laps"])

    # Explicit request for future lap (Lap 25) at t_lap15 must return NO_DATA
    future_payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 25,
        "as_of": t_lap15,
    }
    future_resp = client.post("/api/races/2022_monza/compare", json=future_payload)
    assert future_resp.status_code == 200
    future_data = future_resp.json()
    assert future_data["status"] == "NO_DATA"
    assert len(future_data["laps"]) == 0


def test_compare_trailing_driver_asymmetric_barrier(session_2022):
    """
    Verify asymmetric crossing: if driver A crossed lap 10 but driver B has not yet
    crossed lap 10 at t_as_of, driver B only receives laps up to 9.
    """
    ver_t10 = session_2022.driver_lap_crossing_times["VER"][9]  # 0-indexed lap 10 crossing
    # Query at exact timestamp of VER lap 10 crossing
    payload = {
        "drivers": ["VER", "LAT"],  # LAT is significantly behind VER
        "lap_mode": "range",
        "lap_range": [1, 10],
        "as_of": ver_t10,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    ver_laps = [item["lap_number"] for item in data["laps"] if item["driver"] == "VER"]
    lat_laps = [item["lap_number"] for item in data["laps"] if item["driver"] == "LAT"]

    assert 10 in ver_laps
    assert 10 not in lat_laps  # LAT has not crossed line for lap 10 yet!


def test_compare_fuel_correction(session_2022):
    """Verify fuel correction adds 0.033 * (lap - 1) to lap times."""
    raw_payload = {
        "drivers": ["VER"],
        "lap_mode": "single",
        "lap_number": 10,
        "fuel_corrected": False,
        "as_of": session_2022.T_finish,
    }
    raw_resp = client.post("/api/races/2022_monza/compare", json=raw_payload)
    raw_time = raw_resp.json()["laps"][0]["lap_time_s"]

    corr_payload = {
        "drivers": ["VER"],
        "lap_mode": "single",
        "lap_number": 10,
        "fuel_corrected": True,
        "as_of": session_2022.T_finish,
    }
    corr_resp = client.post("/api/races/2022_monza/compare", json=corr_payload)
    corr_time = corr_resp.json()["laps"][0]["lap_time_s"]

    expected_diff = round(FUEL_CORRECTION_S_PER_LAP * 9, 3)
    assert abs((corr_time - raw_time) - expected_diff) < 1e-3


def test_compare_tyre_age_synchronization(session_2022):
    """
    Verify sync_mode='tyre_age' aligns by tyre_age_laps while preserving original lap_number.
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "range",
        "lap_range": [1, 10],
        "sync_mode": "tyre_age",
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["align_key"] == "tyre_age_laps"
    for item in data["laps"]:
        assert item["tyre_age_laps"] is not None
        assert item["lap_number"] is not None  # Original race lap is preserved


def test_compare_single_lap_tyre_age_interaction(session_2022):
    """
    Verify single lap with tyre_age sync aligns by reference driver's tyre age on that lap.
    """
    # On Lap 5, reference driver (VER) tyre age is 5
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 5,
        "reference_driver": "VER",
        "sync_mode": "tyre_age",
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert len(data["laps"]) == 2
    ver_item = next(item for item in data["laps"] if item["driver"] == "VER")
    lec_item = next(item for item in data["laps"] if item["driver"] == "LEC")

    assert ver_item["tyre_age_laps"] == 5
    assert lec_item["tyre_age_laps"] == 5


def test_compare_missing_reference_lap_delta_is_none(session_2022):
    """
    When reference driver has no lap for a key (e.g. retired driver as reference),
    comparison driver's delta_to_reference_s must be None, without crashing.
    """
    # VET retired early on lap 11 in 2022 Monza. On Lap 30, VET has no lap.
    payload = {
        "drivers": ["VER", "VET"],
        "lap_mode": "single",
        "lap_number": 30,
        "reference_driver": "VET",
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    ver_item = next(item for item in data["laps"] if item["driver"] == "VER")
    assert ver_item["lap_time_s"] is not None
    assert ver_item["delta_to_reference_s"] is None  # Cannot compute delta to absent reference!


def test_compare_input_validation():
    """Verify input error handling and HTTP status codes."""
    # Unknown driver -> 422
    resp_bad_driver = client.post("/api/races/2022_monza/compare", json={
        "drivers": ["XYZ"],
        "lap_mode": "single",
        "lap_number": 5
    })
    assert resp_bad_driver.status_code == 422

    # Empty driver list -> 422
    resp_empty = client.post("/api/races/2022_monza/compare", json={
        "drivers": [],
        "lap_mode": "single",
        "lap_number": 5
    })
    assert resp_empty.status_code == 422

    # Invalid range (start > end) -> 422
    resp_bad_range = client.post("/api/races/2022_monza/compare", json={
        "drivers": ["VER"],
        "lap_mode": "range",
        "lap_range": [15, 5]
    })
    assert resp_bad_range.status_code == 422

    # Unknown race -> 404
    resp_bad_race = client.post("/api/races/1999_spaceway/compare", json={
        "drivers": ["VER"],
        "lap_mode": "single",
        "lap_number": 5
    })
    assert resp_bad_race.status_code == 404

    # Invalid sync_mode -> 422
    resp_bad_sync = client.post("/api/races/2022_monza/compare", json={
        "drivers": ["VER"],
        "lap_mode": "single",
        "lap_number": 5,
        "sync_mode": "invalid_sync_mode"
    })
    assert resp_bad_sync.status_code == 422


def test_compare_multi_stint_tyre_age_range_crossing_pit_stop(session_2022):
    """
    REGRESSION TEST FOR MULTI-STINT TYRE-AGE RANGE SYNCHRONIZATION:
    In 2022 Monza, LEC pitted on lap 12 (stint 1 -> stint 2), while VER pitted on lap 25.
    When querying range [10, 16] with sync_mode='tyre_age' and reference_driver='VER':
    1. Laps 10 and 11: both are in Stint 1. Delata is computed against VER Stint 1 at matching tyre age.
    2. Laps 14+: LEC is in Stint 2. Because VER has no Stint 2 laps in [10, 16], delta_to_reference_s is None.
    3. Stint 2 tyre ages do not overwrite Stint 1 tyre ages in the reference dictionary.
    4. Actual race lap numbers (10, 11, 14, 15) and stint numbers are preserved.
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "range",
        "lap_range": [10, 16],
        "reference_driver": "VER",
        "sync_mode": "tyre_age",
        "clean_only": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["align_key"] == "tyre_age_laps"

    lec_items = [item for item in data["laps"] if item["driver"] == "LEC"]
    ver_items = [item for item in data["laps"] if item["driver"] == "VER"]

    # In [10, 16], VER is entirely in Stint 1
    assert all(item["stint"] == 1 for item in ver_items)

    # LEC has Stint 1 laps (laps 10, 11) and Stint 2 laps (laps 14, 15, 16; lap 12 pit-in & 13 pit-out dirty)
    lec_stint1 = [item for item in lec_items if item["stint"] == 1]
    lec_stint2 = [item for item in lec_items if item["stint"] == 2]

    assert len(lec_stint1) > 0
    assert len(lec_stint2) > 0

    # LEC Stint 1 laps should have non-null deltas against VER Stint 1 laps
    for item in lec_stint1:
        assert item["delta_to_reference_s"] is not None
        assert item["lap_number"] in (10, 11)

    # LEC Stint 2 laps CANNOT match VER Stint 1 laps; VER has no Stint 2 laps in [10, 16].
    # Deltas MUST be None (null), not erroneous deltas against Stint 1 tyre ages!
    for item in lec_stint2:
        assert item["delta_to_reference_s"] is None
        assert item["lap_number"] >= 14
        assert item["stint"] == 2


def test_compare_multi_stint_tyre_age_both_drivers_pit(session_2022):
    """
    Verify wider range [10, 28] where both LEC (pitted lap 12) and VER (pitted lap 25) have Stint 2 laps.
    Stint 2 tyre age 2 for LEC (lap 14) and VER (lap 27) match on composite key (stint=2, tyre_age=2).
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "range",
        "lap_range": [10, 28],
        "reference_driver": "VER",
        "sync_mode": "tyre_age",
        "clean_only": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    lec_items = [item for item in data["laps"] if item["driver"] == "LEC"]
    ver_items = [item for item in data["laps"] if item["driver"] == "VER"]

    # VER has both Stint 1 and Stint 2 laps in this range
    ver_stint1_ages = {item["tyre_age_laps"]: item for item in ver_items if item["stint"] == 1}
    ver_stint2_ages = {item["tyre_age_laps"]: item for item in ver_items if item["stint"] == 2}

    assert len(ver_stint1_ages) > 0
    assert len(ver_stint2_ages) > 0

    # In Stint 2, tyre age 2 occurs for VER on lap 27, and for LEC on lap 14
    lec_s2_a2 = next((item for item in lec_items if item["stint"] == 2 and item["tyre_age_laps"] == 2), None)
    ver_s2_a2 = next((item for item in ver_items if item["stint"] == 2 and item["tyre_age_laps"] == 2), None)

    assert lec_s2_a2 is not None
    assert ver_s2_a2 is not None
    assert lec_s2_a2["lap_number"] == 14  # Preserves actual race lap
    assert ver_s2_a2["lap_number"] == 27  # Preserves actual race lap
    assert lec_s2_a2["delta_to_reference_s"] is not None

    expected_delta = round(lec_s2_a2["lap_time_s"] - ver_s2_a2["lap_time_s"], 3)
    assert abs(lec_s2_a2["delta_to_reference_s"] - expected_delta) < 1e-3


# ==================================================
# Checkpoint 2B-3A: Mini-Sector API & Telemetry Contract Tests
# ==================================================


def test_compare_mini_sectors_backward_compatibility(session_2022):
    """
    Test 1: Existing comparison requests without include_mini_sectors retain exact backward compatibility.
    Mini-sector fields remain None and include_mini_sectors is False.
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["include_mini_sectors"] is False
    assert len(data["laps"]) == 2
    for lap_item in data["laps"]:
        assert lap_item["mini_sectors"] is None
        assert lap_item["telemetry_status"] is None
        assert lap_item["provider"] is None
        assert lap_item["precision_tier"] is None


def test_compare_mini_sectors_valid_single_lap(session_2022):
    """
    Test 2: A valid single-lap request with include_mini_sectors=True returns 20 mini-sector timings
    with valid telemetry status, provider metadata, and correct delta calculations.
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "reference_driver": "VER",
        "include_mini_sectors": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["include_mini_sectors"] is True
    assert len(data["laps"]) == 2

    ver = next(item for item in data["laps"] if item["driver"] == "VER")
    lec = next(item for item in data["laps"] if item["driver"] == "LEC")

    # VER (FastF1 high-precision provider)
    assert ver["telemetry_status"] == "VALID"
    assert ver["provider"] == "FastF1"
    assert ver["precision_tier"] == "HIGH_PRECISION_TRANSPONDER"
    assert len(ver["mini_sectors"]) == 20

    for ms in ver["mini_sectors"]:
        assert 1 <= ms["mini_sector_id"] <= 20
        assert ms["sector"] in (1, 2, 3)
        assert ms["start_m"] < ms["end_m"]
        assert ms["duration_s"] is not None and ms["duration_s"] > 0
        assert ms["status"] == "VALID"
        assert ms["delta_to_reference_s"] == 0.0  # Reference vs self is exactly 0.0

    # LEC deltas vs VER
    assert lec["telemetry_status"] == "VALID"
    assert len(lec["mini_sectors"]) == 20
    for i, ms in enumerate(lec["mini_sectors"]):
        ver_ms = ver["mini_sectors"][i]
        assert ms["status"] == "VALID"
        assert ms["duration_s"] is not None and ms["duration_s"] > 0
        expected_delta = round(ms["duration_s"] - ver_ms["duration_s"], 4)
        assert abs(ms["delta_to_reference_s"] - expected_delta) < 1e-4


def test_compare_mini_sectors_fastest_lap_mode(session_2022):
    """
    Test 3: Fastest-lap mode correctly selects the eligible fastest lap per driver and populates mini-sectors.
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "fastest",
        "clean_only": True,
        "include_mini_sectors": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert len(data["laps"]) == 2
    for item in data["laps"]:
        assert item["mini_sectors"] is not None
        assert len(item["mini_sectors"]) == 20
        assert item["telemetry_status"] == "VALID"


def test_compare_mini_sectors_reference_driver_deltas(session_2022):
    """
    Test 4: Reference-driver deltas are zero only for valid matching mini-sectors.
    Faster sectors produce negative delta, slower sectors produce positive delta.
    """
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "reference_driver": "VER",
        "include_mini_sectors": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    ver = next(item for item in data["laps"] if item["driver"] == "VER")
    lec = next(item for item in data["laps"] if item["driver"] == "LEC")

    # All reference mini-sectors must be 0.0
    for ms in ver["mini_sectors"]:
        assert ms["delta_to_reference_s"] == 0.0

    # LEC deltas verify sign: negative when faster, positive when slower
    deltas = [ms["delta_to_reference_s"] for ms in lec["mini_sectors"]]
    assert any(d < 0 for d in deltas)  # At least one sector faster
    assert any(d > 0 for d in deltas)  # At least one sector slower


def test_compare_mini_sectors_missing_telemetry_graceful(session_2022, monkeypatch):
    """
    Test 5: Missing telemetry for a driver does not crash the request.
    Produces status='MISSING' and null durations/deltas without fabricating values.
    """
    from src.race.api.session_manager import session_manager

    session_manager._mini_sector_cache.clear()
    orig_fn = session_manager._get_mini_sectors_for_lap

    def mock_get_ms(sess, drv, lap_num, lap_time):
        if drv == "LEC":
            return {
                "telemetry_status": "MISSING",
                "provider": None,
                "precision_tier": None,
                "mini_sectors": [],
            }
        return orig_fn(sess, drv, lap_num, lap_time)

    monkeypatch.setattr(session_manager, "_get_mini_sectors_for_lap", mock_get_ms)

    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "reference_driver": "VER",
        "include_mini_sectors": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    lec = next(item for item in data["laps"] if item["driver"] == "LEC")
    assert lec["telemetry_status"] == "MISSING"
    assert len(lec["mini_sectors"]) == 20
    for ms in lec["mini_sectors"]:
        assert ms["status"] == "MISSING"
        assert ms["duration_s"] is None
        assert ms["delta_to_reference_s"] is None
        assert ms["avg_speed_kmh"] is None

    # Reference driver remains completely valid
    ver = next(item for item in data["laps"] if item["driver"] == "VER")
    assert ver["telemetry_status"] == "VALID"
    assert all(ms["delta_to_reference_s"] == 0.0 for ms in ver["mini_sectors"])


def test_compare_mini_sectors_unreliable_data_gap_distinguishable(session_2022, monkeypatch):
    """
    Test 6: UNRELIABLE_DATA_GAP (> 2.0s telemetry gap) is preserved and clearly distinguishable from MISSING.
    """
    from src.race.api.session_manager import session_manager

    session_manager._mini_sector_cache.clear()
    orig_fn = session_manager._get_mini_sectors_for_lap

    def mock_get_ms_gap(sess, drv, lap_num, lap_time):
        if drv == "LEC":
            return {
                "telemetry_status": "UNRELIABLE_DATA_GAP",
                "provider": "FastF1",
                "precision_tier": "HIGH_PRECISION_TRANSPONDER",
                "mini_sectors": [],
            }
        return orig_fn(sess, drv, lap_num, lap_time)

    monkeypatch.setattr(session_manager, "_get_mini_sectors_for_lap", mock_get_ms_gap)

    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "include_mini_sectors": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    lec = next(item for item in data["laps"] if item["driver"] == "LEC")
    assert lec["telemetry_status"] == "UNRELIABLE_DATA_GAP"
    assert len(lec["mini_sectors"]) == 20
    for ms in lec["mini_sectors"]:
        assert ms["status"] == "UNRELIABLE_DATA_GAP"
        assert ms["duration_s"] is None
        assert ms["delta_to_reference_s"] is None


def test_compare_mini_sectors_invalid_lap_mode_rejection():
    """
    Test 7: Mini-sectors requested with unsupported lap modes (range, stint, last_clean)
    produce a clear 422 validation error.
    """
    for mode, extra in [("range", {"lap_range": [5, 10]}), ("stint", {"stint_number": 1}), ("last_clean", {"last_n_clean": 3})]:
        payload = {
            "drivers": ["VER", "LEC"],
            "lap_mode": mode,
            "include_mini_sectors": True,
            **extra,
        }
        resp = client.post("/api/races/2022_monza/compare", json=payload)
        assert resp.status_code == 422
        assert "Mini-sector comparison is not supported for lap_mode" in resp.json()["detail"]


def test_compare_mini_sectors_as_of_chronology_barrier(session_2022):
    """
    Test 8: An as_of timestamp before a lap crossing strictly prevents that lap's telemetry from being exposed.
    """
    # At moment leader crosses Lap 8, Lap 10 has not occurred
    t_lap8 = session_2022.get_leader_crossing_time(8)

    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "include_mini_sectors": True,
        "as_of": t_lap8,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    # Lap 10 was not completed at t_lap8 -> NO_DATA, zero laps
    assert data["status"] == "NO_DATA"
    assert len(data["laps"]) == 0


def test_compare_mini_sectors_unreliable_or_missing_reference_yields_null_delta(session_2022, monkeypatch):
    """
    Test 9: When reference driver observation is missing or unreliable, non-reference drivers
    must have null deltas (not error or zero).
    """
    from src.race.api.session_manager import session_manager

    session_manager._mini_sector_cache.clear()
    orig_fn = session_manager._get_mini_sectors_for_lap

    def mock_get_ms_missing_ref(sess, drv, lap_num, lap_time):
        if drv == "VER":  # VER is reference driver
            return {
                "telemetry_status": "MISSING",
                "provider": None,
                "precision_tier": None,
                "mini_sectors": [],
            }
        return orig_fn(sess, drv, lap_num, lap_time)

    monkeypatch.setattr(session_manager, "_get_mini_sectors_for_lap", mock_get_ms_missing_ref)

    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "reference_driver": "VER",
        "include_mini_sectors": True,
        "as_of": session_2022.T_finish,
    }
    resp = client.post("/api/races/2022_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    ver = next(item for item in data["laps"] if item["driver"] == "VER")
    lec = next(item for item in data["laps"] if item["driver"] == "LEC")

    # VER is missing
    assert ver["telemetry_status"] == "MISSING"
    for ms in ver["mini_sectors"]:
        assert ms["delta_to_reference_s"] is None

    # LEC is VALID, but delta is None because reference has no valid sector
    assert lec["telemetry_status"] == "VALID"
    for ms in lec["mini_sectors"]:
        assert ms["status"] == "VALID"
        assert ms["duration_s"] is not None
        assert ms["delta_to_reference_s"] is None



def test_compare_mini_sectors_openf1_session_integration():
    """
    Test 10: Real cached OpenF1 session (2023 Monza) returns valid mini-sectors with
    ESTIMATED_COORDINATE_DERIVED precision tier and provider metadata.
    """
    session_2023 = ReplaySession(2023, "monza")
    payload = {
        "drivers": ["VER", "LEC"],
        "lap_mode": "single",
        "lap_number": 10,
        "reference_driver": "VER",
        "include_mini_sectors": True,
        "as_of": session_2023.T_finish,
    }
    resp = client.post("/api/races/2023_monza/compare", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["status"] == "SUCCESS"
    assert data["include_mini_sectors"] is True
    assert len(data["laps"]) == 2

    ver = next(item for item in data["laps"] if item["driver"] == "VER")
    assert ver["telemetry_status"] == "VALID"
    assert ver["provider"] == "OpenF1"
    assert ver["precision_tier"] == "ESTIMATED_COORDINATE_DERIVED"
    assert len(ver["mini_sectors"]) == 20
    for ms in ver["mini_sectors"]:
        assert ms["status"] == "VALID"
        assert ms["duration_s"] is not None and ms["duration_s"] > 0
        assert ms["avg_speed_kmh"] is not None and ms["avg_speed_kmh"] > 0
        assert ms["delta_to_reference_s"] == 0.0



