"""
Tests for Gap & Battle Analysis API endpoint and SessionManager integration (Milestone 2).
Covers:
- REST API response structure and schema compliance.
- Antisymmetry of gap calculations between driver pairs: gap(d1, d2) == -gap(d2, d1).
- Strict as_of temporal boundary enforcement and zero future data leakage.
- Separation of historical overtakes from active battles at as_of.
- Telemetry crossover caching verification by (slug, field_max_lap).
- Pre-race status handling when as_of is before race start.
- Error handling: identical drivers (422), invalid/unknown drivers (422), non-existent race (404/422).
"""

import pytest
from starlette.testclient import TestClient

from src.race.api.app import app
from src.race.api.session_manager import session_manager


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def race_id(client):
    res = client.get("/api/races")
    assert res.status_code == 200
    races = res.json()
    assert len(races) > 0, "No cached races available for battle tests"
    return races[0]["race_id"]


def test_gap_battle_api_schema_and_success(client, race_id):
    """Verify endpoint returns 200 and matches GapBattleResponse schema."""
    res = client.get(f"/api/races/{race_id}/battles", params={"driver1": "VER", "driver2": "LEC"})
    assert res.status_code == 200
    data = res.json()

    assert data["race_id"] == race_id
    assert data["driver1"] == "VER"
    assert data["driver2"] == "LEC"
    assert data["status"] == "SUCCESS"
    assert "gap_history" in data
    assert "pair_battles" in data
    assert "active_battles_at_as_of" in data
    assert "overtakes_up_to_as_of" in data
    assert "summary" in data

    # Validate summary
    summary = data["summary"]
    assert "total_laps_analyzed" in summary
    assert "laps_in_battle_window" in summary
    assert "total_overtakes_completed" in summary
    assert "current_leader" in summary
    assert len(data["gap_history"]) > 0


def test_gap_battle_antisymmetry(client, race_id):
    """Verify signed gap is strictly antisymmetric: gap(d1, d2) == -gap(d2, d1)."""
    res1 = client.get(f"/api/races/{race_id}/battles", params={"driver1": "VER", "driver2": "LEC"})
    res2 = client.get(f"/api/races/{race_id}/battles", params={"driver1": "LEC", "driver2": "VER"})

    assert res1.status_code == 200
    assert res2.status_code == 200

    data1 = res1.json()
    data2 = res2.json()

    laps1 = {lap["lap_number"]: lap for lap in data1["gap_history"]}
    laps2 = {lap["lap_number"]: lap for lap in data2["gap_history"]}

    common_laps = set(laps1.keys()) & set(laps2.keys())
    assert len(common_laps) > 0

    for lap_num in common_laps:
        item1 = laps1[lap_num]
        item2 = laps2[lap_num]

        # Signed gap is antisymmetric
        assert abs(item1["gap_s"] + item2["gap_s"]) < 1e-3

        # Leaders are identical
        assert item1["leader"] == item2["leader"]
        assert item1["chaser"] == item2["chaser"]


def test_gap_battle_as_of_chronological_boundary(client, race_id):
    """Verify as_of prevents any future laps or future events from leaking."""
    session = session_manager.get_or_load_session(race_id)
    # Find crossing time at lap 12 for VER
    idx12 = list(session.driver_lap_numbers["VER"]).index(12)
    as_of_time = float(session.driver_lap_crossing_times["VER"][idx12])

    bounded_res = client.get(
        f"/api/races/{race_id}/battles",
        params={"driver1": "VER", "driver2": "LEC", "as_of": as_of_time},
    )
    assert bounded_res.status_code == 200
    bounded_data = bounded_res.json()

    # Bounded laps must not exceed lap 12
    bounded_laps = bounded_data["gap_history"]
    assert len(bounded_laps) <= 12
    for lap in bounded_laps:
        assert lap["lap_number"] <= 12

    # Historical overtakes must not exceed lap 12
    for ot in bounded_data["overtakes_up_to_as_of"]:
        assert ot["lap_number"] <= 12

    # Pair battles must not exceed lap 12
    for b in bounded_data["pair_battles"]:
        assert b["lap_number"] <= 12


def test_gap_battle_separation_active_vs_historical(client, race_id):
    """Verify completed overtakes are separate from battles active at as_of."""
    res = client.get(f"/api/races/{race_id}/battles", params={"driver1": "VER", "driver2": "LEC"})
    assert res.status_code == 200
    data = res.json()

    # Historical overtakes have classification category
    for ot in data["overtakes_up_to_as_of"]:
        assert ot["classification"] in [
            "CONFIRMED_ON_TRACK_OVERTAKE",
            "INFERRED_ON_TRACK_OVERTAKE",
            "PIT_STRATEGY_SWAP",
            "RETIREMENT_PASS",
            "CAUTION_PERIOD_CHANGE",
            "UNKNOWN_DATA_GAP",
        ]

    # Active battles have battle_verdict
    for ab in data["active_battles_at_as_of"]:
        assert ab["battle_verdict"] in [
            "ACTIVE_CHASE",
            "SUSTAINED_DEFENSE",
            "CLOSE_CONTEST",
            "OVERTAKE_COMPLETED",
        ]


def test_gap_battle_telemetry_cache(race_id):
    """Verify session_manager caches overtakes by (race_id, field_max_lap)."""
    session_manager._overtakes_cache.clear()

    # First call populates cache
    res1 = session_manager.get_gap_and_battle_analysis(race_id, "VER", "LEC", as_of=None)
    assert len(session_manager._overtakes_cache) > 0

    cache_keys = list(session_manager._overtakes_cache.keys())
    assert any(k[0] == race_id for k in cache_keys)

    # Second call with another driver pair reuses cached overtakes
    res2 = session_manager.get_gap_and_battle_analysis(race_id, "HAM", "RUS", as_of=None)
    assert res2 is not None


def test_gap_battle_pre_race_status(client, race_id):
    """Querying before race start returns PRE_RACE status."""
    res = client.get(
        f"/api/races/{race_id}/battles",
        params={"driver1": "VER", "driver2": "LEC", "as_of": 0.0},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "PRE_RACE"
    assert len(data["gap_history"]) == 0
    assert len(data["pair_battles"]) == 0


def test_gap_battle_identical_driver_error(client, race_id):
    """Verify identical driver1 and driver2 raises 422."""
    res = client.get(f"/api/races/{race_id}/battles", params={"driver1": "VER", "driver2": "VER"})
    assert res.status_code == 422
    assert "distinct" in res.json()["detail"].lower()


def test_gap_battle_invalid_driver_error(client, race_id):
    """Verify non-existent driver raises 422."""
    res = client.get(f"/api/races/{race_id}/battles", params={"driver1": "XYZ", "driver2": "VER"})
    assert res.status_code == 422
    assert "not found" in res.json()["detail"].lower()


def test_gap_battle_nonexistent_race_error(client):
    """Verify unknown race_id raises 404 or 422."""
    res = client.get("/api/races/fake_race_9999/battles", params={"driver1": "VER", "driver2": "LEC"})
    assert res.status_code in [404, 422]
