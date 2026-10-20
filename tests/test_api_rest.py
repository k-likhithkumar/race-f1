"""
REST API test suite for RACE Replay Engine (Phase 3B Checkpoint A).
Tests endpoint contracts, race catalog discovery, state serialization, and error responses.
"""

import pytest
from fastapi.testclient import TestClient

from src.race.api.app import app

client = TestClient(app)


def test_list_races():
    """Verifies /api/races discovers cached Monza sessions and provides complete metadata."""
    response = client.get("/api/races")
    assert response.status_code == 200
    races = response.json()
    assert isinstance(races, list)
    assert len(races) >= 2

    # Check 2022 Monza exists and has accurate boundary information
    monza_2022 = next((r for r in races if r["race_id"] == "2022_monza"), None)
    assert monza_2022 is not None
    assert monza_2022["year"] == 2022
    assert monza_2022["gp"] == "Monza"
    assert monza_2022["total_laps"] == 53
    assert monza_2022["driver_count"] == 20
    assert monza_2022["T_0"] > 3000.0
    assert monza_2022["T_finish"] > monza_2022["T_0"]


def test_get_replay_state_default():
    """Verifies /api/races/{race_id}/state returns valid RaceState at T_0 when as_of is omitted."""
    response = client.get("/api/races/2022_monza/state")
    assert response.status_code == 200
    data = response.json()
    assert "session_time_s" in data
    assert "leaderboard" in data
    assert len(data["leaderboard"]) == 20
    assert data["track_status"] in ["PRE_RACE", "1", "UNKNOWN"]

    # Verify transponder timing position exists on every driver
    for d in data["leaderboard"]:
        assert "driver" in d
        assert "timing_position" in d
        assert "status" in d
        assert "compound" in d


def test_get_replay_state_with_as_of():
    """Verifies /api/races/{race_id}/state?as_of=... returns point-in-time snapshot without mutating clock."""
    response = client.get("/api/races/2022_monza/state?as_of=4400.0")
    assert response.status_code == 200
    data = response.json()
    assert data["session_time_s"] == 4400.0
    assert len(data["leaderboard"]) == 20


def test_get_clock_status():
    """Verifies /api/races/{race_id}/clock returns authoritative ReplayClock status."""
    response = client.get("/api/races/2022_monza/clock")
    assert response.status_code == 200
    clock = response.json()
    assert clock["state"] in ["PAUSED", "PLAYING"]
    assert clock["current_lap"] >= 1
    assert clock["total_laps"] == 53
    assert clock["T_finish"] > clock["T_0"]


def test_unknown_race_404():
    """Verifies unknown race IDs return 404 Not Found."""
    response = client.get("/api/races/1990_fictional/state")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_invalid_as_of_422():
    """Verifies malformed as_of timestamp returns 422 Unprocessable Entity."""
    response = client.get("/api/races/2022_monza/state?as_of=not_a_float")
    assert response.status_code == 422


def test_get_track_outline_2022_monza():
    """Verifies /api/races/2022_monza/track returns complete track outline schema and coordinates."""
    response = client.get("/api/races/2022_monza/track")
    assert response.status_code == 200
    data = response.json()
    assert data["year"] == 2022
    assert "Monza" in data["circuit_name"] or "ITALIA" in data["circuit_name"]
    assert data["total_distance_m"] > 5000.0
    assert len(data["sectors"]) == 3
    assert len(data["mini_sectors"]) == 20
    assert len(data["points"]) == 1000

    # Verify first point structure
    pt0 = data["points"][0]
    assert "x" in pt0 and "y" in pt0 and "dist_m" in pt0
    assert pt0["dist_m"] == 0.0


def test_get_track_outline_2023_monza():
    """Verifies /api/races/2023_monza/track returns complete track outline schema and coordinates."""
    response = client.get("/api/races/2023_monza/track")
    assert response.status_code == 200
    data = response.json()
    assert data["year"] == 2023
    assert "Monza" in data["circuit_name"]
    assert data["total_distance_m"] > 5000.0
    assert len(data["sectors"]) == 3
    assert len(data["mini_sectors"]) == 20
    assert len(data["points"]) == 1000


def test_get_track_outline_unknown_race_404():
    """Verifies /api/races/{race_id}/track returns 404 for nonexistent race."""
    response = client.get("/api/races/1990_fictional/track")
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_serve_frontend_index():
    """Verifies root endpoint serves the built frontend HTML."""
    response = client.get("/")
    assert response.status_code == 200
    assert "RACE F1" in response.text
    assert "root" in response.text
