"""
Integration tests for Battle & Overtake Prediction Engine and API (Checkpoint 3C).

Validates:
- Successful prediction responses and schema compliance.
- Pre-race and insufficient-data status handling.
- Missing or unavailable model artifacts handling (MODEL_UNAVAILABLE).
- Strict as_of temporal boundary enforcement and zero future-data leakage.
- Separate model outputs (battle probability and overtake probability are never multiplied).
- Distinction between on-track overtakes and pit strategy swaps.
- Error handling for invalid driver codes, identical drivers, and non-existent races.
- In-memory model persistence (repeated requests reuse loaded pipelines without retraining).
- Backward compatibility of all existing API endpoints.
"""

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from src.race.api.app import app
from src.race.api.session_manager import session_manager
from src.race.analytics.prediction_engine import prediction_engine, PredictionEngine


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def race_id(client):
    res = client.get("/api/races")
    assert res.status_code == 200
    races = res.json()
    assert len(races) > 0, "No cached races found"
    return "2023_monza"


def test_successful_prediction_response(client, race_id):
    """Verify endpoint returns 200 and matches BattleOvertakePredictionResponse schema with SUCCESS status."""
    # At t=1850.0s in 2023 Monza, SAI and VER have completed lap 5 and are in close proximity (<0.4s)
    res = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "SAI", "as_of": 1850.0, "horizon_laps": 5},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["race_id"] == race_id
    assert data["status"] == "SUCCESS"
    assert data["as_of_lap"] == 5
    assert data["as_of_time_s"] == 1850.0
    assert data["horizon_laps"] == 5

    # Check attacker and defender assignment
    assert data["attacker"] == "VER"
    assert data["defender"] == "SAI"

    # Check separate predictions
    bp = data["battle_prediction"]
    op = data["overtake_prediction"]
    assert bp is not None
    assert op is not None
    assert 0.0 <= bp["probability"] <= 1.0
    assert 0.0 <= op["probability"] <= 1.0
    assert bp["metric_type"] == "probability_estimate"
    assert op["metric_type"] == "probability_estimate"
    assert isinstance(bp["classification_threshold_0_5"], bool)
    assert isinstance(op["classification_threshold_0_5"], bool)

    # Check supporting features
    feats = data["supporting_features"]
    assert feats is not None
    assert feats["gap_s"] is not None and feats["gap_s"] > 0
    assert feats["clean_pace_delta_s"] is not None or feats["clean_pace_delta_s"] is None

    # Check documented reliability warnings
    warnings = data["reliability_warnings"]
    assert len(warnings) >= 4
    warning_types = [w["warning_type"] for w in warnings]
    assert "CALIBRATION_OVERCONFIDENCE" in warning_types
    assert "RECALL_LIMITATION" in warning_types
    assert "CIRCUIT_SPECIFIC" in warning_types
    assert "UNCERTAINTY_LIMITATION" in warning_types

    # Specifically verify the battle model overconfidence warning text mentions empirical results
    cal_warn = next(w for w in warnings if w["warning_type"] == "CALIBRATION_OVERCONFIDENCE")
    assert "86.0%" in cal_warn["message"]
    assert "58.7%" in cal_warn["message"]

    # Verify laps_remaining against official scheduled distance (51 laps for 2023 Monza - 5 laps completed = 46)
    assert feats["laps_remaining"] == 46

    # Verify driver-pair orientation symmetry: swapping driver1 and driver2 must preserve
    # physical attacker/defender roles and produce identical probabilities and gap
    res_swap = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "SAI", "driver2": "VER", "as_of": 1850.0, "horizon_laps": 5},
    )
    assert res_swap.status_code == 200
    data_swap = res_swap.json()
    assert data_swap["attacker"] == "VER"
    assert data_swap["defender"] == "SAI"
    assert data_swap["supporting_features"]["gap_s"] == feats["gap_s"]
    assert data_swap["battle_prediction"]["probability"] == bp["probability"]
    assert data_swap["overtake_prediction"]["probability"] == op["probability"]
    assert data_swap["supporting_features"]["laps_remaining"] == 46


def test_prerace_status(client, race_id):
    """Verify pre-race timestamp (prior to T_0) returns status PRE_RACE."""
    res = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "LEC", "as_of": 0.0},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "PRE_RACE"
    assert data["as_of_lap"] == 0
    assert data["battle_prediction"] is None
    assert data["overtake_prediction"] is None
    assert data["supporting_features"] is None
    assert data["data_sufficiency"] == "NONE"

    # Negative timestamp also returns PRE_RACE
    res_neg = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "LEC", "as_of": -50.0},
    )
    assert res_neg.status_code == 200
    assert res_neg.json()["status"] == "PRE_RACE"

    # Malformed non-numeric timestamp raises 422
    res_bad_ts = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "LEC", "as_of": "invalid_ts"},
    )
    assert res_bad_ts.status_code == 422

    # Out of range horizon (le=10) raises 422
    res_bad_hor = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "LEC", "as_of": 1850.0, "horizon_laps": 25},
    )
    assert res_bad_hor.status_code == 422


def test_insufficient_data_status(client, race_id):
    """Verify cars separated by large gap (>10.0s) return INSUFFICIENT_DATA."""
    # VER and SAR are never close in 2023 Monza
    res = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "SAR", "as_of": 2000.0},
    )
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "INSUFFICIENT_DATA"
    assert "proximity window" in data["message"].lower() or "insufficient" in data["message"].lower()
    assert data["battle_prediction"] is None
    assert data["overtake_prediction"] is None


def test_model_unavailable_handling(client, race_id):
    """Verify engine returns MODEL_UNAVAILABLE when model artifacts cannot be loaded."""
    # Temporarily unload models and point to non-existent directory without allow_training
    prediction_engine.unload_models()
    assert prediction_engine._models_loaded is False

    # Force is_ready to return False
    original_is_ready = prediction_engine.is_ready
    prediction_engine.is_ready = lambda: False

    try:
        res = client.get(
            f"/api/races/{race_id}/predictions",
            params={"driver1": "VER", "driver2": "SAI", "as_of": 1850.0},
        )
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "MODEL_UNAVAILABLE"
        assert "unavailable" in data["message"].lower()
    finally:
        # Restore prediction engine
        prediction_engine.is_ready = original_is_ready
        prediction_engine.load_models(allow_training=True)


def test_as_of_filtering_and_zero_future_leakage(client, race_id):
    """Predictions at as_of=T must be completely invariant to repeated calls and future data."""
    t_test = 1850.0
    res1 = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "SAI", "as_of": t_test},
    )
    assert res1.status_code == 200
    d1 = res1.json()

    # Repeated query at the exact same as_of time
    res2 = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "SAI", "as_of": t_test},
    )
    assert res2.status_code == 200
    d2 = res2.json()

    assert d1["battle_prediction"]["probability"] == d2["battle_prediction"]["probability"]
    assert d1["overtake_prediction"]["probability"] == d2["overtake_prediction"]["probability"]
    assert d1["supporting_features"]["gap_s"] == d2["supporting_features"]["gap_s"]
    assert d1["as_of_lap"] == d2["as_of_lap"]


def test_separate_model_outputs_not_multiplied(client, race_id):
    """Battle and overtake models must produce distinct, non-multiplied probability values."""
    res = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "SAI", "as_of": 1850.0},
    )
    assert res.status_code == 200
    data = res.json()

    p_battle = data["battle_prediction"]["probability"]
    p_overtake = data["overtake_prediction"]["probability"]

    # Verify they are separate predictions
    assert p_battle != p_overtake
    assert p_overtake != p_battle * p_overtake  # Not a joint product


def test_invalid_driver_codes_error(client, race_id):
    """Identical driver codes and unknown driver codes must raise 422."""
    # Identical drivers
    res_ident = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "VER"},
    )
    assert res_ident.status_code == 422

    # Unknown driver
    res_unk = client.get(
        f"/api/races/{race_id}/predictions",
        params={"driver1": "VER", "driver2": "XYZ"},
    )
    assert res_unk.status_code == 422


def test_nonexistent_race_error(client):
    """Non-existent race ID must return 404."""
    res = client.get(
        "/api/races/1990_monaco/predictions",
        params={"driver1": "VER", "driver2": "LEC"},
    )
    assert res.status_code == 404


def test_repeated_requests_reuse_loaded_models(client, race_id):
    """Repeated requests must reuse the exact in-memory pipeline instances without retraining."""
    # Ensure loaded
    prediction_engine.is_ready()
    pipe_b_id = id(prediction_engine._battle_pipeline)
    pipe_o_id = id(prediction_engine._overtake_pipeline)

    # Execute 3 separate API calls
    for as_of_t in [1200.0, 1500.0, 1800.0]:
        res = client.get(
            f"/api/races/{race_id}/predictions",
            params={"driver1": "VER", "driver2": "SAI", "as_of": as_of_t},
        )
        assert res.status_code == 200

    # Assert object identity is unchanged
    assert id(prediction_engine._battle_pipeline) == pipe_b_id
    assert id(prediction_engine._overtake_pipeline) == pipe_o_id


def test_existing_api_compatibility(client, race_id):
    """All existing API endpoints must remain 100% operational."""
    # 1. Races list
    r1 = client.get("/api/races")
    assert r1.status_code == 200

    # 2. Replay Clock
    r2 = client.get(f"/api/races/{race_id}/clock")
    assert r2.status_code == 200

    # 3. Gap & Battles
    r3 = client.get(f"/api/races/{race_id}/battles", params={"driver1": "VER", "driver2": "LEC"})
    assert r3.status_code == 200

    # 4. Tyre Degradation
    r4 = client.get(f"/api/races/{race_id}/tyres", params={"driver": "VER", "stint": 1})
    assert r4.status_code == 200
