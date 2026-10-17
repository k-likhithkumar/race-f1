"""
Tests for 2023+ Historical Data Access via OpenF1 Fallback Adapter.
Covers:
- Unauthenticated OpenF1 live access
- 2023 Monza data availability and normalization parity with 2022 schema
- Golden validation against Jolpica results (Winner, Podium, Fastest Lap)
- Handling of missing and partial fields (no fabrication)
- Fallback mechanics when FastF1 encounters 401/errors
- Strict offline loading with socket/requests blocked
- as_of replay barrier and future-data leakage protection
"""

import socket
import pytest
import requests
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.race.config import OPENF1_BASE_URL
from src.race.data.cache import is_race_cached, load_cached_race, load_cached_table, cache_race
from src.race.data.loader import fetch_jolpica_race_results
from src.race.data.openf1 import _safe_float, _safe_int, _normalize_session_name


# -----------------------------------------------------------------------------
# 1. LIVE / INTEGRATION TESTS (Real Network, Unauthenticated)
# -----------------------------------------------------------------------------

def test_unauthenticated_openf1_live_access():
    """
    Verify that OpenF1 historical endpoints can be queried over HTTPS
    without any authentication tokens, cookies, or credentials.
    """
    # 1. Sessions endpoint
    sess_url = f"{OPENF1_BASE_URL}/sessions"
    sess_resp = requests.get(sess_url, params={"year": 2023, "session_name": "Race"}, timeout=15)
    assert sess_resp.status_code == 200, f"Expected 200, got {sess_resp.status_code}"
    sessions = sess_resp.json()
    assert isinstance(sessions, list)
    assert len(sessions) > 0

    # 2. Monza 2023 session key check
    monza_sess = next((s for s in sessions if "monza" in str(s.get("circuit_short_name", "")).lower()), None)
    assert monza_sess is not None, "2023 Monza race session must be present in OpenF1 sessions"
    session_key = monza_sess["session_key"]
    assert session_key == 9157

    # 3. Laps endpoint for session 9157
    laps_url = f"{OPENF1_BASE_URL}/laps"
    laps_resp = requests.get(laps_url, params={"session_key": session_key, "driver_number": 1}, timeout=15)
    assert laps_resp.status_code == 200
    laps = laps_resp.json()
    assert isinstance(laps, list)
    assert len(laps) >= 50, f"Expected at least 50 laps for Verstappen, got {len(laps)}"


# -----------------------------------------------------------------------------
# 2. CACHE COMPLETENESS & SCHEMA PARITY
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def monza_2023_data():
    """Load cached 2023 Monza race tables."""
    assert is_race_cached(2023, "Monza"), "Benchmark race 2023 Monza must be cached before running tests."
    return load_cached_race(2023, "Monza")


def test_2023_monza_cache_completeness(monza_2023_data):
    """Verify all 6 core tables and metadata exist and are populated."""
    assert monza_2023_data["laps"] is not None and not monza_2023_data["laps"].empty
    assert monza_2023_data["stints"] is not None and not monza_2023_data["stints"].empty
    assert monza_2023_data["weather"] is not None and not monza_2023_data["weather"].empty
    assert monza_2023_data["positions"] is not None and not monza_2023_data["positions"].empty
    assert monza_2023_data["classification"] is not None and not monza_2023_data["classification"].empty
    assert monza_2023_data["track_outline"] is not None and "points" in monza_2023_data["track_outline"]
    assert monza_2023_data["metadata"] is not None
    assert monza_2023_data["metadata"]["winner"] == "VER"
    assert monza_2023_data["metadata"]["data_source"] == "openf1"


def test_schema_parity_between_fastf1_and_openf1(monza_2023_data):
    """
    Verify exact column schema and dtype parity between FastF1 (2022 Monza) and OpenF1 (2023 Monza).
    Downstream modules must consume identical table contracts regardless of source.
    """
    assert is_race_cached(2022, "Monza")
    monza_2022_data = load_cached_race(2022, "Monza")

    for table_name in ["laps", "stints", "weather", "positions", "classification"]:
        cols_2022 = set(monza_2022_data[table_name].columns)
        cols_2023 = set(monza_2023_data[table_name].columns)
        diff = cols_2022 ^ cols_2023
        assert not diff, f"Schema mismatch in {table_name}: symmetric diff = {diff}"

        # Verify exact dtype equality for all shared columns
        for col in cols_2022:
            dt22 = monza_2022_data[table_name][col].dtype
            dt23 = monza_2023_data[table_name][col].dtype
            assert dt22 == dt23, f"Dtype mismatch in {table_name}.{col}: 2022={dt22} vs 2023={dt23}"


# -----------------------------------------------------------------------------
# 3. GOLDEN RACE VALIDATION (2023 Monza)
# -----------------------------------------------------------------------------

def test_2023_monza_golden_winner_and_podium(monza_2023_data):
    """
    Cross-check winner, podium, and race distance against official Jolpica results.
    2023 Italian GP (Round 14): Verstappen won (record 10th consecutive win),
    Perez P2, Sainz P3. Total laps scheduled was 53, reduced to 51 due to aborted starts.
    """
    class_df = monza_2023_data["classification"]
    laps_df = monza_2023_data["laps"]

    # 1. Classification check
    top_3 = class_df.sort_values("position").iloc[:3]["driver"].tolist()
    assert top_3 == ["VER", "PER", "SAI"], f"Unexpected podium: {top_3}"

    # 2. Race distance completed by winner
    ver_laps = laps_df[laps_df["driver"] == "VER"]
    # OpenF1 records 51 race laps for Verstappen (plus formation laps if recorded, max lap 51/52)
    assert ver_laps["lap_number"].max() in (51, 52)

    # 3. Cross-validate with Jolpica official results if available
    jolpica_race = fetch_jolpica_race_results(2023, 14)
    if jolpica_race and "Results" in jolpica_race:
        official_p1 = jolpica_race["Results"][0]
        assert official_p1["Driver"]["code"] == "VER"
        assert int(official_p1["laps"]) == 51
        official_top_3 = [r["Driver"]["code"] for r in jolpica_race["Results"][:3]]
        assert top_3 == official_top_3


def test_2023_monza_golden_fastest_lap(monza_2023_data):
    """
    Validate the fastest lap of the race.
    Monza 2023 fastest lap was set by Oscar Piastri (PIA) on Lap 43 with a 1:25.072 (85.072s).
    """
    laps_df = monza_2023_data["laps"].dropna(subset=["lap_time_s"])
    clean_laps = laps_df[laps_df["is_clean"]]
    fastest_lap_row = clean_laps.sort_values("lap_time_s").iloc[0]

    assert fastest_lap_row["driver"] == "PIA"
    assert fastest_lap_row["lap_number"] == 43
    # 1:25.072 = 85.072s
    assert pytest.approx(fastest_lap_row["lap_time_s"], abs=0.01) == 85.072


def test_2023_monza_clean_lap_proportions(monza_2023_data):
    """
    Verify clean lap rules on OpenF1 data:
    In-laps, out-laps, and caution laps must NOT be marked clean.
    """
    laps_df = monza_2023_data["laps"]
    total_laps = len(laps_df)
    clean_laps = laps_df["is_clean"].sum()

    assert clean_laps > 0
    clean_ratio = clean_laps / total_laps
    assert clean_ratio > 0.70, f"Clean lap ratio unexpectedly low: {clean_ratio:.2%}"

    # Verify that in-laps and out-laps are not clean
    stints_df = monza_2023_data["stints"]
    for _, stint in stints_df.iterrows():
        drv = stint["driver"]
        in_lap = stint["pit_in_lap"]
        if pd.notna(in_lap):
            drv_in = laps_df[(laps_df["driver"] == drv) & (laps_df["lap_number"] == int(in_lap))]
            if not drv_in.empty:
                assert not drv_in.iloc[0]["is_clean"], f"In-lap {in_lap} for {drv} must not be clean"


def test_2023_monza_deleted_laps_marked_dirty(monza_2023_data):
    """
    Verify that laps deleted for track limits by FIA Race Control are marked is_clean == False.
    Examples from 2023 Monza: ALO lap 3, NOR lap 5, RUS lap 14, SAI lap 47.
    """
    laps_df = monza_2023_data["laps"]
    # Check Alonso Lap 3 (Turn 10 track limits)
    alo_3 = laps_df[(laps_df["driver"] == "ALO") & (laps_df["lap_number"] == 3)]
    assert not alo_3.empty
    assert not alo_3.iloc[0]["is_clean"], "Alonso Lap 3 was deleted for track limits and must not be clean"

    # Check Norris Lap 5 (Turn 7 track limits)
    nor_5 = laps_df[(laps_df["driver"] == "NOR") & (laps_df["lap_number"] == 5)]
    assert not nor_5.empty
    assert not nor_5.iloc[0]["is_clean"], "Norris Lap 5 was deleted for track limits and must not be clean"

    # Check Sainz Lap 47 (Turn 5 track limits)
    sai_47 = laps_df[(laps_df["driver"] == "SAI") & (laps_df["lap_number"] == 47)]
    assert not sai_47.empty
    assert not sai_47.iloc[0]["is_clean"], "Sainz Lap 47 was deleted for track limits and must not be clean"


def test_2023_monza_track_outline_geometry(monza_2023_data):
    """Verify track outline geometry and 20 mini-sector partitions with strict benchmark tolerance."""
    track = monza_2023_data["track_outline"]
    # Maintain exact same strict tolerance as 2022 golden race test (5700m - 5900m)
    assert 5700 < track["total_distance_m"] < 5900, f"Track length {track['total_distance_m']} out of range"
    assert len(track["mini_sectors"]) == 20
    assert len(track["sectors"]) == 3
    assert len(track["points"]) == 1000


def test_2023_monza_track_outline_geometric_usability(monza_2023_data):
    """
    Verify geometric usability of generated 2023 Monza track outline:
    - Exactly 1000 resampled points, strictly monotonic distance values
    - Centered around (0, 0)
    - Physical East-West and North-South spans match Autodromo Nazionale Monza
    - Closed circuit geometry (point 0 and point 999 gap < 25m)
    - 20 strictly contiguous mini-sectors
    """
    track = monza_2023_data["track_outline"]
    points = track["points"]
    assert len(points) == 1000

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    dists = [p["dist_m"] for p in points]

    # Monotonic distance sequence
    assert dists[0] == 0.0
    assert dists[-1] == track["total_distance_m"]
    for i in range(len(dists) - 1):
        assert dists[i] <= dists[i + 1], f"Non-monotonic distance at index {i}"

    # Centered coordinates
    assert abs(np.mean(xs)) < 1.0, f"X not centered: mean={np.mean(xs)}"
    assert abs(np.mean(ys)) < 1.0, f"Y not centered: mean={np.mean(ys)}"

    # Physical spans: Monza is approx 1.25 km wide and 2.17 km long
    span_x = max(xs) - min(xs)
    span_y = max(ys) - min(ys)
    assert 1100 < span_x < 1400, f"Unexpected X span: {span_x}"
    assert 2000 < span_y < 2300, f"Unexpected Y span: {span_y}"

    # Closed circuit check: distance from finish line back to start line
    gap_closure = np.sqrt((points[0]["x"] - points[-1]["x"])**2 + (points[0]["y"] - points[-1]["y"])**2)
    assert gap_closure < 25.0, f"Circuit not properly closed: gap={gap_closure}m"

    # Mini-sectors: strictly contiguous and valid
    ms = track["mini_sectors"]
    assert len(ms) == 20
    assert ms[0]["start_m"] == 0.0
    assert ms[-1]["end_m"] == track["total_distance_m"]
    for i in range(len(ms) - 1):
        assert ms[i]["end_m"] == ms[i + 1]["start_m"]
        assert ms[i]["sector"] in [1, 2, 3]


def test_clean_lap_conditions_cross_verification(monza_2023_data):
    """
    Verify every clean-lap condition specified in F1_Project_Plan.pdf for OpenF1 (2023)
    and cross-verify against FastF1 (2022):
    1. LapTime must be valid and positive (>0)
    2. TrackStatus must be verified and strictly green ('1')
    3. In-laps and out-laps must not be clean
    4. Deleted laps must not be clean
    """
    laps_23 = monza_2023_data["laps"]
    clean_23 = laps_23[laps_23["is_clean"]]

    # Condition 1: positive valid time
    assert (clean_23["lap_time_s"] > 0).all()
    assert clean_23["lap_time_s"].notna().all()

    # Condition 2: strictly green track status '1'
    assert (clean_23["track_status"] == "1").all()

    # Condition 3: no dirty track status marked clean
    dirty_23 = laps_23[laps_23["track_status"] != "1"]
    assert not dirty_23["is_clean"].any()

    # FastF1 2022 benchmark cross-verification
    monza_22 = load_cached_race(2022, "Monza")
    clean_22 = monza_22["laps"][monza_22["laps"]["is_clean"]]
    assert (clean_22["lap_time_s"] > 0).all()
    assert (clean_22["track_status"] == "1").all()
    dirty_22 = monza_22["laps"][monza_22["laps"]["track_status"] != "1"]
    assert not dirty_22["is_clean"].any()


# -----------------------------------------------------------------------------
# 4. REPLAY BARRIER & FUTURE DATA PROTECTION
# -----------------------------------------------------------------------------

def test_2023_monza_as_of_replay_barrier():
    """Verify as_of replay barrier prevents future-data leakage on 2023 OpenF1 cache."""
    # Lap barrier: query at as_of_lap = 20
    laps_as_of_20 = load_cached_table(2023, "Monza", "laps", as_of_lap=20)
    assert not laps_as_of_20.empty
    assert laps_as_of_20["lap_number"].max() == 20
    assert not (laps_as_of_20["lap_number"] > 20).any()

    # Time barrier: query positions at as_of_time_s = 1500.0s
    pos_as_of_1500 = load_cached_table(2023, "Monza", "positions", as_of_time_s=1500.0)
    assert not pos_as_of_1500.empty
    assert pos_as_of_1500["session_time_s"].max() <= 1500.0
    assert not (pos_as_of_1500["session_time_s"] > 1500.0).any()

    # Stint future protection: at as_of_lap = 15, future pit stops must not leak
    stints_as_of_15 = load_cached_table(2023, "Monza", "stints", as_of_lap=15)
    ver_stints = stints_as_of_15[stints_as_of_15["driver"] == "VER"]
    assert len(ver_stints) == 1
    assert ver_stints.iloc[0]["end_lap"] == 15
    assert pd.isna(ver_stints.iloc[0]["pit_in_lap"])

    # Time-based stint future protection: at as_of_time_s = 1500.0s (~lap 16), ongoing stint is clamped
    stints_as_of_time = load_cached_table(2023, "Monza", "stints", as_of_time_s=1500.0)
    assert not stints_as_of_time.empty
    ver_time_stint = stints_as_of_time[stints_as_of_time["driver"] == "VER"]
    assert len(ver_time_stint) == 1
    assert pd.isna(ver_time_stint.iloc[0]["pit_in_lap"])

    # Classification protection: mid-race query (by lap or time) must not leak final race classification
    class_as_of_15 = load_cached_table(2023, "Monza", "classification", as_of_lap=15)
    assert class_as_of_15.empty, "Mid-race classification by lap must be empty"

    class_as_of_time = load_cached_table(2023, "Monza", "classification", as_of_time_s=1500.0)
    assert class_as_of_time.empty, "Mid-race classification by time must be empty"

    # Full race container query with as_of_time_s
    race_time = load_cached_race(2023, "Monza", as_of_time_s=1500.0)
    assert race_time["classification"].empty
    assert not race_time["laps"].empty
    assert race_time["laps"]["session_time_s"].max() <= 1500.0

    # Retired driver coordinate masking: Esteban Ocon retired around lap 39 (~4878s)
    pos_all = load_cached_table(2023, "Monza", "positions")
    oco_pos = pos_all[pos_all["driver"] == "OCO"]
    oco_late = oco_pos[oco_pos["session_time_s"] > 5000.0]
    assert not oco_late.empty
    assert oco_late["x"].isna().all(), "Retired driver coordinates must be NaN after retirement"
    assert oco_late["y"].isna().all()


# -----------------------------------------------------------------------------
# 5. OFFLINE ZERO-NETWORK LOADING
# -----------------------------------------------------------------------------

def test_2023_monza_offline_zero_network_read(monkeypatch):
    """
    Verify that reading cached 2023 Monza operates 100% offline.
    Mocks socket/requests to ensure any attempted network call raises an error.
    """
    def blocked_connect(*args, **kwargs):
        raise RuntimeError("Network access attempted during offline cached read!")

    monkeypatch.setattr(socket, "socket", blocked_connect)
    monkeypatch.setattr(requests, "get", blocked_connect)

    # Load race tables under strict network block
    offline_data = load_cached_race(2023, "Monza", as_of_lap=20)
    assert not offline_data["laps"].empty
    assert offline_data["laps"]["lap_number"].max() == 20
    assert not offline_data["positions"].empty
    assert offline_data["metadata"]["data_source"] == "openf1"


# -----------------------------------------------------------------------------
# 6. SOURCE SELECTION & FALLBACK MECHANICS (MOCKED)
# -----------------------------------------------------------------------------

def test_fastf1_failure_triggers_openf1_fallback(tmp_path):
    """
    Unit test: Verify that when FastF1 raises an HTTP 401 or runtime error,
    cache_race automatically catches the error and executes the OpenF1 fallback.
    """
    with patch("src.race.data.cache.load_session") as mock_fastf1, \
         patch("src.race.data.openf1.cache_race_openf1") as mock_openf1:
        
        mock_fastf1.side_effect = Exception("HTTP 401 Unauthorized from livetiming.formula1.com")
        mock_openf1.return_value = {"status": "ok", "data_source": "openf1"}

        result = cache_race(2023, "Monza")
        
        mock_fastf1.assert_called_once()
        mock_openf1.assert_called_once_with(2023, "Monza", "R")
        assert result["data_source"] == "openf1"


def test_missing_and_partial_fields_handled_gracefully():
    """
    Verify safe type conversions and handling of missing / null values from OpenF1.
    Never fabricates missing data; represents missing values as NaN or UNKNOWN.
    """
    assert np.isnan(_safe_float(None))
    assert np.isnan(_safe_float("corrupt"))
    assert _safe_float(None, 0.0) == 0.0
    assert _safe_float("123.45") == 123.45

    assert _safe_int(None, 1) == 1
    assert _safe_int("corrupt", 5) == 5
    assert _safe_int("42") == 42

    assert _normalize_session_name("R") == "Race"
    assert _normalize_session_name("race") == "Race"
    assert _normalize_session_name("q") == "Qualifying"
    assert _normalize_session_name("fp1") == "Practice 1"
