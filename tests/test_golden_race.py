"""
Integration and Golden Race Validation Suite for Stage 1.
Validates cached 2022 Monza against Jolpica-F1 official results and verifies offline behavior.
"""

import pytest
import pandas as pd

from src.race.data.cache import is_race_cached, load_cached_table, load_cached_race
from src.race.data.loader import fetch_jolpica_race_results, fetch_jolpica_pitstops


@pytest.fixture(scope="module")
def monza_data():
    """Load cached 2022 Monza race tables."""
    assert is_race_cached(2022, "Monza"), "Benchmark race 2022 Monza must be cached before running golden tests."
    return load_cached_race(2022, "Monza")


def test_parquet_cache_completeness(monza_data):
    """Verify all 6 core tables and metadata exist and are populated."""
    assert monza_data["laps"] is not None and not monza_data["laps"].empty
    assert monza_data["stints"] is not None and not monza_data["stints"].empty
    assert monza_data["weather"] is not None and not monza_data["weather"].empty
    assert monza_data["positions"] is not None and not monza_data["positions"].empty
    assert monza_data["classification"] is not None and not monza_data["classification"].empty
    assert monza_data["track_outline"] is not None and "points" in monza_data["track_outline"]
    assert monza_data["metadata"] is not None and monza_data["metadata"]["winner"] == "VER"


def test_golden_race_winner_and_distance(monza_data):
    """
    Cross-check winner and race distance against Jolpica official results.
    2022 Italian GP (Round 16): Verstappen won, 53 laps completed.
    """
    laps_df = monza_data["laps"]
    class_df = monza_data["classification"]

    # 1. Official Classification in FastF1
    winner_row = class_df[class_df["position"] == 1]
    assert not winner_row.empty
    assert winner_row.iloc[0]["driver"] == "VER"
    assert winner_row.iloc[0]["status"] == "Finished"

    # 2. Race distance
    winner_laps = laps_df[laps_df["driver"] == "VER"]
    assert winner_laps["lap_number"].max() == 53

    # 3. Cross-validate with Jolpica if network is available
    jolpica_race = fetch_jolpica_race_results(2022, 16)
    if jolpica_race and "Results" in jolpica_race:
        official_p1 = jolpica_race["Results"][0]
        assert official_p1["Driver"]["code"] == "VER"
        assert int(official_p1["laps"]) == 53


def test_golden_classification_matching(monza_data):
    """
    Validate top-5 finishing order against official Jolpica results.
    Monza 2022 Top 5: P1 VER, P2 LEC, P3 RUS, P4 SAI, P5 HAM.
    """
    class_df = monza_data["classification"]
    top_5_fastf1 = class_df[class_df["position"].isin([1, 2, 3, 4, 5])].sort_values("position")
    fastf1_order = top_5_fastf1["driver"].tolist()

    jolpica_race = fetch_jolpica_race_results(2022, 16)
    if jolpica_race and "Results" in jolpica_race:
        jolpica_order = [r["Driver"]["code"] for r in jolpica_race["Results"][:5]]
        assert fastf1_order == jolpica_order, f"Classification mismatch: {fastf1_order} vs {jolpica_order}"
    else:
        # Fallback expected order
        assert fastf1_order == ["VER", "LEC", "RUS", "SAI", "HAM"]


def test_golden_fastest_lap(monza_data):
    """
    Validate the fastest lap of the race.
    Monza 2022 fastest lap was set by Sergio Perez on Lap 46 (1:24.030).
    """
    laps_df = monza_data["laps"].dropna(subset=["lap_time_s"])
    fastest_lap_row = laps_df.sort_values("lap_time_s").iloc[0]

    assert fastest_lap_row["driver"] == "PER"
    assert fastest_lap_row["lap_number"] == 46
    # 1:24.030 = 84.030s
    assert pytest.approx(fastest_lap_row["lap_time_s"], abs=0.005) == 84.030


def test_clean_lap_proportions(monza_data):
    """
    Verify clean-lap identification.
    In Monza 2022, Daniel Ricciardo stopped on track on lap 47, causing a Safety Car until finish.
    Laps 48-53 should be marked dirty (non-green / SC), while earlier green laps are clean.
    """
    laps_df = monza_data["laps"]
    total_laps = len(laps_df)
    clean_laps = laps_df["is_clean"].sum()

    # Verify that clean laps exist and are realistically proportioned (> 65% of race laps)
    clean_ratio = clean_laps / total_laps
    assert clean_ratio > 0.65, f"Clean lap ratio unexpectedly low: {clean_ratio:.2%}"

    # Verify that safety car finish laps (Lap 49-53) are marked dirty
    sc_laps = laps_df[laps_df["lap_number"] >= 49]
    assert not sc_laps["is_clean"].any(), "Safety car laps at race finish must not be marked clean"


def test_as_of_replay_barrier():
    """Verify load_cached_table enforces strict as_of bounds on the cached benchmark."""
    # Lap barrier: query at as_of_lap = 20
    laps_as_of_20 = load_cached_table(2022, "Monza", "laps", as_of_lap=20)
    assert not laps_as_of_20.empty
    assert laps_as_of_20["lap_number"].max() == 20
    assert not (laps_as_of_20["lap_number"] > 20).any()

    # Time barrier: query positions at as_of_time_s = 2000.0s
    pos_as_of_2000 = load_cached_table(2022, "Monza", "positions", as_of_time_s=2000.0)
    assert not pos_as_of_2000.empty
    assert pos_as_of_2000["session_time_s"].max() <= 2000.0
    assert not (pos_as_of_2000["session_time_s"] > 2000.0).any()

    # Stint future protection: at as_of_lap = 15, Verstappen's future pit on lap 25 must NOT leak
    stints_as_of_15 = load_cached_table(2022, "Monza", "stints", as_of_lap=15)
    ver_stints = stints_as_of_15[stints_as_of_15["driver"] == "VER"]
    assert len(ver_stints) == 1, "Future stints must not be visible"
    assert ver_stints.iloc[0]["end_lap"] == 15, "Ongoing stint end_lap must be clamped to as_of_lap"
    assert pd.isna(ver_stints.iloc[0]["pit_in_lap"]), "Future pit_in_lap must not be visible"
    assert pd.isna(ver_stints.iloc[0]["pit_stop_duration_s"]), "Future pit stop duration must not be visible"

    # Classification protection: mid-race query must not leak final race classification
    class_as_of_15 = load_cached_table(2022, "Monza", "classification", as_of_lap=15)
    assert class_as_of_15.empty, "Final race classification must not leak mid-race"

    # Retired driver coordinate masking: Vettel retired on lap 10 (finished running at ~4786s); all points after 4800s must be NaN
    pos_all = load_cached_table(2022, "Monza", "positions")
    vet_pos = pos_all[pos_all["driver"] == "VET"]
    vet_late = vet_pos[vet_pos["session_time_s"] > 4800.0]
    assert not vet_late.empty
    assert vet_late["x"].isna().all(), "Retired driver coordinates must be NaN after retirement"
    assert vet_late["y"].isna().all()


def test_track_outline_geometry(monza_data):
    """Verify track outline geometry and 20 mini-sector partitions."""
    track = monza_data["track_outline"]
    circuit_upper = track["circuit_name"].upper()
    assert any(term in circuit_upper for term in ["MONZA", "ITALIA", "ITALIAN"])
    # Monza official length is ~5.793 km (5793 meters)
    assert 5700 < track["total_distance_m"] < 5900

    # Must contain exactly 20 mini-sectors (PDF Topic 4 & 5)
    mini_sectors = track["mini_sectors"]
    assert len(mini_sectors) == 20
    assert mini_sectors[0]["start_m"] == 0.0
    assert mini_sectors[-1]["end_m"] == track["total_distance_m"]

    # Sectors 1, 2, 3 must be present
    sectors = track["sectors"]
    assert len(sectors) == 3
    assert sectors[0]["sector"] == 1
    assert sectors[1]["sector"] == 2
    assert sectors[2]["sector"] == 3


def test_offline_zero_network_read(monkeypatch):
    """
    Verify that reading a cached race operates 100% offline.
    Mocks socket/requests to ensure any attempted network call raises an error.
    """
    import socket
    import requests

    def blocked_connect(*args, **kwargs):
        raise RuntimeError("Network access attempted during offline cached read!")

    monkeypatch.setattr(socket, "socket", blocked_connect)
    monkeypatch.setattr(requests, "get", blocked_connect)

    # Load race tables under strict network block
    offline_data = load_cached_race(2022, "Monza", as_of_lap=15)
    assert not offline_data["laps"].empty
    assert offline_data["laps"]["lap_number"].max() == 15
    assert not offline_data["positions"].empty

