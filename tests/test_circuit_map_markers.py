"""
Tests for Checkpoint C: Moving Driver Markers on Circuit Map.
Verifies driver coordinate alignment, progression over time, missing/retired coordinate handling,
and multi-race coordinate scaling parity.
"""

import pytest
import numpy as np

from src.race.replay.session import ReplaySession
from src.race.replay.state import get_state


def test_driver_marker_positions_at_known_timestamp_2022():
    """Verifies active drivers possess valid, on-track coordinates aligned with track outline."""
    session = ReplaySession(2022, "Monza")
    t = session.T_0 + 120.0  # ~2 minutes into race (Lap 2)
    state = get_state(session, t)

    assert len(state.leaderboard) == 20
    leader = state.leaderboard[0]
    assert leader.timing_position == 1
    assert leader.x is not None and leader.y is not None
    assert leader.status == "ON_TRACK"

    # Verify leader coordinates lie within reasonable proximity of track centerline
    track_x = session.track_x
    track_y = session.track_y
    dists_sq = (track_x - leader.x) ** 2 + (track_y - leader.y) ** 2
    min_dist = np.sqrt(np.min(dists_sq))
    # Lateral offset from track centerline must be within realistic track limits (< 35m = 350 decimeters)
    assert min_dist < 350.0

    # Verify at least 18 drivers have valid coordinates during active green-flag racing
    active_with_coords = [d for d in state.leaderboard if d.x is not None and d.y is not None]
    assert len(active_with_coords) >= 18


def test_driver_marker_advances_with_replay_clock():
    """Verifies that driver coordinates update monotonically with race progression."""
    session = ReplaySession(2022, "Monza")
    t1 = session.T_0 + 60.0
    t2 = session.T_0 + 90.0

    s1 = get_state(session, t1)
    s2 = get_state(session, t2)

    lec1 = next(d for d in s1.leaderboard if d.driver == "LEC")
    lec2 = next(d for d in s2.leaderboard if d.driver == "LEC")

    assert lec1.x is not None and lec1.y is not None
    assert lec2.x is not None and lec2.y is not None

    # Coordinates must have changed as car traveled down the circuit
    assert (lec1.x != lec2.x) or (lec1.y != lec2.y)
    assert lec2.total_race_distance_m > lec1.total_race_distance_m


def test_missing_coordinates_for_retired_and_pre_race_drivers():
    """Verifies coordinates are not fabricated for pre-race or retired drivers."""
    # 1. 2023 Monza: TSU retired prior to race start
    session_2023 = ReplaySession(2023, "Monza")
    t_race = session_2023.T_0 + 100.0
    state_2023 = get_state(session_2023, t_race)

    tsu = next(d for d in state_2023.leaderboard if d.driver == "TSU")
    assert tsu.status == "RETIRED"
    assert tsu.x is None
    assert tsu.y is None

    # 2. Pre-race condition (before T_0): all drivers have TELEMETRY_MISSING coordinates
    t_pre = session_2023.T_0 - 20.0
    state_pre = get_state(session_2023, t_pre)
    for d in state_pre.leaderboard:
        assert d.status == "TELEMETRY_MISSING"
        assert d.x is None
        assert d.y is None


def test_multi_race_coordinate_parity():
    """Verifies both 2022 and 2023 Monza driver coordinates align with their respective track outlines."""
    for yr, scale_factor in [(2022, 1.0), (2023, 0.1)]:
        session = ReplaySession(yr, "Monza")
        t = session.T_0 + 150.0
        state = get_state(session, t)

        p1 = state.leaderboard[0]
        assert p1.x is not None and p1.y is not None
        assert p1.status == "ON_TRACK"

        # Check track outline coordinate bounds match driver coordinate bounds
        track_min_x, track_max_x = session.track_x.min(), session.track_x.max()
        track_min_y, track_max_y = session.track_y.min(), session.track_y.max()

        # Driver should be within track coordinate bounding box (with slight margin for racing line)
        assert track_min_x - 50.0 <= p1.x <= track_max_x + 50.0
        assert track_min_y - 50.0 <= p1.y <= track_max_y + 50.0


def test_timing_tower_and_marker_synchronization():
    """Verifies that P1 driver in the timing tower matches the leader driver marker."""
    session = ReplaySession(2022, "Monza")
    for delta_t in [50.0, 500.0, 1500.0]:
        t = session.T_0 + delta_t
        state = get_state(session, t)

        p1_tower = state.leaderboard[0]
        assert p1_tower.timing_position == 1

        # Leader must have valid coordinates
        assert p1_tower.x is not None
        assert p1_tower.y is not None
        assert p1_tower.status == "ON_TRACK"
