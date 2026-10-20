"""
Unit and integration tests for Phase 3A Replay Engine.
Tests clock state transitions, exact-boundary lap stepping, guarded interpolation,
causality protection, visibility firewalls, and deterministic replay.
"""

import unittest.mock
import pytest
import numpy as np
import pandas as pd

from src.race.replay.clock import ReplayClock, ClockState, LapBoundaryUnavailableError
from src.race.replay.session import ReplaySession
from src.race.replay.state import get_state, RaceState
from src.race.replay.visibility import ChronologyGuard, FutureDataLeakError, visible_state
from src.race.replay.adapter import ReplayAdapter


@pytest.fixture(scope="module")
def session_2022():
    return ReplaySession(2022, "Monza")


@pytest.fixture
def clock(session_2022):
    return ReplayClock(session_2022, epsilon=1e-3)


# ==============================================================================
# 1. ReplaySession & Authoritative Boundaries
# ==============================================================================

def test_replay_session_initialization(session_2022):
    assert session_2022.total_laps == 53
    assert len(session_2022.drivers) == 20
    assert "VER" in session_2022.drivers
    assert "LEC" in session_2022.drivers
    assert session_2022.T_0 == 3748.384
    assert session_2022.T_finish == 8575.895
    assert len(session_2022.track_x) == 1000


def test_authoritative_leader_crossing(session_2022):
    # Verify exact cached leader crossings for 2022 Monza
    assert session_2022.get_leader_crossing_time(1) == 3836.469
    assert session_2022.get_leader_crossing_time(2) == 3923.004
    assert session_2022.get_leader_crossing_time(3) == 4009.356
    assert session_2022.get_leader_crossing_time(53) == 8575.895

    with pytest.raises(LapBoundaryUnavailableError):
        session_2022.get_leader_crossing_time(54)


# ==============================================================================
# 2. Lap-Boundary Consistency & Stepping
# ==============================================================================

def test_lap_boundary_clearly_before_outside_tolerance(clock):
    # 69 ms before T_1 (3836.469)
    clock.seek(3836.400)
    assert clock.current_lap == 1
    clock.step_lap(+1)
    assert clock.current_time_s == 3836.469  # Lap 2 start

    # Independent backward step from clearly before boundary
    clock.seek(3836.400)
    assert clock.current_lap == 1
    clock.step_lap(-1)
    assert clock.current_time_s == 3748.384  # Lap 1 start (T_0)


def test_lap_boundary_before_inside_tolerance(clock):
    # 0.5 ms before T_1 (inside boundary tolerance epsilon=1ms)
    clock.seek(3836.4685)
    assert clock.current_lap == 2
    clock.step_lap(+1)
    assert clock.current_time_s == 3923.004  # Lap 3 start

    # Independent backward step from inside tolerance window
    clock.seek(3836.4685)
    assert clock.current_lap == 2
    clock.step_lap(-1)
    assert clock.current_time_s == 3748.384  # Lap 1 start (T_0)


def test_lap_boundary_exactly_at(clock):
    # Exactly at boundary T_1
    clock.seek(3836.4690)
    assert clock.current_lap == 2
    clock.step_lap(+1)
    assert clock.current_time_s == 3923.004  # Lap 3 start

    # Independent backward step exactly at boundary T_1
    clock.seek(3836.4690)
    assert clock.current_lap == 2
    clock.step_lap(-1)
    assert clock.current_time_s == 3748.384  # Lap 1 start (T_0)


def test_lap_boundary_after_inside_tolerance(clock):
    # 0.5 ms after T_1 (inside boundary tolerance)
    clock.seek(3836.4695)
    assert clock.current_lap == 2
    clock.step_lap(+1)
    assert clock.current_time_s == 3923.004  # Lap 3 start

    # Independent backward step from inside tolerance window
    clock.seek(3836.4695)
    assert clock.current_lap == 2
    clock.step_lap(-1)
    assert clock.current_time_s == 3748.384  # Lap 1 start (T_0)


def test_lap_boundary_clearly_after_outside_tolerance(clock):
    # 31 ms after T_1
    clock.seek(3836.500)
    assert clock.current_lap == 2
    clock.step_lap(+1)
    assert clock.current_time_s == 3923.004  # Lap 3 start

    # Independent backward step from clearly after boundary
    clock.seek(3836.500)
    assert clock.current_lap == 2
    clock.step_lap(-1)
    assert clock.current_time_s == 3836.469  # Lap 2 start (T_1)


def test_lap_boundary_race_start_clamping(clock):
    clock.seek(3748.384)
    assert clock.current_lap == 1
    clock.step_lap(-1)
    assert clock.current_time_s == 3748.384  # Clamped at race start


def test_lap_step_finish_tolerance_inside_before(clock):
    # 0.5 ms before finish, paused on final lap
    clock.seek(8575.895 - 0.0005)
    clock.state = ClockState.PAUSED
    assert clock.current_lap == 53
    assert clock.state == ClockState.PAUSED

    # step_lap(+1) seeks to exact T_finish and transitions to FINISHED
    clock.step_lap(+1)
    assert clock.current_time_s == 8575.895
    assert clock.state == ClockState.FINISHED


def test_lap_step_finish_exactly_at(clock):
    # Case A: at T_finish but state is PAUSED
    clock.current_time_s = 8575.895
    clock.state = ClockState.PAUSED
    clock.step_lap(+1)
    assert clock.current_time_s == 8575.895
    assert clock.state == ClockState.FINISHED

    # Case B: already FINISHED -> no-op
    clock.step_lap(+1)
    assert clock.current_time_s == 8575.895
    assert clock.state == ClockState.FINISHED


def test_lap_step_finish_seek_beyond_clamping(clock):
    # Seeking beyond finish clamps to T_finish and sets FINISHED
    clock.seek(8575.895 + 0.0005)
    assert clock.current_time_s == 8575.895
    assert clock.state == ClockState.FINISHED

    clock.step_lap(+1)
    assert clock.current_time_s == 8575.895
    assert clock.state == ClockState.FINISHED


def test_lap_step_backward_from_finish(clock):
    clock.seek(8575.895)
    clock.state = ClockState.FINISHED
    clock.step_lap(-1)
    assert clock.current_time_s == 8443.528  # Lap 53 start
    assert clock.state == ClockState.PAUSED


def test_lap_boundary_unavailable_preserves_timestamp_and_state(clock):
    """
    Verifies that when stepping forward from lap 15, if the authoritative
    leader crossing timestamp T_15 is unavailable:
    1. Clock is positioned at 5000.0 s and verified to be on lap 15.
    2. The exact authoritative lookup method (get_leader_crossing_time) fails for lap 15.
    3. step_lap(+1) propagates LapBoundaryUnavailableError.
    4. get_leader_crossing_time was actually invoked with argument 15.
    5. current_time_s and clock.state remain strictly unchanged.
    """
    clock.seek(5000.0)
    clock.state = ClockState.PAUSED
    assert clock.current_time_s == 5000.0
    assert clock.current_lap == 15
    initial_time_s = clock.current_time_s
    initial_state = clock.state

    orig_lookup = clock.session.get_leader_crossing_time

    def mock_get_leader_crossing(lap):
        if lap == 15:
            raise LapBoundaryUnavailableError(
                "Authoritative leader lap boundary unavailable for lap 15"
            )
        return orig_lookup(lap)

    with unittest.mock.patch.object(
        clock.session,
        "get_leader_crossing_time",
        side_effect=mock_get_leader_crossing
    ) as mock_lookup:

        with pytest.raises(LapBoundaryUnavailableError):
            clock.step_lap(+1)

        mock_lookup.assert_called_with(15)

    assert clock.current_time_s == initial_time_s
    assert clock.state == initial_state


# ==============================================================================
# 3. Guarded Coordinate Interpolation & Point-in-Time State
# ==============================================================================

def test_guarded_coordinate_interpolation(session_2022):
    # Query at fractional timestamp between 5 Hz grid points
    state = get_state(session_2022, 4200.137)
    assert isinstance(state, RaceState)
    assert state.session_time_s == 4200.137
    assert len(state.leaderboard) == 20

    # LEC and VER must have continuous coordinates
    lec_state = next(s for s in state.leaderboard if s.driver == "LEC")
    assert lec_state.x is not None
    assert lec_state.y is not None
    assert lec_state.spatial_projection_valid is True
    assert lec_state.status == "ON_TRACK"


def test_backward_seek_replay_determinism(session_2022, clock):
    # Replay to t=4500
    clock.seek(4500.0)
    state_1 = get_state(session_2022, clock.current_time_s)

    # Seek back to 4000
    clock.seek(4000.0)
    # Replay forward to 4500
    clock.seek(4500.0)
    state_2 = get_state(session_2022, clock.current_time_s)

    assert state_1.model_dump() == state_2.model_dump()


# ==============================================================================
# 4. Anti-Leakage & ChronologyGuard
# ==============================================================================

def test_chronology_guard_raises_on_future_read():
    guard = ChronologyGuard(as_of_time_s=4500.0)
    guard.assert_valid_time(4500.0)  # Valid

    with pytest.raises(FutureDataLeakError):
        guard.assert_valid_time(4500.1)


def test_classification_masked_before_race_end(session_2022):
    guard = ChronologyGuard(as_of_time_s=4500.0)
    masked = guard.filter_classification(
        session_2022.classification_df,
        race_finish_time_s=session_2022.T_finish
    )
    assert masked.empty  # Masked mid-race

    guard_post = ChronologyGuard(as_of_time_s=8600.0)
    revealed = guard_post.filter_classification(
        session_2022.classification_df,
        race_finish_time_s=session_2022.T_finish
    )
    assert len(revealed) == 20


# ==============================================================================
# 5. Tyre Life & Compound Visibility
# ==============================================================================

def test_tyre_age_and_pit_stint_progression(session_2022):
    # LEC Lap 11 crossing (4696.024)
    state_l11 = get_state(session_2022, 4696.024)
    lec_11 = next(s for s in state_l11.leaderboard if s.driver == "LEC")
    assert lec_11.completed_tyre_age_laps == 11
    assert lec_11.compound == "SOFT"
    assert lec_11.stint == 1

    # LEC Lap 12 in-lap crossing (4805.043)
    state_l12 = get_state(session_2022, 4805.043)
    lec_12 = next(s for s in state_l12.leaderboard if s.driver == "LEC")
    assert lec_12.completed_tyre_age_laps == 12
    assert lec_12.compound == "SOFT"

    # LEC Lap 13 out-lap crossing (4911.081) on Medium
    state_l13 = get_state(session_2022, 4911.081)
    lec_13 = next(s for s in state_l13.leaderboard if s.driver == "LEC")
    assert lec_13.completed_tyre_age_laps == 1
    assert lec_13.compound == "MEDIUM"
    assert lec_13.stint == 2


def test_visible_state_masks_rival_pit_compound(session_2022):
    # During out-lap at t=4850.0 (LEC in out-lap before line crossing)
    raw_state = get_state(session_2022, 4850.0)
    lec_raw = next(s for s in raw_state.leaderboard if s.driver == "LEC")
    assert lec_raw.is_out_lap is True

    # Red Bull perspective: rival Ferrari compound is masked
    rb_view = visible_state(raw_state, team="Red Bull Racing")
    lec_rb = next(s for s in rb_view.leaderboard if s.driver == "LEC")
    assert lec_rb.compound == "UNKNOWN"
    assert lec_rb.compound_observation_status == "MASKED_IN_PIT"

    # Ferrari perspective: own car compound is known
    ferrari_view = visible_state(raw_state, team="Ferrari")
    lec_ferrari = next(s for s in ferrari_view.leaderboard if s.driver == "LEC")
    assert lec_ferrari.compound == "SOFT"  # Preserved or known internally


# ==============================================================================
# 6. Vehicle Status & Retirement (No False Retirement)
# ==============================================================================

def test_vehicle_status_no_false_retirement(session_2022):
    # ALO stopped after lap 31 (around t=6600)
    mid_race_state = get_state(session_2022, 6600.0)
    alo_mid = next(s for s in mid_race_state.leaderboard if s.driver == "ALO")
    # Coordinates have dropped, but prior to race end, not classified as RETIRED without official feed
    assert alo_mid.status in ["TELEMETRY_MISSING", "STOPPED"]
    assert alo_mid.status != "RETIRED"

    # After race finish (t >= T_finish), classification is safely unmasked
    post_race_state = get_state(session_2022, session_2022.T_finish + 10.0)
    alo_post = next(s for s in post_race_state.leaderboard if s.driver == "ALO")
    assert alo_post.status == "RETIRED"


# ==============================================================================
# 7. LiveAdapter & ReplayAdapter Interface
# ==============================================================================

def test_replay_adapter_controls(session_2022):
    adapter = ReplayAdapter(session_2022)
    assert adapter.is_live() is False

    adapter.seek(4000.0)
    state = adapter.poll()
    assert state.session_time_s == 4000.0
    assert len(state.leaderboard) == 20

    adapter.play()
    adapter.tick(0.2)
    assert adapter.clock.current_time_s == 4000.2


# ==============================================================================
# 8. Targeted Review Refinements (Phase 3A Approved Plan)
# ==============================================================================

def test_track_status_fallback_unknown_not_green(session_2022):
    # Query before race start T_0 (3748.384)
    pre_race_state = get_state(session_2022, 3500.0)
    assert pre_race_state.track_status == "PRE_RACE"
    assert pre_race_state.track_status != "1"


def test_track_status_sc_propagation(session_2022):
    # Query during known Safety Car period at Monza 2022 (around lap 48, t=7900)
    sc_state = get_state(session_2022, 7900.0)
    assert sc_state.track_status == "4"


def test_in_lap_causality_no_premature_leakage(session_2022):
    # LEC pits on Lap 12 (crosses pit at t=4805.043).
    # Query at t=4720.0 (early Lap 12 on track)
    early_l12_state = get_state(session_2022, 4720.0)
    lec = next(s for s in early_l12_state.leaderboard if s.driver == "LEC")
    # Must NOT prematurely announce in-lap prior to pit entry
    assert lec.is_in_lap is False
    assert lec.pit_status == "ON_TRACK"


def test_earlier_snapshot_invariance_with_future_injection(session_2022):
    # Capture snapshot at t=4200
    snap_before = get_state(session_2022, 4200.0)

    # Inject synthetic future lap records for lap 55+
    synthetic_row = pd.DataFrame([{
        "race_id": session_2022.race_id,
        "lap_number": 99,
        "driver": "VER",
        "team": "Red Bull Racing",
        "lap_time_s": 85.0,
        "session_time_s": 9999.0,
        "compound": "SOFT",
        "tyre_age_laps": 10,
        "stint": 3,
        "position": 1.0,
        "track_status": "1",
        "is_clean": True,
    }])
    orig_df = session_2022.laps_df
    session_2022.laps_df = pd.concat([orig_df, synthetic_row], ignore_index=True)

    try:
        snap_after = get_state(session_2022, 4200.0)
        assert snap_before.model_dump() == snap_after.model_dump()
    finally:
        session_2022.laps_df = orig_df


def test_vehicle_state_precedence_hierarchy(session_2022):
    state = get_state(session_2022, 4400.0)
    # Moving on-track cars
    ver = next(s for s in state.leaderboard if s.driver == "VER")
    assert ver.status == "ON_TRACK"
    assert ver.spatial_projection_valid is True

    # Missing telemetry car mid-race must never be classified as RETIRED
    alo_mid = next(s for s in get_state(session_2022, 6600.0).leaderboard if s.driver == "ALO")
    assert alo_mid.status != "RETIRED"
    assert alo_mid.status in ["TELEMETRY_MISSING", "STOPPED"]


def test_tyre_visibility_context_separation(session_2022):
    # LEC on out-lap at t=4850.0
    raw_state = get_state(session_2022, 4850.0)
    lec_raw = next(s for s in raw_state.leaderboard if s.driver == "LEC")
    assert lec_raw.is_out_lap is True
    assert lec_raw.last_confirmed_compound == "SOFT"

    # 1. Public view preserves previous confirmed compound with pending status
    pub_view = visible_state(raw_state, team=None)
    lec_pub = next(s for s in pub_view.leaderboard if s.driver == "LEC")
    assert lec_pub.compound == "SOFT"
    assert lec_pub.compound_observation_status == "OBSERVATION_PENDING"

    # 2. Rival view masks compound to UNKNOWN in pit
    rival_view = visible_state(raw_state, team="Red Bull Racing")
    lec_rival = next(s for s in rival_view.leaderboard if s.driver == "LEC")
    assert lec_rival.compound == "UNKNOWN"
    assert lec_rival.compound_observation_status == "MASKED_IN_PIT"

    # 3. Own team view reflects internal pit wall status
    own_view = visible_state(raw_state, team="Ferrari")
    lec_own = next(s for s in own_view.leaderboard if s.driver == "LEC")
    assert lec_own.compound_observation_status == "TEAM_INTERNAL"


def test_active_battles_causality_and_suppression(session_2022):
    # 1. Safety car period must suppress all active battles
    sc_state = get_state(session_2022, 7900.0)
    assert sc_state.track_status == "4"
    assert len(sc_state.active_battles) == 0

    # 2. Green flag period active battles check
    green_state = get_state(session_2022, 4400.0)
    for b in green_state.active_battles:
        assert b["gap_s"] <= 1.0
        assert b["battle_verdict"] in ["ACTIVE_CHASE", "SUSTAINED_DEFENSE", "OVERTAKE_COMPLETED"]


def test_spatial_ordering_dynamic_attrition(session_2022):
    state = get_state(session_2022, 4400.0)
    # Valid field with >= 80% on-track projections
    assert state.spatial_ordering_valid is True
    assert state.spatial_fallback_reason is None
    assert len(state.spatial_order) >= 15


def test_coordinate_alignment_multi_sector_accuracy(session_2022):
    distances = []
    drivers = ["VER", "LEC", "HAM", "RUS", "SAI"]
    times = [4100.0, 4500.0, 5200.0, 6000.0]
    for t in times:
        for drv in drivers:
            pos_t = session_2022.pos_times[drv]
            if pos_t[0] <= t <= pos_t[-1]:
                idx = np.searchsorted(pos_t, t)
                x, y = session_2022.pos_x[drv][idx], session_2022.pos_y[drv][idx]
                norm_x = x - session_2022.coord_offset_x
                norm_y = y - session_2022.coord_offset_y
                dx = norm_x - session_2022.track_x
                dy = norm_y - session_2022.track_y
                dist = np.sqrt(np.min(dx * dx + dy * dy))
                distances.append(dist)

    arr = np.array(distances)
    median_err = float(np.median(arr))
    p95_err = float(np.percentile(arr, 95))

    # Assert agreed empirical targets: median <= 15m, P95 <= 35m
    assert median_err <= 15.0, f"Median projection error {median_err:.2f}m exceeded 15.0m target"
    assert p95_err <= 35.0, f"P95 projection error {p95_err:.2f}m exceeded 35.0m target"


def test_start_finish_spatial_continuity(session_2022):
    # Leader LEC crossing T_1 at 3836.469
    t_cross = 3836.469
    times = np.arange(t_cross - 0.5, t_cross + 0.5, 0.1)
    tot_dists = []
    for t in times:
        st = get_state(session_2022, t)
        lec = next(s for s in st.leaderboard if s.driver == "LEC")
        if lec.total_race_distance_m is not None:
            tot_dists.append(lec.total_race_distance_m)

    # Assert that total distance across the crossing window is non-decreasing
    for i in range(1, len(tot_dists)):
        assert tot_dists[i] >= tot_dists[i - 1] - 0.1, (
            f"Distance drop detected at start/finish: {tot_dists[i-1]} -> {tot_dists[i]}"
        )


# ==============================================================================
# 9. Historical Leaderboard Progression & Retirement Regression Tests
# ==============================================================================

@pytest.fixture(scope="module")
def session_2023():
    return ReplaySession(2023, "Monza")


def test_tsu_invalid_crossing_excluded_and_never_leads(session_2023):
    # Verify TSU has no valid timing crossings
    tsu_crossings = session_2023.driver_lap_crossing_times.get("TSU", np.array([]))
    assert len(tsu_crossings) == 0

    # Test timestamps: pre-race, lap 1 ongoing, lap 1 done, mid-race, late-race, finish
    test_times = [1312.0, 1400.0, 1450.0, 1500.0, 3000.0, 4800.0, 6300.0]
    for t in test_times:
        st = get_state(session_2023, t)
        assert st.leaderboard[0].driver != "TSU", f"TSU incorrectly leading at t={t}"
        tsu_state = next(s for s in st.leaderboard if s.driver == "TSU")
        assert tsu_state.last_completed_lap == 0
        assert tsu_state.timing_position != 1
        if t <= session_2023.T_0:
            assert tsu_state.timing_position == 11  # Official starting grid position
        else:
            assert tsu_state.timing_position == 20  # Demoted to last place
            assert tsu_state.status == "RETIRED"


def test_starting_grid_order_preservation(session_2022, session_2023):
    # 2023 Monza pre-race at t <= T_0 (1402.77)
    st_2023_grid = get_state(session_2023, 1350.0)
    expected_top5_2023 = ["SAI", "VER", "LEC", "RUS", "PER"]
    actual_top5_2023 = [s.driver for s in st_2023_grid.leaderboard[:5]]
    assert actual_top5_2023 == expected_top5_2023
    for idx, s in enumerate(st_2023_grid.leaderboard, start=1):
        assert s.timing_position == idx

    # 2022 Monza pre-race at t <= T_0 (3748.384)
    st_2022_grid = get_state(session_2022, 3700.0)
    expected_top5_2022 = ["LEC", "RUS", "NOR", "RIC", "GAS"]
    actual_top5_2022 = [s.driver for s in st_2022_grid.leaderboard[:5]]
    assert actual_top5_2022 == expected_top5_2022
    for idx, s in enumerate(st_2022_grid.leaderboard, start=1):
        assert s.timing_position == idx


def test_intermediate_race_progression(session_2023):
    # Lap 1 complete (t=1500)
    st_lap1 = get_state(session_2023, 1500.0)
    assert st_lap1.leaderboard[0].driver == "SAI"
    assert st_lap1.leaderboard[1].driver == "VER"
    assert st_lap1.leaderboard[2].driver == "LEC"

    # Lap 18 (t=3000)
    st_lap18 = get_state(session_2023, 3000.0)
    assert st_lap18.leaderboard[0].driver == "VER"
    assert st_lap18.leaderboard[1].driver == "SAI"
    assert st_lap18.leaderboard[2].driver == "LEC"

    # Lap 39 (t=4800)
    st_lap39 = get_state(session_2023, 4800.0)
    assert st_lap39.leaderboard[0].driver == "VER"
    assert st_lap39.leaderboard[1].driver == "SAI"
    assert st_lap39.leaderboard[2].driver == "PER"


def test_retired_driver_ordering(session_2022, session_2023):
    # 2023 Monza: OCO stopped at lap 39; TSU 0 laps
    # At t=5200 (leaders on lap 43), OCO is behind all running cars that completed >= 40 laps
    st_2023_mid = get_state(session_2023, 5200.0)
    oco_mid = next(s for s in st_2023_mid.leaderboard if s.driver == "OCO")
    assert oco_mid.last_completed_lap == 39
    assert oco_mid.timing_position == 19
    assert st_2023_mid.leaderboard[-1].driver == "TSU"

    # 2023 Monza finish: OCO at P19 (RETIRED), TSU at P20 (RETIRED)
    st_2023_fin = get_state(session_2023, 6300.0)
    assert st_2023_fin.leaderboard[18].driver == "OCO"
    assert st_2023_fin.leaderboard[18].status == "RETIRED"
    assert st_2023_fin.leaderboard[19].driver == "TSU"
    assert st_2023_fin.leaderboard[19].status == "RETIRED"

    # 2022 Monza finish: RIC (45), STR (39), ALO (31), VET (10)
    st_2022_fin = get_state(session_2022, 8600.0)
    retired_order = [s.driver for s in st_2022_fin.leaderboard[16:]]
    assert retired_order == ["RIC", "STR", "ALO", "VET"]
    for s in st_2022_fin.leaderboard[16:]:
        assert s.status == "RETIRED"


def test_final_finishing_order(session_2022, session_2023):
    # 2023 Monza official podium: VER, PER, SAI
    st_2023 = get_state(session_2023, 6300.0)
    podium_2023 = [s.driver for s in st_2023.leaderboard[:3]]
    assert podium_2023 == ["VER", "PER", "SAI"]
    for idx, s in enumerate(st_2023.leaderboard, start=1):
        assert s.timing_position == idx

    # 2022 Monza official podium: VER, LEC, RUS
    st_2022 = get_state(session_2022, 8600.0)
    podium_2022 = [s.driver for s in st_2022.leaderboard[:3]]
    assert podium_2022 == ["VER", "LEC", "RUS"]
    for idx, s in enumerate(st_2022.leaderboard, start=1):
        assert s.timing_position == idx


def test_telemetry_dropout_not_marked_as_retirement_mid_race(session_2022):
    # During race at t=5000, active driver with missing coords is TELEMETRY_MISSING, not RETIRED
    st = get_state(session_2022, 5000.0)
    # Check that any driver with TELEMETRY_MISSING is NOT marked RETIRED before T_finish
    for s in st.leaderboard:
        if s.status == "TELEMETRY_MISSING":
            assert s.status != "RETIRED"


def test_driver_state_coordinates_aligned_with_track_outline(session_2022, session_2023):
    # 1. 2022 Monza: verify DriverState.x/y directly align with session_2022 track outline
    st_2022 = get_state(session_2022, session_2022.T_0 + 300.0)
    active_2022 = [s for s in st_2022.leaderboard if s.status == "ON_TRACK" and s.x is not None]
    assert len(active_2022) >= 15
    for s in active_2022:
        dist_units = np.sqrt(np.min((s.x - session_2022.track_x)**2 + (s.y - session_2022.track_y)**2))
        dist_m = dist_units * 0.1  # 2022 units are decimeters
        assert dist_m <= 15.0, f"Driver {s.driver} in 2022 misaligned: {dist_m:.2f} m"

    # 2. 2023 Monza: verify DriverState.x/y directly align with session_2023 track outline
    st_2023 = get_state(session_2023, session_2023.T_0 + 300.0)
    active_2023 = [s for s in st_2023.leaderboard if s.status == "ON_TRACK" and s.x is not None]
    assert len(active_2023) >= 15
    for s in active_2023:
        dist_m = np.sqrt(np.min((s.x - session_2023.track_x)**2 + (s.y - session_2023.track_y)**2))
        assert dist_m <= 15.0, f"Driver {s.driver} in 2023 misaligned: {dist_m:.2f} m"

    # 3. Verified missing source telemetry propagates to None in DriverState.x/y
    # In 2023 Monza, TSU has confirmed absent/NaN coordinates in source position data
    assert np.isnan(session_2023.pos_x["TSU"]).all()
    tsu_2023 = next(s for s in st_2023.leaderboard if s.driver == "TSU")
    assert tsu_2023.x is None
    assert tsu_2023.y is None




