import pytest
import time
from src.race.replay.session import ReplaySession
from src.race.replay.clock import ReplayClock, ClockState
from src.race.api.session_manager import SessionManager
from starlette.testclient import TestClient
from src.race.api.app import app

client = TestClient(app)


@pytest.fixture
def session():
    return ReplaySession(2022, "monza")


@pytest.fixture
def clock(session):
    return ReplayClock(session)


def test_step_lap_minus_one_while_playing(clock, session):
    """
    Requirement: -1 Lap while playing moves back exactly one lap and continues playback.
    """
    # Start playback and seek to middle of Lap 5
    t_lap5_start = session.get_leader_crossing_time(4)
    t_lap4_start = session.get_leader_crossing_time(3)
    clock.seek(t_lap5_start + 45.0)
    clock.play()
    assert clock.state == ClockState.PLAYING
    assert clock.current_lap == 5

    # Step backward by 1 lap
    clock.step_lap(-1)

    # Should land exactly at Lap 4 start and remain PLAYING
    assert clock.current_time_s == t_lap4_start
    assert clock.current_lap == 4
    assert clock.state == ClockState.PLAYING


def test_step_lap_minus_one_while_paused(clock, session):
    """
    Requirement: -1 Lap while paused moves back and stays paused.
    Mid-lap paused -> steps to current lap start; at boundary -> steps to previous lap start.
    """
    t_lap5_start = session.get_leader_crossing_time(4)
    t_lap4_start = session.get_leader_crossing_time(3)

    # 1. At mid-lap 5, paused
    clock.seek(t_lap5_start + 30.0)
    clock.pause()
    assert clock.state == ClockState.PAUSED
    assert clock.current_lap == 5

    clock.step_lap(-1)
    assert clock.current_time_s == t_lap5_start
    assert clock.state == ClockState.PAUSED

    # 2. Already at lap 5 boundary, step again
    clock.step_lap(-1)
    assert clock.current_time_s == t_lap4_start
    assert clock.state == ClockState.PAUSED


def test_repeated_backward_steps_while_playing(clock, session):
    """
    Requirement: Repeated backward steps work reliably without getting trapped.
    """
    t_lap10_start = session.get_leader_crossing_time(9)
    clock.seek(t_lap10_start + 10.0)
    clock.play()
    assert clock.state == ClockState.PLAYING
    assert clock.current_lap == 10

    # Step backward repeatedly 5 times while playing
    for expected_lap in [9, 8, 7, 6, 5]:
        clock.step_lap(-1)
        expected_t = session.get_leader_crossing_time(expected_lap - 1)
        assert clock.current_lap == expected_lap
        assert clock.current_time_s == expected_t
        assert clock.state == ClockState.PLAYING


def test_step_lap_minus_one_clamping_at_session_start(clock, session):
    """
    Requirement: Ensure time never goes below the session start (T_0) and repeated steps at Lap 1 clamp safely.
    """
    clock.seek(clock.T_0 + 15.0)
    clock.play()
    assert clock.current_lap == 1
    assert clock.state == ClockState.PLAYING

    # Step back from mid-lap 1
    clock.step_lap(-1)
    assert clock.current_time_s == clock.T_0
    assert clock.current_lap == 1
    assert clock.state == ClockState.PLAYING

    # Step back again while at T_0
    clock.step_lap(-1)
    assert clock.current_time_s == clock.T_0
    assert clock.current_lap == 1
    assert clock.state == ClockState.PLAYING


def test_step_lap_boundary_behavior(clock, session):
    """
    Requirement: Explicit test for lap boundary behavior.
    """
    t_lap3_start = session.get_leader_crossing_time(2)
    t_lap2_start = session.get_leader_crossing_time(1)

    # Exactly at boundary T_2 while playing
    clock.seek(t_lap3_start)
    clock.play()
    assert clock.current_lap == 3
    clock.step_lap(-1)
    assert clock.current_time_s == t_lap2_start
    assert clock.state == ClockState.PLAYING


def test_reset_tick_delta_prevents_stale_jumps():
    """
    Requirement: Prevent stale monotonic tick deltas from immediately advancing clock unexpectedly after seek/step.
    """
    mgr = SessionManager()
    slug = "2022_monza"
    adapter = mgr.get_or_create_adapter(slug)
    adapter.seek(4000.0)
    adapter.play()

    # Simulate tick delta reset
    mgr.reset_tick_delta(slug)
    base_t = mgr._last_tick_times.get(slug)
    assert base_t is not None
    assert abs(time.monotonic() - base_t) < 0.1

    # Simulate elapsed time but with reset called just before tick
    time.sleep(0.05)
    mgr.reset_tick_delta(slug)
    now = time.monotonic()
    dt = max(0.0, now - mgr._last_tick_times[slug])
    assert dt < 0.02  # Delta is virtually zero, preventing clock jump


def test_websocket_step_lap_while_playing():
    """
    Requirement: WebSocket STEP_LAP -1 while PLAYING decrements lap, preserves PLAYING, and resets tick delta.
    """
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        _ = ws.receive_json()

        # 1. Seek to Lap 4 and start PLAY
        ws.send_json({"action": "SEEK", "target_time_s": 4100.0})
        _ = ws.receive_json()  # COMMAND_ACK
        _ = ws.receive_json()  # STATE_UPDATE

        ws.send_json({"action": "PLAY"})
        ack_play = ws.receive_json()
        assert ack_play["action"] == "PLAY"
        state_play = ws.receive_json()
        assert state_play["clock"]["state"] == "PLAYING"
        initial_lap = state_play["clock"]["current_lap"]

        # 2. Step backward while playing
        ws.send_json({"action": "STEP_LAP", "direction": -1})
        ack_step = ws.receive_json()
        assert ack_step["action"] == "STEP_LAP"
        assert ack_step["status"] == "SUCCESS"
        assert ack_step["current_lap"] == initial_lap - 1

        state_step = ws.receive_json()
        assert state_step["clock"]["state"] == "PLAYING"
        assert state_step["clock"]["current_lap"] == initial_lap - 1

        # Pause at the end
        ws.send_json({"action": "PAUSE"})
        _ = ws.receive_json()
        _ = ws.receive_json()
