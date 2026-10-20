"""
WebSocket lifecycle and playback test suite for RACE Replay Engine (Checkpoint B).
Tests connection initialization, command handling, pause/play, seek, step lap, speed control,
structured error responses, and clean disconnection.
"""

import pytest
from fastapi.testclient import TestClient

from src.race.api.app import app
from src.race.api.session_manager import session_manager

client = TestClient(app)


def test_ws_initial_state():
    """Verifies that on connection, the client receives an INITIAL_STATE message with full state."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        msg = ws.receive_json()
        assert msg["type"] == "INITIAL_STATE"
        assert "clock" in msg
        assert "state" in msg

        clock = msg["clock"]
        assert clock["state"] in ["PAUSED", "PLAYING"]
        assert clock["total_laps"] == 53
        assert clock["speed"] == 1.0

        state = msg["state"]
        assert "leaderboard" in state
        assert len(state["leaderboard"]) == 20
        assert "track_status" in state


def test_ws_play_pause_commands():
    """Verifies PLAY and PAUSE command acknowledgements and state transitions."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        # Drain initial message
        init_msg = ws.receive_json()
        assert init_msg["type"] == "INITIAL_STATE"

        # 1. Send PLAY
        ws.send_json({"action": "PLAY"})
        ack = ws.receive_json()
        assert ack["type"] == "COMMAND_ACK"
        assert ack["action"] == "PLAY"
        assert ack["status"] == "SUCCESS"

        # Drain the broadcast update
        update = ws.receive_json()
        assert update["type"] == "STATE_UPDATE"
        assert update["clock"]["state"] == "PLAYING"

        # 2. Send PAUSE
        ws.send_json({"action": "PAUSE"})
        ack_pause = ws.receive_json()
        assert ack_pause["type"] == "COMMAND_ACK"
        assert ack_pause["action"] == "PAUSE"
        assert ack_pause["status"] == "SUCCESS"

        update_pause = ws.receive_json()
        assert update_pause["type"] == "STATE_UPDATE"
        assert update_pause["clock"]["state"] == "PAUSED"


def test_ws_seek_forward_and_backward():
    """Verifies forward seek and backward seek consistency."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        _ = ws.receive_json()  # drain initial

        # Forward seek to 4500.0 s
        ws.send_json({"action": "SEEK", "target_time_s": 4500.0})
        ack = ws.receive_json()
        assert ack["type"] == "COMMAND_ACK"
        assert ack["action"] == "SEEK"
        assert ack["session_time_s"] == 4500.0

        update = ws.receive_json()
        assert update["type"] == "STATE_UPDATE"
        assert update["clock"]["session_time_s"] == 4500.0

        # Backward seek to 4000.0 s
        ws.send_json({"action": "SEEK", "target_time_s": 4000.0})
        ack_back = ws.receive_json()
        assert ack_back["type"] == "COMMAND_ACK"
        assert ack_back["action"] == "SEEK"
        assert ack_back["session_time_s"] == 4000.0

        update_back = ws.receive_json()
        assert update_back["type"] == "STATE_UPDATE"
        assert update_back["clock"]["session_time_s"] == 4000.0


def test_ws_step_lap():
    """Verifies step lap forward (+1) and backward (-1) boundary jumps."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        _ = ws.receive_json()

        # Step forward
        ws.send_json({"action": "STEP_LAP", "direction": 1})
        ack = ws.receive_json()
        assert ack["type"] == "COMMAND_ACK"
        assert ack["action"] == "STEP_LAP"
        assert ack["status"] == "SUCCESS"
        assert ack["current_lap"] >= 1

        update = ws.receive_json()
        assert update["type"] == "STATE_UPDATE"

        # Step backward
        ws.send_json({"action": "STEP_LAP", "direction": -1})
        ack_back = ws.receive_json()
        assert ack_back["type"] == "COMMAND_ACK"
        assert ack_back["action"] == "STEP_LAP"
        assert ack_back["status"] == "SUCCESS"

        update_back = ws.receive_json()
        assert update_back["type"] == "STATE_UPDATE"


def test_ws_set_speed_and_validation():
    """Verifies speed multiplier updates and validation of invalid values."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        _ = ws.receive_json()

        # Valid speed
        ws.send_json({"action": "SET_SPEED", "speed": 2.0})
        ack = ws.receive_json()
        assert ack["type"] == "COMMAND_ACK"
        assert ack["speed"] == 2.0

        update = ws.receive_json()
        assert update["clock"]["speed"] == 2.0

        # Invalid negative speed
        ws.send_json({"action": "SET_SPEED", "speed": -1.0})
        err = ws.receive_json()
        assert err["type"] == "ERROR"
        assert err["code"] == "INVALID_SPEED"


def test_ws_unknown_action():
    """Verifies structured ERROR returned on unsupported action."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        _ = ws.receive_json()

        ws.send_json({"action": "INVALID_ACTION_123"})
        err = ws.receive_json()
        assert err["type"] == "ERROR"
        assert err["code"] == "UNKNOWN_ACTION"


def test_ws_unknown_race():
    """Verifies connecting to an unknown race returns RACE_NOT_FOUND error."""
    with client.websocket_connect("/ws/replay/9999_atlantis") as ws:
        err = ws.receive_json()
        assert err["type"] == "ERROR"
        assert err["code"] == "RACE_NOT_FOUND"


def test_ws_disconnect_cleanup():
    """Verifies subscriber set is cleanly updated upon client disconnection."""
    slug = session_manager.normalize_race_id("2022_monza")
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        _ = ws.receive_json()
        assert len(session_manager._subscribers.get(slug, set())) >= 1

    # After exit from context manager, socket is disconnected
    assert len(session_manager._subscribers.get(slug, set())) == 0


def test_ws_multi_client_synchronization():
    """Verifies that multiple clients connected to the same race observe identical synchronized state."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws_a:
        init_a = ws_a.receive_json()
        assert init_a["type"] == "INITIAL_STATE"

        with client.websocket_connect("/ws/replay/2022_monza") as ws_b:
            init_b = ws_b.receive_json()
            assert init_b["type"] == "INITIAL_STATE"
            # Both start with the same session time and clock state
            assert init_b["clock"]["session_time_s"] == init_a["clock"]["session_time_s"]

            # Client A commands a SEEK
            ws_a.send_json({"action": "SEEK", "target_time_s": 4250.0})
            ack_a = ws_a.receive_json()
            assert ack_a["type"] == "COMMAND_ACK"
            assert ack_a["action"] == "SEEK"

            update_a = ws_a.receive_json()
            assert update_a["type"] == "STATE_UPDATE"
            assert update_a["clock"]["session_time_s"] == 4250.0

            # Client B, without sending any command, MUST receive the synchronized STATE_UPDATE
            update_b = ws_b.receive_json()
            assert update_b["type"] == "STATE_UPDATE"
            assert update_b["clock"]["session_time_s"] == 4250.0
            assert update_b["clock"]["session_time_s"] == update_a["clock"]["session_time_s"]

            # Client B commands SET_SPEED to 3.0
            ws_b.send_json({"action": "SET_SPEED", "speed": 3.0})
            ack_b = ws_b.receive_json()
            assert ack_b["type"] == "COMMAND_ACK"
            assert ack_b["speed"] == 3.0

            update_b2 = ws_b.receive_json()
            assert update_b2["clock"]["speed"] == 3.0

            # Client A receives the broadcast with speed 3.0
            update_a2 = ws_a.receive_json()
            assert update_a2["clock"]["speed"] == 3.0


def test_ws_playback_finished_lifecycle():
    """Verifies that reaching T_finish transitions state to FINISHED and broadcasts PLAYBACK_FINISHED."""
    with client.websocket_connect("/ws/replay/2022_monza") as ws:
        init = ws.receive_json()
        assert init["type"] == "INITIAL_STATE"
        t_finish = init["clock"]["T_finish"]
        total_laps = init["clock"]["total_laps"]

        # Seek to exact T_finish
        ws.send_json({"action": "SEEK", "target_time_s": t_finish})
        ack = ws.receive_json()
        assert ack["type"] == "COMMAND_ACK"
        assert ack["action"] == "SEEK"

        # Receive STATE_UPDATE indicating FINISHED state
        update = ws.receive_json()
        assert update["type"] == "STATE_UPDATE"
        assert update["clock"]["state"] == "FINISHED"
        assert update["clock"]["session_time_s"] == t_finish

        # Receive the PLAYBACK_FINISHED lifecycle event broadcast
        event = ws.receive_json()
        assert event["type"] == "PLAYBACK_FINISHED"
        assert event["session_time_s"] == t_finish
        assert event["total_laps"] == total_laps

        # Verify backward seek resets FINISHED state to PAUSED
        ws.send_json({"action": "SEEK", "target_time_s": t_finish - 500.0})
        ack_seek = ws.receive_json()
        assert ack_seek["type"] == "COMMAND_ACK"

        update_reset = ws.receive_json()
        assert update_reset["type"] == "STATE_UPDATE"
        assert update_reset["clock"]["state"] == "PAUSED"

