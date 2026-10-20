"""
WebSocket endpoint and message handling for the RACE Replay Engine.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.race.api.session_manager import session_manager
from src.race.replay.clock import ClockState

logger = logging.getLogger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/replay/{race_id}")
async def replay_websocket(websocket: WebSocket, race_id: str):
    await websocket.accept()
    slug = session_manager.normalize_race_id(race_id)

    try:
        session_manager.get_or_create_adapter(slug)
    except FileNotFoundError:
        await websocket.send_json({
            "type": "ERROR",
            "code": "RACE_NOT_FOUND",
            "message": f"Race session '{race_id}' not found in cache."
        })
        await websocket.close()
        return
    except Exception as e:
        await websocket.send_json({
            "type": "ERROR",
            "code": "SESSION_INIT_FAILED",
            "message": str(e)
        })
        await websocket.close()
        return

    # 1. Register subscriber
    session_manager.add_subscriber(slug, websocket)

    # 2. Push immediate initial state to new subscriber
    adapter = session_manager.get_or_create_adapter(slug)
    clock = adapter.clock
    state = session_manager.get_public_state(slug)
    await websocket.send_json({
        "type": "INITIAL_STATE",
        "clock": {
            "session_time_s": round(clock.current_time_s, 3),
            "state": clock.state.value,
            "speed": clock.speed,
            "current_lap": clock.current_lap,
            "total_laps": clock.total_laps,
            "T_0": round(clock.T_0, 3),
            "T_finish": round(clock.T_finish, 3),
        },
        "state": state.model_dump(),
    })

    # 3. Command loop
    try:
        while True:
            data = await websocket.receive_json()
            if not isinstance(data, dict):
                await websocket.send_json({
                    "type": "ERROR",
                    "code": "INVALID_PAYLOAD",
                    "message": "Message payload must be a JSON object."
                })
                continue

            action = str(data.get("action", "")).upper()

            if action == "PLAY":
                session_manager.reset_tick_delta(slug)
                adapter.play()
                await websocket.send_json({
                    "type": "COMMAND_ACK",
                    "action": "PLAY",
                    "status": "SUCCESS",
                    "session_time_s": round(clock.current_time_s, 3)
                })
                await session_manager.broadcast_state(slug, "STATE_UPDATE")

            elif action == "PAUSE":
                adapter.pause()
                await websocket.send_json({
                    "type": "COMMAND_ACK",
                    "action": "PAUSE",
                    "status": "SUCCESS",
                    "session_time_s": round(clock.current_time_s, 3)
                })
                await session_manager.broadcast_state(slug, "STATE_UPDATE")

            elif action == "SEEK":
                if "target_time_s" not in data:
                    await websocket.send_json({
                        "type": "ERROR",
                        "code": "MISSING_ARGUMENT",
                        "message": "Seek action requires 'target_time_s' field."
                    })
                    continue
                try:
                    target_time = float(data["target_time_s"])
                    session_manager.reset_tick_delta(slug)
                    adapter.seek(target_time)
                    await websocket.send_json({
                        "type": "COMMAND_ACK",
                        "action": "SEEK",
                        "status": "SUCCESS",
                        "session_time_s": round(clock.current_time_s, 3)
                    })
                    await session_manager.broadcast_state(slug, "STATE_UPDATE")
                except ValueError:
                    await websocket.send_json({
                        "type": "ERROR",
                        "code": "INVALID_TIMESTAMP",
                        "message": f"Invalid target_time_s: {data.get('target_time_s')}"
                    })

            elif action == "STEP_LAP":
                direction = data.get("direction", 1)
                try:
                    dir_int = int(direction)
                    if dir_int not in (+1, -1):
                        raise ValueError("Direction must be +1 or -1")
                    session_manager.reset_tick_delta(slug)
                    adapter.step_lap(dir_int)
                    await websocket.send_json({
                        "type": "COMMAND_ACK",
                        "action": "STEP_LAP",
                        "status": "SUCCESS",
                        "direction": dir_int,
                        "current_lap": clock.current_lap,
                        "session_time_s": round(clock.current_time_s, 3)
                    })
                    await session_manager.broadcast_state(slug, "STATE_UPDATE")
                except Exception as e:
                    await websocket.send_json({
                        "type": "ERROR",
                        "code": "STEP_LAP_FAILED",
                        "message": str(e)
                    })

            elif action == "SET_SPEED":
                speed_val = data.get("speed")
                try:
                    spd = float(speed_val)
                    if spd <= 0:
                        raise ValueError("Speed must be positive")
                    adapter.set_speed(spd)
                    await websocket.send_json({
                        "type": "COMMAND_ACK",
                        "action": "SET_SPEED",
                        "status": "SUCCESS",
                        "speed": clock.speed
                    })
                    await session_manager.broadcast_state(slug, "STATE_UPDATE")
                except Exception as e:
                    await websocket.send_json({
                        "type": "ERROR",
                        "code": "INVALID_SPEED",
                        "message": str(e)
                    })

            else:
                await websocket.send_json({
                    "type": "ERROR",
                    "code": "UNKNOWN_ACTION",
                    "message": f"Unrecognized action '{action}'."
                })

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from replay {slug}")
    except Exception as e:
        logger.error(f"WebSocket unexpected error for {slug}: {e}")
    finally:
        session_manager.remove_subscriber(slug, websocket)
