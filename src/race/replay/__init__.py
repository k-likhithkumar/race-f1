"""
Core Replay Engine for RACE (Phase 3A).
Provides point-in-time race state reconstruction, causality-preserving replay clock,
authoritative boundary stepping, and visibility firewalls.
"""

from src.race.replay.clock import ReplayClock, ClockState, LapBoundaryUnavailableError
from src.race.replay.session import ReplaySession
from src.race.replay.state import RaceState, DriverState, get_state
from src.race.replay.visibility import ChronologyGuard, FutureDataLeakError, visible_state
from src.race.replay.adapter import LiveAdapter, ReplayAdapter

__all__ = [
    "ReplayClock",
    "ClockState",
    "LapBoundaryUnavailableError",
    "ReplaySession",
    "RaceState",
    "DriverState",
    "get_state",
    "ChronologyGuard",
    "FutureDataLeakError",
    "visible_state",
    "LiveAdapter",
    "ReplayAdapter",
]
