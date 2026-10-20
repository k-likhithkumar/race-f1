"""
ReplayClock module for RACE (Phase 3A).
Provides a causality-preserving, continuous floating-point simulation clock
with authoritative exact-boundary lap stepping and speed control.
"""

from enum import Enum
import logging
from typing import Optional

from src.race.replay.session import ReplaySession, LapBoundaryUnavailableError

logger = logging.getLogger(__name__)


class ClockState(str, Enum):
    IDLE = "IDLE"
    PLAYING = "PLAYING"
    PAUSED = "PAUSED"
    SEEKING = "SEEKING"
    FINISHED = "FINISHED"


class ReplayClock:
    """
    Simulates race elapsed time for replay and what-if simulation.
    Operates on continuous floating-point seconds.
    """

    def __init__(self, session: ReplaySession, epsilon: float = 1e-3):
        self.session: ReplaySession = session
        self.epsilon: float = epsilon
        self.speed: float = 1.0
        self.state: ClockState = ClockState.PAUSED

        # Race boundary references
        self.T_0: float = session.T_0
        self.T_finish: float = session.T_finish
        self.total_laps: int = session.total_laps

        # Initial clock position at race start
        self.current_time_s: float = self.T_0

    @property
    def current_lap(self) -> int:
        """Returns the active leader lap according to the unified boundary policy."""
        return self.get_current_lap(self.current_time_s)

    def get_current_lap(self, t: float) -> int:
        """
        Unified boundary comparison policy for current-lap classification.
        For boundary T_l:
        - t < T_l - epsilon (Clearly Before) -> lap l
        - |t - T_l| <= epsilon (At Boundary)  -> lap l + 1
        - t > T_l + epsilon (Clearly After)   -> lap l + 1
        """
        if t <= self.T_0 + self.epsilon:
            return 1

        if t >= self.T_finish - self.epsilon:
            return self.total_laps

        for l in range(1, self.total_laps):
            t_cross = self.session.get_leader_crossing_time(l)
            if t < t_cross - self.epsilon:
                return l

        return self.total_laps

    def seek(self, target_time_s: float) -> None:
        """
        Clamps target time to [T_0, T_finish] and transitions state consistently.
        """
        clamped = max(self.T_0, min(self.T_finish, float(target_time_s)))
        self.current_time_s = clamped

        if clamped >= self.T_finish:
            self.state = ClockState.FINISHED
        elif self.state == ClockState.FINISHED:
            self.state = ClockState.PAUSED

    def play(self) -> None:
        """Resume or begin playback if not finished."""
        if self.state == ClockState.FINISHED:
            return
        self.state = ClockState.PLAYING

    def pause(self) -> None:
        """Pause playback."""
        if self.state == ClockState.FINISHED:
            return
        self.state = ClockState.PAUSED

    def set_speed(self, speed: float) -> None:
        """Set playback speed multiplier (positive float)."""
        if speed <= 0:
            raise ValueError(f"Speed multiplier must be positive, got {speed}")
        self.speed = float(speed)

    def tick(self, dt_real_s: float) -> float:
        """
        Advance simulation time by (dt_real_s * speed).
        Continuous floating-point advancement.
        """
        if self.state != ClockState.PLAYING:
            return self.current_time_s

        advance = dt_real_s * self.speed
        new_time = self.current_time_s + advance
        self.seek(new_time)
        return self.current_time_s

    def step_lap(self, direction: int) -> None:
        """
        Step forward or backward by lap boundary.
        Uses the exact authoritative leader crossing timestamps.
        If boundary lookup fails, raises LapBoundaryUnavailableError and leaves
        current_time_s and state completely unmutated.
        """
        if direction == +1:
            # If already FINISHED, step_lap(+1) is a strict no-op
            if self.state == ClockState.FINISHED:
                return

            l_curr = self.current_lap
            if l_curr < self.total_laps:
                # Resolve target timestamp BEFORE mutating clock or state
                target_time_s = self.session.get_leader_crossing_time(l_curr)
                self.seek(target_time_s)
            else:
                # On final lap and not already FINISHED:
                # Seek to exact T_finish and transition to FINISHED,
                # even if current timestamp is within the finish tolerance.
                target_time_s = self.session.get_leader_crossing_time(self.total_laps)
                self.seek(target_time_s)
                self.state = ClockState.FINISHED

        elif direction == -1:
            if self.state == ClockState.FINISHED or abs(self.current_time_s - self.T_finish) <= self.epsilon:
                # Step backward from race finish returns to start of final lap
                target_time_s = self.session.get_leader_crossing_time(self.total_laps - 1)
                self.seek(target_time_s)
                self.state = ClockState.PAUSED
            elif self.state == ClockState.PLAYING:
                # Active playback: step backward by exactly one lap and continue playing
                l_curr = self.current_lap
                target_lap = max(1, l_curr - 1)
                target_time_s = (
                    self.T_0
                    if target_lap == 1
                    else self.session.get_leader_crossing_time(target_lap - 1)
                )
                self.seek(target_time_s)
                self.state = ClockState.PLAYING
            else:
                # Paused state: maintain boundary tolerance precision
                l_curr = self.current_lap
                t_lap_start = (
                    self.T_0
                    if l_curr == 1
                    else self.session.get_leader_crossing_time(l_curr - 1)
                )

                if abs(self.current_time_s - t_lap_start) <= self.epsilon:
                    # Clock is already at current lap's start boundary: step to previous lap
                    if l_curr > 1:
                        t_prev_start = (
                            self.T_0
                            if l_curr == 2
                            else self.session.get_leader_crossing_time(l_curr - 2)
                        )
                        self.seek(t_prev_start)
                    else:
                        self.seek(self.T_0)
                else:
                    # Clock is strictly inside current lap (current_time_s > t_lap_start + epsilon):
                    self.seek(t_lap_start)

        else:
            raise ValueError(f"Direction must be +1 or -1, got {direction}")
