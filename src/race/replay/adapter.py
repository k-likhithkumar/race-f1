"""
Adapter interfaces for RACE (Phase 3A).
Provides the abstract LiveAdapter and concrete cache-backed ReplayAdapter.
"""

from abc import ABC, abstractmethod
import logging
from typing import Optional

from src.race.replay.clock import ReplayClock
from src.race.replay.session import ReplaySession
from src.race.replay.state import RaceState, get_state

logger = logging.getLogger(__name__)


class LiveAdapter(ABC):
    """
    Abstract interface unifying live streaming feeds and historical replay engines.
    Downstream consumers (FastAPI, WebSockets, AI assistant) interact exclusively with LiveAdapter.
    """

    @abstractmethod
    def poll(self) -> RaceState:
        """Poll the adapter for the current RaceState snapshot."""
        pass

    @abstractmethod
    def seek(self, target_time_s: float) -> None:
        """Seek the adapter to a specific timestamp if supported."""
        pass

    @abstractmethod
    def is_live(self) -> bool:
        """Returns True if connected to an active live feed, False for replay."""
        pass


class ReplayAdapter(LiveAdapter):
    """
    Concrete replay-backed adapter driving state from ReplayClock and ReplaySession.
    """

    def __init__(self, session: ReplaySession, clock: Optional[ReplayClock] = None):
        self.session: ReplaySession = session
        self.clock: ReplayClock = clock if clock is not None else ReplayClock(session)

    def poll(self) -> RaceState:
        """Constructs and returns RaceState at clock.current_time_s."""
        return get_state(self.session, self.clock.current_time_s)

    def seek(self, target_time_s: float) -> None:
        """Seek replay clock to target_time_s."""
        self.clock.seek(target_time_s)

    def is_live(self) -> bool:
        """ReplayAdapter is offline cache-backed, not live."""
        return False

    def tick(self, dt_real_s: float) -> float:
        """Advance clock by dt_real_s."""
        return self.clock.tick(dt_real_s)

    def play(self) -> None:
        self.clock.play()

    def pause(self) -> None:
        self.clock.pause()

    def step_lap(self, direction: int) -> None:
        self.clock.step_lap(direction)

    def set_speed(self, speed: float) -> None:
        self.clock.set_speed(speed)
