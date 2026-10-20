"""
Visibility firewall and causality guard module for RACE (Phase 3A).
Prevents future-data leakage and enforces team-private observation boundaries.
"""

import copy
import logging
from typing import Optional
import pandas as pd

from src.race.replay.state import RaceState

logger = logging.getLogger(__name__)


class FutureDataLeakError(Exception):
    """Raised when an operation attempts to read data past the as_of replay boundary."""
    pass


class ChronologyGuard:
    """
    Guards replay and simulation routines against accidental future data access.
    """

    def __init__(self, as_of_time_s: float):
        self.as_of_time_s: float = float(as_of_time_s)

    def assert_valid_time(self, timestamp_s: float) -> None:
        """Raises FutureDataLeakError if timestamp_s is in the future relative to as_of_time_s."""
        if timestamp_s > self.as_of_time_s + 1e-4:
            raise FutureDataLeakError(
                f"Causality violation: attempted to access timestamp {timestamp_s:.3f}s "
                f"which is in the future relative to as_of {self.as_of_time_s:.3f}s."
            )

    def filter_dataframe(self, df: pd.DataFrame, time_col: str = "session_time_s") -> pd.DataFrame:
        """Filters a dataframe to exclude all records strictly after as_of_time_s."""
        if df.empty or time_col not in df.columns:
            return df
        return df[df[time_col] <= self.as_of_time_s].copy()

    def filter_classification(self, classification_df: pd.DataFrame, race_finish_time_s: float) -> pd.DataFrame:
        """
        Ensures final classification is masked until the race is officially finished.
        """
        if self.as_of_time_s < race_finish_time_s:
            return pd.DataFrame(columns=classification_df.columns)
        return classification_df.copy()


def visible_state(state: RaceState, team: Optional[str] = None) -> RaceState:
    """
    Returns a cloned RaceState filtered to what the specified team could legitimately know.
    If team is None, returns the public timing feed.
    """
    filtered_state = copy.deepcopy(state)

    if team is None:
        # Public feed: timing towers retain previous confirmed compound during out-lap
        for d in filtered_state.leaderboard:
            if d.is_out_lap or d.compound_observation_status == "UNCONFIRMED_OUT_LAP":
                d.compound = d.last_confirmed_compound if d.last_confirmed_compound != "UNKNOWN" else "UNKNOWN"
                d.compound_observation_status = "OBSERVATION_PENDING"
        return filtered_state

    # Team-specific perspective
    normalized_team = team.strip().lower()

    for d in filtered_state.leaderboard:
        is_own_car = (d.team.strip().lower() == normalized_team)

        if not is_own_car:
            # Rival car: mask unobserved tyre compounds during pit transitions and out-lap
            if d.is_out_lap or d.pit_status == "IN_PIT" or d.compound_observation_status == "UNCONFIRMED_OUT_LAP":
                d.compound = "UNKNOWN"
                d.compound_observation_status = "MASKED_IN_PIT"
        else:
            # Own car: internal pit wall status
            if d.is_out_lap:
                d.compound_observation_status = "TEAM_INTERNAL"

    return filtered_state
