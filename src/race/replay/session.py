"""
ReplaySession module for RACE (Phase 3A).
Loads cached Parquet race data and builds in-memory pre-sorted indexing structures
for O(log N) binary search lookups without mutating on-disk schemas.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
import numpy as np
import pandas as pd

from src.race.config import get_race_cache_dir

logger = logging.getLogger(__name__)


class LapBoundaryUnavailableError(Exception):
    """Raised when an authoritative leader lap boundary cannot be established."""
    pass


class ReplaySession:
    """
    In-memory representation of a cached race session.
    Manages cached dataframes and fast sorted numpy arrays for deterministic replay.
    """

    def __init__(self, year: int, gp: str):
        self.year = year
        self.gp = gp
        self.cache_dir: Path = get_race_cache_dir(year, gp)

        if not self.cache_dir.exists():
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")

        # 1. Load metadata
        meta_path = self.cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {self.cache_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata: Dict[str, Any] = json.load(f)

        self.race_id: str = f"{self.year}_{self.gp.lower().replace(' ', '_')}"
        self.total_laps: int = int(self.metadata.get("total_laps", 0))

        # 2. Load cached Parquet tables
        self.laps_df: pd.DataFrame = pd.read_parquet(self.cache_dir / "laps.parquet")
        self.stints_df: pd.DataFrame = pd.read_parquet(self.cache_dir / "stints.parquet")
        self.positions_df: pd.DataFrame = pd.read_parquet(self.cache_dir / "positions.parquet")
        self.weather_df: pd.DataFrame = pd.read_parquet(self.cache_dir / "weather.parquet")
        self.classification_df: pd.DataFrame = pd.read_parquet(self.cache_dir / "classification.parquet")

        # 3. Load track outline
        track_path = self.cache_dir / "track_outline.json"
        with open(track_path, "r", encoding="utf-8") as f:
            self.track_outline: Dict[str, Any] = json.load(f)

        self.total_circuit_distance_m: float = float(self.track_outline.get("total_distance_m", 5000.0))
        pts = self.track_outline.get("points", [])
        self.track_x = np.array([p["x"] for p in pts], dtype=np.float64)
        self.track_y = np.array([p["y"] for p in pts], dtype=np.float64)
        self.track_dist = np.array([p["dist_m"] for p in pts], dtype=np.float64)

        # Compute coordinate alignment offset between raw positions and centered track outline
        outline_center_x = float((self.track_x.max() + self.track_x.min()) / 2.0)
        outline_center_y = float((self.track_y.max() + self.track_y.min()) / 2.0)

        clean_pos = self.positions_df.dropna(subset=["x", "y"])
        if not clean_pos.empty:
            pos_center_x = float((clean_pos["x"].max() + clean_pos["x"].min()) / 2.0)
            pos_center_y = float((clean_pos["y"].max() + clean_pos["y"].min()) / 2.0)
            self.coord_offset_x = pos_center_x - outline_center_x
            self.coord_offset_y = pos_center_y - outline_center_y
        else:
            self.coord_offset_x = 0.0
            self.coord_offset_y = 0.0

        # 4. Extract driver list and drivers' team mapping
        self.drivers: List[str] = sorted(self.laps_df["driver"].unique().tolist())
        self.driver_teams: Dict[str, str] = {}
        for drv, grp in self.laps_df.groupby("driver"):
            teams = grp["team"].dropna().unique()
            self.driver_teams[drv] = str(teams[0]) if len(teams) > 0 else "Unknown"

        # 5. Build in-memory pre-sorted NumPy arrays for O(log N) coordinate queries
        self.pos_times: Dict[str, np.ndarray] = {}
        self.pos_x: Dict[str, np.ndarray] = {}
        self.pos_y: Dict[str, np.ndarray] = {}

        for drv, grp in self.positions_df.groupby("driver"):
            sorted_grp = grp.sort_values("session_time_s")
            self.pos_times[drv] = sorted_grp["session_time_s"].to_numpy(dtype=np.float64)
            self.pos_x[drv] = sorted_grp["x"].to_numpy(dtype=np.float64)
            self.pos_y[drv] = sorted_grp["y"].to_numpy(dtype=np.float64)

        # 6. Build in-memory pre-sorted lap tables & fast dict lookups per driver
        self.driver_laps: Dict[str, pd.DataFrame] = {}
        self.driver_lap_dict: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.driver_lap_crossing_times: Dict[str, np.ndarray] = {}
        self.driver_lap_numbers: Dict[str, np.ndarray] = {}

        for drv, grp in self.laps_df.groupby("driver"):
            sorted_laps = grp.sort_values("lap_number").reset_index(drop=True)
            self.driver_laps[drv] = sorted_laps
            self.driver_lap_dict[drv] = {
                int(row["lap_number"]): row.to_dict()
                for _, row in sorted_laps.iterrows()
            }
            valid_crossings = sorted_laps.dropna(subset=["session_time_s", "lap_time_s"])
            valid_crossings = valid_crossings[valid_crossings["lap_time_s"] > 0]
            self.driver_lap_crossing_times[drv] = valid_crossings["session_time_s"].to_numpy(dtype=np.float64)
            self.driver_lap_numbers[drv] = valid_crossings["lap_number"].to_numpy(dtype=np.int32)

        # 7. Fast stint lookups
        self.driver_pit_in_laps: Dict[str, Set[int]] = {}
        for drv, grp in self.stints_df.groupby("driver"):
            self.driver_pit_in_laps[drv] = set(
                grp["pit_in_lap"].dropna().astype(int).tolist()
            )

        # 8. Weather sorted time array
        sorted_weather = self.weather_df.sort_values("session_time_s")
        self.weather_times: np.ndarray = sorted_weather["session_time_s"].to_numpy(dtype=np.float64)
        self.sorted_weather_df: pd.DataFrame = sorted_weather.reset_index(drop=True)

        # 8b. Pre-indexed starting grid positions and classified status
        self.driver_grid_positions: Dict[str, int] = {}
        self.driver_classified_status: Dict[str, str] = {}
        for _, row in self.classification_df.iterrows():
            d_name = str(row["driver"])
            g_pos = row.get("grid_position")
            if pd.notna(g_pos) and float(g_pos) > 0:
                self.driver_grid_positions[d_name] = int(float(g_pos))
            else:
                self.driver_grid_positions[d_name] = 20
            self.driver_classified_status[d_name] = str(row.get("status", "")).strip()

        # 9. Compute Authoritative Boundaries (T_0, T_1, ... T_L)
        self._boundary_cache: Dict[int, float] = {}
        self.T_0: float = self._compute_race_start()
        self.T_finish: float = self.get_leader_crossing_time(self.total_laps)

    def _compute_race_start(self) -> float:
        """Derive authoritative race start (T_0) from Leader Lap 1 crossing minus Lap 1 time."""
        t1 = self.get_leader_crossing_time(1)
        lead_lap1 = self.laps_df[(self.laps_df["lap_number"] == 1) & (self.laps_df["position"] == 1.0)]
        if not lead_lap1.empty and pd.notna(lead_lap1.iloc[0]["lap_time_s"]):
            t_start = t1 - float(lead_lap1.iloc[0]["lap_time_s"])
            return round(t_start, 3)
        return round(t1 - 90.0, 3)

    def get_leader_crossing_time(self, lap: int) -> float:
        """
        Authoritative lookup for the leader crossing timestamp T_l.
        Strict rule: Never substitutes another driver's crossing time.
        Raises LapBoundaryUnavailableError if unavailable.
        """
        if lap in self._boundary_cache:
            return self._boundary_cache[lap]

        if lap < 1 or lap > self.total_laps:
            raise LapBoundaryUnavailableError(
                f"Lap {lap} is out of valid range [1, {self.total_laps}]"
            )

        leader_row = self.laps_df[
            (self.laps_df["lap_number"] == lap) & (self.laps_df["position"] == 1.0)
        ]

        if not leader_row.empty:
            t_cross = leader_row.iloc[0].get("session_time_s")
            if pd.notna(t_cross):
                val = float(t_cross)
                self._boundary_cache[lap] = val
                return val

        raise LapBoundaryUnavailableError(
            f"Authoritative leader lap boundary unavailable for lap {lap}"
        )

    def get_boundary_timestamps(self) -> List[float]:
        """Returns the full ordered list [T_0, T_1, ..., T_L]."""
        boundaries = [self.T_0]
        for l in range(1, self.total_laps + 1):
            boundaries.append(self.get_leader_crossing_time(l))
        return boundaries
