"""
Gap dynamics, closing rates, and historical battle detection module for RACE.
Calculates antisymmetric lap-by-lap interval gaps and identifies active on-track battles
without conflating proximity with overtakes.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.race.config import (
    BATTLE_GAP_THRESHOLD_S,
    BATTLE_CLOSING_RATE_THRESHOLD_S,
    POSITION_MAX_INTERPOLATION_GAP_S,
)

logger = logging.getLogger(__name__)


def compute_gap_history(
    laps_df: pd.DataFrame,
    driver1: str,
    driver2: str,
    as_of_lap: Optional[int] = None,
) -> pd.DataFrame:
    """
    Computes lap-by-lap gap and closing rate between two drivers.

    Timing Interpretation & Sign Convention:
    gap_s = time_driver2 - time_driver1
    If driver 1 reaches the finish line earlier (leading driver 2):
    gap_s > 0 (Driver 1 leads Driver 2).
    Antisymmetry: gap(d2, d1) = -gap(d1, d2).

    Closing rate:
    Rate at which the chasing car is closing the gap (s/lap):
    If driver 1 leads: closing_rate = gap(l-1) - gap(l).
    Positive closing rate means the gap is shrinking.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Cached laps table with 'driver', 'lap_number', 'session_time_s', 'is_clean'.
    driver1, driver2 : str
        Three-letter driver codes.
    as_of_lap : int, optional
        Replay barrier. Laps beyond as_of_lap are excluded.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['lap_number', 'gap_s', 'closing_rate_s_per_lap', 'leader', 'chaser', 'is_clean_pair']
    """
    empty_df = pd.DataFrame(columns=[
        "lap_number", "gap_s", "closing_rate_s_per_lap", "leader", "chaser", "is_clean_pair"
    ])
    if laps_df is None or laps_df.empty:
        return empty_df

    df = laps_df.copy()
    if as_of_lap is not None:
        df = df[df["lap_number"] <= as_of_lap]

    d1_laps = df[df["driver"] == driver1].dropna(subset=["session_time_s"]).set_index("lap_number")
    d2_laps = df[df["driver"] == driver2].dropna(subset=["session_time_s"]).set_index("lap_number")

    common_laps = sorted(list(set(d1_laps.index).intersection(set(d2_laps.index))))
    if not common_laps:
        return empty_df

    records = []
    prev_gap = None

    for lap in common_laps:
        t1 = float(d1_laps.loc[lap, "session_time_s"])
        t2 = float(d2_laps.loc[lap, "session_time_s"])
        # gap > 0 means driver1 is ahead of driver2
        gap = round(t2 - t1, 3)

        clean1 = bool(d1_laps.loc[lap, "is_clean"]) if "is_clean" in d1_laps.columns else True
        clean2 = bool(d2_laps.loc[lap, "is_clean"]) if "is_clean" in d2_laps.columns else True
        is_clean_pair = clean1 and clean2

        if gap > 0:
            leader = driver1
            chaser = driver2
            closing_rate = round(prev_gap - gap, 3) if prev_gap is not None and prev_gap > 0 else 0.0
        elif gap < 0:
            leader = driver2
            chaser = driver1
            closing_rate = round(gap - prev_gap, 3) if prev_gap is not None and prev_gap < 0 else 0.0
        else:
            leader = "DEAD_HEAT"
            chaser = "DEAD_HEAT"
            closing_rate = 0.0

        records.append({
            "lap_number": lap,
            "gap_s": gap,
            "closing_rate_s_per_lap": closing_rate,
            "leader": leader,
            "chaser": chaser,
            "is_clean_pair": is_clean_pair,
        })
        prev_gap = gap

    return pd.DataFrame(records)


def classify_overtakes(
    laps_df: pd.DataFrame,
    stints_df: Optional[pd.DataFrame] = None,
    as_of_lap: Optional[int] = None,
    telemetry_loader_func: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Classifies all historical position changes across laps:
    - Strictly distinguishes genuine on-track passes from pit lane swaps,
      retirements/crashes, caution periods, and telemetry gaps.
    - Classifies evidence as CONFIRMED, INFERRED, or UNKNOWN.
    - Requires continuous telemetry crossover without gap > 2s for CONFIRMED_ON_TRACK.
    - Classifies lap-end timing-loop changes during green running as INFERRED_ON_TRACK
      when telemetry is uninspected or unavailable.
    - Identifies PIT_STRATEGY_SWAP when either driver was in pit lane.
    - Identifies RETIREMENT_PASS when overtaken car failed to complete subsequent laps.
    - Identifies CAUTION_PERIOD_CHANGE when position change occurred under SC/VSC/Yellow.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Cached laps table.
    stints_df : pd.DataFrame, optional
        Cached stints table for pit stop detection.
    as_of_lap : int, optional
        Replay boundary.
    telemetry_loader_func : callable, optional
        Optional function (driver, lap) -> pd.DataFrame to inspect telemetry for crossover.

    Returns
    -------
    list of dict
        Classified overtake events with explicit evidence and confidence rating.
    """
    if laps_df is None or laps_df.empty:
        return []

    max_lap = as_of_lap if as_of_lap is not None else int(laps_df["lap_number"].max())
    df = laps_df[laps_df["lap_number"] <= max_lap].copy()

    # Pre-build pit lap sets
    pit_laps_map = {}
    if stints_df is not None and not stints_df.empty:
        for _, s_row in stints_df.iterrows():
            drv = s_row.get("driver")
            p_in = s_row.get("pit_in_lap")
            if pd.notna(p_in):
                p_val = int(p_in)
                if drv not in pit_laps_map:
                    pit_laps_map[drv] = set()
                pit_laps_map[drv].add(p_val)
                pit_laps_map[drv].add(p_val + 1)

    # Max lap per driver to detect retirements
    max_lap_per_driver = laps_df.groupby("driver")["lap_number"].max().to_dict()

    overtakes = []

    # Check consecutive laps
    for lap in range(2, max_lap + 1):
        prev_df = df[df["lap_number"] == lap - 1].dropna(subset=["session_time_s"]).sort_values("session_time_s")
        curr_df = df[df["lap_number"] == lap].dropna(subset=["session_time_s"]).sort_values("session_time_s")

        if prev_df.empty or curr_df.empty:
            continue

        prev_order = prev_df["driver"].tolist()
        curr_order = curr_df["driver"].tolist()

        common_drivers = [d for d in curr_order if d in prev_order]
        for idx_a in range(len(common_drivers)):
            drv_a = common_drivers[idx_a]
            curr_pos_a = curr_order.index(drv_a)
            prev_pos_a = prev_order.index(drv_a)

            for idx_b in range(idx_a + 1, len(common_drivers)):
                drv_b = common_drivers[idx_b]
                curr_pos_b = curr_order.index(drv_b)
                prev_pos_b = prev_order.index(drv_b)

                # Check if drv_a passed drv_b:
                # On prev lap: drv_b was ahead of drv_a (prev_pos_b < prev_pos_a)
                # On curr lap: drv_a is ahead of drv_b (curr_pos_a < curr_pos_b)
                if prev_pos_b < prev_pos_a and curr_pos_a < curr_pos_b:
                    event = _classify_swap(
                        df, pit_laps_map, max_lap_per_driver, lap, drv_a, drv_b, telemetry_loader_func
                    )
                    overtakes.append(event)

    return overtakes


def _classify_swap(
    df: pd.DataFrame,
    pit_laps_map: Dict[str, set],
    max_lap_per_driver: Dict[str, int],
    lap: int,
    overtaking_driver: str,
    overtaken_driver: str,
    telemetry_loader_func: Optional[Any] = None,
) -> Dict[str, Any]:
    """Evaluates the evidence for a single driver position change on a given lap."""
    curr_lap_a = df[(df["driver"] == overtaking_driver) & (df["lap_number"] == lap)]
    curr_lap_b = df[(df["driver"] == overtaken_driver) & (df["lap_number"] == lap)]

    # 1. Check for Pit Stops (Pit Strategy Swap)
    is_pit_a = False
    is_pit_b = False
    if overtaking_driver in pit_laps_map and (lap in pit_laps_map[overtaking_driver] or (lap - 1) in pit_laps_map[overtaking_driver]):
        is_pit_a = True
    if overtaken_driver in pit_laps_map and (lap in pit_laps_map[overtaken_driver] or (lap - 1) in pit_laps_map[overtaken_driver]):
        is_pit_b = True

    if not curr_lap_a.empty and (bool(curr_lap_a.iloc[0].get("is_pit_in", False)) or bool(curr_lap_a.iloc[0].get("is_pit_out", False))):
        is_pit_a = True
    if not curr_lap_b.empty and (bool(curr_lap_b.iloc[0].get("is_pit_in", False)) or bool(curr_lap_b.iloc[0].get("is_pit_out", False))):
        is_pit_b = True

    if is_pit_b or is_pit_a:
        return {
            "lap_number": lap,
            "overtaking_driver": overtaking_driver,
            "overtaken_driver": overtaken_driver,
            "classification": "PIT_STRATEGY_SWAP",
            "confidence": "CONFIRMED",
            "is_on_track_overtake": False,
            "evidence": f"Car {overtaken_driver if is_pit_b else overtaking_driver} was in pit lane during passing window.",
        }

    # 2. Check for Retirements / DNF
    max_b = max_lap_per_driver.get(overtaken_driver, 0)
    max_race_lap = max(max_lap_per_driver.values()) if max_lap_per_driver else lap
    t_b = curr_lap_b.iloc[0].get("lap_time_s") if not curr_lap_b.empty else None

    # Car retired if it failed to record lap time on lap, or stopped while other cars continued
    is_retired = (t_b is None or pd.isna(t_b) or t_b <= 0) or (max_b < max_race_lap and max_b == lap)
    if is_retired:
        return {
            "lap_number": lap,
            "overtaking_driver": overtaking_driver,
            "overtaken_driver": overtaken_driver,
            "classification": "RETIREMENT_PASS",
            "confidence": "CONFIRMED",
            "is_on_track_overtake": False,
            "evidence": f"Car {overtaken_driver} retired or stopped running on lap {lap}.",
        }

    # 3. Check for Caution Period
    status_a = str(curr_lap_a.iloc[0].get("track_status", "1")).strip() if not curr_lap_a.empty else "1"
    status_b = str(curr_lap_b.iloc[0].get("track_status", "1")).strip() if not curr_lap_b.empty else "1"
    if status_a != "1" or status_b != "1":
        return {
            "lap_number": lap,
            "overtaking_driver": overtaking_driver,
            "overtaken_driver": overtaken_driver,
            "classification": "CAUTION_PERIOD_CHANGE",
            "confidence": "INFERRED",
            "is_on_track_overtake": False,
            "evidence": f"Position change occurred under caution (track status {status_a}/{status_b}).",
        }

    # 4. On-Track Racing (Green flag, neither car in pit, both running)
    if telemetry_loader_func is not None:
        try:
            tel_a = telemetry_loader_func(overtaking_driver, lap)
            tel_b = telemetry_loader_func(overtaken_driver, lap)
            if tel_a is not None and not tel_a.empty and tel_b is not None and not tel_b.empty:
                gap_a = np.diff(tel_a["time_s"].values) if "time_s" in tel_a.columns else np.array([])
                gap_b = np.diff(tel_b["time_s"].values) if "time_s" in tel_b.columns else np.array([])
                if (gap_a > POSITION_MAX_INTERPOLATION_GAP_S).any() or (gap_b > POSITION_MAX_INTERPOLATION_GAP_S).any():
                    return {
                        "lap_number": lap,
                        "overtaking_driver": overtaking_driver,
                        "overtaken_driver": overtaken_driver,
                        "classification": "UNKNOWN_DATA_GAP",
                        "confidence": "UNKNOWN",
                        "is_on_track_overtake": False,
                        "crossover_distance_m": None,
                        "evidence": "Telemetry contains time gap exceeding 2.0s during passing window.",
                    }

                # Distance alignment: establish physical crossover evidence
                if "distance_m" in tel_a.columns and "distance_m" in tel_b.columns:
                    tel_a_clean = tel_a.dropna(subset=["distance_m", "time_s"]).drop_duplicates(subset=["distance_m"]).sort_values("distance_m")
                    tel_b_clean = tel_b.dropna(subset=["distance_m", "time_s"]).drop_duplicates(subset=["distance_m"]).sort_values("distance_m")

                    if len(tel_a_clean) >= 2 and len(tel_b_clean) >= 2:
                        d_min = max(float(tel_a_clean["distance_m"].min()), float(tel_b_clean["distance_m"].min()))
                        d_max = min(float(tel_a_clean["distance_m"].max()), float(tel_b_clean["distance_m"].max()))

                        if (d_max - d_min) >= 500.0:
                            # Retrieve session start timestamps for both cars on this lap
                            prev_lap_a = df[(df["driver"] == overtaking_driver) & (df["lap_number"] == lap - 1)]
                            prev_lap_b = df[(df["driver"] == overtaken_driver) & (df["lap_number"] == lap - 1)]

                            s_start_a = None
                            s_start_b = None
                            if not prev_lap_a.empty and pd.notna(prev_lap_a.iloc[0].get("session_time_s")):
                                s_start_a = float(prev_lap_a.iloc[0]["session_time_s"])
                            elif not curr_lap_a.empty and pd.notna(curr_lap_a.iloc[0].get("session_time_s")) and pd.notna(curr_lap_a.iloc[0].get("lap_time_s")):
                                s_start_a = float(curr_lap_a.iloc[0]["session_time_s"]) - float(curr_lap_a.iloc[0]["lap_time_s"])

                            if not prev_lap_b.empty and pd.notna(prev_lap_b.iloc[0].get("session_time_s")):
                                s_start_b = float(prev_lap_b.iloc[0]["session_time_s"])
                            elif not curr_lap_b.empty and pd.notna(curr_lap_b.iloc[0].get("session_time_s")) and pd.notna(curr_lap_b.iloc[0].get("lap_time_s")):
                                s_start_b = float(curr_lap_b.iloc[0]["session_time_s"]) - float(curr_lap_b.iloc[0]["lap_time_s"])

                            if s_start_a is not None and s_start_b is not None:
                                grid = np.linspace(d_min, d_max, 500)
                                t_a_grid = np.interp(grid, tel_a_clean["distance_m"].values, tel_a_clean["time_s"].values)
                                t_b_grid = np.interp(grid, tel_b_clean["distance_m"].values, tel_b_clean["time_s"].values)

                                T_a_grid = s_start_a + t_a_grid
                                T_b_grid = s_start_b + t_b_grid
                                # Delta = T_b - T_a.
                                # Delta < 0 => Car B reached point earlier (Car B ahead).
                                # Delta > 0 => Car A reached point earlier (Car A ahead).
                                delta = T_b_grid - T_a_grid

                                crossing_indices = np.where((delta[:-1] <= 0) & (delta[1:] > 0))[0]
                                if len(crossing_indices) > 0:
                                    cross_idx = crossing_indices[0]
                                    d_i = grid[cross_idx]
                                    d_next = grid[cross_idx + 1]
                                    val_i = delta[cross_idx]
                                    val_next = delta[cross_idx + 1]
                                    if abs(val_next - val_i) > 1e-6:
                                        s_star = d_i - val_i * ((d_next - d_i) / (val_next - val_i))
                                    else:
                                        s_star = d_i

                                    return {
                                        "lap_number": lap,
                                        "overtaking_driver": overtaking_driver,
                                        "overtaken_driver": overtaken_driver,
                                        "classification": "CONFIRMED_ON_TRACK_OVERTAKE",
                                        "confidence": "CONFIRMED",
                                        "is_on_track_overtake": True,
                                        "crossover_distance_m": round(float(s_star), 1),
                                        "evidence": f"Telemetry confirms physical on-track crossover at {round(float(s_star), 1)}m under green flag.",
                                    }
                                else:
                                    # Continuous telemetry does NOT demonstrate an on-track crossover
                                    return {
                                        "lap_number": lap,
                                        "overtaking_driver": overtaking_driver,
                                        "overtaken_driver": overtaken_driver,
                                        "classification": "INFERRED_ON_TRACK_OVERTAKE",
                                        "confidence": "INFERRED",
                                        "is_on_track_overtake": True,
                                        "crossover_distance_m": None,
                                        "evidence": "Continuous telemetry inspected but did not demonstrate a distinct on-track crossing; retained as inferred lap-end position swap.",
                                    }
        except Exception as exc:
            logger.debug(f"Telemetry inspection failed for overtake on lap {lap}: {exc}")

    # Fallback to timing transponder evidence: inferred on-track pass
    return {
        "lap_number": lap,
        "overtaking_driver": overtaking_driver,
        "overtaken_driver": overtaken_driver,
        "classification": "INFERRED_ON_TRACK_OVERTAKE",
        "confidence": "INFERRED",
        "is_on_track_overtake": True,
        "crossover_distance_m": None,
        "evidence": "Green-flag timing loops confirm lap-end position swap with both cars running and neither pitting; telemetry uninspected.",
    }


def detect_battles(
    laps_df: pd.DataFrame,
    as_of_lap: Optional[int] = None,
    gap_threshold_s: float = BATTLE_GAP_THRESHOLD_S,
    closing_rate_threshold_s: float = BATTLE_CLOSING_RATE_THRESHOLD_S,
    stints_df: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """
    Detects active historical battles at a given as_of_lap.

    A battle is an active competitive contest between two cars in close proximity,
    not an automatic overtake.

    Requirements per PDF:
    - Cars running on the same lap (excludes lapped traffic).
    - Strictly green flag running (track_status == '1').
    - Neither car in pit lane.
    - Gap <= gap_threshold_s (default 1.0s).
    - Active contest: closing rate >= closing_rate_threshold_s (0.05 s/lap) OR
      tight gap <= 0.7s maintained for 2+ laps.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Cached laps DataFrame.
    as_of_lap : int, optional
        Target lap to evaluate. If omitted, uses the latest completed lap in laps_df.
    gap_threshold_s : float
        Proximity limit in seconds (default: 1.0s).
    closing_rate_threshold_s : float
        Closing rate threshold in s/lap (default: 0.05 s/lap).
    stints_df : pd.DataFrame, optional
        Cached stints DataFrame to vet pit stops.

    Returns
    -------
    list of dict
        Detected battles with driver pair, gap, closing rate, battle verdict,
        and overtake classification if a position swap occurred.
    """
    if laps_df is None or laps_df.empty:
        return []

    target_lap = as_of_lap if as_of_lap is not None else int(laps_df["lap_number"].max())
    df = laps_df[laps_df["lap_number"] <= target_lap].copy()

    # Get standings on target_lap
    current_lap_df = df[df["lap_number"] == target_lap].dropna(subset=["session_time_s"]).sort_values("session_time_s").reset_index(drop=True)
    if len(current_lap_df) < 2:
        return []

    pit_laps_map = {}
    if stints_df is not None and not stints_df.empty:
        for _, s_row in stints_df.iterrows():
            drv = s_row.get("driver")
            p_in = s_row.get("pit_in_lap")
            if pd.notna(p_in):
                p_val = int(p_in)
                if drv not in pit_laps_map:
                    pit_laps_map[drv] = set()
                pit_laps_map[drv].add(p_val)
                pit_laps_map[drv].add(p_val + 1)

    max_lap_per_driver = laps_df.groupby("driver")["lap_number"].max().to_dict()

    battles = []

    for i in range(len(current_lap_df) - 1):
        car_ahead = current_lap_df.iloc[i]
        car_behind = current_lap_df.iloc[i + 1]

        # Exclude pit laps
        if bool(car_ahead.get("is_pit_in", False)) or bool(car_ahead.get("is_pit_out", False)):
            continue
        if bool(car_behind.get("is_pit_in", False)) or bool(car_behind.get("is_pit_out", False)):
            continue

        # Exclude caution laps (must be strictly green)
        if str(car_ahead.get("track_status", "")).strip() != "1" or str(car_behind.get("track_status", "")).strip() != "1":
            continue

        t_ahead = float(car_ahead["session_time_s"])
        t_behind = float(car_behind["session_time_s"])
        gap = round(t_behind - t_ahead, 3)

        if gap < 0 or gap > gap_threshold_s:
            continue

        # Compute closing rate from previous lap
        prev_lap = target_lap - 1
        closing_rate = 0.0
        tight_run_laps = 1

        prev_df = df[df["lap_number"] == prev_lap]
        prev_ahead = prev_df[prev_df["driver"] == car_ahead["driver"]]
        prev_behind = prev_df[prev_df["driver"] == car_behind["driver"]]

        position_swap_detected = False
        overtake_classification = None

        if not prev_ahead.empty and not prev_behind.empty:
            pt_ahead = float(prev_ahead.iloc[0]["session_time_s"])
            pt_behind = float(prev_behind.iloc[0]["session_time_s"])
            prev_gap = round(pt_behind - pt_ahead, 3)

            # Check if order reversed on track between previous lap and target lap
            if prev_gap < 0:
                # car_ahead was behind car_behind on previous lap -> car_ahead overtook car_behind
                position_swap_detected = True
                overtake_classification = _classify_swap(
                    df, pit_laps_map, max_lap_per_driver, target_lap, car_ahead["driver"], car_behind["driver"]
                )
            else:
                closing_rate = round(prev_gap - gap, 3)
                if prev_gap <= 0.7:
                    tight_run_laps += 1

        # Check battle condition: closing rate > threshold OR tight proximity maintained
        is_closing = closing_rate >= closing_rate_threshold_s
        is_sustained_pressure = tight_run_laps >= 2 or gap <= 0.5

        if is_closing or is_sustained_pressure or position_swap_detected:
            # Active battle verdict
            if position_swap_detected and overtake_classification and overtake_classification.get("is_on_track_overtake"):
                battle_verdict = "OVERTAKE_COMPLETED"
            elif is_closing:
                battle_verdict = "ACTIVE_CHASE"
            elif is_sustained_pressure:
                battle_verdict = "SUSTAINED_DEFENSE"
            else:
                battle_verdict = "CLOSE_CONTEST"

            battles.append({
                "lap_number": target_lap,
                "leader": car_ahead["driver"],
                "chaser": car_behind["driver"],
                "gap_s": gap,
                "closing_rate_s_per_lap": closing_rate,
                "is_closing": is_closing,
                "sustained_pressure": is_sustained_pressure,
                "battle_verdict": battle_verdict,
                "position_swap_detected": position_swap_detected,
                "is_overtake_confirmed": bool(overtake_classification and overtake_classification.get("classification") == "CONFIRMED_ON_TRACK_OVERTAKE"),
                "overtake_classification": overtake_classification,
            })

    return battles
