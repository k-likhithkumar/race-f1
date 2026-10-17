"""
Telemetry trace comparison and section performance analytics module for RACE.
Calculates distance-aligned cumulative time delta traces and corner/section statistics.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_delta_trace(
    tel1: pd.DataFrame,
    tel2: pd.DataFrame,
    step_m: float = 10.0,
    total_distance_m: Optional[float] = None,
) -> pd.DataFrame:
    """
    Computes the cumulative time delta along lap distance between driver 1 and driver 2.
    Delta is defined as: delta_s = time1_s - time2_s.
    A positive delta means driver 1 is slower (taking more time to reach distance s).

    Parameters
    ----------
    tel1, tel2 : pd.DataFrame
        Telemetry DataFrames containing 'distance_m' and 'time_s'.
    step_m : float
        Uniform distance interpolation step (default: 10.0 meters).
    total_distance_m : float, optional
        Track length in meters. If omitted, uses the overlapping maximum distance.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ['distance_m', 'time1_s', 'time2_s', 'delta_s', 'speed1_kmh', 'speed2_kmh']
    """
    empty_res = pd.DataFrame(columns=["distance_m", "time1_s", "time2_s", "delta_s", "speed1_kmh", "speed2_kmh"])
    if tel1 is None or tel1.empty or tel2 is None or tel2.empty:
        return empty_res

    d1 = tel1.dropna(subset=["distance_m", "time_s"]).sort_values("distance_m")
    d2 = tel2.dropna(subset=["distance_m", "time_s"]).sort_values("distance_m")
    if len(d1) < 2 or len(d2) < 2:
        return empty_res

    max_d1 = float(d1["distance_m"].max())
    max_d2 = float(d2["distance_m"].max())
    max_dist = total_distance_m if total_distance_m is not None else min(max_d1, max_d2)

    grid = np.arange(0.0, max_dist + step_m, step_m)
    if len(grid) == 0:
        return empty_res

    # Interpolate time along distance grid
    t1_interp = np.interp(grid, d1["distance_m"].values, d1["time_s"].values)
    t2_interp = np.interp(grid, d2["distance_m"].values, d2["time_s"].values)
    delta_arr = np.round(t1_interp - t2_interp, 4)

    # Interpolate speed if available
    s1_interp = np.interp(grid, d1["distance_m"].values, d1["speed_kmh"].values) if "speed_kmh" in d1.columns and d1["speed_kmh"].notna().any() else np.full_like(grid, np.nan)
    s2_interp = np.interp(grid, d2["distance_m"].values, d2["speed_kmh"].values) if "speed_kmh" in d2.columns and d2["speed_kmh"].notna().any() else np.full_like(grid, np.nan)

    return pd.DataFrame({
        "distance_m": np.round(grid, 1),
        "time1_s": np.round(t1_interp, 4),
        "time2_s": np.round(t2_interp, 4),
        "delta_s": delta_arr,
        "speed1_kmh": np.round(s1_interp, 1),
        "speed2_kmh": np.round(s2_interp, 1),
    })


def compute_section_stats(
    section_id: str,
    telemetry_dict: Dict[str, pd.DataFrame],
    track_outline: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Calculates driver performance metrics within a specified track section or corner.

    Parameters
    ----------
    section_id : str
        Identifier, e.g. 'T1', 'Sector 1', or 'mini_sector_4'.
    telemetry_dict : dict
        Dict mapping driver code -> lap telemetry DataFrame.
    track_outline : dict
        Cached track_outline.json with distance boundaries.

    Returns
    -------
    dict
        Dict containing fastest driver, minimum cornering speeds, braking points,
        and time deltas.
    """
    # Resolve boundary distances from track outline
    start_m = None
    end_m = None

    if "sectors" in track_outline:
        for s in track_outline["sectors"]:
            if str(s.get("sector")) == str(section_id) or f"Sector {s.get('sector')}" == section_id:
                start_m = float(s["start_m"])
                end_m = float(s["end_m"])
                break

    if start_m is None and "mini_sectors" in track_outline:
        for ms in track_outline["mini_sectors"]:
            if str(ms.get("mini_sector_id")) == str(section_id) or f"mini_sector_{ms.get('mini_sector_id')}" == section_id:
                start_m = float(ms["start_m"])
                end_m = float(ms["end_m"])
                break

    if start_m is None or end_m is None:
        # Default fallback to first 500m if section is not found
        start_m = 0.0
        end_m = 500.0

    driver_metrics = {}
    fastest_time = float("inf")
    fastest_driver = None

    for drv, tel in telemetry_dict.items():
        if tel is None or tel.empty:
            continue

        df_sec = tel[(tel["distance_m"] >= start_m) & (tel["distance_m"] <= end_m)].sort_values("distance_m")
        if len(df_sec) < 2:
            continue

        t_start = float(np.interp(start_m, tel["distance_m"].values, tel["time_s"].values))
        t_end = float(np.interp(end_m, tel["distance_m"].values, tel["time_s"].values))
        duration = round(t_end - t_start, 4)

        if duration < fastest_time:
            fastest_time = duration
            fastest_driver = drv

        # Min and max speed
        if "speed_kmh" in df_sec.columns and df_sec["speed_kmh"].notna().any():
            valid_speeds = df_sec["speed_kmh"].dropna()
            min_speed = round(float(valid_speeds.min()), 1) if not valid_speeds.empty else np.nan
            max_speed = round(float(valid_speeds.max()), 1) if not valid_speeds.empty else np.nan
        else:
            min_speed = np.nan
            max_speed = np.nan

        # Braking point detection
        braking_point = None
        if "brake_pct" in df_sec.columns and df_sec["brake_pct"].notna().any():
            brake_rows = df_sec[df_sec["brake_pct"] > 5.0]
            if not brake_rows.empty:
                braking_point = round(float(brake_rows.iloc[0]["distance_m"]), 1)
        
        # Fallback to deceleration derivative if brake channel is unavailable
        if braking_point is None and "speed_kmh" in df_sec.columns and len(df_sec) > 3:
            v_ms = df_sec["speed_kmh"].values / 3.6
            dt = np.diff(df_sec["time_s"].values)
            dv = np.diff(v_ms)
            valid = dt > 0
            if valid.any():
                accel = np.zeros_like(v_ms)
                accel[1:][valid] = dv[valid] / dt[valid]
                decel_idx = np.where(accel < -2.0)[0]
                if len(decel_idx) > 0:
                    braking_point = round(float(df_sec["distance_m"].iloc[decel_idx[0]]), 1)

        driver_metrics[drv] = {
            "duration_s": duration,
            "min_speed_kmh": min_speed,
            "max_speed_kmh": max_speed,
            "braking_point_m": braking_point,
            "delta_to_fastest_s": 0.0,
        }

    # Compute deltas relative to fastest driver
    if fastest_driver is not None:
        for drv, m in driver_metrics.items():
            m["delta_to_fastest_s"] = round(m["duration_s"] - fastest_time, 4)

    return {
        "section_id": section_id,
        "start_m": start_m,
        "end_m": end_m,
        "fastest_driver": fastest_driver,
        "fastest_time_s": fastest_time if fastest_driver is not None else None,
        "drivers": driver_metrics,
    }
