"""
Track outline and circuit geometry extraction module for RACE.
Generates smoothed coordinates, sector splits, and 20 mini-sector distance boundaries.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import fastf1

logger = logging.getLogger(__name__)


def build_track_outline(session: fastf1.core.Session, num_mini_sectors: int = 20) -> Dict[str, Any]:
    """
    Build a smoothed track outline and distance-based section boundaries from the fastest clean lap.

    Parameters
    ----------
    session : fastf1.core.Session
        FastF1 session object with loaded laps and telemetry.
    num_mini_sectors : int
        Number of mini-sectors to partition the lap into (default 20, per PDF spec).

    Returns
    -------
    dict
        Dictionary containing circuit metadata, total distance, sector splits,
        mini-sector boundaries, and normalized (X, Y) coordinate points.
    """
    clean_laps = session.laps.pick_quicklaps()
    if clean_laps.empty:
        clean_laps = session.laps.dropna(subset=["LapTime"])

    if clean_laps.empty:
        raise ValueError("Cannot build track outline: no valid laps found in session.")

    fastest_lap = clean_laps.pick_fastest()
    tel = fastest_lap.get_telemetry()

    if tel.empty or "X" not in tel.columns or "Y" not in tel.columns:
        raise ValueError("Cannot build track outline: telemetry coordinates missing for fastest lap.")

    # Extract coordinates and distance
    df_coords = tel[["Distance", "X", "Y"]].dropna().sort_values("Distance").drop_duplicates(subset=["Distance"])
    total_distance_m = float(df_coords["Distance"].max())

    # Evenly resample track geometry to ~1000 points for smooth SVG rendering
    target_points = 1000
    resampled_dist = np.linspace(0, total_distance_m, target_points)
    resampled_x = np.interp(resampled_dist, df_coords["Distance"].values, df_coords["X"].values)
    resampled_y = np.interp(resampled_dist, df_coords["Distance"].values, df_coords["Y"].values)

    # Center coordinates around (0, 0) for uniform UI viewport scaling
    x_center = float(np.mean(resampled_x))
    y_center = float(np.mean(resampled_y))
    norm_x = (resampled_x - x_center).round(2).tolist()
    norm_y = (resampled_y - y_center).round(2).tolist()

    # Determine Sector 1 and Sector 2 split distances from telemetry markers if available
    s1_dist_m = float(total_distance_m / 3.0)  # Default fallback if telemetry missing split markers
    s2_dist_m = float(2.0 * total_distance_m / 3.0)

    # Attempt to locate exact sector distance splits from the lap
    try:
        s1_time = fastest_lap["Sector1Time"]
        s2_time = fastest_lap["Sector2Time"]
        lap_start_time = fastest_lap["LapStartTime"]
        if pd.notna(s1_time) and pd.notna(lap_start_time):
            s1_boundary_time = lap_start_time + s1_time
            s1_tel = tel[tel["Time"] <= s1_boundary_time]
            if not s1_tel.empty:
                s1_dist_m = float(s1_tel["Distance"].max())
        if pd.notna(s2_time) and pd.notna(lap_start_time):
            s2_boundary_time = lap_start_time + s1_time + s2_time
            s2_tel = tel[tel["Time"] <= s2_boundary_time]
            if not s2_tel.empty:
                s2_dist_m = float(s2_tel["Distance"].max())
    except Exception as exc:
        logger.debug(f"Could not extract exact sector distance markers; using proportional estimate: {exc}")

    # Build 20 equal-distance mini-sector boundaries (PDF Topic 4 & 5)
    mini_sector_boundaries: List[Dict[str, Any]] = []
    mini_sector_step = total_distance_m / num_mini_sectors
    for i in range(num_mini_sectors):
        start_m = round(i * mini_sector_step, 2)
        end_m = round((i + 1) * mini_sector_step, 2) if i < num_mini_sectors - 1 else round(total_distance_m, 2)
        # Determine parent sector (1, 2, or 3)
        mid_m = (start_m + end_m) / 2.0
        if mid_m <= s1_dist_m:
            sector = 1
        elif mid_m <= s2_dist_m:
            sector = 2
        else:
            sector = 3

        mini_sector_boundaries.append({
            "mini_sector_id": i + 1,
            "start_m": start_m,
            "end_m": end_m,
            "sector": sector,
        })

    points_data = [
        {"dist_m": round(float(d), 2), "x": x, "y": y}
        for d, x, y in zip(resampled_dist, norm_x, norm_y)
    ]

    circuit_name = getattr(session.event, "OfficialEventName", getattr(session.event, "EventName", "Circuit"))

    return {
        "circuit_name": circuit_name,
        "year": int(session.event.year),
        "total_distance_m": round(total_distance_m, 2),
        "sectors": [
            {"sector": 1, "start_m": 0.0, "end_m": round(s1_dist_m, 2)},
            {"sector": 2, "start_m": round(s1_dist_m, 2), "end_m": round(s2_dist_m, 2)},
            {"sector": 3, "start_m": round(s2_dist_m, 2), "end_m": round(total_distance_m, 2)},
        ],
        "mini_sectors": mini_sector_boundaries,
        "points": points_data,
    }
