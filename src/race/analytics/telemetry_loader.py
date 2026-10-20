"""
Offline telemetry loader and schema normalizer for RACE.
Provides high-resolution single-lap car telemetry for both FastF1 and OpenF1 sessions
without initiating network requests.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.race.config import (
    RAW_FASTF1_CACHE_DIR,
    RAW_OPENF1_CACHE_DIR,
    get_race_cache_dir,
)
from src.race.data.loader import load_session

logger = logging.getLogger(__name__)

# In-memory session cache for fast repeated lap lookups
_SESSION_CACHE: Dict[str, Any] = {}
_LAP_TELEMETRY_CACHE: Dict[Tuple[int, str, str, int], pd.DataFrame] = {}


def _get_cached_fastf1_session(year: int, gp: str, session_type: str = "R") -> Any:
    cache_key = f"{year}_{gp}_{session_type}"
    if cache_key not in _SESSION_CACHE:
        _SESSION_CACHE[cache_key] = load_session(year, gp, session_type)
    return _SESSION_CACHE[cache_key]


def load_lap_telemetry(
    year: int,
    gp: str,
    driver: str,
    lap_number: int,
) -> pd.DataFrame:
    """
    Load normalized single-lap telemetry for a driver completely offline.

    Normalized DataFrame Columns:
    - distance_m: Cumulative distance along the lap in meters
    - time_s: Elapsed time from start of lap in seconds
    - speed_kmh: Speed in km/h
    - throttle_pct: Throttle application (0-100%) or NaN if unavailable
    - brake_pct: Brake application (0-100% or boolean) or NaN if unavailable
    - gear: Current gear (1-8) or NaN if unavailable
    - drs: DRS indicator or NaN if unavailable

    DataFrame Attributes (df.attrs):
    - provider: 'FastF1' | 'OpenF1'
    - channels_available: List[str]
    - channels_unavailable: List[str]
    """
    cache_key = (year, gp.lower(), driver.upper(), lap_number)
    if cache_key in _LAP_TELEMETRY_CACHE:
        return _LAP_TELEMETRY_CACHE[cache_key].copy()

    cache_dir = get_race_cache_dir(year, gp)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found for race: {year} {gp}")

    # Inspect metadata to determine ingestion source
    meta_path = cache_dir / "metadata.json"
    is_openf1 = False
    session_key = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            if meta.get("data_source") == "openf1" or meta.get("source") == "openf1":
                is_openf1 = True
                session_key = meta.get("session_key")

    if is_openf1:
        if session_key is None:
            # Look up session_key from cached drivers files in raw_openf1
            for p in RAW_OPENF1_CACHE_DIR.glob("drivers_*.json"):
                try:
                    s_key_str = p.stem.split("_")[1]
                    session_key = int(s_key_str)
                    break
                except (IndexError, ValueError):
                    pass
        if session_key is None:
            session_key = 9157

        result = _load_openf1_lap_telemetry(year, gp, session_key, driver, lap_number, cache_dir)
    else:
        result = _load_fastf1_lap_telemetry(year, gp, driver, lap_number)

    _LAP_TELEMETRY_CACHE[cache_key] = result
    return result.copy()


def _load_fastf1_lap_telemetry(
    year: int,
    gp: str,
    driver: str,
    lap_number: int,
) -> pd.DataFrame:
    """Load and normalize telemetry from FastF1 local cache."""
    try:
        session = _get_cached_fastf1_session(year, gp, "R")
    except Exception as exc:
        logger.warning(f"Failed to load FastF1 session for {year} {gp}: {exc}")
        return pd.DataFrame()

    if not hasattr(session, "laps") or session.laps.empty:
        return pd.DataFrame()

    driver_laps = session.laps.pick_driver(driver)
    matching_lap = driver_laps[driver_laps["LapNumber"] == lap_number]
    if matching_lap.empty:
        logger.warning(f"Lap {lap_number} not found for driver {driver} in FastF1 {year} {gp}")
        return pd.DataFrame()

    try:
        raw_tel = matching_lap.iloc[0].get_telemetry()
    except Exception as exc:
        logger.warning(f"Could not extract telemetry for {driver} lap {lap_number}: {exc}")
        return pd.DataFrame()

    if raw_tel.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["distance_m"] = raw_tel["Distance"].astype(float)
    if "Time" in raw_tel.columns and pd.api.types.is_timedelta64_dtype(raw_tel["Time"]):
        df["time_s"] = raw_tel["Time"].dt.total_seconds().astype(float)
    else:
        df["time_s"] = pd.to_numeric(raw_tel["Time"], errors="coerce")

    # Shift time_s so lap start is 0.0s
    if not df.empty and df["time_s"].notna().any():
        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]

    df["speed_kmh"] = pd.to_numeric(raw_tel.get("Speed"), errors="coerce").astype(float)
    df["throttle_pct"] = pd.to_numeric(raw_tel.get("Throttle"), errors="coerce").astype(float)
    
    brake_val = raw_tel.get("Brake")
    if brake_val is not None:
        df["brake_pct"] = brake_val.astype(float) if not brake_val.dtype == bool else brake_val.astype(int) * 100.0
    else:
        df["brake_pct"] = np.nan

    df["gear"] = pd.to_numeric(raw_tel.get("nGear"), errors="coerce")
    df["drs"] = pd.to_numeric(raw_tel.get("DRS"), errors="coerce")

    df = df.dropna(subset=["distance_m", "time_s"]).sort_values("distance_m").reset_index(drop=True)
    df.attrs["provider"] = "FastF1"
    df.attrs["channels_available"] = ["distance_m", "time_s", "speed_kmh", "throttle_pct", "brake_pct", "gear", "drs"]
    df.attrs["channels_unavailable"] = []
    return df


def _load_openf1_lap_telemetry(
    year: int,
    gp: str,
    session_key: int,
    driver: str,
    lap_number: int,
    cache_dir: Path,
) -> pd.DataFrame:
    """Reconstruct single-lap distance and speed telemetry from OpenF1 local location coordinates."""
    # Find driver number from driver code
    drivers_file = RAW_OPENF1_CACHE_DIR / f"drivers_{session_key}.json"
    driver_number = None
    if drivers_file.exists():
        with open(drivers_file, "r", encoding="utf-8") as f:
            drv_list = json.load(f)
            for d in drv_list:
                if d.get("name_acronym") == driver or str(d.get("driver_number")) == driver:
                    driver_number = int(d["driver_number"])
                    break

    if driver_number is None:
        logger.warning(f"Could not map driver code {driver} to driver number for OpenF1 session {session_key}")
        return pd.DataFrame()

    # Load laps to get date_start and lap_duration
    laps_path = cache_dir / "laps.parquet"
    if not laps_path.exists():
        return pd.DataFrame()

    laps_df = pd.read_parquet(laps_path)
    drv_laps = laps_df[(laps_df["driver"] == driver) & (laps_df["lap_number"] == lap_number)]
    if drv_laps.empty:
        return pd.DataFrame()

    lap_row = drv_laps.iloc[0]
    lap_duration = lap_row.get("lap_time_s")
    if pd.isna(lap_duration) or lap_duration <= 0:
        return pd.DataFrame()

    # Read raw location data
    loc_file = RAW_OPENF1_CACHE_DIR / f"loc_{session_key}_{driver_number}.json"
    if not loc_file.exists():
        logger.warning(f"Missing OpenF1 location cache {loc_file}")
        return pd.DataFrame()

    with open(loc_file, "r", encoding="utf-8") as f:
        loc_data = json.load(f)

    if not loc_data:
        return pd.DataFrame()

    df_loc = pd.DataFrame(loc_data)[["date", "x", "y"]].dropna()
    df_loc["dt"] = pd.to_datetime(df_loc["date"], format="ISO8601")

    # Match timing window using raw laps json or elapsed session time
    raw_laps_file = RAW_OPENF1_CACHE_DIR / f"laps_{session_key}.json"
    start_dt = None
    if raw_laps_file.exists():
        with open(raw_laps_file, "r", encoding="utf-8") as f:
            raw_laps = json.load(f)
            for r in raw_laps:
                if int(r.get("driver_number", -1)) == driver_number and int(r.get("lap_number", -1)) == lap_number:
                    s_str = r.get("date_start")
                    if s_str:
                        start_dt = pd.to_datetime(s_str, format="ISO8601")
                        break

    if start_dt is None:
        # Fallback to matching session time
        sess_file = RAW_OPENF1_CACHE_DIR / f"sess_{session_key}.json"
        if sess_file.exists():
            with open(sess_file, "r", encoding="utf-8") as f:
                s_info = json.load(f)
                if s_info and s_info[0].get("date_start"):
                    sess_start_dt = pd.to_datetime(s_info[0]["date_start"], format="ISO8601")
                    sess_time = lap_row.get("session_time_s")
                    if pd.notna(sess_time):
                        end_dt = sess_start_dt + pd.Timedelta(seconds=sess_time)
                        start_dt = end_dt - pd.Timedelta(seconds=lap_duration)

    if start_dt is None:
        return pd.DataFrame()

    end_dt = start_dt + pd.Timedelta(seconds=lap_duration)
    lap_loc = df_loc[(df_loc["dt"] >= start_dt) & (df_loc["dt"] <= end_dt)].sort_values("dt").reset_index(drop=True)
    if len(lap_loc) < 2:
        return pd.DataFrame()

    # Scale coordinates from decimeters to meters
    x_m = lap_loc["x"].astype(float).values / 10.0
    y_m = lap_loc["y"].astype(float).values / 10.0
    t_s = (lap_loc["dt"] - start_dt).dt.total_seconds().values

    # Compute cumulative path distance
    dx = np.diff(x_m)
    dy = np.diff(y_m)
    step_dist = np.sqrt(dx**2 + dy**2)
    dist_m = np.concatenate([[0.0], np.cumsum(step_dist)])

    # Compute numerical speed in km/h
    dt_arr = np.diff(t_s)
    speed_kmh = np.zeros_like(dist_m)
    valid_dt = dt_arr > 0
    raw_speeds = np.zeros_like(dt_arr)
    raw_speeds[valid_dt] = (step_dist[valid_dt] / dt_arr[valid_dt]) * 3.6
    # Smooth speed over 3 samples to mitigate GPS quantization jitter
    if len(raw_speeds) > 2:
        smoothed = np.convolve(raw_speeds, np.ones(3) / 3.0, mode="same")
    else:
        smoothed = raw_speeds
    speed_kmh[1:] = smoothed
    speed_kmh[0] = speed_kmh[1] if len(speed_kmh) > 1 else 0.0

    res = pd.DataFrame({
        "distance_m": np.round(dist_m, 2),
        "time_s": np.round(t_s, 3),
        "speed_kmh": np.round(speed_kmh, 1),
        "throttle_pct": np.nan,
        "brake_pct": np.nan,
        "gear": np.nan,
        "drs": np.nan,
    })

    res.attrs["provider"] = "OpenF1"
    res.attrs["channels_available"] = ["distance_m", "time_s", "speed_kmh"]
    res.attrs["channels_unavailable"] = ["throttle_pct", "brake_pct", "gear", "drs"]
    return res
