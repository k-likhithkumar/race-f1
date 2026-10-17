"""
Data processing and normalization module for RACE.
Transforms raw FastF1 sessions into clean, typed, deterministic DataFrames.
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import fastf1

from src.race.config import POSITION_RESAMPLE_HZ, POSITION_MAX_INTERPOLATION_GAP_S

logger = logging.getLogger(__name__)

# Canonical compound mapping
COMPOUND_MAP = {
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE",
    "INTER": "INTERMEDIATE",
    "WET": "WET",
}


def normalize_compound(compound: Any) -> str:
    """Normalize tyre compound string to canonical values or 'UNKNOWN'."""
    if not isinstance(compound, str) or not compound.strip():
        return "UNKNOWN"
    return COMPOUND_MAP.get(compound.strip().upper(), "UNKNOWN")


def is_track_status_strictly_green(status_val: Any) -> bool:
    """
    Safely determine if a TrackStatus value represents strictly green-flag conditions.
    Missing, ambiguous, or combined status values are NEVER treated as green.

    In FastF1, TrackStatus can be a string composed of digits:
    '1' = Track Clear / Green Flag
    '2' = Yellow Flag
    '4' = Safety Car
    '5' = Red Flag
    '6' = Virtual Safety Car Deployed
    '7' = Virtual Safety Car Ending

    A lap is strictly green only if status is verified and strictly equals '1'.
    """
    if status_val is None or pd.isna(status_val):
        return False

    status_str = str(status_val).strip()
    if not status_str:
        return False

    # Strictly '1' means track was clear throughout the entire lap.
    # Any combined status like '12' (yellow occurred) or '14' (SC called) is NOT green.
    return status_str == "1"


def get_laps(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract and normalize lap timing, sectors, tyres, and clean-lap status.

    Returns DataFrame with columns:
    [race_id, lap_number, driver, team, lap_time_s, s1_s, s2_s, s3_s,
     compound, tyre_age_laps, stint, position, track_status, is_clean]
    """
    laps = session.laps.copy()
    if laps.empty:
        return pd.DataFrame()

    race_id = f"{session.event.year}_{session.event.EventName.lower().replace(' ', '_')}"

    # Extract seconds from Timedelta series safely
    def to_sec(td_series: pd.Series) -> pd.Series:
        return td_series.dt.total_seconds()

    laps_df = pd.DataFrame()
    laps_df["race_id"] = race_id
    laps_df["lap_number"] = laps["LapNumber"].astype(int)
    laps_df["driver"] = laps["Driver"].astype(str)
    laps_df["team"] = laps["Team"].astype(str)

    laps_df["lap_time_s"] = to_sec(laps["LapTime"])
    laps_df["s1_s"] = to_sec(laps["Sector1Time"])
    laps_df["s2_s"] = to_sec(laps["Sector2Time"])
    laps_df["s3_s"] = to_sec(laps["Sector3Time"])
    # Elapsed session time at lap completion
    laps_df["session_time_s"] = to_sec(laps["Time"])

    laps_df["compound"] = laps["Compound"].apply(normalize_compound)
    laps_df["tyre_age_laps"] = pd.to_numeric(laps["TyreLife"], errors="coerce").fillna(0).astype(int)
    laps_df["stint"] = pd.to_numeric(laps["Stint"], errors="coerce").fillna(1).astype(int)
    laps_df["position"] = pd.to_numeric(laps["Position"], errors="coerce")

    # Raw TrackStatus preservation as string
    laps_df["track_status"] = laps["TrackStatus"].fillna("").astype(str)

    # Clean-lap logic per PDF rules:
    # 1. LapTime must be valid and positive
    # 2. PitInTime must be null (not an in-lap)
    # 3. PitOutTime must be null (not an out-lap)
    # 4. Deleted must not be True (not deleted for track limits)
    # 5. TrackStatus must be verified and strictly green ('1')
    has_valid_time = laps_df["lap_time_s"].notna() & (laps_df["lap_time_s"] > 0)
    not_pit_in = laps["PitInTime"].isna()
    not_pit_out = laps["PitOutTime"].isna()

    if "Deleted" in laps.columns:
        not_deleted = ~laps["Deleted"].fillna(False).astype(bool)
    else:
        not_deleted = pd.Series(True, index=laps.index)

    strictly_green = laps_df["track_status"].apply(is_track_status_strictly_green)

    laps_df["is_clean"] = (
        has_valid_time & not_pit_in & not_pit_out & not_deleted & strictly_green
    )

    return laps_df.sort_values(by=["lap_number", "position"]).reset_index(drop=True)


def get_telemetry(session: fastf1.core.Session, driver: str, lap_number: int) -> pd.DataFrame:
    """
    Extract raw car telemetry for a specific driver and lap.

    Returns DataFrame with columns:
    [driver, lap_number, session_time_s, distance_m, speed_kmh, throttle_pct, brake_pct, gear, drs]
    """
    driver_laps = session.laps.pick_driver(driver)
    lap = driver_laps[driver_laps["LapNumber"] == lap_number]
    if lap.empty:
        return pd.DataFrame()

    try:
        tel = lap.iloc[0].get_telemetry()
    except Exception as exc:
        logger.warning(f"Could not load telemetry for {driver} lap {lap_number}: {exc}")
        return pd.DataFrame()

    if tel.empty:
        return pd.DataFrame()

    tel_df = pd.DataFrame()
    tel_df["driver"] = driver
    tel_df["lap_number"] = lap_number
    tel_df["session_time_s"] = tel["SessionTime"].dt.total_seconds()
    tel_df["distance_m"] = pd.to_numeric(tel["Distance"], errors="coerce")
    tel_df["speed_kmh"] = pd.to_numeric(tel["Speed"], errors="coerce")
    tel_df["throttle_pct"] = pd.to_numeric(tel["Throttle"], errors="coerce")
    tel_df["brake_pct"] = pd.to_numeric(tel["Brake"], errors="coerce")
    tel_df["gear"] = pd.to_numeric(tel["nGear"], errors="coerce").fillna(0).astype(int)

    # DRS state: 10, 12, 14 in FastF1 indicates DRS active/open
    if "DRS" in tel.columns:
        tel_df["drs"] = pd.to_numeric(tel["DRS"], errors="coerce").isin([10, 12, 14])
    else:
        tel_df["drs"] = False

    return tel_df.reset_index(drop=True)


def get_positions(session: fastf1.core.Session, driver: Optional[str] = None) -> pd.DataFrame:
    """
    Extract car coordinates (X, Y) resampled to 5 Hz (0.2s grid).
    Strictly preserves missing data: intervals where gap > 2.0s are NOT interpolated.

    Returns DataFrame with columns:
    [session_time_s, driver, x, y]
    """
    drivers_to_process = [driver] if driver else list(session.drivers)
    resampled_records: List[pd.DataFrame] = []

    dt_step = 1.0 / POSITION_RESAMPLE_HZ  # 0.2 seconds

    for drv_num in drivers_to_process:
        try:
            # Check pos_data first, fallback to car_data
            drv_info = session.get_driver(drv_num)
            drv_code = drv_info["Abbreviation"] if "Abbreviation" in drv_info else str(drv_num)

            raw_pos = session.pos_data.get(str(drv_num))
            if raw_pos is None or raw_pos.empty:
                raw_pos = session.pos_data.get(int(drv_num))

            if raw_pos is None or raw_pos.empty or "X" not in raw_pos.columns:
                continue

            pos_df = raw_pos[["SessionTime", "X", "Y"]].dropna().copy()
            if len(pos_df) < 2:
                continue

            pos_df["t"] = pos_df["SessionTime"].dt.total_seconds()
            pos_df = pos_df.sort_values(by="t").drop_duplicates(subset=["t"])

            t_min = pos_df["t"].min()
            t_max = pos_df["t"].max()

            # Align to 5 Hz uniform grid
            grid_times = np.arange(np.ceil(t_min * POSITION_RESAMPLE_HZ) / POSITION_RESAMPLE_HZ,
                                   t_max,
                                   dt_step)
            if len(grid_times) == 0:
                continue

            # Linear interpolation of coordinates
            orig_t = pos_df["t"].values
            orig_x = pos_df["X"].values
            orig_y = pos_df["Y"].values

            interp_x = np.interp(grid_times, orig_t, orig_x)
            interp_y = np.interp(grid_times, orig_t, orig_y)

            # Mask out intervals where consecutive source points had gap > 2.0s
            gap_indices = np.where(np.diff(orig_t) > POSITION_MAX_INTERPOLATION_GAP_S)[0]
            for gap_idx in gap_indices:
                t_gap_start = orig_t[gap_idx]
                t_gap_end = orig_t[gap_idx + 1]
                mask = (grid_times > t_gap_start) & (grid_times < t_gap_end)
                interp_x[mask] = np.nan
                interp_y[mask] = np.nan

            # Bound active running: do not extrapolate before race start or after retirement
            if hasattr(session, "laps") and not session.laps.empty:
                try:
                    driver_laps = session.laps.pick_driver(drv_code)
                    valid_laps = driver_laps.dropna(subset=["Time"])
                    if not valid_laps.empty:
                        # Start of active race running
                        t_start_active = (
                            valid_laps.iloc[0]["LapStartTime"].total_seconds()
                            if pd.notna(valid_laps.iloc[0]["LapStartTime"])
                            else t_min
                        )
                        # Check retirement status
                        drv_result = (
                            session.results[session.results["Abbreviation"] == drv_code]
                            if hasattr(session, "results") and not session.results.empty
                            else pd.DataFrame()
                        )
                        status_str = str(drv_result.iloc[0]["Status"]) if not drv_result.empty else "Finished"
                        if status_str != "Finished" and not status_str.startswith("+"):
                            t_end_active = valid_laps["Time"].max().total_seconds()
                        else:
                            t_end_active = t_max

                        mask_inactive = (grid_times < t_start_active) | (grid_times > t_end_active)
                        interp_x[mask_inactive] = np.nan
                        interp_y[mask_inactive] = np.nan
                except Exception as exc:
                    logger.debug(f"Could not bound active running for {drv_code}: {exc}")

            drv_resampled = pd.DataFrame({
                "session_time_s": np.round(grid_times, 3),
                "driver": drv_code,
                "x": interp_x,
                "y": interp_y,
            })
            resampled_records.append(drv_resampled)

        except Exception as exc:
            logger.warning(f"Failed to resample positions for driver {drv_num}: {exc}")
            continue

    if not resampled_records:
        return pd.DataFrame(columns=["session_time_s", "driver", "x", "y"])

    combined = pd.concat(resampled_records, ignore_index=True)
    return combined.sort_values(by=["session_time_s", "driver"]).reset_index(drop=True)


def get_pits_stints(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract tyre stint history and pit stop events per driver.

    Returns DataFrame with columns:
    [driver, stint, compound, start_lap, end_lap, lap_count, pit_in_lap, pit_stop_duration_s]
    """
    laps = session.laps
    if laps.empty:
        return pd.DataFrame()

    stint_records = []

    for drv, drv_laps in laps.groupby("Driver"):
        drv_laps = drv_laps.sort_values("LapNumber")
        for stint_num, stint_laps in drv_laps.groupby("Stint"):
            if stint_laps.empty:
                continue

            compound = normalize_compound(stint_laps["Compound"].iloc[0])
            start_lap = int(stint_laps["LapNumber"].min())
            end_lap = int(stint_laps["LapNumber"].max())
            lap_count = len(stint_laps)

            # Check if an in-lap occurred at the end of this stint
            pit_in_rows = stint_laps[stint_laps["PitInTime"].notna()]
            if not pit_in_rows.empty:
                pit_in_lap = int(pit_in_rows.iloc[-1]["LapNumber"])
            else:
                pit_in_lap = None

            # Estimate pit stop duration if pit stop times exist
            pit_duration = None
            if pit_in_lap is not None:
                # Find the corresponding out-lap in the next stint
                next_stint_laps = drv_laps[drv_laps["Stint"] == stint_num + 1]
                if not next_stint_laps.empty and next_stint_laps.iloc[0]["PitOutTime"] is not pd.NaT:
                    pit_in_time = pit_in_rows.iloc[-1]["PitInTime"]
                    pit_out_time = next_stint_laps.iloc[0]["PitOutTime"]
                    if pd.notna(pit_in_time) and pd.notna(pit_out_time):
                        pit_duration = (pit_out_time - pit_in_time).total_seconds()

            stint_records.append({
                "driver": str(drv),
                "stint": int(stint_num),
                "compound": compound,
                "start_lap": start_lap,
                "end_lap": end_lap,
                "lap_count": lap_count,
                "pit_in_lap": pit_in_lap,
                "pit_stop_duration_s": pit_duration,
            })

    stints_df = pd.DataFrame(stint_records)
    if stints_df.empty:
        return pd.DataFrame()
    return stints_df.sort_values(by=["driver", "stint"]).reset_index(drop=True)


def get_weather_track_status(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract weather and track status conditions along session elapsed time.

    Returns DataFrame with columns:
    [session_time_s, air_temp_c, track_temp_c, humidity_pct, rainfall, track_status]
    """
    weather = session.weather_data.copy()
    if weather.empty:
        return pd.DataFrame(columns=["session_time_s", "air_temp_c", "track_temp_c", "humidity_pct", "rainfall", "track_status"])

    weather_df = pd.DataFrame()
    weather_df["session_time_s"] = weather["Time"].dt.total_seconds().round(1)
    weather_df["air_temp_c"] = pd.to_numeric(weather["AirTemp"], errors="coerce")
    weather_df["track_temp_c"] = pd.to_numeric(weather["TrackTemp"], errors="coerce")
    weather_df["humidity_pct"] = pd.to_numeric(weather["Humidity"], errors="coerce")
    weather_df["rainfall"] = weather["Rainfall"].fillna(False).astype(bool)

    # Associate nearest TrackStatus from session.track_status if available
    if hasattr(session, "track_status") and not session.track_status.empty:
        ts = session.track_status.copy()
        ts["t"] = ts["Time"].dt.total_seconds()
        # Merge as-of backward
        weather_df = pd.merge_asof(
            weather_df.sort_values("session_time_s"),
            ts[["t", "Status"]].rename(columns={"Status": "track_status"}),
            left_on="session_time_s",
            right_on="t",
            direction="backward",
        )
        weather_df["track_status"] = weather_df["track_status"].fillna("1").astype(str)
        weather_df = weather_df.drop(columns=["t"])
    else:
        weather_df["track_status"] = "1"

    return weather_df.reset_index(drop=True)


def get_official_classification(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Extract the official classification results from FastF1 session.results.
    Used for golden checks and official standings rather than inferring from laps.

    Returns DataFrame with columns:
    [position, classified_position, driver, driver_number, team, status, points, grid_position]
    """
    results = session.results.copy()
    if results.empty:
        return pd.DataFrame()

    res_df = pd.DataFrame()
    res_df["position"] = pd.to_numeric(results["Position"], errors="coerce")
    res_df["classified_position"] = results["ClassifiedPosition"].astype(str)
    res_df["driver"] = results["Abbreviation"].astype(str)
    res_df["driver_number"] = results["DriverNumber"].astype(str)
    res_df["team"] = results["TeamName"].astype(str)
    res_df["status"] = results["Status"].astype(str)
    res_df["points"] = pd.to_numeric(results["Points"], errors="coerce").fillna(0.0)
    res_df["grid_position"] = pd.to_numeric(results["GridPosition"], errors="coerce")

    return res_df.sort_values(by="position", na_position="last").reset_index(drop=True)
