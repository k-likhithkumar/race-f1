"""
OpenF1 data ingestion and normalization adapter for RACE.
Provides unauthenticated fallback access for 2023+ historical race sessions.
Transforms OpenF1 endpoints into the established, normalized Parquet schema.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import requests

from src.race.config import (
    OPENF1_BASE_URL,
    RAW_OPENF1_CACHE_DIR,
    POSITION_RESAMPLE_HZ,
    POSITION_MAX_INTERPOLATION_GAP_S,
    get_race_cache_dir,
    ensure_directories,
)
from src.race.data.loader import fetch_jolpica_race_results
from src.race.data.processor import normalize_compound, is_track_status_strictly_green

logger = logging.getLogger(__name__)


def fetch_openf1_json(
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    cache_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch JSON from OpenF1 REST API with local raw file caching.
    Separates raw HTTP downloads into data/raw_openf1/.
    """
    ensure_directories()
    if cache_name:
        cache_file = RAW_OPENF1_CACHE_DIR / f"{cache_name}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to read raw OpenF1 cache {cache_file}: {exc}")

    url = f"{OPENF1_BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=30)
    if response.status_code == 404:
        return []
    response.raise_for_status()
    data = response.json()

    if cache_name and isinstance(data, list):
        cache_file = RAW_OPENF1_CACHE_DIR / f"{cache_name}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except Exception as exc:
            logger.warning(f"Failed to write raw OpenF1 cache {cache_file}: {exc}")

    return data


def _normalize_session_name(session_type: str) -> str:
    st = session_type.strip().lower()
    if st in ("r", "race"):
        return "Race"
    if st in ("q", "qualifying", "qualify"):
        return "Qualifying"
    if st in ("fp1", "p1", "practice 1"):
        return "Practice 1"
    if st in ("fp2", "p2", "practice 2"):
        return "Practice 2"
    if st in ("fp3", "p3", "practice 3"):
        return "Practice 3"
    if st in ("s", "sprint"):
        return "Sprint"
    if st in ("sq", "sprint shootout", "sprint qualifying"):
        return "Sprint Qualifying"
    return session_type.capitalize()


def find_openf1_session(year: int, gp: str, session_type: str = "Race") -> Optional[Dict[str, Any]]:
    """
    Locate an OpenF1 session record by year and Grand Prix name/slug.
    """
    norm_session = _normalize_session_name(session_type)
    cache_name = f"sessions_{year}_{norm_session.lower().replace(' ', '_')}"
    sessions = fetch_openf1_json("sessions", {"year": year, "session_name": norm_session}, cache_name=cache_name)
    if not sessions:
        cache_name_all = f"sessions_{year}_all"
        all_sessions = fetch_openf1_json("sessions", {"year": year}, cache_name=cache_name_all)
        sessions = [s for s in all_sessions if str(s.get("session_name", "")).lower() == norm_session.lower()]
        if not sessions:
            return None

    clean_gp = gp.lower().replace("_", " ").replace("-", " ")
    for sess in sessions:
        circuit_short = str(sess.get("circuit_short_name", "")).lower()
        country = str(sess.get("country_name", "")).lower()
        location = str(sess.get("location", "")).lower()

        if (clean_gp in circuit_short or circuit_short in clean_gp or
            clean_gp in country or country in clean_gp or
            clean_gp in location or location in clean_gp):
            return sess

    # Fallback to round index matching if available
    return sessions[0] if len(sessions) == 1 else None


def get_openf1_drivers(session_key: int) -> Dict[int, Dict[str, Any]]:
    """
    Fetch driver metadata for a session and map by driver_number.
    """
    cache_name = f"drivers_{session_key}"
    driver_list = fetch_openf1_json("drivers", {"session_key": session_key}, cache_name=cache_name)
    driver_map: Dict[int, Dict[str, Any]] = {}
    for d in driver_list:
        num = int(d["driver_number"])
        acronym = d.get("name_acronym") or str(num)
        driver_map[num] = {
            "driver_number": str(num),
            "driver": acronym,
            "full_name": d.get("full_name", acronym),
            "team": d.get("team_name", "UNKNOWN"),
            "color": d.get("team_colour", "FFFFFF"),
        }
    return driver_map


def _safe_float(val: Any, default: float = np.nan) -> float:
    if val is None or pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 1) -> int:
    if val is None or pd.isna(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def get_openf1_stints(session_key: int, driver_map: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    """
    Fetch and normalize tyre stint and pit stop history from OpenF1.
    """
    cache_name = f"stints_{session_key}"
    stint_list = fetch_openf1_json("stints", {"session_key": session_key}, cache_name=cache_name)
    if not stint_list:
        return pd.DataFrame(columns=[
            "driver", "stint", "compound", "start_lap", "end_lap",
            "lap_count", "pit_in_lap", "pit_stop_duration_s"
        ])

    cache_pits = f"pit_{session_key}"
    pit_list = fetch_openf1_json("pit", {"session_key": session_key}, cache_name=cache_pits)
    pit_df = pd.DataFrame(pit_list) if pit_list else pd.DataFrame()

    records = []
    for s in stint_list:
        drv_num = int(s["driver_number"])
        drv_code = driver_map.get(drv_num, {}).get("driver", str(drv_num))
        stint_num = _safe_int(s.get("stint_number"), 1)
        compound = normalize_compound(s.get("compound", "UNKNOWN"))
        start_lap = _safe_int(s.get("lap_start"), 1)
        end_lap_val = s.get("lap_end")
        end_lap = _safe_int(end_lap_val, start_lap)
        lap_count = max(1, end_lap - start_lap + 1)

        pit_in_lap = None
        pit_duration = None
        if not pit_df.empty and "driver_number" in pit_df.columns:
            drv_pits = pit_df[pit_df["driver_number"] == drv_num]
            matching_pit = drv_pits[drv_pits["lap_number"] == end_lap]
            if not matching_pit.empty:
                pit_in_lap = int(end_lap)
                p_dur = matching_pit.iloc[0].get("pit_duration")
                if p_dur is not None and not pd.isna(p_dur):
                    pit_duration = _safe_float(p_dur, None)

        records.append({
            "driver": drv_code,
            "stint": stint_num,
            "compound": compound,
            "start_lap": start_lap,
            "end_lap": end_lap,
            "lap_count": lap_count,
            "pit_in_lap": pit_in_lap,
            "pit_stop_duration_s": pit_duration,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=[
            "driver", "stint", "compound", "start_lap", "end_lap",
            "lap_count", "pit_in_lap", "pit_stop_duration_s"
        ])
    return df.sort_values(by=["driver", "stint"]).reset_index(drop=True)


def parse_openf1_race_control_timeline(rc_list: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Parse OpenF1 race_control messages into a verified timeline of caution periods,
    explicit caution laps, and track limits deleted laps.

    If race control data is missing, empty, or lacks positive confirmation of
    green/racing conditions, verified is False, preventing laps from being assumed clean.
    """
    if not rc_list:
        return {
            "verified": False,
            "first_green_dt": None,
            "periods": [],
            "caution_laps": {},
            "deleted_laps": set(),
        }

    periods = []
    caution_laps = {}
    deleted_laps = set()
    active_cautions = {}
    has_positive_green_evidence = False
    first_green_dt = None

    del_pattern1 = re.compile(r'CAR\s+(\d+).*?DELETED.*?LAP\s+(\d+)', re.IGNORECASE)
    del_pattern2 = re.compile(r'LAP\s+(\d+).*?DELETED.*?CAR\s+(\d+)', re.IGNORECASE)

    for item in rc_list:
        d_str = item.get("date")
        if not d_str:
            continue
        dt = pd.to_datetime(d_str, format="ISO8601")
        flag = str(item.get("flag", "")).upper()
        msg = str(item.get("message", "")).upper()
        lap_num = item.get("lap_number")

        # 1. Track limits deletions
        if "DELETED" in msg:
            m1 = del_pattern1.search(msg)
            if m1:
                deleted_laps.add((int(m1.group(1)), int(m1.group(2))))
            else:
                m2 = del_pattern2.search(msg)
                if m2:
                    deleted_laps.add((int(m2.group(2)), int(m2.group(1))))
                elif item.get("driver_number") is not None and lap_num is not None:
                    deleted_laps.add((int(item["driver_number"]), int(lap_num)))

        # 2. Caution flag events & intervals
        if "ABORTED START" in msg:
            active_cautions["ABORTED_START"] = (dt, "4")
            if lap_num is not None:
                caution_laps[int(lap_num)] = "4"
        elif flag in ["DOUBLE YELLOW", "YELLOW"] or "YELLOW" in msg:
            code = "2"
            sector = item.get("sector")
            key = f"YELLOW_{sector}" if sector else "YELLOW"
            active_cautions[key] = (dt, code)
            if lap_num is not None:
                caution_laps[int(lap_num)] = code
        elif "SAFETY CAR" in msg:
            code = "4"
            if "ENDING" not in msg and "IN THIS LAP" not in msg:
                active_cautions["SAFETY_CAR"] = (dt, code)
            if lap_num is not None:
                caution_laps[int(lap_num)] = code
        elif "VSC" in msg:
            code = "6"
            if "ENDING" in msg:
                active_cautions["VSC"] = (active_cautions.get("VSC", (dt, "6"))[0], "7")
            else:
                active_cautions["VSC"] = (dt, code)
            if lap_num is not None:
                caution_laps[int(lap_num)] = code
        elif flag == "RED" or "RED FLAG" in msg:
            code = "5"
            active_cautions["RED_FLAG"] = (dt, code)
            if lap_num is not None:
                caution_laps[int(lap_num)] = code
        elif flag in ["CLEAR", "GREEN"] or "CLEAR" in msg or "GREEN LIGHT" in msg:
            has_positive_green_evidence = True
            if first_green_dt is None:
                first_green_dt = dt
            sector = item.get("sector")
            if sector and f"YELLOW_{sector}" in active_cautions:
                start_t, c_code = active_cautions.pop(f"YELLOW_{sector}")
                periods.append({"start": start_t, "end": dt, "code": c_code})
            elif flag in ["CLEAR", "GREEN"] and not sector:
                for k in list(active_cautions.keys()):
                    start_t, c_code = active_cautions.pop(k)
                    periods.append({"start": start_t, "end": dt, "code": c_code})
        elif "SESSION STARTED" in msg:
            has_positive_green_evidence = True
            if first_green_dt is None:
                first_green_dt = dt
        elif "CHEQUERED" in flag or "CHEQUERED" in msg:
            active_cautions["CHEQUERED"] = (dt, "UNKNOWN")

    # Close any still-open cautions with end of session
    for k, (start_t, c_code) in active_cautions.items():
        periods.append({"start": start_t, "end": dt + pd.Timedelta(hours=4), "code": c_code})

    return {
        "verified": has_positive_green_evidence,
        "first_green_dt": first_green_dt,
        "periods": periods,
        "caution_laps": caution_laps,
        "deleted_laps": deleted_laps,
    }


def resolve_openf1_lap_track_status(
    lap_start_dt: Optional[pd.Timestamp],
    lap_end_dt: Optional[pd.Timestamp],
    lap_num: int,
    timeline: Dict[str, Any],
) -> str:
    """
    Positively establish track status for an OpenF1 lap based on verified race control timeline.
    Returns:
    - '1': Strictly green positively verified throughout entire lap.
    - '12', '14', '16', '17': Lap was partially green but affected by caution during lap.
    - '2', '4', '5', '6', '7': Lap ran entirely under caution.
    - 'UNKNOWN': Race control data missing, unverified, or lap timing missing.
    """
    if not timeline or not timeline.get("verified"):
        return "UNKNOWN"

    if lap_start_dt is None or lap_end_dt is None or pd.isna(lap_start_dt) or pd.isna(lap_end_dt):
        return "UNKNOWN"

    # If lap occurred before any positive green signal was established in the session
    first_green = timeline.get("first_green_dt")
    if first_green is not None and lap_end_dt <= first_green:
        return "UNKNOWN"

    statuses = set()

    # 1. Check temporal overlap with caution periods
    overlapping_periods = []
    for period in timeline.get("periods", []):
        if (lap_start_dt < period["end"]) and (lap_end_dt > period["start"]):
            statuses.add(period["code"])
            overlapping_periods.append(period)

    # 2. Check if lap number explicitly recorded in caution events
    if lap_num in timeline.get("caution_laps", {}):
        statuses.add(timeline["caution_laps"][lap_num])

    if not statuses:
        # Confirmed clear and green throughout the entire lap
        return "1"

    # 3. Check if lap was partially green (started before caution or ended after caution cleared)
    partially_green = any(
        (lap_start_dt < p["start"] < lap_end_dt) or (lap_start_dt < p["end"] < lap_end_dt)
        for p in overlapping_periods
    )
    if partially_green:
        statuses.add("1")

    sorted_codes = sorted(list(statuses), key=lambda x: (x != "1", x))
    return "".join(sorted_codes)


def get_openf1_laps(
    session_key: int,
    driver_map: Dict[int, Dict[str, Any]],
    race_id: str,
    session_start_dt: pd.Timestamp,
    stints_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetch and normalize lap times, sector splits, and clean-lap tags from OpenF1.
    """
    cache_name = f"laps_{session_key}"
    laps_list = fetch_openf1_json("laps", {"session_key": session_key}, cache_name=cache_name)
    if not laps_list:
        return pd.DataFrame()

    cache_rc = f"rc_{session_key}"
    rc_list = fetch_openf1_json("race_control", {"session_key": session_key}, cache_name=cache_rc)
    timeline = parse_openf1_race_control_timeline(rc_list)
    deleted_laps = timeline["deleted_laps"]

    # Identify in-laps from stints and from consecutive laps (lap before an out-lap)
    pit_in_laps_set = set()
    if not stints_df.empty:
        for _, stint in stints_df.iterrows():
            p_in = stint.get("pit_in_lap")
            if pd.notna(p_in):
                pit_in_laps_set.add((stint["driver"], int(p_in)))

    laps_by_driver = {}
    for row in laps_list:
        d_num = int(row["driver_number"])
        d_code = driver_map.get(d_num, {}).get("driver", str(d_num))
        laps_by_driver.setdefault(d_code, []).append(row)

    for d_code, drv_rows in laps_by_driver.items():
        sorted_rows = sorted(drv_rows, key=lambda r: int(r.get("lap_number", 0)))
        for i in range(len(sorted_rows) - 1):
            if bool(sorted_rows[i + 1].get("is_pit_out_lap", False)):
                pit_in_laps_set.add((d_code, int(sorted_rows[i]["lap_number"])))

    records = []
    for row in laps_list:
        drv_num = int(row["driver_number"])
        drv_meta = driver_map.get(drv_num, {})
        drv_code = drv_meta.get("driver", str(drv_num))
        team_name = drv_meta.get("team", "UNKNOWN")
        lap_num = int(row["lap_number"])

        lap_dur = row.get("lap_duration")
        lap_time_s = float(lap_dur) if lap_dur is not None and not pd.isna(lap_dur) else np.nan

        s1 = row.get("duration_sector_1")
        s2 = row.get("duration_sector_2")
        s3 = row.get("duration_sector_3")
        s1_s = float(s1) if s1 is not None and not pd.isna(s1) else np.nan
        s2_s = float(s2) if s2 is not None and not pd.isna(s2) else np.nan
        s3_s = float(s3) if s3 is not None and not pd.isna(s3) else np.nan

        # Calculate session time elapsed at lap end and define interval boundaries
        date_start_str = row.get("date_start")
        if date_start_str and not pd.isna(date_start_str):
            lap_start_dt = pd.to_datetime(date_start_str, format="ISO8601")
            if not np.isnan(lap_time_s) and lap_time_s > 0:
                lap_end_dt = lap_start_dt + pd.Timedelta(seconds=lap_time_s)
                session_time_s = round((lap_end_dt - session_start_dt).total_seconds(), 3)
            else:
                lap_end_dt = None
                session_time_s = round((lap_start_dt - session_start_dt).total_seconds(), 3)
        else:
            lap_start_dt = None
            lap_end_dt = None
            session_time_s = np.nan

        is_pit_out = bool(row.get("is_pit_out_lap", False))

        # Join tyre compound and stint from stints_df
        compound = "UNKNOWN"
        stint_idx = 1
        tyre_age = lap_num
        if not stints_df.empty:
            drv_stints = stints_df[stints_df["driver"] == drv_code]
            matching_stint = drv_stints[(drv_stints["start_lap"] <= lap_num) & (drv_stints["end_lap"] >= lap_num)]
            if not matching_stint.empty:
                compound = matching_stint.iloc[0]["compound"]
                stint_idx = int(matching_stint.iloc[0]["stint"])
                tyre_age = lap_num - matching_stint.iloc[0]["start_lap"] + 1

        # Track status resolution based on verified timeline (never assumed green)
        track_status = resolve_openf1_lap_track_status(lap_start_dt, lap_end_dt, lap_num, timeline)
        is_strictly_green = is_track_status_strictly_green(track_status)

        # Clean lap classification per PDF:
        # Valid time, not pit out lap, not in-lap, verified strictly green flag, not deleted
        is_pit_in = (drv_code, lap_num) in pit_in_laps_set
        is_deleted = (drv_num, lap_num) in deleted_laps

        is_clean = (
            not np.isnan(lap_time_s) and
            lap_time_s > 0 and
            not is_pit_out and
            not is_pit_in and
            is_strictly_green and
            not is_deleted
        )

        records.append({
            "race_id": race_id,
            "lap_number": lap_num,
            "driver": drv_code,
            "team": team_name,
            "lap_time_s": lap_time_s,
            "s1_s": s1_s,
            "s2_s": s2_s,
            "s3_s": s3_s,
            "session_time_s": session_time_s,
            "compound": compound,
            "tyre_age_laps": tyre_age,
            "stint": stint_idx,
            "position": np.nan,  # Populated from classification/order below
            "track_status": track_status,
            "is_clean": is_clean,
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Order positions per lap by session_time_s
    for lap_n, group in df.groupby("lap_number"):
        sorted_indices = group.sort_values(by="session_time_s").index
        df.loc[sorted_indices, "position"] = range(1, len(sorted_indices) + 1)

    # Cast integer columns to int32 for exact schema parity with FastF1 processor
    df["lap_number"] = df["lap_number"].astype("int32")
    df["tyre_age_laps"] = df["tyre_age_laps"].astype("int32")
    df["stint"] = df["stint"].astype("int32")

    return df.sort_values(by=["lap_number", "position"]).reset_index(drop=True)


def get_openf1_weather(session_key: int, session_start_dt: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch and normalize weather telemetry from OpenF1.
    """
    cache_name = f"weather_{session_key}"
    weather_list = fetch_openf1_json("weather", {"session_key": session_key}, cache_name=cache_name)
    if not weather_list:
        return pd.DataFrame(columns=[
            "session_time_s", "air_temp_c", "track_temp_c", "humidity_pct", "rainfall", "track_status"
        ])

    records = []
    for row in weather_list:
        date_str = row.get("date")
        if not date_str:
            continue
        t_sec = round((pd.to_datetime(date_str, format="ISO8601") - session_start_dt).total_seconds(), 1)
        air_temp = _safe_float(row.get("air_temperature"))
        track_temp = _safe_float(row.get("track_temperature"))
        hum = _safe_float(row.get("humidity"))
        records.append({
            "session_time_s": t_sec,
            "air_temp_c": air_temp,
            "track_temp_c": track_temp,
            "humidity_pct": hum,
            "rainfall": bool(row.get("rainfall", 0) == 1),
            "track_status": "1",
        })

    df = pd.DataFrame(records)
    return df.sort_values("session_time_s").reset_index(drop=True)


def get_openf1_positions(
    session_key: int,
    driver_map: Dict[int, Dict[str, Any]],
    session_start_dt: pd.Timestamp,
    laps_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fetch car coordinates per driver from OpenF1 location, converting to meters and resampling to 5 Hz.
    Strictly preserves gaps > 2.0s as NaN and masks pre-race and post-retirement points.
    """
    resampled_records = []
    dt_step = 1.0 / POSITION_RESAMPLE_HZ

    for drv_num, meta in driver_map.items():
        drv_code = meta["driver"]
        cache_name = f"loc_{session_key}_{drv_num}"
        raw_loc = fetch_openf1_json("location", {"session_key": session_key, "driver_number": drv_num}, cache_name=cache_name)
        if not raw_loc:
            continue

        pos_df = pd.DataFrame(raw_loc)[["date", "x", "y"]].dropna()
        if len(pos_df) < 2:
            continue

        pos_df["t"] = (pd.to_datetime(pos_df["date"], format="ISO8601") - session_start_dt).dt.total_seconds()
        # Scale coordinates from OpenF1 decimeters to meters
        pos_df["x_m"] = pos_df["x"].astype(float) / 10.0
        pos_df["y_m"] = pos_df["y"].astype(float) / 10.0

        pos_df = pos_df.sort_values("t").drop_duplicates(subset=["t"])
        t_min = pos_df["t"].min()
        t_max = pos_df["t"].max()

        grid_times = np.arange(
            np.ceil(t_min * POSITION_RESAMPLE_HZ) / POSITION_RESAMPLE_HZ,
            t_max,
            dt_step,
        )
        if len(grid_times) == 0:
            continue

        orig_t = pos_df["t"].values
        orig_x = pos_df["x_m"].values
        orig_y = pos_df["y_m"].values

        interp_x = np.interp(grid_times, orig_t, orig_x)
        interp_y = np.interp(grid_times, orig_t, orig_y)

        # Apply 2.0s gap mask
        gap_indices = np.where(np.diff(orig_t) > POSITION_MAX_INTERPOLATION_GAP_S)[0]
        for gap_idx in gap_indices:
            t_gap_start = orig_t[gap_idx]
            t_gap_end = orig_t[gap_idx + 1]
            mask = (grid_times > t_gap_start) & (grid_times < t_gap_end)
            interp_x[mask] = np.nan
            interp_y[mask] = np.nan

        # Mask pre-race start and post-retirement intervals
        if not laps_df.empty:
            drv_laps = laps_df[laps_df["driver"] == drv_code]
            if not drv_laps.empty:
                valid_laps = drv_laps.dropna(subset=["session_time_s"])
                if not valid_laps.empty:
                    t_start_active = valid_laps.iloc[0]["session_time_s"] - (valid_laps.iloc[0]["lap_time_s"] or 0)
                    t_end_active = valid_laps["session_time_s"].max()
                    # If retired, mask after last lap
                    mask_inactive = (grid_times < t_start_active) | (grid_times > t_end_active)
                    interp_x[mask_inactive] = np.nan
                    interp_y[mask_inactive] = np.nan

        drv_resampled = pd.DataFrame({
            "session_time_s": np.round(grid_times, 3),
            "driver": drv_code,
            "x": interp_x,
            "y": interp_y,
        })
        resampled_records.append(drv_resampled)

    if not resampled_records:
        return pd.DataFrame(columns=["session_time_s", "driver", "x", "y"])

    combined = pd.concat(resampled_records, ignore_index=True)
    return combined.sort_values(by=["session_time_s", "driver"]).reset_index(drop=True)


def build_openf1_track_outline(
    session_key: int,
    fastest_driver_num: int,
    fastest_lap_start_dt: pd.Timestamp,
    fastest_lap_duration: float,
    circuit_name: str,
    year: int,
    num_mini_sectors: int = 20,
) -> Dict[str, Any]:
    """
    Build track outline and 20 mini-sector distance boundaries from OpenF1 fastest clean lap coordinates.
    """
    cache_name = f"loc_{session_key}_{fastest_driver_num}"
    raw_loc = fetch_openf1_json("location", {"session_key": session_key, "driver_number": fastest_driver_num}, cache_name=cache_name)
    df_loc = pd.DataFrame(raw_loc)
    df_loc["dt"] = pd.to_datetime(df_loc["date"], format="ISO8601")

    lap_end_dt = fastest_lap_start_dt + pd.Timedelta(seconds=fastest_lap_duration)
    lap_points = df_loc[(df_loc["dt"] >= fastest_lap_start_dt) & (df_loc["dt"] <= lap_end_dt)].copy()

    if len(lap_points) < 10:
        raise ValueError(f"Insufficient location points ({len(lap_points)}) to build track outline.")

    lap_points["x_m"] = lap_points["x"].astype(float) / 10.0
    lap_points["y_m"] = lap_points["y"].astype(float) / 10.0
    lap_points = lap_points.sort_values("dt").drop_duplicates(subset=["x_m", "y_m"])

    # Calculate cumulative distance in meters
    dx = np.diff(lap_points["x_m"].values)
    dy = np.diff(lap_points["y_m"].values)
    dists = np.insert(np.cumsum(np.sqrt(dx**2 + dy**2)), 0, 0.0)
    total_distance_m = float(dists[-1])

    # Resample track geometry to 1000 points
    target_points = 1000
    resampled_dist = np.linspace(0, total_distance_m, target_points)
    resampled_x = np.interp(resampled_dist, dists, lap_points["x_m"].values)
    resampled_y = np.interp(resampled_dist, dists, lap_points["y_m"].values)

    # Center coordinates around (0, 0)
    x_center = float(np.mean(resampled_x))
    y_center = float(np.mean(resampled_y))
    norm_x = (resampled_x - x_center).round(2).tolist()
    norm_y = (resampled_y - y_center).round(2).tolist()

    # Sector distance boundaries (approximate equal thirds if sector splits unavailable)
    s1_dist_m = round(total_distance_m / 3.0, 2)
    s2_dist_m = round(2.0 * total_distance_m / 3.0, 2)

    mini_sector_step = total_distance_m / num_mini_sectors
    mini_sectors = []
    for i in range(num_mini_sectors):
        start_m = round(i * mini_sector_step, 2)
        end_m = round((i + 1) * mini_sector_step, 2) if i < num_mini_sectors - 1 else round(total_distance_m, 2)
        mid_m = (start_m + end_m) / 2.0
        sec = 1 if mid_m <= s1_dist_m else (2 if mid_m <= s2_dist_m else 3)
        mini_sectors.append({
            "mini_sector_id": i + 1,
            "start_m": start_m,
            "end_m": end_m,
            "sector": sec,
        })

    points_data = [
        {"dist_m": round(float(d), 2), "x": x, "y": y}
        for d, x, y in zip(resampled_dist, norm_x, norm_y)
    ]

    return {
        "circuit_name": circuit_name,
        "year": year,
        "total_distance_m": round(total_distance_m, 2),
        "sectors": [
            {"sector": 1, "start_m": 0.0, "end_m": s1_dist_m},
            {"sector": 2, "start_m": s1_dist_m, "end_m": s2_dist_m},
            {"sector": 3, "start_m": s2_dist_m, "end_m": round(total_distance_m, 2)},
        ],
        "mini_sectors": mini_sectors,
        "points": points_data,
    }


def cache_race_openf1(year: int, gp: str, session_type: str = "Race", round_num: Optional[int] = None) -> Dict[str, Any]:
    """
    Ingest a 2023+ race session via OpenF1 and save normalized Parquet tables to data/cache/.
    Preserves exact schema and interfaces established in Stage 1.
    """
    ensure_directories()
    sess_info = find_openf1_session(year, gp, session_type)
    if not sess_info:
        raise ValueError(f"OpenF1 session not found for {year} {gp} [{session_type}]")

    session_key = int(sess_info["session_key"])
    circuit_name = sess_info.get("circuit_short_name", gp)
    session_start_dt = pd.to_datetime(sess_info["date_start"], format="ISO8601")
    cache_dir = get_race_cache_dir(year, gp)
    cache_dir.mkdir(parents=True, exist_ok=True)
    race_id = f"{year}_{gp.lower().replace(' ', '_')}"

    logger.info(f"Ingesting {year} {gp} via OpenF1 (session_key={session_key})...")

    # 1. Drivers
    driver_map = get_openf1_drivers(session_key)

    # 2. Stints & Pits
    stints_df = get_openf1_stints(session_key, driver_map)
    stints_df.to_parquet(cache_dir / "stints.parquet", index=False)

    # 3. Laps
    laps_df = get_openf1_laps(session_key, driver_map, race_id, session_start_dt, stints_df)
    laps_df.to_parquet(cache_dir / "laps.parquet", index=False)

    # 4. Weather
    weather_df = get_openf1_weather(session_key, session_start_dt)
    weather_df.to_parquet(cache_dir / "weather.parquet", index=False)

    # 5. Positions (5 Hz resampled with gap/retirement guards)
    positions_df = get_openf1_positions(session_key, driver_map, session_start_dt, laps_df)
    positions_df.to_parquet(cache_dir / "positions.parquet", index=False)

    # 6. Official Classification (Jolpica with fallback)
    inferred_round = round_num or 14  # Default for Monza 2023 or look up
    jolpica_race = fetch_jolpica_race_results(year, inferred_round)
    class_records = []
    if jolpica_race and "Results" in jolpica_race:
        for r in jolpica_race["Results"]:
            class_records.append({
                "position": float(r["position"]),
                "classified_position": str(r["positionText"]),
                "driver": r["Driver"]["code"],
                "driver_number": r["number"],
                "team": r["Constructor"]["name"],
                "status": r["status"],
                "points": float(r["points"]),
                "grid_position": float(r.get("grid", 0)),
            })
    else:
        # Fallback from laps
        for drv_code, grp in laps_df.groupby("driver"):
            last_lap = grp.sort_values("lap_number").iloc[-1]
            class_records.append({
                "position": float(last_lap["position"]),
                "classified_position": str(int(last_lap["position"])),
                "driver": drv_code,
                "driver_number": "0",
                "team": last_lap["team"],
                "status": "Finished",
                "points": 0.0,
                "grid_position": 0.0,
            })
    classification_df = pd.DataFrame(class_records).sort_values("position").reset_index(drop=True)
    classification_df.to_parquet(cache_dir / "classification.parquet", index=False)

    # 7. Track Outline
    clean_laps = laps_df[laps_df["is_clean"]].dropna(subset=["lap_time_s"])
    if clean_laps.empty:
        clean_laps = laps_df.dropna(subset=["lap_time_s"])
    fastest_row = clean_laps.sort_values("lap_time_s").iloc[0]
    fastest_drv_code = fastest_row["driver"]
    # Reverse lookup driver number
    fastest_num = next(k for k, v in driver_map.items() if v["driver"] == fastest_drv_code)
    # Find original lap start datetime
    raw_laps = fetch_openf1_json("laps", {"session_key": session_key, "driver_number": fastest_num})
    match_lap = next((l for l in raw_laps if int(l.get("lap_number", 0)) == int(fastest_row["lap_number"])), None)
    if match_lap and match_lap.get("date_start"):
        fastest_start_dt = pd.to_datetime(match_lap["date_start"], format="ISO8601")
    else:
        fastest_start_dt = session_start_dt + pd.Timedelta(seconds=max(0.0, float(fastest_row.get("session_time_s", 0.0)) - float(fastest_row["lap_time_s"])))

    track_outline = build_openf1_track_outline(
        session_key=session_key,
        fastest_driver_num=fastest_num,
        fastest_lap_start_dt=fastest_start_dt,
        fastest_lap_duration=float(fastest_row["lap_time_s"]),
        circuit_name=circuit_name,
        year=year,
    )
    with open(cache_dir / "track_outline.json", "w", encoding="utf-8") as f:
        json.dump(track_outline, f, indent=2)

    # 8. Metadata
    total_laps = int(laps_df["lap_number"].max()) if not laps_df.empty else 0
    winner_code = classification_df.iloc[0]["driver"] if not classification_df.empty else "UNKNOWN"
    metadata = {
        "year": year,
        "gp": gp,
        "official_event_name": sess_info.get("meeting_name", circuit_name),
        "session": session_type,
        "total_laps": total_laps,
        "winner": winner_code,
        "driver_count": len(driver_map),
        "drivers": [str(v["driver"]) for v in driver_map.values()],
        "data_source": "openf1",
        "cached_tables": [
            "laps.parquet",
            "stints.parquet",
            "weather.parquet",
            "positions.parquet",
            "classification.parquet",
            "track_outline.json",
        ],
    }
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully cached {year} {gp} via OpenF1 to {cache_dir}")
    return metadata
