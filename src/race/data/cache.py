"""
Parquet caching and offline data access module for RACE.
Saves processed tables to disk and enforces the strict as_of replay principle.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.race.config import get_race_cache_dir, ensure_directories
from src.race.data.loader import load_session
from src.race.data.processor import (
    get_laps,
    get_positions,
    get_pits_stints,
    get_weather_track_status,
    get_official_classification,
)
from src.race.data.track import build_track_outline

logger = logging.getLogger(__name__)


def cache_race(year: int, gp: str, session_type: str = "R") -> Dict[str, Any]:
    """
    Ingest a race session and cache all processed tables to local Parquet files.
    Once cached, downstream modules (M2, M3, M4, M5) read completely offline.

    Parameters
    ----------
    year : int
        Championship year (e.g. 2023)
    gp : str
        Grand Prix name or round number (e.g. "Monza")
    session_type : str
        Session type (default "R" for Race)

    Returns
    -------
    dict
        Summary of cached tables and file sizes.
    """
    ensure_directories()
    cache_dir = get_race_cache_dir(year, gp)
    cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting ingestion and caching for {year} {gp}...")

    # FastF1 is primary: attempt loading via FastF1 first
    try:
        session = load_session(year, gp, session_type)
        if hasattr(session, "laps") and not session.laps.empty:
            return _cache_race_fastf1(session, year, gp, session_type, cache_dir)
        else:
            logger.warning(f"FastF1 returned empty laps for {year} {gp}. Attempting OpenF1 fallback...")
    except Exception as exc:
        logger.warning(f"FastF1 ingestion failed for {year} {gp}: {exc}. Attempting OpenF1 fallback...")

    # OpenF1 fallback (free, unauthenticated support for 2023+ seasons per PDF)
    from src.race.data.openf1 import cache_race_openf1
    return cache_race_openf1(year, gp, session_type)


def _cache_race_fastf1(session: Any, year: int, gp: str, session_type: str, cache_dir: Path) -> Dict[str, Any]:
    """Internal helper to process and cache FastF1 session data."""
    # 1. Process Laps
    laps_df = get_laps(session)
    laps_path = cache_dir / "laps.parquet"
    laps_df.to_parquet(laps_path, index=False)
    logger.info(f"Saved {len(laps_df)} laps to {laps_path.name}")

    # 2. Process Stints & Pits
    stints_df = get_pits_stints(session)
    stints_path = cache_dir / "stints.parquet"
    stints_df.to_parquet(stints_path, index=False)
    logger.info(f"Saved {len(stints_df)} stint records to {stints_path.name}")

    # 3. Process Weather & Track Status
    weather_df = get_weather_track_status(session)
    weather_path = cache_dir / "weather.parquet"
    weather_df.to_parquet(weather_path, index=False)
    logger.info(f"Saved {len(weather_df)} weather points to {weather_path.name}")

    # 4. Process 5 Hz Resampled Positions
    positions_df = get_positions(session)
    pos_path = cache_dir / "positions.parquet"
    positions_df.to_parquet(pos_path, index=False)
    logger.info(f"Saved {len(positions_df)} position points to {pos_path.name}")

    # 5. Process Official Classification
    classification_df = get_official_classification(session)
    class_path = cache_dir / "classification.parquet"
    classification_df.to_parquet(class_path, index=False)
    logger.info(f"Saved classification for {len(classification_df)} drivers to {class_path.name}")

    # 6. Build and Save Track Outline
    track_outline = build_track_outline(session)
    track_path = cache_dir / "track_outline.json"
    with open(track_path, "w", encoding="utf-8") as f:
        json.dump(track_outline, f, indent=2)
    logger.info(f"Saved track outline to {track_path.name}")

    # 7. Metadata summary
    total_laps = int(laps_df["lap_number"].max()) if not laps_df.empty else 0
    winner_row = classification_df.iloc[0] if not classification_df.empty else None
    winner_code = winner_row["driver"] if winner_row is not None else "UNKNOWN"

    metadata = {
        "year": year,
        "gp": gp,
        "official_event_name": str(session.event.EventName),
        "session": session_type,
        "total_laps": total_laps,
        "winner": winner_code,
        "driver_count": len(session.drivers),
        "drivers": [str(d) for d in session.drivers],
        "cached_tables": [
            "laps.parquet",
            "stints.parquet",
            "weather.parquet",
            "positions.parquet",
            "classification.parquet",
            "track_outline.json",
        ],
    }
    meta_path = cache_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Successfully cached race to {cache_dir}")
    return metadata


def is_race_cached(year: int, gp: str) -> bool:
    """Check if all required Parquet tables exist in the local cache."""
    cache_dir = get_race_cache_dir(year, gp)
    required_files = [
        "laps.parquet",
        "stints.parquet",
        "weather.parquet",
        "positions.parquet",
        "classification.parquet",
        "track_outline.json",
        "metadata.json",
    ]
    return all((cache_dir / f).exists() for f in required_files)


def load_cached_table(
    year: int,
    gp: str,
    table_name: str,
    as_of_lap: Optional[int] = None,
    as_of_time_s: Optional[float] = None,
) -> Any:
    """
    Load a cached table completely offline, enforcing the as_of boundary.

    Parameters
    ----------
    year : int
        Championship year
    gp : str
        Grand Prix name or slug
    table_name : str
        Table identifier (e.g. 'laps', 'stints', 'weather', 'positions', 'classification', 'track_outline')
    as_of_lap : int, optional
        Filter records to those up to and including this lap number.
    as_of_time_s : float, optional
        Filter records to those up to and including this session elapsed time.

    Returns
    -------
    pd.DataFrame or dict
        Clean, filtered dataset visible at the specified as_of state.
    """
    cache_dir = get_race_cache_dir(year, gp)
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory not found for race: {year} {gp}")

    if table_name == "track_outline":
        track_path = cache_dir / "track_outline.json"
        if not track_path.exists():
            raise FileNotFoundError(f"Missing track_outline.json in {cache_dir}")
        with open(track_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if table_name == "metadata":
        meta_path = cache_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing metadata.json in {cache_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    file_path = cache_dir / f"{table_name}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"Cached table {table_name}.parquet not found in {cache_dir}")

    df = pd.read_parquet(file_path)

    # 1. Classification guard: do not leak final race classification mid-race
    if table_name == "classification":
        meta = load_cached_table(year, gp, "metadata")
        total_laps = meta.get("total_laps", 0)
        if as_of_lap is not None and as_of_lap < total_laps:
            # Classification is final post-race result; mid-race standings come from laps.parquet
            return pd.DataFrame(columns=df.columns)
        if as_of_time_s is not None:
            all_laps = pd.read_parquet(cache_dir / "laps.parquet")
            if not all_laps.empty and "session_time_s" in all_laps.columns:
                final_time = all_laps["session_time_s"].dropna().max()
                if as_of_time_s < final_time:
                    return pd.DataFrame(columns=df.columns)

    # 2. Lap-based filtering
    if as_of_lap is not None:
        if "lap_number" in df.columns:
            df = df[df["lap_number"] <= as_of_lap].copy()
        elif table_name == "stints" and "start_lap" in df.columns:
            # Exclude future stints
            df = df[df["start_lap"] <= as_of_lap].copy()
            # For active ongoing stint, mask future pit stop timing and clamp end_lap
            active_mask = df["end_lap"] > as_of_lap
            if active_mask.any():
                df.loc[active_mask, "end_lap"] = as_of_lap
                df.loc[active_mask, "lap_count"] = as_of_lap - df.loc[active_mask, "start_lap"] + 1
                df.loc[active_mask, "pit_in_lap"] = None
                df.loc[active_mask, "pit_stop_duration_s"] = None

            # Mask future pit-in laps that haven't occurred yet
            future_pits = df["pit_in_lap"] > as_of_lap
            if future_pits.any():
                df.loc[future_pits, "pit_in_lap"] = None
                df.loc[future_pits, "pit_stop_duration_s"] = None

    # Stints fallback for as_of_time_s when as_of_lap is omitted
    elif as_of_time_s is not None and table_name == "stints":
        all_laps = pd.read_parquet(cache_dir / "laps.parquet")
        laps_until = all_laps[all_laps["session_time_s"] <= as_of_time_s]
        if not laps_until.empty:
            derived_as_of_lap = int(laps_until["lap_number"].max())
            return load_cached_table(year, gp, "stints", as_of_lap=derived_as_of_lap)
        else:
            return pd.DataFrame(columns=df.columns)

    # 3. Time-based filtering
    if as_of_time_s is not None:
        if "session_time_s" in df.columns:
            df = df[df["session_time_s"] <= as_of_time_s].copy()

    return df.reset_index(drop=True)


def load_cached_race(
    year: int,
    gp: str,
    as_of_lap: Optional[int] = None,
    as_of_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Load all cached tables for a race at a given as_of replay state.
    Operates strictly offline and synchronizes lap/time boundaries to prevent future leaks.
    """
    derived_time_s = as_of_time_s
    derived_lap = as_of_lap
    if as_of_lap is not None and derived_time_s is None:
        raw_laps = load_cached_table(year, gp, "laps", as_of_lap=as_of_lap)
        if not raw_laps.empty and "session_time_s" in raw_laps.columns:
            valid_times = raw_laps["session_time_s"].dropna()
            if not valid_times.empty:
                derived_time_s = float(valid_times.max())
    elif as_of_time_s is not None and derived_lap is None:
        raw_laps = load_cached_table(year, gp, "laps", as_of_time_s=as_of_time_s)
        if not raw_laps.empty and "lap_number" in raw_laps.columns:
            derived_lap = int(raw_laps["lap_number"].max())

    return {
        "metadata": load_cached_table(year, gp, "metadata"),
        "laps": load_cached_table(year, gp, "laps", as_of_lap=as_of_lap, as_of_time_s=derived_time_s),
        "stints": load_cached_table(year, gp, "stints", as_of_lap=derived_lap),
        "weather": load_cached_table(year, gp, "weather", as_of_time_s=derived_time_s),
        "positions": load_cached_table(year, gp, "positions", as_of_time_s=derived_time_s),
        "classification": load_cached_table(year, gp, "classification", as_of_lap=derived_lap, as_of_time_s=derived_time_s),
        "track_outline": load_cached_table(year, gp, "track_outline"),
    }
