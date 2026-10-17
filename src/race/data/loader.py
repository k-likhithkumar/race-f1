"""
Data loader module for RACE.
Interacts with FastF1 for telemetry/timing and Jolpica-F1 for official golden checks.
M1 is the only module that accesses the network.
"""

import logging
from typing import Any, Dict, Optional
import fastf1
import requests

from src.race.config import (
    RAW_FASTF1_CACHE_DIR,
    JOLPICA_BASE_URL,
    ensure_directories,
)

logger = logging.getLogger(__name__)


def setup_fastf1_cache() -> None:
    """Enable and configure FastF1 internal raw network cache."""
    ensure_directories()
    fastf1.Cache.enable_cache(str(RAW_FASTF1_CACHE_DIR))


def load_session(year: int, gp: str, session_type: str = "R") -> fastf1.core.Session:
    """
    Load a session object from FastF1 with raw caching enabled.

    Parameters
    ----------
    year : int
        Championship year (e.g. 2023)
    gp : str
        Grand Prix name or round number (e.g. "Monza" or "Italian Grand Prix")
    session_type : str
        Session identifier ("R" for Race, "Q" for Qualifying, etc.)

    Returns
    -------
    fastf1.core.Session
        Loaded session object with laps, telemetry, and weather loaded.
    """
    setup_fastf1_cache()
    logger.info(f"Loading session {year} {gp} [{session_type}] via FastF1...")
    session = fastf1.get_session(year, gp, session_type)
    session.load(laps=True, telemetry=True, weather=True, messages=True)
    logger.info(f"Successfully loaded {year} {gp} [{session_type}].")
    return session


def fetch_jolpica_race_results(year: int, round_num: int) -> Dict[str, Any]:
    """
    Fetch official race results from the Jolpica-F1 API (Ergast API replacement).
    Used strictly for independent verification and golden dataset checks.
    """
    url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/results.json"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        race_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if race_data:
            return race_data[0]
        return {}
    except Exception as exc:
        logger.warning(f"Failed to fetch Jolpica results from {url}: {exc}")
        return {}


def fetch_jolpica_pitstops(year: int, round_num: int) -> Dict[str, Any]:
    """
    Fetch official pit stop data from the Jolpica-F1 API.
    Used for independent verification of pit stop counts and timing.
    """
    url = f"{JOLPICA_BASE_URL}/{year}/{round_num}/pitstops.json?limit=100"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        race_data = data.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        if race_data:
            return race_data[0].get("PitStops", [])
        return []
    except Exception as exc:
        logger.warning(f"Failed to fetch Jolpica pitstops from {url}: {exc}")
        return []
