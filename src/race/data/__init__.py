"""
Data ingestion, normalization, and caching module for RACE.
"""

from src.race.data.loader import load_session
from src.race.data.processor import (
    get_laps,
    get_telemetry,
    get_positions,
    get_pits_stints,
    get_weather_track_status,
)
from src.race.data.track import build_track_outline
from src.race.data.cache import cache_race, load_cached_race, load_cached_table

__all__ = [
    "load_session",
    "get_laps",
    "get_telemetry",
    "get_positions",
    "get_pits_stints",
    "get_weather_track_status",
    "build_track_outline",
    "cache_race",
    "load_cached_race",
    "load_cached_table",
]
