"""
Stage 2 Analytics package for RACE: Historical Race Analysis (M2 Analytics).
"""

from src.race.analytics.telemetry_loader import load_lap_telemetry
from src.race.analytics.mini_sectors import compute_mini_sectors
from src.race.analytics.telemetry import compute_delta_trace, compute_section_stats
from src.race.analytics.comparison import compare_laps
from src.race.analytics.tyres import compute_tyre_degradation
from src.race.analytics.gaps import compute_gap_history, detect_battles, classify_overtakes
from src.race.analytics.strategy import analyze_pit_stops, compute_time_loss_breakdown

__all__ = [
    "load_lap_telemetry",
    "compute_mini_sectors",
    "compute_delta_trace",
    "compute_section_stats",
    "compare_laps",
    "compute_tyre_degradation",
    "compute_gap_history",
    "detect_battles",
    "classify_overtakes",
    "analyze_pit_stops",
    "compute_time_loss_breakdown",
]
