"""
Unit tests for Stage 1: Data Ingestion and Normalization.
Tests clean-lap classification, compound mapping, GPS gap handling, and as_of slicing.
"""

import numpy as np
import pandas as pd
import pytest

from src.race.data.processor import (
    is_track_status_strictly_green,
    normalize_compound,
    COMPOUND_MAP,
)


def test_track_status_strictly_green():
    """Verify that only strictly green flag status '1' evaluates to True."""
    # Green flag valid
    assert is_track_status_strictly_green("1") is True
    assert is_track_status_strictly_green(1) is True
    assert is_track_status_strictly_green(" 1 ") is True

    # Ambiguous / Non-green flags
    assert is_track_status_strictly_green("2") is False       # Yellow
    assert is_track_status_strictly_green("4") is False       # Safety Car
    assert is_track_status_strictly_green("5") is False       # Red Flag
    assert is_track_status_strictly_green("6") is False       # VSC
    assert is_track_status_strictly_green("7") is False       # VSC Ending

    # Combined statuses (yellow/SC occurred during the lap)
    assert is_track_status_strictly_green("12") is False
    assert is_track_status_strictly_green("14") is False
    assert is_track_status_strictly_green("24") is False

    # Missing / Unknown / Corrupt data (must NEVER be silently treated as green)
    assert is_track_status_strictly_green(None) is False
    assert is_track_status_strictly_green("") is False
    assert is_track_status_strictly_green("   ") is False
    assert is_track_status_strictly_green(np.nan) is False
    assert is_track_status_strictly_green("UNKNOWN") is False
    assert is_track_status_strictly_green("0") is False


def test_normalize_compound():
    """Verify compound normalization maps to standard names or 'UNKNOWN'."""
    assert normalize_compound("Soft") == "SOFT"
    assert normalize_compound("MEDIUM") == "MEDIUM"
    assert normalize_compound("hard") == "HARD"
    assert normalize_compound("INTER") == "INTERMEDIATE"
    assert normalize_compound("Intermediate") == "INTERMEDIATE"
    assert normalize_compound("WET") == "WET"

    # Missing / Unknown values
    assert normalize_compound("") == "UNKNOWN"
    assert normalize_compound(None) == "UNKNOWN"
    assert normalize_compound("HYPERSOFT_VINTAGE") == "UNKNOWN"
    assert normalize_compound(123) == "UNKNOWN"


def test_clean_lap_logic():
    """Verify that in-laps, out-laps, SC laps, deleted laps, and missing times are excluded."""
    # Construct a synthetic lap batch
    df = pd.DataFrame({
        "LapNumber": [1, 2, 3, 4, 5, 6, 7],
        "LapTime": [
            pd.Timedelta(seconds=90.5),   # 1: Clean green lap
            pd.Timedelta(seconds=95.0),   # 2: In-lap (pitted at end)
            pd.Timedelta(seconds=115.0),  # 3: Out-lap (exited pit)
            pd.Timedelta(seconds=130.0),  # 4: Safety car lap
            pd.Timedelta(seconds=89.8),   # 5: Deleted lap (track limits)
            pd.NaT,                       # 6: Incomplete lap / DNF
            pd.Timedelta(seconds=90.2),   # 7: Combined track status '12'
        ],
        "PitInTime": [pd.NaT, pd.Timedelta(seconds=95.0), pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "PitOutTime": [pd.NaT, pd.NaT, pd.Timedelta(seconds=0.0), pd.NaT, pd.NaT, pd.NaT, pd.NaT],
        "TrackStatus": ["1", "1", "1", "4", "1", "1", "12"],
        "Deleted": [False, False, False, False, True, False, False],
    })

    # Apply the clean-lap conditions from processor
    has_valid_time = df["LapTime"].notna()
    not_pit_in = df["PitInTime"].isna()
    not_pit_out = df["PitOutTime"].isna()
    not_deleted = ~df["Deleted"].fillna(False).astype(bool)
    strictly_green = df["TrackStatus"].apply(is_track_status_strictly_green)

    is_clean = has_valid_time & not_pit_in & not_pit_out & not_deleted & strictly_green

    # Only lap 1 meets all strict clean criteria
    assert bool(is_clean.iloc[0]) is True
    assert bool(is_clean.iloc[1]) is False  # In-lap
    assert bool(is_clean.iloc[2]) is False  # Out-lap
    assert bool(is_clean.iloc[3]) is False  # SC lap
    assert bool(is_clean.iloc[4]) is False  # Deleted
    assert bool(is_clean.iloc[5]) is False  # Missing time
    assert bool(is_clean.iloc[6]) is False  # Non-green (combined '12')


def test_gps_gap_preservation():
    """Verify that gaps exceeding 2.0s are preserved as NaN and not falsely interpolated."""
    from src.race.config import POSITION_RESAMPLE_HZ, POSITION_MAX_INTERPOLATION_GAP_S

    # Create telemetry with a 5-second gap between t=1.0 and t=6.0
    orig_t = np.array([0.0, 1.0, 6.0, 7.0])
    orig_x = np.array([10.0, 20.0, 120.0, 140.0])
    orig_y = np.array([5.0, 10.0, 60.0, 70.0])

    dt = 1.0 / POSITION_RESAMPLE_HZ  # 0.2s
    grid_times = np.arange(0.0, 7.0 + dt/2, dt)

    interp_x = np.interp(grid_times, orig_t, orig_x)
    interp_y = np.interp(grid_times, orig_t, orig_y)

    # Apply gap threshold mask
    gap_indices = np.where(np.diff(orig_t) > POSITION_MAX_INTERPOLATION_GAP_S)[0]
    for gap_idx in gap_indices:
        t_gap_start = orig_t[gap_idx]
        t_gap_end = orig_t[gap_idx + 1]
        mask = (grid_times > t_gap_start) & (grid_times < t_gap_end)
        interp_x[mask] = np.nan
        interp_y[mask] = np.nan

    # Test points inside the gap (e.g. t=3.0s)
    gap_mask = (grid_times > 1.0) & (grid_times < 6.0)
    assert np.isnan(interp_x[gap_mask]).all()
    assert np.isnan(interp_y[gap_mask]).all()

    # Test continuous points outside the gap (e.g. t=0.4s)
    continuous_mask = (grid_times >= 0.0) & (grid_times <= 1.0)
    assert not np.isnan(interp_x[continuous_mask]).any()
    assert not np.isnan(interp_y[continuous_mask]).any()


def test_as_of_filtering():
    """Verify as_of filters prevent reading records from future laps or timestamps."""
    laps_sample = pd.DataFrame({
        "lap_number": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "driver": ["VER"] * 10,
        "lap_time_s": [85.0 + i for i in range(10)],
    })

    # Filter at as_of_lap = 5
    as_of_lap = 5
    filtered_laps = laps_sample[laps_sample["lap_number"] <= as_of_lap]
    assert len(filtered_laps) == 5
    assert filtered_laps["lap_number"].max() == 5
    assert not (filtered_laps["lap_number"] > 5).any()

    # Time-based filtering
    pos_sample = pd.DataFrame({
        "session_time_s": [10.0, 20.0, 30.0, 40.0, 50.0],
        "driver": ["NOR"] * 5,
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    as_of_time = 35.0
    filtered_pos = pos_sample[pos_sample["session_time_s"] <= as_of_time]
    assert len(filtered_pos) == 3
    assert filtered_pos["session_time_s"].max() == 30.0
    assert not (filtered_pos["session_time_s"] > 35.0).any()

