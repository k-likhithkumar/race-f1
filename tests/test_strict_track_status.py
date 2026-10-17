"""
Regression tests for strict track-status and green-flag detection.
Validates both FastF1 and OpenF1 implementations against F1_Project_Plan clean-lap rules:
- Only strictly TrackStatus == '1' is clean.
- Missing, unknown, composite, or ambiguous status values must never silently count as green.
- Caution periods (Yellow, Double Yellow, SC, VSC, Red Flag) must mark all overlapping laps dirty.
- Status changes during a lap (e.g. '12', '14') must result in is_clean == False.
- Missing race control data must result in track_status == 'UNKNOWN' and is_clean == False.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.race.data.processor import is_track_status_strictly_green
from src.race.data.openf1 import (
    parse_openf1_race_control_timeline,
    resolve_openf1_lap_track_status,
)


# ============================================================================
# 1. FASTF1 STRICT GREEN FLAG DETECTION UNIT TESTS
# ============================================================================

def test_fastf1_strictly_green_flag_detection():
    """Verify that only strictly green flag status '1' evaluates to True."""
    # Explicit green
    assert is_track_status_strictly_green("1") is True
    assert is_track_status_strictly_green(1) is True
    assert is_track_status_strictly_green(" 1 ") is True

    # Single caution flags
    assert is_track_status_strictly_green("2") is False       # Yellow
    assert is_track_status_strictly_green("4") is False       # Safety Car
    assert is_track_status_strictly_green("5") is False       # Red Flag
    assert is_track_status_strictly_green("6") is False       # VSC
    assert is_track_status_strictly_green("7") is False       # VSC Ending

    # Composite statuses (status change during a lap)
    assert is_track_status_strictly_green("12") is False      # Green then Yellow
    assert is_track_status_strictly_green("14") is False      # Green then SC
    assert is_track_status_strictly_green("16") is False      # Green then VSC
    assert is_track_status_strictly_green("24") is False      # Yellow to SC
    assert is_track_status_strictly_green("124") is False     # Complex caution

    # Missing / Unknown / Corrupt data (must NEVER silently count as green)
    assert is_track_status_strictly_green(None) is False
    assert is_track_status_strictly_green("") is False
    assert is_track_status_strictly_green("   ") is False
    assert is_track_status_strictly_green(np.nan) is False
    assert is_track_status_strictly_green("UNKNOWN") is False
    assert is_track_status_strictly_green("0") is False
    assert is_track_status_strictly_green("?") is False


def test_fastf1_lap_clean_logic_matrix():
    """
    Test clean-lap evaluation across a synthetic matrix of FastF1 lap conditions.
    A lap is clean IF AND ONLY IF:
    1. LapTime is valid (> 0)
    2. PitIn is null
    3. PitOut is null
    4. Deleted is False
    5. TrackStatus is strictly '1'
    """
    test_cases = [
        # (lap_time, pit_in, pit_out, deleted, track_status, expected_clean)
        (85.0, None, None, False, "1", True),         # Perfect clean lap
        (85.0, 85.0, None, False, "1", False),        # Pit-in lap -> dirty
        (85.0, None, 0.0, False, "1", False),         # Pit-out lap -> dirty
        (85.0, None, None, True, "1", False),         # Deleted lap -> dirty
        (85.0, None, None, False, "2", False),        # Yellow flag -> dirty
        (85.0, None, None, False, "4", False),        # Safety car -> dirty
        (85.0, None, None, False, "5", False),        # Red flag -> dirty
        (85.0, None, None, False, "6", False),        # VSC -> dirty
        (85.0, None, None, False, "12", False),       # Status change (green to yellow) -> dirty
        (85.0, None, None, False, "14", False),       # Status change (green to SC) -> dirty
        (85.0, None, None, False, "", False),         # Missing track status -> dirty
        (85.0, None, None, False, "UNKNOWN", False),  # Unknown track status -> dirty
        (np.nan, None, None, False, "1", False),      # Missing lap time -> dirty
        (0.0, None, None, False, "1", False),         # Zero lap time -> dirty
    ]

    for lap_time, pit_in, pit_out, deleted, status, expected in test_cases:
        has_valid_time = pd.notna(lap_time) and lap_time > 0
        not_pit_in = pd.isna(pit_in)
        not_pit_out = pd.isna(pit_out)
        not_deleted = not deleted
        strictly_green = is_track_status_strictly_green(status)
        is_clean = has_valid_time and not_pit_in and not_pit_out and not_deleted and strictly_green
        assert is_clean == expected, f"Failed for case: status={status}, time={lap_time}, del={deleted}, exp={expected}"


# ============================================================================
# 2. OPENF1 TIMELINE AND TRACK STATUS POSITIVE VERIFICATION TESTS
# ============================================================================

def test_openf1_missing_race_control_events_marks_unknown():
    """Missing or empty race control data must mark track status UNKNOWN and laps dirty."""
    # Empty list
    timeline_empty = parse_openf1_race_control_timeline([])
    assert timeline_empty["verified"] is False
    status_empty = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:00:00+00:00"),
        pd.Timestamp("2023-09-03 13:01:30+00:00"),
        1,
        timeline_empty,
    )
    assert status_empty == "UNKNOWN"
    assert is_track_status_strictly_green(status_empty) is False

    # None
    timeline_none = parse_openf1_race_control_timeline(None)
    assert timeline_none["verified"] is False
    status_none = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:00:00+00:00"),
        pd.Timestamp("2023-09-03 13:01:30+00:00"),
        1,
        timeline_none,
    )
    assert status_none == "UNKNOWN"
    assert is_track_status_strictly_green(status_none) is False


def test_openf1_ambiguous_race_control_without_green_signal_marks_unknown():
    """Race control messages without positive confirmation of green/racing must not assume green."""
    rc_ambiguous = [
        {"date": "2023-09-03T12:00:00+00:00", "message": "WEATHER UPDATE: NO RAIN"},
        {"date": "2023-09-03T12:05:00+00:00", "message": "PIT LANE OPEN"},
    ]
    timeline = parse_openf1_race_control_timeline(rc_ambiguous)
    # Without any GREEN LIGHT, SESSION STARTED, or CLEAR message, green is NOT verified
    assert timeline["verified"] is False

    status = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:00:00+00:00"),
        pd.Timestamp("2023-09-03 13:01:30+00:00"),
        1,
        timeline,
    )
    assert status == "UNKNOWN"
    assert is_track_status_strictly_green(status) is False


def test_openf1_explicit_green_timeline():
    """Positively verified green timeline marks lap strictly '1'."""
    rc_clean = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT - PIT EXIT OPEN"},
        {"date": "2023-09-03T13:00:01+00:00", "message": "SESSION STARTED"},
    ]
    timeline = parse_openf1_race_control_timeline(rc_clean)
    assert timeline["verified"] is True

    # Lap running well after green light
    status = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:05:00+00:00"),
        pd.Timestamp("2023-09-03 13:06:25+00:00"),
        2,
        timeline,
    )
    assert status == "1"
    assert is_track_status_strictly_green(status) is True


def test_openf1_lap_before_session_start_marks_unknown():
    """A lap occurring before the green flag / session start cannot be marked clean green."""
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT - PIT EXIT OPEN"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)
    assert timeline["verified"] is True

    # Lap occurring before 13:00:00
    status = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 12:55:00+00:00"),
        pd.Timestamp("2023-09-03 12:56:30+00:00"),
        1,
        timeline,
    )
    assert status == "UNKNOWN"
    assert is_track_status_strictly_green(status) is False


def test_openf1_yellow_and_double_yellow_periods():
    """Yellow flag intervals (single sector or track) must mark overlapping laps dirty."""
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"},
        {"date": "2023-09-03T13:10:00+00:00", "flag": "DOUBLE YELLOW", "sector": 4, "message": "DOUBLE YELLOW SECTOR 4"},
        {"date": "2023-09-03T13:15:00+00:00", "flag": "CLEAR", "sector": 4, "message": "CLEAR SECTOR 4"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)

    # Lap entirely inside yellow period (13:11:00 to 13:12:30)
    status_inside = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:11:00+00:00"),
        pd.Timestamp("2023-09-03 13:12:30+00:00"),
        5,
        timeline,
    )
    assert status_inside == "2"
    assert is_track_status_strictly_green(status_inside) is False

    # Lap after clear (13:16:00 to 13:17:30)
    status_after = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:16:00+00:00"),
        pd.Timestamp("2023-09-03 13:17:30+00:00"),
        8,
        timeline,
    )
    assert status_after == "1"
    assert is_track_status_strictly_green(status_after) is True


def test_openf1_red_flag_period():
    """Red flag stops racing and must mark overlapping laps dirty with code '5'."""
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"},
        {"date": "2023-09-03T13:20:00+00:00", "flag": "RED", "message": "RED FLAG - SESSION SUSPENDED"},
        {"date": "2023-09-03T13:50:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT - RESUMED"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)

    # Lap running during red flag
    status_red = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:21:00+00:00"),
        pd.Timestamp("2023-09-03 13:23:00+00:00"),
        12,
        timeline,
    )
    assert status_red == "5"
    assert is_track_status_strictly_green(status_red) is False


def test_openf1_multi_lap_safety_car_interval_continuous_coverage():
    """
    CRITICAL FAILURE MODE TEST:
    A Safety Car deployed at t=10 and ending at t=25 spans multiple laps (Laps 8, 9, 10).
    Even though intermediate laps (e.g. Lap 9) have no per-lap race control message,
    they MUST NOT be inferred as green. The continuous interval must mark Lap 9 as '4'.
    """
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"},
        {"date": "2023-09-03T13:10:00+00:00", "message": "SAFETY CAR DEPLOYED"},
        # No message on lap 9!
        {"date": "2023-09-03T13:25:00+00:00", "flag": "CLEAR", "message": "TRACK CLEAR"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)

    # Lap 8: starts right after SC deployed (13:11:00 -> 13:13:00)
    s8 = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:11:00+00:00"),
        pd.Timestamp("2023-09-03 13:13:00+00:00"),
        8,
        timeline,
    )
    assert s8 == "4"
    assert is_track_status_strictly_green(s8) is False

    # Lap 9: intermediate lap with NO direct message (13:14:00 -> 13:16:00)
    s9 = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:14:00+00:00"),
        pd.Timestamp("2023-09-03 13:16:00+00:00"),
        9,
        timeline,
    )
    # MUST BE '4', NOT '1'!
    assert s9 == "4"
    assert is_track_status_strictly_green(s9) is False

    # Lap 10: intermediate lap with NO direct message (13:18:00 -> 13:20:00)
    s10 = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:18:00+00:00"),
        pd.Timestamp("2023-09-03 13:20:00+00:00"),
        10,
        timeline,
    )
    assert s10 == "4"
    assert is_track_status_strictly_green(s10) is False

    # Lap 12: well after SC cleared (13:26:00 -> 13:27:25)
    s12 = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:26:00+00:00"),
        pd.Timestamp("2023-09-03 13:27:25+00:00"),
        12,
        timeline,
    )
    assert s12 == "1"
    assert is_track_status_strictly_green(s12) is True


def test_openf1_vsc_period_and_vsc_ending():
    """VSC deployment ('6') and VSC ending ('7') must mark laps dirty."""
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"},
        {"date": "2023-09-03T13:10:00+00:00", "message": "VSC DEPLOYED"},
        {"date": "2023-09-03T13:14:00+00:00", "message": "VSC ENDING"},
        {"date": "2023-09-03T13:15:00+00:00", "flag": "CLEAR", "message": "TRACK CLEAR"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)

    # Lap during active VSC
    s_vsc = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:11:00+00:00"),
        pd.Timestamp("2023-09-03 13:13:00+00:00"),
        6,
        timeline,
    )
    assert "6" in s_vsc or "7" in s_vsc
    assert is_track_status_strictly_green(s_vsc) is False


def test_openf1_status_change_during_lap_emits_composite():
    """
    When a caution is called while a car is mid-lap, or cleared while mid-lap,
    the lap has both green and caution portions.
    Must emit composite status (e.g. '12' or '14') and evaluate to is_clean == False.
    """
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"},
        # Yellow deployed at 13:05:30
        {"date": "2023-09-03T13:05:30+00:00", "flag": "YELLOW", "message": "YELLOW FLAG"},
        # Yellow cleared at 13:08:30
        {"date": "2023-09-03T13:08:30+00:00", "flag": "CLEAR", "message": "TRACK CLEAR"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)

    # Lap starts green at 13:04:45, yellow called at 13:05:30, lap finishes at 13:06:10
    s_green_to_yellow = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:04:45+00:00"),
        pd.Timestamp("2023-09-03 13:06:10+00:00"),
        3,
        timeline,
    )
    assert s_green_to_yellow == "12"
    assert is_track_status_strictly_green(s_green_to_yellow) is False

    # Lap starts under yellow at 13:08:00, clears at 13:08:30, lap finishes green at 13:09:25
    s_yellow_to_green = resolve_openf1_lap_track_status(
        pd.Timestamp("2023-09-03 13:08:00+00:00"),
        pd.Timestamp("2023-09-03 13:09:25+00:00"),
        5,
        timeline,
    )
    assert s_yellow_to_green == "12"
    assert is_track_status_strictly_green(s_yellow_to_green) is False


def test_openf1_missing_lap_timing_returns_unknown():
    """If lap start or end timestamp is NaN or None, return UNKNOWN."""
    rc = [{"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"}]
    timeline = parse_openf1_race_control_timeline(rc)

    s1 = resolve_openf1_lap_track_status(None, pd.Timestamp("2023-09-03 13:05:00+00:00"), 1, timeline)
    assert s1 == "UNKNOWN"

    s2 = resolve_openf1_lap_track_status(pd.Timestamp("2023-09-03 13:05:00+00:00"), None, 1, timeline)
    assert s2 == "UNKNOWN"

    s3 = resolve_openf1_lap_track_status(pd.NaT, pd.NaT, 1, timeline)
    assert s3 == "UNKNOWN"


def test_openf1_track_limits_deleted_laps():
    """Ensure track limits deletions extracted from messages are recorded in timeline."""
    rc = [
        {"date": "2023-09-03T13:00:00+00:00", "flag": "GREEN", "message": "GREEN LIGHT"},
        {"date": "2023-09-03T13:29:28+00:00", "message": "CAR 14 (ALO) TIME 1:27.546 DELETED - TRACK LIMITS AT TURN 10 LAP 3"},
        {"date": "2023-09-03T13:31:29+00:00", "message": "CAR 4 (NOR) TIME 1:27.142 DELETED - TRACK LIMITS AT TURN 7 LAP 5"},
    ]
    timeline = parse_openf1_race_control_timeline(rc)
    assert (14, 3) in timeline["deleted_laps"]
    assert (4, 5) in timeline["deleted_laps"]
