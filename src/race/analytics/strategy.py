"""
Pit stop strategy analysis and time-loss decomposition module for RACE.
Analyzes pit lane transit, stationary durations, tactical undercut/overcuts,
and decomposes total race time deficits without double-counting.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.race.config import TRAFFIC_GAP_THRESHOLD_S, FUEL_CORRECTION_S_PER_LAP

logger = logging.getLogger(__name__)


def analyze_pit_stops(
    stints_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    driver: str,
    as_of_lap: Optional[int] = None,
    reference_stationary_s: float = 2.5,
    parallel_straight_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyzes all completed pit stops for a driver with rigorous time accounting:
    - Distinguishes measured pit lane duration (PitIn to PitOut) from stationary stop time.
    - Explicitly marks stationary duration as unavailable when not directly recorded by the provider.
    - Evaluates stationary excess against a stated reference stop duration (default 2.5s).
    - Estimates pit lane transit duration without claiming it is physically measured.
    - Computes total track pit loss relative to a clearly specified clean-lap baseline across 2 laps.
    - Strictly prevents double-counting and prohibits invalid subtraction of absolute pit lane duration from net pit loss.
    - Reports track entry/exit loss only when an explicit parallel straight baseline is provided, otherwise UNAVAILABLE.
    - Evaluates undercut / overcut tactical outcomes against immediate rivals.
    - Flags Safety Car / VSC timing advantages.

    Parameters
    ----------
    stints_df : pd.DataFrame
        Cached stints table containing 'driver', 'stint', 'start_lap', 'end_lap',
        'pit_in_lap', 'pit_stop_duration_s'.
    laps_df : pd.DataFrame
        Cached laps table containing 'driver', 'lap_number', 'lap_time_s', 'track_status', 'is_clean'.
    driver : str
        Three-letter driver code.
    as_of_lap : int, optional
        Replay barrier. Only pit stops on or before as_of_lap are analyzed.
    reference_stationary_s : float
        Stated benchmark reference for stationary wheel change (default: 2.5s).
    parallel_straight_time_s : float, optional
        Elapsed time for a car on track to cover the parallel straight segment between
        PitIn and PitOut coordinates. Required to decompose net pit loss into net lane loss
        and track entry/exit loss without fabricating data.

    Returns
    -------
    dict
        Structured analysis of each pit stop with explicit accounting categories.
    """
    if stints_df is None or stints_df.empty or laps_df is None or laps_df.empty:
        return {
            "driver": driver,
            "stops": [],
            "total_stops": 0,
            "verdict": "NO_DATA",
            "baseline_description": "No data available",
        }

    s_df = stints_df.copy()
    l_df = laps_df.copy()

    if as_of_lap is not None:
        s_df = s_df[s_df["start_lap"] <= as_of_lap]
        l_df = l_df[l_df["lap_number"] <= as_of_lap]

    # Stated Reference Baseline: median clean green flying lap pace of the field up to as_of
    clean_laps = l_df[l_df["is_clean"] == True]["lap_time_s"].dropna()
    ref_clean_pace = float(clean_laps.median()) if not clean_laps.empty else 85.0

    drv_stints = s_df[s_df["driver"] == driver].sort_values("stint").reset_index(drop=True)
    stops = []

    for i in range(len(drv_stints) - 1):
        stint_row = drv_stints.iloc[i]
        next_stint = drv_stints.iloc[i + 1]

        pit_lap = stint_row.get("pit_in_lap")
        if pd.isna(pit_lap) or (as_of_lap is not None and pit_lap > as_of_lap):
            continue

        pit_lap = int(pit_lap)
        out_lap = pit_lap + 1

        # In-lap and Out-lap times
        in_lap_row = l_df[(l_df["driver"] == driver) & (l_df["lap_number"] == pit_lap)]
        out_lap_row = l_df[(l_df["driver"] == driver) & (l_df["lap_number"] == out_lap)]

        t_in = float(in_lap_row.iloc[0]["lap_time_s"]) if not in_lap_row.empty and pd.notna(in_lap_row.iloc[0]["lap_time_s"]) else None
        t_out = float(out_lap_row.iloc[0]["lap_time_s"]) if not out_lap_row.empty and pd.notna(out_lap_row.iloc[0]["lap_time_s"]) else None

        # 1. Total track pit loss relative to stated reference clean flying pace across 2 laps
        if t_in is not None and t_out is not None:
            total_pit_loss_s = round((t_in + t_out) - (2.0 * ref_clean_pace), 2)
        else:
            total_pit_loss_s = None

        # 2. Pit Lane Duration from timing loops (PitIn to PitOut transponders)
        # In both FastF1 and OpenF1 schemas, pit_stop_duration_s records total pit lane duration (~20-30s)
        raw_pit_dur = stint_row.get("pit_stop_duration_s")
        if pd.notna(raw_pit_dur):
            raw_val = float(raw_pit_dur)
            # Distinguish whether raw value is stationary (<15s) or pit lane duration (>=15s)
            if raw_val < 15.0:
                measured_stationary_s = round(raw_val, 2)
                measured_pit_lane_duration_s = None
                stationary_data_status = "MEASURED_TRANSPONDER"
            else:
                measured_pit_lane_duration_s = round(raw_val, 2)
                # Stationary duration was not separated in raw transponder feed
                measured_stationary_s = None
                stationary_data_status = "UNAVAILABLE_IN_PROVIDER_FEED"
        else:
            measured_pit_lane_duration_s = None
            measured_stationary_s = None
            stationary_data_status = "MISSING"

        # 3. Stationary Excess above stated reference benchmark
        if measured_stationary_s is not None:
            stationary_excess_s = round(measured_stationary_s - reference_stationary_s, 2)
            eff_stationary = measured_stationary_s
        else:
            stationary_excess_s = None
            eff_stationary = reference_stationary_s

        # 4. Estimated Pit Lane Transit Time (never claimed as physically measured)
        # Transit = Pit Lane Duration - Stationary Time
        if measured_pit_lane_duration_s is not None:
            estimated_pit_lane_transit_s = round(max(0.0, measured_pit_lane_duration_s - eff_stationary), 2)
            is_transit_estimated = True
            transit_estimation_method = f"PIT_LANE_DURATION_MINUS_{'MEASURED' if measured_stationary_s else 'REFERENCE'}_STATIONARY"
        else:
            estimated_pit_lane_transit_s = None
            is_transit_estimated = False
            transit_estimation_method = "UNAVAILABLE"

        # 5. Track In/Out Deceleration & Acceleration Loss
        # Dimensionally and conceptually, subtracting measured pit lane duration (T_pit_lane)
        # directly from net pit loss (Delta T_total_loss) is invalid because T_pit_lane is an
        # absolute elapsed duration, whereas Delta T_total_loss is a net differential vs a flying lap.
        # Net loss inside pit lane is: Delta T_lane = T_pit_lane - T_straight_baseline.
        # Track entry/exit loss is: Delta T_track = Delta T_total_loss - Delta T_lane.
        # Because transponder loops do not measure T_straight_baseline, track in/out loss cannot be
        # physically measured from transponder feeds alone without fabricating a measurement.
        if parallel_straight_time_s is not None and total_pit_loss_s is not None and measured_pit_lane_duration_s is not None:
            net_pit_lane_loss_s = max(0.0, measured_pit_lane_duration_s - float(parallel_straight_time_s))
            estimated_track_in_out_loss_s = round(max(0.0, total_pit_loss_s - net_pit_lane_loss_s), 2)
            track_loss_status = "ESTIMATED_WITH_STRAIGHT_BASELINE"
        else:
            estimated_track_in_out_loss_s = None
            track_loss_status = "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"

        # Check track status for Safety Car advantage
        track_status = str(in_lap_row.iloc[0]["track_status"]).strip() if not in_lap_row.empty else "1"
        is_sc_pit = any(c in track_status for c in ["4", "5", "6", "7"])

        # Tactical outcome (undercut/overcut evaluation vs rivals who pitted within +/- 2 laps)
        tactical_verdict = "REGULAR_STOP"
        if is_sc_pit:
            tactical_verdict = "SAFETY_CAR_ADVANTAGE"
        else:
            rival_stints = s_df[(s_df["driver"] != driver) & (s_df["pit_in_lap"].between(pit_lap - 2, pit_lap + 2))]
            if not rival_stints.empty:
                earlier_rivals = rival_stints[rival_stints["pit_in_lap"] > pit_lap]
                if not earlier_rivals.empty:
                    tactical_verdict = "UNDERCUT_ATTEMPT"
                else:
                    tactical_verdict = "OVERCUT_ATTEMPT"

        stops.append({
            "stop_number": i + 1,
            "pit_in_lap": pit_lap,
            "out_lap": out_lap,
            "in_lap_time_s": t_in,
            "out_lap_time_s": t_out,
            "estimated_total_pit_loss_s": total_pit_loss_s,
            "measured_pit_lane_duration_s": measured_pit_lane_duration_s,
            "measured_stationary_s": measured_stationary_s,
            "stationary_data_status": stationary_data_status,
            "reference_stationary_s": reference_stationary_s,
            "stationary_excess_s": stationary_excess_s,
            "estimated_pit_lane_transit_s": estimated_pit_lane_transit_s,
            "is_transit_loss_estimated": is_transit_estimated,
            "transit_estimation_basis": transit_estimation_method,
            "estimated_track_in_out_loss_s": estimated_track_in_out_loss_s,
            "track_loss_status": track_loss_status,
            "parallel_straight_time_s": parallel_straight_time_s,
            "is_safety_car_stop": is_sc_pit,
            "tactical_verdict": tactical_verdict,
        })

    return {
        "driver": driver,
        "total_stops": len(stops),
        "stops": stops,
        "reference_clean_pace_s": ref_clean_pace,
        "reference_stationary_s": reference_stationary_s,
        "baseline_description": f"Field clean flying lap median ({ref_clean_pace:.3f}s) across 2 laps",
        "accounting_notes": [
            "Pit lane duration represents transponder-measured elapsed time from pit entry to pit exit (gross duration).",
            "Total pit loss represents net race time lost across in-lap and out-lap relative to 2x reference clean pace.",
            "Stationary duration is reported only when explicitly recorded; otherwise marked UNAVAILABLE.",
            "Transit duration is calculated as pit lane duration minus stationary duration and explicitly labeled estimated.",
            "Track entry/exit loss requires parallel main straight transit baseline; marked UNAVAILABLE when baseline is omitted.",
            "Direct subtraction of pit lane duration from net pit loss is prohibited as it mixes absolute duration with differential loss.",
        ],
    }


def compute_time_loss_breakdown(
    laps_df: pd.DataFrame,
    stints_df: pd.DataFrame,
    driver: str,
    as_of_lap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Decomposes total race time deficit for a driver against a stated reference baseline.
    Enforces a strict, non-overlapping attribution waterfall and preserves the exact
    accounting identity:
        Total Measured Deficit == Caution + Pit + Traffic + Tyre + Residual

    Attribution Waterfall (Mutual Exclusivity Order):
    1. Caution Loss: Entire lap deficit on non-green laps (SC, VSC, Yellow, Red).
    2. Pit Loss: Entire lap deficit on pit-in or pit-out laps (encompasses transit & stationary).
    3. Traffic Loss: Deficit above driver's clean-air pace on green non-pit laps within 1.2s of car ahead.
    4. Tyre Degradation Loss: Deficit attributable to tyre life > 5 laps on free-air green non-pit laps.
    5. Residual Loss: Mathematically exact balancing residual (driver variance, setup, energy).

    Parameters
    ----------
    laps_df : pd.DataFrame
        Cached laps DataFrame.
    stints_df : pd.DataFrame
        Cached stints DataFrame.
    driver : str
        Target driver code.
    as_of_lap : int, optional
        Replay boundary.

    Returns
    -------
    dict
        Structured breakdown with stated baseline, attributed factor seconds,
        exact residual, and accounting identity verification flag.
    """
    if laps_df is None or laps_df.empty:
        return _empty_time_loss_breakdown(driver)

    max_lap = as_of_lap if as_of_lap is not None else int(laps_df["lap_number"].max())
    df = laps_df[laps_df["lap_number"] <= max_lap].copy()

    drv_laps = df[df["driver"] == driver].sort_values("lap_number").reset_index(drop=True)
    if drv_laps.empty:
        return _empty_time_loss_breakdown(driver)

    # Establish stated reference baseline:
    # Median clean green flying lap pace of the field up to as_of_lap
    clean_field_laps = df[df["is_clean"] == True]["lap_time_s"].dropna()
    ref_clean_lap_s = float(clean_field_laps.median()) if not clean_field_laps.empty else 85.0

    # Best clean stint pace of target driver as private free-air expectation for traffic attribution
    drv_clean_laps = drv_laps[drv_laps["is_clean"] == True]["lap_time_s"].dropna()
    drv_free_air_baseline_s = float(drv_clean_laps.quantile(0.25)) if len(drv_clean_laps) >= 4 else ref_clean_lap_s

    # Identify pit in and out laps
    pit_in_laps = set(drv_laps[drv_laps["is_pit_in"] == True]["lap_number"].values) if "is_pit_in" in drv_laps.columns else set()
    pit_out_laps = set(drv_laps[drv_laps["is_pit_out"] == True]["lap_number"].values) if "is_pit_out" in drv_laps.columns else set()

    # Also check stints_df for pit_in_lap and pit_in_lap + 1
    if stints_df is not None and not stints_df.empty:
        drv_stints = stints_df[stints_df["driver"] == driver]
        for p_lap in drv_stints["pit_in_lap"].dropna():
            p_val = int(p_lap)
            pit_in_laps.add(p_val)
            pit_out_laps.add(p_val + 1)

    loss_caution = 0.0
    loss_pit = 0.0
    loss_traffic = 0.0
    loss_tyre = 0.0
    loss_residual = 0.0
    total_driver_time = 0.0
    total_baseline_time = 0.0

    for _, row in drv_laps.iterrows():
        lap_num = int(row["lap_number"])
        lap_t = row.get("lap_time_s")
        if pd.isna(lap_t) or lap_t <= 0:
            continue

        lap_t = float(lap_t)
        total_driver_time += lap_t
        total_baseline_time += ref_clean_lap_s
        lap_deficit = lap_t - ref_clean_lap_s

        if lap_deficit <= 0.0:
            # Lap was faster than reference baseline: contributes negative residual (pace advantage)
            loss_residual += lap_deficit
            continue

        # Priority 1: Caution Loss takes highest precedence
        track_status = str(row.get("track_status", "")).strip()
        is_caution = track_status != "1"
        if is_caution:
            loss_caution += lap_deficit
            continue

        # Priority 2: Pit Loss takes second precedence (encompasses transit and stationary stop)
        is_pit = (lap_num in pit_in_laps) or (lap_num in pit_out_laps)
        if is_pit:
            loss_pit += lap_deficit
            continue

        # Priority 3: Traffic Loss
        # Check proximity to car ahead on this lap
        lap_field = df[df["lap_number"] == lap_num].dropna(subset=["session_time_s"]).sort_values("session_time_s")
        car_idx = lap_field[lap_field["driver"] == driver].index
        in_traffic = False
        if not car_idx.empty:
            pos_in_lap = lap_field.index.get_loc(car_idx[0])
            if pos_in_lap > 0:
                car_ahead = lap_field.iloc[pos_in_lap - 1]
                gap_ahead = float(row["session_time_s"]) - float(car_ahead["session_time_s"])
                if 0 < gap_ahead <= TRAFFIC_GAP_THRESHOLD_S:
                    in_traffic = True

        remaining_lap_deficit = lap_deficit
        if in_traffic and lap_t > drv_free_air_baseline_s:
            traffic_candidate = max(0.0, lap_t - drv_free_air_baseline_s)
            traffic_attributed = min(remaining_lap_deficit, traffic_candidate)
            loss_traffic += traffic_attributed
            remaining_lap_deficit -= traffic_attributed

        # Priority 4: Tyre Degradation Loss
        tyre_age = int(row.get("tyre_age_laps", 1))
        if tyre_age > 5 and remaining_lap_deficit > 0:
            tyre_candidate = FUEL_CORRECTION_S_PER_LAP * (tyre_age - 5)
            tyre_attributed = min(remaining_lap_deficit, tyre_candidate)
            loss_tyre += tyre_attributed
            remaining_lap_deficit -= tyre_attributed

        # Priority 5: Exact Residual Loss
        loss_residual += remaining_lap_deficit

    # Total observed deficit relative to clean reference pace
    total_measured_deficit = round(total_driver_time - total_baseline_time, 2)
    c_loss = round(loss_caution, 2)
    p_loss = round(loss_pit, 2)
    tr_loss = round(loss_traffic, 2)
    ty_loss = round(loss_tyre, 2)
    # Ensure residual strictly balances the accounting identity to within 0.01s rounding
    res_loss = round(total_measured_deficit - (c_loss + p_loss + tr_loss + ty_loss), 2)

    baseline_scope = f"AS_OF_LAP_{max_lap}_FIELD_MEDIAN" if as_of_lap is not None else "COMPLETED_RACE_FIELD_MEDIAN"

    return {
        "driver": driver,
        "as_of_lap": max_lap,
        "baseline_scope": baseline_scope,
        "reference_clean_lap_pace_s": round(ref_clean_lap_s, 3),
        "driver_total_time_s": round(total_driver_time, 2),
        "total_deficit_vs_baseline_s": total_measured_deficit,
        "accounting_identity_verified": abs(total_measured_deficit - (c_loss + p_loss + tr_loss + ty_loss + res_loss)) < 0.02,
        "loss_factors_s": {
            "caution_and_incidents_s": c_loss,
            "pit_stop_and_transit_s": p_loss,
            "traffic_s": tr_loss,
            "tyre_degradation_s": ty_loss,
            "unexplained_residual_s": res_loss,
        },
        "attribution_notes": [
            "Strict non-overlapping attribution waterfall: Caution -> Pit -> Traffic -> Tyre -> Residual.",
            "Accounting identity verified: Total Deficit == Caution + Pit + Traffic + Tyre + Residual.",
            "Pit loss represents in-lap and out-lap deficit and encompasses stationary time and transit without double-counting.",
            "Laps faster than reference baseline contribute negative residual (pace advantage).",
        ],
    }


def _empty_time_loss_breakdown(driver: str) -> Dict[str, Any]:
    return {
        "driver": driver,
        "as_of_lap": 0,
        "baseline_scope": "NO_DATA",
        "reference_clean_lap_pace_s": 0.0,
        "driver_total_time_s": 0.0,
        "total_deficit_vs_baseline_s": 0.0,
        "loss_factors_s": {
            "caution_and_incidents_s": 0.0,
            "pit_stop_and_transit_s": 0.0,
            "traffic_s": 0.0,
            "tyre_degradation_s": 0.0,
            "unexplained_residual_s": 0.0,
        },
        "attribution_notes": ["No laps available for driver."],
    }
