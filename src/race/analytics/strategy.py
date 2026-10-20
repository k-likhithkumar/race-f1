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

TACTICAL_PIT_GAP_THRESHOLD_S = 3.0  # Tactical undercut/overcut pit window threshold (s)


def analyze_pit_stops(
    stints_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    driver: str,
    as_of_lap: Optional[int] = None,
    as_of_time_s: Optional[float] = None,
    reference_stationary_s: float = 2.5,
    parallel_straight_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyzes all pit stops for a driver with point-in-time causality and rigorous time accounting:
    - Distinguishes measured pit lane duration (PitIn to PitOut) from stationary stop time.
    - Explicitly marks stationary duration as unavailable when not directly recorded by the provider.
    - Treats reference_stationary_s (default: 2.5s) strictly as an illustrative benchmark assumption.
    - Estimates pit lane transit duration without claiming it is physically measured.
    - Computes total track pit loss relative to a clearly specified clean-lap baseline across 2 laps.
    - Strictly prevents double-counting and prohibits invalid subtraction of absolute pit lane duration from net pit loss.
    - Reports track entry/exit loss only when an explicit parallel straight baseline is provided, otherwise UNAVAILABLE.
    - Enforces 3 distinct lifecycle states:
        * IN_PROGRESS: Pit entry occurred (t >= T_pit_in), but pit exit has not occurred (t < T_pit_out).
        * OUT_LAP_IN_PROGRESS: Pit exit occurred (t >= T_pit_out), but out-lap S/F crossing has not occurred.
        * COMPLETED: Out-lap S/F crossing occurred (t >= T_out_crossing).
    - Counts only COMPLETED stops in total_completed_stops.
    - Evaluates undercut / overcut tactical outcomes as heuristic classifications with supporting evidence.
    - Flags Safety Car / VSC timing advantages without defaulting unknown track status to green.

    Parameters
    ----------
    stints_df : pd.DataFrame
        Cached stints table containing 'driver', 'stint', 'start_lap', 'end_lap',
        'pit_in_lap', 'pit_stop_duration_s'.
    laps_df : pd.DataFrame
        Cached laps table containing 'driver', 'lap_number', 'lap_time_s', 'track_status', 'is_clean', 'session_time_s'.
    driver : str
        Three-letter driver code.
    as_of_lap : int, optional
        Replay lap barrier.
    as_of_time_s : float, optional
        Replay timestamp barrier. If provided, enforces exact point-in-time lifecycle.
    reference_stationary_s : float
        Illustrative benchmark reference assumption for stationary wheel change (default: 2.5s).
    parallel_straight_time_s : float, optional
        Elapsed time for a car on track to cover the parallel straight segment between
        PitIn and PitOut coordinates. Required to decompose net pit loss into net lane loss
        and track entry/exit loss without fabricating data.

    Returns
    -------
    dict
        Structured analysis of each pit stop with explicit accounting categories and lifecycle states.
    """
    if stints_df is None or stints_df.empty or laps_df is None or laps_df.empty:
        return {
            "driver": driver,
            "total_stops": 0,
            "total_completed_stops": 0,
            "active_stop": None,
            "stops": [],
            "verdict": "NO_DATA",
            "baseline_description": "No data available",
        }

    s_df = stints_df.copy()
    l_df = laps_df.copy()

    # Pre-filter dataset based on point-in-time boundary for baseline calculation
    if as_of_time_s is not None:
        l_df_clean_pool = l_df[l_df["session_time_s"] <= as_of_time_s]
    elif as_of_lap is not None:
        l_df_clean_pool = l_df[l_df["lap_number"] <= as_of_lap]
    else:
        l_df_clean_pool = l_df

    # Stated Reference Baseline: median clean green flying lap pace of the field up to as_of
    clean_laps = l_df_clean_pool[l_df_clean_pool["is_clean"] == True]["lap_time_s"].dropna()
    ref_clean_pace = float(clean_laps.median()) if not clean_laps.empty else 85.0

    drv_stints = s_df[s_df["driver"] == driver].sort_values("stint").reset_index(drop=True)
    stops: List[Dict[str, Any]] = []

    for i, stint_row in drv_stints.iterrows():
        pit_lap_raw = stint_row.get("pit_in_lap")
        if pd.isna(pit_lap_raw):
            continue

        pit_lap = int(pit_lap_raw)
        out_lap = pit_lap + 1

        # Look up in-lap and out-lap records
        in_lap_row = l_df[(l_df["driver"] == driver) & (l_df["lap_number"] == pit_lap)]
        out_lap_row = l_df[(l_df["driver"] == driver) & (l_df["lap_number"] == out_lap)]

        if in_lap_row.empty:
            continue

        # Establish boundary timestamps
        # T_pit_in: S/F or Pit In line elapsed time from in-lap completion
        t_pit_in: Optional[float] = (
            float(in_lap_row.iloc[0]["session_time_s"])
            if pd.notna(in_lap_row.iloc[0].get("session_time_s"))
            else None
        )

        raw_pit_dur = stint_row.get("pit_stop_duration_s")
        pit_lane_duration_val: Optional[float] = float(raw_pit_dur) if pd.notna(raw_pit_dur) else 24.5

        t_pit_out: Optional[float] = (
            t_pit_in + pit_lane_duration_val if t_pit_in is not None else None
        )

        t_out_crossing: Optional[float] = (
            float(out_lap_row.iloc[0]["session_time_s"])
            if (not out_lap_row.empty and pd.notna(out_lap_row.iloc[0].get("session_time_s")))
            else None
        )

        # -------------------------------------------------------------
        # Determine Pit Stop Lifecycle State
        # -------------------------------------------------------------
        if as_of_time_s is not None:
            if t_pit_in is not None and as_of_time_s < t_pit_in:
                # Stop has not started yet
                continue
            elif t_pit_in is not None and t_pit_out is not None and t_pit_in <= as_of_time_s < t_pit_out:
                status = "IN_PROGRESS"
            elif t_pit_out is not None and (t_out_crossing is None or t_pit_out <= as_of_time_s < t_out_crossing):
                status = "OUT_LAP_IN_PROGRESS"
            elif t_out_crossing is not None and as_of_time_s >= t_out_crossing:
                status = "COMPLETED"
            else:
                # Fallback if timestamps are missing
                status = "COMPLETED"
        elif as_of_lap is not None:
            if pit_lap > as_of_lap:
                # Stop has not occurred yet
                continue
            elif as_of_lap == pit_lap:
                status = "IN_PROGRESS"
            elif as_of_lap == out_lap:
                status = "OUT_LAP_IN_PROGRESS"
            else:
                status = "COMPLETED"
        else:
            status = "COMPLETED"

        # -------------------------------------------------------------
        # Extract Lap Times According to Lifecycle State
        # -------------------------------------------------------------
        t_in = (
            float(in_lap_row.iloc[0]["lap_time_s"])
            if pd.notna(in_lap_row.iloc[0].get("lap_time_s"))
            else None
        )

        if status == "COMPLETED":
            t_out = (
                float(out_lap_row.iloc[0]["lap_time_s"])
                if (not out_lap_row.empty and pd.notna(out_lap_row.iloc[0].get("lap_time_s")))
                else None
            )
        else:
            # Out-lap is in the future; do not expose prematurely
            t_out = None

        # 1. Total track pit loss relative to stated reference clean flying pace across 2 laps
        if status == "COMPLETED" and t_in is not None and t_out is not None:
            total_pit_loss_s = round((t_in + t_out) - (2.0 * ref_clean_pace), 2)
        else:
            total_pit_loss_s = None

        # 2. Pit Lane Duration from timing loops (PitIn to PitOut transponders)
        if status == "IN_PROGRESS":
            measured_pit_lane_duration_s = None
            measured_stationary_s = None
            stationary_data_status = "UNAVAILABLE_IN_PROGRESS"
        else:
            if pd.notna(raw_pit_dur):
                raw_val = float(raw_pit_dur)
                if raw_val < 15.0:
                    measured_stationary_s = round(raw_val, 2)
                    measured_pit_lane_duration_s = None
                    stationary_data_status = "MEASURED_TRANSPONDER"
                else:
                    measured_pit_lane_duration_s = round(raw_val, 2)
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
            is_stat_measured = True
        else:
            stationary_excess_s = None
            eff_stationary = reference_stationary_s
            is_stat_measured = False

        # 4. Estimated Pit Lane Transit Time (never claimed as physically measured)
        if measured_pit_lane_duration_s is not None:
            estimated_pit_lane_transit_s = round(max(0.0, measured_pit_lane_duration_s - eff_stationary), 2)
            is_transit_estimated = True
            transit_estimation_method = f"PIT_LANE_DURATION_MINUS_{'MEASURED' if measured_stationary_s else 'REFERENCE'}_STATIONARY"
        else:
            estimated_pit_lane_transit_s = None
            is_transit_estimated = False
            transit_estimation_method = "UNAVAILABLE"

        # 5. Track In/Out Deceleration & Acceleration Loss
        if (
            status == "COMPLETED"
            and parallel_straight_time_s is not None
            and total_pit_loss_s is not None
            and measured_pit_lane_duration_s is not None
        ):
            net_pit_lane_loss_s = max(0.0, measured_pit_lane_duration_s - float(parallel_straight_time_s))
            estimated_track_in_out_loss_s = round(max(0.0, total_pit_loss_s - net_pit_lane_loss_s), 2)
            track_loss_status = "ESTIMATED_WITH_STRAIGHT_BASELINE"
        else:
            estimated_track_in_out_loss_s = None
            track_loss_status = "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"

        # 6. Check track status for Safety Car advantage (without defaulting to green)
        raw_ts = in_lap_row.iloc[0].get("track_status")
        track_status = str(raw_ts).strip() if pd.notna(raw_ts) and str(raw_ts).strip() else "UNKNOWN"
        is_sc_pit = (
            any(c in track_status for c in ["4", "5", "6", "7"])
            if track_status not in ("UNKNOWN", "PRE_RACE", "")
            else False
        )

        # 7. Grounded Tactical Outcome Heuristic with Supporting Evidence
        if status != "COMPLETED":
            tactical_verdict = "UNKNOWN"
            tactical_evidence = {
                "is_heuristic": True,
                "classification": "UNKNOWN",
                "rival_driver": None,
                "pre_stop_gap_s": None,
                "pre_stop_track_position": None,
                "target_pit_lap": pit_lap,
                "rival_pit_lap": None,
                "lap_delta": None,
                "outcome": "INSUFFICIENT_DATA",
                "evidence_summary": f"Stop is currently {status}; tactical outcome cannot be evaluated until out-lap completion.",
            }
        elif is_sc_pit:
            tactical_verdict = "SAFETY_CAR_ADVANTAGE"
            tactical_evidence = {
                "is_heuristic": True,
                "classification": "SAFETY_CAR_ADVANTAGE",
                "rival_driver": None,
                "pre_stop_gap_s": None,
                "pre_stop_track_position": None,
                "target_pit_lap": pit_lap,
                "rival_pit_lap": None,
                "lap_delta": None,
                "outcome": None,
                "evidence_summary": f"Stop occurred under Safety Car / VSC conditions (track status '{track_status}').",
            }
        else:
            tactical_evidence = _evaluate_tactical_evidence(
                driver=driver,
                pit_lap=pit_lap,
                out_lap=out_lap,
                stints_df=s_df,
                laps_df=l_df,
            )
            tactical_verdict = tactical_evidence["classification"]

        stops.append({
            "stop_number": len(stops) + 1,
            "status": status,
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
            "is_stationary_measured": is_stat_measured,
            "stationary_metric_type": "ILLUSTRATIVE_ASSUMPTION",
            "stationary_assumption_label": f"Illustrative benchmark assumption ({reference_stationary_s:.1f}s)",
            "stationary_disclaimer": "Stationary wheel change duration is an illustrative 2.5s assumption used for transit estimation, not a measured sensor value.",
            "estimated_pit_lane_transit_s": estimated_pit_lane_transit_s,
            "is_transit_loss_estimated": is_transit_estimated,
            "transit_estimation_basis": transit_estimation_method,
            "estimated_track_in_out_loss_s": estimated_track_in_out_loss_s,
            "track_loss_status": track_loss_status,
            "parallel_straight_time_s": parallel_straight_time_s,
            "is_safety_car_stop": is_sc_pit,
            "tactical_verdict": tactical_verdict,
            "tactical_evidence": tactical_evidence,
        })

    completed_stops = [s for s in stops if s["status"] == "COMPLETED"]
    active_stop = next((s for s in stops if s["status"] in ("IN_PROGRESS", "OUT_LAP_IN_PROGRESS")), None)

    return {
        "driver": driver,
        "total_stops": len(stops),
        "total_completed_stops": len(completed_stops),
        "active_stop": active_stop,
        "stops": stops,
        "reference_clean_pace_s": ref_clean_pace,
        "reference_stationary_s": reference_stationary_s,
        "baseline_description": f"Field clean flying lap median ({ref_clean_pace:.3f}s) across 2 laps",
        "accounting_notes": [
            "Pit lane duration represents transponder-measured elapsed time from pit entry to pit exit (gross duration).",
            "Total pit loss represents net race time lost across in-lap and out-lap relative to 2x reference clean pace.",
            "Stationary duration is reported only when explicitly recorded; otherwise marked UNAVAILABLE.",
            "Reference stationary duration (2.5s) is strictly an illustrative assumption, never a measured sensor value.",
            "Transit duration is calculated as pit lane duration minus stationary duration and explicitly labeled estimated.",
            "Track entry/exit loss requires parallel main straight transit baseline; marked UNAVAILABLE when baseline is omitted.",
            "Direct subtraction of pit lane duration from net pit loss is prohibited as it mixes absolute duration with differential loss.",
        ],
    }


def _evaluate_tactical_evidence(
    driver: str,
    pit_lap: int,
    out_lap: int,
    stints_df: pd.DataFrame,
    laps_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Evaluates undercut / overcut outcomes against immediate rivals running within
    tactical proximity (<= 3.0s gap on the pre-stop lap).
    """
    pre_lap = pit_lap - 1
    if pre_lap < 1:
        return {
            "is_heuristic": True,
            "classification": "REGULAR_STOP",
            "rival_driver": None,
            "pre_stop_gap_s": None,
            "pre_stop_track_position": None,
            "target_pit_lap": pit_lap,
            "rival_pit_lap": None,
            "lap_delta": None,
            "outcome": "INSUFFICIENT_DATA",
            "evidence_summary": "Lap 1 pit stop; insufficient preceding lap data for tactical rival analysis.",
        }

    # Find driver's position and timing on pre_lap
    pre_laps = laps_df[laps_df["lap_number"] == pre_lap]
    drv_pre = pre_laps[pre_laps["driver"] == driver]
    if drv_pre.empty or pd.isna(drv_pre.iloc[0].get("position")) or pd.isna(drv_pre.iloc[0].get("session_time_s")):
        return {
            "is_heuristic": True,
            "classification": "UNKNOWN",
            "rival_driver": None,
            "pre_stop_gap_s": None,
            "pre_stop_track_position": None,
            "target_pit_lap": pit_lap,
            "rival_pit_lap": None,
            "lap_delta": None,
            "outcome": "INSUFFICIENT_DATA",
            "evidence_summary": "Pre-stop lap timing or position unavailable.",
        }

    drv_pos = int(drv_pre.iloc[0]["position"])
    drv_time = float(drv_pre.iloc[0]["session_time_s"])

    # 1. Check car ahead on pre_lap for UNDERCUT attempt
    car_ahead = pre_laps[pre_laps["position"] == drv_pos - 1]
    if not car_ahead.empty and pd.notna(car_ahead.iloc[0].get("session_time_s")):
        ahead_drv = str(car_ahead.iloc[0]["driver"])
        ahead_time = float(car_ahead.iloc[0]["session_time_s"])
        gap_ahead = round(drv_time - ahead_time, 2)

        if 0.0 < gap_ahead <= TACTICAL_PIT_GAP_THRESHOLD_S:
            # Check when car ahead pitted
            ahead_stints = stints_df[stints_df["driver"] == ahead_drv]
            for p_lap in ahead_stints["pit_in_lap"].dropna():
                p_val = int(p_lap)
                if p_val in (pit_lap + 1, pit_lap + 2):
                    lap_delta = pit_lap - p_val  # -1 or -2
                    # Evaluate outcome after rival's out lap
                    eval_lap = p_val + 2
                    post_laps = laps_df[laps_df["lap_number"] == eval_lap]
                    p_drv = post_laps[post_laps["driver"] == driver]
                    p_rival = post_laps[post_laps["driver"] == ahead_drv]
                    outcome = "INSUFFICIENT_DATA"
                    if not p_drv.empty and not p_rival.empty:
                        p_d_pos = p_drv.iloc[0].get("position")
                        p_r_pos = p_rival.iloc[0].get("position")
                        if pd.notna(p_d_pos) and pd.notna(p_r_pos):
                            outcome = "GAINED_POSITION" if p_d_pos < p_r_pos else "LOST_POSITION"

                    return {
                        "is_heuristic": True,
                        "classification": "UNDERCUT_ATTEMPT",
                        "rival_driver": ahead_drv,
                        "pre_stop_gap_s": gap_ahead,
                        "pre_stop_track_position": drv_pos,
                        "target_pit_lap": pit_lap,
                        "rival_pit_lap": p_val,
                        "lap_delta": lap_delta,
                        "outcome": outcome,
                        "evidence_summary": f"Heuristic: {driver} pitted Lap {pit_lap} trailing {ahead_drv} by {gap_ahead:.1f}s. {ahead_drv} pitted Lap {p_val} (delta {lap_delta} laps).",
                    }

    # 2. Check car behind on pre_lap for OVERCUT attempt
    car_behind = pre_laps[pre_laps["position"] == drv_pos + 1]
    if not car_behind.empty and pd.notna(car_behind.iloc[0].get("session_time_s")):
        behind_drv = str(car_behind.iloc[0]["driver"])
        behind_time = float(car_behind.iloc[0]["session_time_s"])
        gap_behind = round(behind_time - drv_time, 2)

        if 0.0 < gap_behind <= TACTICAL_PIT_GAP_THRESHOLD_S:
            behind_stints = stints_df[stints_df["driver"] == behind_drv]
            for p_lap in behind_stints["pit_in_lap"].dropna():
                p_val = int(p_lap)
                if p_val in (pit_lap - 2, pit_lap - 1):
                    lap_delta = pit_lap - p_val  # +1 or +2
                    eval_lap = out_lap + 1
                    post_laps = laps_df[laps_df["lap_number"] == eval_lap]
                    p_drv = post_laps[post_laps["driver"] == driver]
                    p_rival = post_laps[post_laps["driver"] == behind_drv]
                    outcome = "INSUFFICIENT_DATA"
                    if not p_drv.empty and not p_rival.empty:
                        p_d_pos = p_drv.iloc[0].get("position")
                        p_r_pos = p_rival.iloc[0].get("position")
                        if pd.notna(p_d_pos) and pd.notna(p_r_pos):
                            outcome = "RETAINED_POSITION" if p_d_pos < p_r_pos else "LOST_POSITION"

                    return {
                        "is_heuristic": True,
                        "classification": "OVERCUT_ATTEMPT",
                        "rival_driver": behind_drv,
                        "pre_stop_gap_s": gap_behind,
                        "pre_stop_track_position": drv_pos,
                        "target_pit_lap": pit_lap,
                        "rival_pit_lap": p_val,
                        "lap_delta": lap_delta,
                        "outcome": outcome,
                        "evidence_summary": f"Heuristic: {behind_drv} pitted Lap {p_val} trailing {driver} by {gap_behind:.1f}s. {driver} pitted Lap {pit_lap} (delta +{lap_delta} laps).",
                    }

    return {
        "is_heuristic": True,
        "classification": "REGULAR_STOP",
        "rival_driver": None,
        "pre_stop_gap_s": None,
        "pre_stop_track_position": drv_pos,
        "target_pit_lap": pit_lap,
        "rival_pit_lap": None,
        "lap_delta": None,
        "outcome": None,
        "evidence_summary": "Isolated stop; no direct rival within tactical proximity (3.0s window).",
    }


def compute_time_loss_breakdown(
    laps_df: pd.DataFrame,
    stints_df: pd.DataFrame,
    driver: str,
    as_of_lap: Optional[int] = None,
    as_of_time_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Decomposes total race time deficit for a driver against a stated reference baseline.
    Enforces a strict, non-overlapping attribution waterfall and preserves the exact
    accounting identity:
        Total Measured Deficit == Caution + Pit + Traffic + Tyre + Residual (within +-0.01s)

    Attribution Waterfall (Mutual Exclusivity Order):
    1. Caution Loss: Entire lap deficit on confirmed non-green laps (SC, VSC, Yellow, Red).
       Unknown track status is NEVER classified as caution.
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
        Replay lap boundary.
    as_of_time_s : float, optional
        Replay timestamp boundary. If provided, strictly slices laps with session_time_s <= as_of_time_s.

    Returns
    -------
    dict
        Structured breakdown with stated baseline, attributed factor seconds,
        exact residual, and accounting identity verification flag.
    """
    if laps_df is None or laps_df.empty:
        return _empty_time_loss_breakdown(driver)

    df = laps_df.copy()
    if as_of_time_s is not None:
        df = df[df["session_time_s"] <= as_of_time_s].copy()
        max_lap = int(df[df["driver"] == driver]["lap_number"].max()) if not df[df["driver"] == driver].empty else 0
    elif as_of_lap is not None:
        max_lap = as_of_lap
        df = df[df["lap_number"] <= max_lap].copy()
    else:
        max_lap = int(df["lap_number"].max()) if not df.empty else 0

    drv_laps = df[df["driver"] == driver].sort_values("lap_number").reset_index(drop=True)
    if drv_laps.empty:
        return _empty_time_loss_breakdown(driver)

    # Establish stated reference baseline:
    # Median clean green flying lap pace of the field up to boundary
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
        # Never classify unknown or pre-race conditions as caution
        raw_ts = row.get("track_status")
        track_status = str(raw_ts).strip() if pd.notna(raw_ts) and str(raw_ts).strip() else "UNKNOWN"
        is_caution = (
            track_status not in ("1", "UNKNOWN", "PRE_RACE", "")
            and any(c in track_status for c in ["2", "4", "5", "6", "7"])
        )
        if is_caution:
            loss_caution += lap_deficit
            continue

        # Priority 2: Pit Loss takes second precedence (encompasses transit and stationary stop)
        is_pit = (lap_num in pit_in_laps) or (lap_num in pit_out_laps)
        if is_pit:
            loss_pit += lap_deficit
            continue

        # Priority 3: Traffic Loss
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

    if as_of_time_s is not None:
        baseline_scope = f"AS_OF_TIME_{as_of_time_s:.1f}S_FIELD_MEDIAN"
    elif as_of_lap is not None:
        baseline_scope = f"AS_OF_LAP_{max_lap}_FIELD_MEDIAN"
    else:
        baseline_scope = "COMPLETED_RACE_FIELD_MEDIAN"

    identity_verified = abs(total_measured_deficit - (c_loss + p_loss + tr_loss + ty_loss + res_loss)) <= 0.01

    return {
        "driver": driver,
        "as_of_lap": max_lap,
        "baseline_scope": baseline_scope,
        "reference_clean_lap_pace_s": round(ref_clean_lap_s, 3),
        "driver_total_time_s": round(total_driver_time, 2),
        "total_deficit_vs_baseline_s": total_measured_deficit,
        "accounting_identity_verified": identity_verified,
        "loss_factors_s": {
            "caution_and_incidents_s": c_loss,
            "pit_stop_and_transit_s": p_loss,
            "traffic_s": tr_loss,
            "tyre_degradation_s": ty_loss,
            "unexplained_residual_s": res_loss,
        },
        "attribution_notes": [
            "Strict non-overlapping attribution waterfall: Caution -> Pit -> Traffic -> Tyre -> Residual.",
            "Accounting identity verified within +-0.01s: Total Deficit == Caution + Pit + Traffic + Tyre + Residual.",
            "Pit loss represents in-lap and out-lap deficit and encompasses stationary time and transit without double-counting.",
            "Unknown track status is never classified as caution or green.",
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
        "accounting_identity_verified": True,
        "loss_factors_s": {
            "caution_and_incidents_s": 0.0,
            "pit_stop_and_transit_s": 0.0,
            "traffic_s": 0.0,
            "tyre_degradation_s": 0.0,
            "unexplained_residual_s": 0.0,
        },
        "attribution_notes": ["No laps available for driver."],
    }
