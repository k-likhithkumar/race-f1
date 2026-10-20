"""
RaceState assembly module for RACE (Phase 3A).
Constructs causality-preserving, point-in-time race state snapshots
with separate recorded timing order and estimated spatial position.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel

from src.race.replay.session import ReplaySession

logger = logging.getLogger(__name__)

# Heuristic lateral distance threshold from track centerline (meters)
HEURISTIC_OFF_TRACK_DISTANCE_M = 35.0


class DriverState(BaseModel):
    driver: str
    team: str

    # 1. Official Recorded Timing Order (from transponder crossing)
    timing_position: int
    last_completed_lap: int
    last_lap_time_s: Optional[float] = None
    last_crossing_time_s: Optional[float] = None
    timing_gap_to_leader_s: Optional[float] = None
    timing_interval_s: Optional[float] = None
    recorded_gap_age_s: Optional[float] = None

    # 2. Instantaneous Spatial State (Mid-Lap Estimate)
    spatial_position: Optional[int] = None
    is_spatial_estimate: bool = True
    spatial_projection_valid: bool = False
    track_distance_m: Optional[float] = None
    total_race_distance_m: Optional[float] = None
    x: Optional[float] = None
    y: Optional[float] = None
    estimated_gap_to_leader_s: Optional[float] = None
    estimated_interval_s: Optional[float] = None
    gap_estimate_status: str = "RECORDED_ONLY"

    # 3. Operational & Vehicle Status
    status: str = "ON_TRACK"  # "ON_TRACK", "OFF_TRACK", "STOPPED", "IN_PIT", "TELEMETRY_MISSING", "RETIRED"
    pit_status: str = "ON_TRACK"  # "ON_TRACK", "IN_PIT", "UNKNOWN"
    is_in_lap: bool = False
    is_out_lap: bool = False

    # 4. Tyre & Stint State
    last_confirmed_compound: str = "UNKNOWN"
    compound: str = "UNKNOWN"
    compound_observation_status: str = "UNKNOWN"
    completed_tyre_age_laps: int = 0
    active_tyre_lap: int = 1
    stint: int = 1


class RaceState(BaseModel):
    session_time_s: float
    race_id: str
    track_status: str
    weather: Dict[str, Any]
    leaderboard: List[DriverState]
    spatial_order: List[str]
    active_battles: List[Dict[str, Any]]
    spatial_ordering_valid: bool = True
    spatial_fallback_reason: Optional[str] = None


def get_state(session: ReplaySession, as_of_time_s: float) -> RaceState:
    """
    Construct an immutable, point-in-time RaceState snapshot at as_of_time_s.
    Strictly excludes future information past as_of_time_s.
    """
    t_as_of = float(as_of_time_s)

    # 1. Query latest Weather and TrackStatus up to t_as_of
    weather_dict: Dict[str, Any] = {
        "air_temp_c": None,
        "track_temp_c": None,
        "humidity_pct": None,
        "rainfall": False,
    }
    if t_as_of < session.T_0:
        track_status = "PRE_RACE"
    else:
        track_status = "UNKNOWN"
        if len(session.weather_times) > 0:
            w_idx = np.searchsorted(session.weather_times, t_as_of, side="right") - 1
            if w_idx >= 0:
                w_row = session.sorted_weather_df.iloc[w_idx]
                weather_dict = {
                    "air_temp_c": float(w_row["air_temp_c"]) if pd.notna(w_row["air_temp_c"]) else None,
                    "track_temp_c": float(w_row["track_temp_c"]) if pd.notna(w_row["track_temp_c"]) else None,
                    "humidity_pct": float(w_row["humidity_pct"]) if pd.notna(w_row["humidity_pct"]) else None,
                    "rainfall": bool(w_row["rainfall"]) if pd.notna(w_row["rainfall"]) else False,
                }
                if pd.notna(w_row.get("track_status")) and str(w_row["track_status"]).strip():
                    track_status = str(w_row["track_status"]).strip()

    # 2. Build DriverState for each driver using in-memory pre-indexed lookups
    driver_states: List[DriverState] = []
    spatial_candidates: List[Tuple[str, float]] = []

    for drv in session.drivers:
        team = session.driver_teams.get(drv, "Unknown")
        crossing_times = session.driver_lap_crossing_times.get(drv, np.array([]))
        lap_numbers = session.driver_lap_numbers.get(drv, np.array([]))
        drv_lap_dict = session.driver_lap_dict.get(drv, {})
        pit_set = session.driver_pit_in_laps.get(drv, set())

        # Determine last completed lap strictly at or before t_as_of
        lap_idx = np.searchsorted(crossing_times, t_as_of, side="right") - 1

        last_completed_lap = 0
        last_lap_time = None
        last_crossing_time = None
        timing_pos = 20
        completed_tyre_age = 0
        active_tyre_lap = 1
        stint_num = 1
        last_confirmed_compound = "UNKNOWN"
        compound = "UNKNOWN"
        compound_obs = "UNKNOWN"

        if lap_idx >= 0 and len(lap_numbers) > lap_idx:
            last_completed_lap = int(lap_numbers[lap_idx])
            last_crossing_time = float(crossing_times[lap_idx])
            lap_record = drv_lap_dict.get(last_completed_lap)
            if lap_record is not None:
                last_lap_time = float(lap_record["lap_time_s"]) if pd.notna(lap_record.get("lap_time_s")) else None
                timing_pos = int(lap_record["position"]) if pd.notna(lap_record.get("position")) else 20
                completed_tyre_age = int(lap_record["tyre_age_laps"]) if pd.notna(lap_record.get("tyre_age_laps")) else 0
                stint_num = int(lap_record["stint"]) if pd.notna(lap_record.get("stint")) else 1
                last_confirmed_compound = str(lap_record["compound"]) if pd.notna(lap_record.get("compound")) else "UNKNOWN"
                compound = last_confirmed_compound
                compound_obs = "CONFIRMED"
                active_tyre_lap = completed_tyre_age + 1
        else:
            # Lap 1 ongoing before first line crossing: use starting grid position
            grid_pos = int(session.driver_grid_positions.get(drv, 20))
            timing_pos = grid_pos if grid_pos > 0 else 20
            lap1_record = drv_lap_dict.get(1)
            if lap1_record is not None:
                last_confirmed_compound = str(lap1_record["compound"]) if pd.notna(lap1_record.get("compound")) else "UNKNOWN"
                compound = last_confirmed_compound
                compound_obs = "CONFIRMED"

        # Check in-lap / out-lap flags causally:
        # An in-lap cannot be known before pit entry; on active in-progress lap, default is_in_lap=False
        is_in_lap = False
        is_out_lap = last_completed_lap in pit_set
        pit_status = "IN_PIT" if is_in_lap else "ON_TRACK"

        if is_out_lap:
            # On out-lap, newly fitted compound is pending confirmation from timing transponder
            compound_obs = "UNCONFIRMED_OUT_LAP"
            # Maintain last confirmed compound until transponder confirmation
            compound = last_confirmed_compound

        # 3. Guarded Coordinate Interpolation & Kinematic Motion Evidence
        x_val, y_val = None, None
        is_stationary = False
        pos_t = session.pos_times.get(drv)
        pos_x = session.pos_x.get(drv)
        pos_y = session.pos_y.get(drv)

        if pos_t is not None and len(pos_t) > 0 and pos_t[0] <= t_as_of <= pos_t[-1]:
            p_idx = np.searchsorted(pos_t, t_as_of)
            if p_idx < len(pos_t) and pos_t[p_idx] == t_as_of:
                x_cand, y_cand = pos_x[p_idx], pos_y[p_idx]
                if not (np.isnan(x_cand) or np.isnan(y_cand)):
                    x_val, y_val = float(x_cand), float(y_cand)
            elif p_idx > 0:
                t0, t1 = pos_t[p_idx - 1], pos_t[p_idx]
                dt = t1 - t0
                x0, x1 = pos_x[p_idx - 1], pos_x[p_idx]
                y0, y1 = pos_y[p_idx - 1], pos_y[p_idx]

                if dt <= 2.0 and not (np.isnan(x0) or np.isnan(x1) or np.isnan(y0) or np.isnan(y1)):
                    disp = np.hypot(x1 - x0, y1 - y0)
                    if disp <= 200.0:
                        alpha = (t_as_of - t0) / dt
                        x_val = float(x0 + alpha * (x1 - x0))
                        y_val = float(y0 + alpha * (y1 - y0))

            # Kinematic motion evaluation over previous 3.0 seconds
            if x_val is not None and y_val is not None:
                t_window_start = t_as_of - 3.0
                p_win_idx = np.searchsorted(pos_t, t_window_start)
                if p_idx > p_win_idx and p_win_idx < len(pos_t):
                    x_prev, y_prev = pos_x[p_win_idx], pos_y[p_win_idx]
                    if not (np.isnan(x_prev) or np.isnan(y_prev)):
                        disp_window = np.hypot(x_val - x_prev, y_val - y_prev)
                        if disp_window <= 2.0:
                            is_stationary = True

        # 4. Track Outline Projection & Operational Vehicle Status
        track_dist_m = None
        total_dist_m = None
        spatial_valid = False
        drv_status = "ON_TRACK"
        norm_x, norm_y = None, None

        if x_val is not None and y_val is not None:
            norm_x = x_val - session.coord_offset_x
            norm_y = y_val - session.coord_offset_y

            dx = norm_x - session.track_x
            dy = norm_y - session.track_y
            dist_sq = dx * dx + dy * dy
            min_idx = int(np.argmin(dist_sq))
            min_dist = float(np.sqrt(dist_sq[min_idx]))

            if min_dist <= HEURISTIC_OFF_TRACK_DISTANCE_M:
                track_dist_m = float(session.track_dist[min_idx])
                total_dist_m = (last_completed_lap * session.total_circuit_distance_m) + track_dist_m
                spatial_valid = True
                if is_stationary:
                    drv_status = "STOPPED"
                else:
                    drv_status = "ON_TRACK"
                spatial_candidates.append((drv, total_dist_m))
            else:
                spatial_valid = False
                if is_stationary:
                    drv_status = "STOPPED"
                else:
                    drv_status = "OFF_TRACK"
        else:
            c_status = session.driver_classified_status.get(drv, "")
            is_dns = c_status.lower() in ["did not start", "dns"]
            if is_dns and t_as_of > session.T_0:
                drv_status = "RETIRED"
            elif t_as_of >= session.T_finish:
                if c_status not in ["Finished", "+1 Lap", "+2 Laps"]:
                    drv_status = "RETIRED"
                else:
                    drv_status = "TELEMETRY_MISSING"
            else:
                drv_status = "TELEMETRY_MISSING"

        recorded_age = (t_as_of - last_crossing_time) if last_crossing_time is not None else None

        d_state = DriverState(
            driver=drv,
            team=team,
            timing_position=timing_pos,
            last_completed_lap=last_completed_lap,
            last_lap_time_s=last_lap_time,
            last_crossing_time_s=last_crossing_time,
            recorded_gap_age_s=round(recorded_age, 3) if recorded_age is not None else None,
            spatial_position=None,
            is_spatial_estimate=True,
            spatial_projection_valid=spatial_valid,
            track_distance_m=round(track_dist_m, 2) if track_dist_m is not None else None,
            total_race_distance_m=round(total_dist_m, 2) if total_dist_m is not None else None,
            x=round(norm_x, 2) if norm_x is not None else None,
            y=round(norm_y, 2) if norm_y is not None else None,
            status=drv_status,
            pit_status=pit_status,
            is_in_lap=is_in_lap,
            is_out_lap=is_out_lap,
            last_confirmed_compound=last_confirmed_compound,
            compound=compound,
            compound_observation_status=compound_obs,
            completed_tyre_age_laps=completed_tyre_age,
            active_tyre_lap=active_tyre_lap,
            stint=stint_num,
        )
        driver_states.append(d_state)

    # 5. Assign Estimated Spatial Positions (Strictly for valid projections on racing circuit)
    spatial_candidates.sort(key=lambda item: item[1], reverse=True)
    spatial_order = [item[0] for item in spatial_candidates]

    driver_map = {s.driver: s for s in driver_states}
    for rank, (d_code, _) in enumerate(spatial_candidates, start=1):
        driver_map[d_code].spatial_position = rank

    # 6. Authoritative Multi-Key Leaderboard Sorting
    def leaderboard_sort_key(s: DriverState):
        grid_pos = session.driver_grid_positions.get(s.driver, 20)
        grid_pos = grid_pos if grid_pos > 0 else 20

        # Before race start, strictly preserve starting grid order
        if t_as_of <= session.T_0:
            return (0, grid_pos)

        # In-race or post-race: active cars ahead of retired cars
        is_retired = 1 if s.status == "RETIRED" else 0
        if is_retired:
            # Retired cars ordered by laps completed desc, then grid position asc
            return (1, -s.last_completed_lap, grid_pos)

        if s.last_completed_lap > 0:
            # Running cars with completed laps: laps completed desc, timing position asc
            return (0, -s.last_completed_lap, s.timing_position)
        else:
            # Running cars on Lap 1: live spatial track position if valid, else starting grid
            lap0_pos = s.spatial_position if (s.spatial_position is not None and s.spatial_projection_valid) else grid_pos
            return (0, 0, lap0_pos)

    driver_states.sort(key=leaderboard_sort_key)

    # Synchronize authoritative timing_position with final leaderboard ranking
    for rank, s in enumerate(driver_states, start=1):
        s.timing_position = rank

    # 7. Calculate recorded timing gaps to leader and interval
    if len(driver_states) > 0:
        leader_state = driver_states[0]
        leader_cross = leader_state.last_crossing_time_s
        leader_lap = leader_state.last_completed_lap

        for i, s in enumerate(driver_states):
            if i == 0:
                s.timing_gap_to_leader_s = 0.0
                s.timing_interval_s = 0.0
            else:
                prev_state = driver_states[i - 1]
                if s.last_completed_lap == leader_lap and leader_cross is not None and s.last_crossing_time_s is not None:
                    s.timing_gap_to_leader_s = round(s.last_crossing_time_s - leader_cross, 3)
                elif s.last_completed_lap < leader_lap:
                    s.timing_gap_to_leader_s = None

                if s.last_completed_lap == prev_state.last_completed_lap and s.last_crossing_time_s is not None and prev_state.last_crossing_time_s is not None:
                    s.timing_interval_s = round(s.last_crossing_time_s - prev_state.last_crossing_time_s, 3)
                else:
                    s.timing_interval_s = None

    # 7. Detect Active Battles (Aligned with Stage 2 detect_battles)
    active_battles: List[Dict[str, Any]] = []
    if track_status == "1":
        for i in range(1, len(driver_states)):
            d_curr = driver_states[i]     # Chaser (behind in timing)
            d_prev = driver_states[i - 1] # Leader (ahead in timing)

            # Both cars must be on the same completed lap and active on track
            if (
                d_curr.last_completed_lap == d_prev.last_completed_lap
                and d_curr.last_completed_lap > 0
                and d_curr.status == "ON_TRACK"
                and d_prev.status == "ON_TRACK"
                and not d_curr.is_in_lap
                and not d_curr.is_out_lap
                and not d_prev.is_in_lap
                and not d_prev.is_out_lap
            ):
                gap_s = d_curr.timing_interval_s
                if gap_s is not None and 0.0 < gap_s <= 1.0:
                    curr_lap = d_curr.last_completed_lap
                    prev_lap_num = curr_lap - 1

                    closing_rate = None
                    verdict = None

                    if prev_lap_num > 0:
                        drv_curr_lap_dict = session.driver_lap_dict.get(d_curr.driver, {})
                        drv_prev_lap_dict = session.driver_lap_dict.get(d_prev.driver, {})
                        prev_curr_rec = drv_curr_lap_dict.get(prev_lap_num)
                        prev_prev_rec = drv_prev_lap_dict.get(prev_lap_num)

                        if prev_curr_rec is not None and prev_prev_rec is not None:
                            pt_curr = prev_curr_rec.get("session_time_s")
                            pt_prev = prev_prev_rec.get("session_time_s")
                            if pd.notna(pt_curr) and pd.notna(pt_prev):
                                prev_gap = round(float(pt_curr) - float(pt_prev), 3)
                                if prev_gap < 0:
                                    verdict = "OVERTAKE_COMPLETED"
                                else:
                                    closing_rate = round(prev_gap - gap_s, 3)
                                    if closing_rate >= 0.05:
                                        verdict = "ACTIVE_CHASE"
                                    elif (prev_gap <= 0.7 and gap_s <= 0.7) or gap_s <= 0.5:
                                        verdict = "SUSTAINED_DEFENSE"

                    if verdict is None and gap_s <= 0.5:
                        verdict = "SUSTAINED_DEFENSE"

                    if verdict is not None:
                        active_battles.append({
                            "leader": d_prev.driver,
                            "chaser": d_curr.driver,
                            "attacker": d_curr.driver,
                            "defender": d_prev.driver,
                            "gap_s": gap_s,
                            "interval_s": gap_s,
                            "closing_rate_s_per_lap": closing_rate,
                            "battle_verdict": verdict,
                            "lap": curr_lap,
                        })

    # 8. Compute Dynamic Spatial Ordering Validity
    active_on_track_count = sum(
        1 for d in driver_states if d.status in ["ON_TRACK", "OFF_TRACK", "TELEMETRY_MISSING"]
    )
    if active_on_track_count > 0:
        valid_ratio = len(spatial_order) / float(active_on_track_count)
        is_spatial_valid = (valid_ratio >= 0.80) and (len(spatial_order) >= 1)
        fallback_reason = None if is_spatial_valid else (
            "TELEMETRY_DROPOUT" if active_on_track_count >= 2 else "INSUFFICIENT_ON_TRACK_CARS"
        )
    else:
        is_spatial_valid = False
        fallback_reason = "NO_ACTIVE_CARS"

    return RaceState(
        session_time_s=round(t_as_of, 3),
        race_id=session.race_id,
        track_status=track_status,
        weather=weather_dict,
        leaderboard=driver_states,
        spatial_order=spatial_order,
        active_battles=active_battles,
        spatial_ordering_valid=is_spatial_valid,
        spatial_fallback_reason=fallback_reason,
    )
