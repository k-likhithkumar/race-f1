"""
Deterministic Tool Registry and tool execution adapters.
Directly wraps SessionManager methods with strict validation, allowlists,
safe JSON serialization, and authoritative as_of enforcement.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel, Field



def _get_sm():
    from src.race.api.session_manager import session_manager
    return session_manager


def sanitize_for_json(val: Any) -> Any:
    """Recursively converts NumPy / Pandas / float NaN values to JSON-safe Python primitives."""
    if val is None:
        return None
    if isinstance(val, (np.ndarray, list)):
        return [sanitize_for_json(item) for item in val]
    if isinstance(val, dict):
        return {str(k): sanitize_for_json(v) for k, v in val.items()}
    if hasattr(val, "model_dump"):
        return sanitize_for_json(val.model_dump())
    if hasattr(val, "dict"):
        return sanitize_for_json(val.dict())
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        if math.isnan(val) or math.isinf(val):
            return None
        return round(float(val), 4)
    # Check pd.isna only for scalars
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


class ToolDefinition(BaseModel):
    """Specification of an allowed deterministic tool."""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolRegistry:
    """
    Registry of approved deterministic tools for the AI Strategy Assistant.
    Enforces allowlist, parameter validation, server-side as_of timestamps,
    and safe serialization.
    """

    def __init__(self, race_id: str, authoritative_as_of: float):
        sm = _get_sm()
        self.race_id = sm.normalize_race_id(race_id)
        self.authoritative_as_of = float(authoritative_as_of)
        self.session = sm.get_or_load_session(self.race_id)

    def get_registered_tool_names(self) -> List[str]:
        return list(self._tool_definitions.keys())

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [t.model_dump() for t in self._tool_definitions.values()]

    def get_gemini_tool_declarations(self) -> List[Dict[str, Any]]:
        """Returns tool declarations formatted for Gemini / Mock LLM providers."""
        return [t.model_dump() for t in self._tool_definitions.values()]

    # -------------------------------------------------------------------------
    # Tool Execution Handlers
    # -------------------------------------------------------------------------

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validates arguments and executes the requested tool."""
        if tool_name not in self._tool_definitions:
            return {
                "status": "ERROR",
                "error": True,
                "code": "UNKNOWN_TOOL",
                "message": f"Unknown tool '{tool_name}'. Allowed: {list(self._tool_definitions.keys())}",
            }

        handler = getattr(self, f"_handle_{tool_name}", None)
        if handler is None:
            return {
                "status": "ERROR",
                "error": True,
                "code": "HANDLER_MISSING",
                "message": f"No implementation found for tool '{tool_name}'.",
            }

        try:
            raw_result = handler(arguments)
            if isinstance(raw_result, dict) and "status" not in raw_result:
                raw_result["status"] = "SUCCESS"
            return sanitize_for_json(raw_result)
        except ValueError as e:
            return {
                "status": "ERROR",
                "error": True,
                "code": "VALIDATION_ERROR",
                "message": str(e),
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "error": True,
                "code": "EXECUTION_ERROR",
                "message": f"Failed executing {tool_name}: {str(e)}",
            }

    # -------------------------------------------------------------------------
    # 1. get_race_leaderboard
    # -------------------------------------------------------------------------
    def _handle_get_race_leaderboard(self, args: Dict[str, Any]) -> Dict[str, Any]:
        sm = _get_sm()
        state = sm.get_public_state(self.race_id, as_of=self.authoritative_as_of)

        leaderboard_summary = []
        for item in (state.leaderboard or []):
            leaderboard_summary.append({
                "position": item.timing_position,
                "driver": item.driver,
                "team": item.team,
                "gap_to_leader_s": item.timing_gap_to_leader_s,
                "interval_s": item.timing_interval_s,
                "tyre_compound": item.compound,
                "tyre_age_laps": item.completed_tyre_age_laps,
                "pit_status": item.pit_status,
                "status": item.status,
                "last_completed_lap": item.last_completed_lap,
            })

        curr_lap = max(1, max((d.last_completed_lap for d in state.leaderboard), default=1)) if state.leaderboard else 1
        weather_info = state.weather if isinstance(state.weather, dict) else {}

        return {
            "race_id": self.race_id,
            "as_of_time_s": self.authoritative_as_of,
            "current_lap": curr_lap,
            "total_laps": self.session.total_laps,
            "track_status": state.track_status,
            "weather": {
                "air_temp_c": weather_info.get("air_temp_c"),
                "track_temp_c": weather_info.get("track_temp_c"),
                "rainfall": weather_info.get("rainfall", False),
            },
            "leaderboard": leaderboard_summary,
        }

    # -------------------------------------------------------------------------
    # 2. compare_driver_laps
    # -------------------------------------------------------------------------
    def _handle_compare_driver_laps(self, args: Dict[str, Any]) -> Dict[str, Any]:
        d1 = str(args.get("driver1") or args.get("driver_a", "")).strip().upper()
        d2 = str(args.get("driver2") or args.get("driver_b", "")).strip().upper()
        mode = str(args.get("mode", "single")).strip().lower()

        if d1 not in self.session.drivers:
            raise ValueError(f"Driver '{d1}' not found in race '{self.race_id}'. Available: {self.session.drivers}")
        if d2 not in self.session.drivers:
            raise ValueError(f"Driver '{d2}' not found in race '{self.race_id}'. Available: {self.session.drivers}")
        if d1 == d2:
            raise ValueError("driver1 and driver2 must be distinct drivers.")

        total_race_laps = self.session.total_laps

        lap_number = args.get("lap_number") if args.get("lap_number") is not None else args.get("lap")
        start_lap = args.get("start_lap")
        end_lap = args.get("end_lap")

        # Point-in-time enforcement: resolve current completed lap
        sm = _get_sm()
        current_state = sm.get_public_state(self.race_id, as_of=self.authoritative_as_of)
        max_allowed_lap = max(1, max((d.last_completed_lap for d in current_state.leaderboard), default=1)) if current_state.leaderboard else 1

        if mode == "single":
            if lap_number is None:
                lap_number = max_allowed_lap
            else:
                lap_number = int(lap_number)
            if lap_number < 1 or lap_number > total_race_laps:
                raise ValueError(f"Lap number {lap_number} out of valid race bounds [1, {total_race_laps}].")
            if lap_number > max_allowed_lap:
                raise ValueError(f"Lap {lap_number} is in the future relative to replay cutoff (current lap: {max_allowed_lap}). Requested lap must be before or at the replay cutoff.")
            start_lap = lap_number
            end_lap = lap_number
        else:
            mode = "range"
            start_lap = int(start_lap) if start_lap is not None else max(1, max_allowed_lap - 5)
            end_lap = int(end_lap) if end_lap is not None else max_allowed_lap
            if start_lap < 1 or end_lap > total_race_laps:
                raise ValueError(f"Lap range [{start_lap}, {end_lap}] exceeds race bounds [1, {total_race_laps}].")
            if end_lap > max_allowed_lap:
                raise ValueError(f"End lap {end_lap} is in the future relative to replay cutoff (current lap: {max_allowed_lap}). Requested lap must be before or at the replay cutoff.")
            if start_lap > end_lap:
                raise ValueError(f"start_lap ({start_lap}) cannot be greater than end_lap ({end_lap}).")

        from src.race.api.schemas import ComparisonRequest
        req = ComparisonRequest(
            drivers=[d1, d2],
            lap_mode=mode,
            lap_number=lap_number if mode == "single" else None,
            lap_range=[start_lap, end_lap] if mode == "range" else None,
            reference_driver=d1,
            clean_only=bool(args.get("clean_only", True)),
            fuel_corrected=bool(args.get("fuel_corrected", True)),
            as_of=self.authoritative_as_of,
        )

        resp = sm.get_driver_comparison(self.race_id, req)

        # Extract delta relative to reference
        d1_laps = [l for l in resp.laps if l.driver == d1 and l.lap_time_s is not None]
        d2_laps = [l for l in resp.laps if l.driver == d2 and l.lap_time_s is not None]
        lap_a_time = d1_laps[0].lap_time_s if d1_laps else None
        lap_b_time = d2_laps[0].lap_time_s if d2_laps else None

        if lap_a_time is not None and lap_b_time is not None:
            mean_delta = round(lap_a_time - lap_b_time, 4)
        else:
            deltas = [l.delta_to_reference_s for l in d2_laps if l.delta_to_reference_s is not None]
            mean_delta = round(float(np.mean(deltas)), 4) if deltas else 0.0

        return {
            "status": "SUCCESS",
            "driver1": d1,
            "driver2": d2,
            "driver_a": d1,
            "driver_b": d2,
            "reference_driver": resp.reference_driver,
            "mode": mode,
            "start_lap": start_lap,
            "end_lap": end_lap,
            "lap_delta_s": mean_delta,
            "mean_delta_s": mean_delta,
            "lap_time_a_s": lap_a_time,
            "lap_time_b_s": lap_b_time,
            "reference_lap_time_s": lap_a_time,
            "d2_lap_time_s": lap_b_time,
            "clean_laps_count": len(d1_laps) + len(d2_laps),
            "fuel_corrected": resp.fuel_corrected,
        }

    # -------------------------------------------------------------------------
    # 3. get_tyre_stint_degradation
    # -------------------------------------------------------------------------
    def _handle_get_tyre_stint_degradation(self, args: Dict[str, Any]) -> Dict[str, Any]:
        driver = str(args.get("driver", "")).strip().upper()
        stint = int(args.get("stint", 1))

        if driver not in self.session.drivers:
            raise ValueError(f"Driver '{driver}' not found in race '{self.race_id}'. Available: {self.session.drivers}")
        if stint < 1:
            raise ValueError("Stint number must be >= 1.")

        sm = _get_sm()
        resp = sm.get_tyre_degradation(
            self.race_id,
            driver=driver,
            stint=stint,
            as_of=self.authoritative_as_of,
        )

        active_stint = resp.active_stint
        return {
            "driver": driver,
            "stint": stint,
            "compound": active_stint.compound if active_stint else "UNKNOWN",
            "slope_s_per_lap": active_stint.slope_s_per_lap if active_stint else None,
            "wear_slope_s_per_lap": active_stint.slope_s_per_lap if active_stint else None,
            "fit_r2": active_stint.r2 if active_stint else None,
            "clean_laps_count": active_stint.clean_lap_count if active_stint else 0,
            "total_stint_laps": active_stint.total_laps if active_stint else 0,
            "fuel_correction_applied_s_per_lap": resp.fuel_correction_applied_s_per_lap,
            "status": resp.status,
        }

    # -------------------------------------------------------------------------
    # 4. get_pit_strategy_and_waterfall
    # -------------------------------------------------------------------------
    def _handle_get_pit_strategy_and_waterfall(self, args: Dict[str, Any]) -> Dict[str, Any]:
        driver = str(args.get("driver", "")).strip().upper()
        ref_stationary = float(args.get("reference_stationary_s", 2.5))

        if driver not in self.session.drivers:
            raise ValueError(f"Driver '{driver}' not found in race '{self.race_id}'. Available: {self.session.drivers}")

        sm = _get_sm()
        resp = sm.get_pit_strategy(
            self.race_id,
            driver=driver,
            as_of=self.authoritative_as_of,
            reference_stationary_s=ref_stationary,
        )

        stops_summary = []
        for s in resp.stops:
            stops_summary.append({
                "stop_number": s.stop_number,
                "status": s.status,
                "in_lap": s.pit_in_lap,
                "out_lap": s.out_lap,
                "measured_pit_lane_duration_s": s.measured_pit_lane_duration_s,
                "reference_stationary_s": s.reference_stationary_s,
                "estimated_total_pit_loss_s": s.estimated_total_pit_loss_s,
                "tactical_verdict": s.tactical_evidence.classification if s.tactical_evidence else "UNKNOWN",
            })

        breakdown = resp.time_loss_breakdown
        factors = breakdown.loss_factors_s if breakdown else None
        net_loss = factors.pit_stop_and_transit_s if factors else 0.0

        return {
            "driver": driver,
            "status": resp.status,
            "total_completed_stops": resp.total_completed_stops,
            "active_stop_state": resp.active_stop.status if resp.active_stop else "ON_TRACK",
            "stops": stops_summary,
            "net_pit_loss_s": net_loss,
            "waterfall": {
                "caution_and_incidents_s": factors.caution_and_incidents_s if factors else 0.0,
                "pit_stop_and_transit_s": factors.pit_stop_and_transit_s if factors else 0.0,
                "traffic_s": factors.traffic_s if factors else 0.0,
                "tyre_degradation_s": factors.tyre_degradation_s if factors else 0.0,
                "unexplained_residual_s": factors.unexplained_residual_s if factors else 0.0,
            } if factors else None,
            "reference_clean_pace_s": resp.reference_clean_pace_s,
            "reference_stationary_s": resp.reference_stationary_s,
            "accounting_notes": resp.accounting_notes,
        }

    # -------------------------------------------------------------------------
    # 5. get_battle_and_overtake_prediction
    # -------------------------------------------------------------------------
    def _handle_get_battle_and_overtake_prediction(self, args: Dict[str, Any]) -> Dict[str, Any]:
        d1 = str(args.get("chaser") or args.get("driver1", "")).strip().upper()
        d2 = str(args.get("leader") or args.get("driver2", "")).strip().upper()
        horizon = int(args.get("horizon_laps", 5))

        if d1 not in self.session.drivers:
            raise ValueError(f"Driver '{d1}' not found in race '{self.race_id}'. Available: {self.session.drivers}")
        if d2 not in self.session.drivers:
            raise ValueError(f"Driver '{d2}' not found in race '{self.race_id}'. Available: {self.session.drivers}")
        if d1 == d2:
            raise ValueError("driver1 and driver2 must be distinct drivers.")

        sm = _get_sm()
        resp = sm.get_battle_and_overtake_prediction(
            self.race_id,
            driver1=d1,
            driver2=d2,
            as_of=self.authoritative_as_of,
            horizon_laps=horizon,
        )

        p_battle = resp.battle_prediction.probability if resp.battle_prediction else None
        p_overtake = resp.overtake_prediction.probability if resp.overtake_prediction else None
        gap = resp.supporting_features.gap_s if resp.supporting_features else None
        closing = resp.supporting_features.closing_rate_s_per_lap if resp.supporting_features else None

        return {
            "status": resp.status,
            "attacker": resp.attacker,
            "defender": resp.defender,
            "chaser": resp.attacker,
            "leader": resp.defender,
            "horizon_laps": resp.horizon_laps,
            "battle_probability_5_laps": p_battle,
            "overtake_probability_5_laps": p_overtake,
            "p_battle": p_battle,
            "p_overtake": p_overtake,
            "current_gap_s": gap,
            "gap_s": gap,
            "closing_rate_s_per_lap": closing,
            "clean_pace_delta_s": resp.supporting_features.clean_pace_delta_s if resp.supporting_features else None,
            "tyre_age_delta_laps": resp.supporting_features.tyre_age_delta_laps if resp.supporting_features else None,
            "reliability_warnings": [w.message for w in resp.reliability_warnings],
        }

    # -------------------------------------------------------------------------
    # Allowlist Tool Definitions
    # -------------------------------------------------------------------------
    _tool_definitions: Dict[str, ToolDefinition] = {
        "get_race_leaderboard": ToolDefinition(
            name="get_race_leaderboard",
            description="Get the current official point-in-time race leaderboard, positions, gaps, tyre compound and age, track status, and weather.",
            parameters={
                "type": "OBJECT",
                "properties": {},
            },
        ),
        "compare_driver_laps": ToolDefinition(
            name="compare_driver_laps",
            description="Compare telemetry and lap times between two drivers on a single completed lap or across a range of laps up to the current simulation cutoff.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "driver1": {
                        "type": "STRING",
                        "description": "Three-letter code for first driver / reference (e.g. 'VER')",
                    },
                    "driver2": {
                        "type": "STRING",
                        "description": "Three-letter code for second driver (e.g. 'LEC')",
                    },
                    "mode": {
                        "type": "STRING",
                        "enum": ["single", "range"],
                        "description": "Comparison mode: 'single' lap or lap 'range'",
                    },
                    "lap_number": {
                        "type": "INTEGER",
                        "description": "Lap number when mode is 'single'",
                    },
                    "start_lap": {
                        "type": "INTEGER",
                        "description": "Start lap when mode is 'range'",
                    },
                    "end_lap": {
                        "type": "INTEGER",
                        "description": "End lap when mode is 'range' (must be <= current replay lap)",
                    },
                },
                "required": ["driver1", "driver2"],
            },
        ),
        "get_tyre_stint_degradation": ToolDefinition(
            name="get_tyre_stint_degradation",
            description="Get tyre wear degradation slope (seconds per lap) and fit confidence (R2) for a driver on a specific stint up to the current simulation cutoff.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "driver": {
                        "type": "STRING",
                        "description": "Three-letter driver code (e.g. 'VER')",
                    },
                    "stint": {
                        "type": "INTEGER",
                        "description": "Stint number (1, 2, ...)",
                    },
                },
                "required": ["driver"],
            },
        ),
        "get_pit_strategy_and_waterfall": ToolDefinition(
            name="get_pit_strategy_and_waterfall",
            description="Get completed pit stops, transponder pit lane durations, illustrative 2.5s wheel-change assumptions, and the 5-category time-loss waterfall for a driver.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "driver": {
                        "type": "STRING",
                        "description": "Three-letter driver code (e.g. 'VER')",
                    },
                },
                "required": ["driver"],
            },
        ),
        "get_battle_and_overtake_prediction": ToolDefinition(
            name="get_battle_and_overtake_prediction",
            description="Get 5-lap horizon probabilistic predictions for an on-track battle and overtake between two drivers, with supporting gap, closing rate, and pace features.",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "driver1": {
                        "type": "STRING",
                        "description": "Three-letter code for first driver (e.g. 'VER')",
                    },
                    "driver2": {
                        "type": "STRING",
                        "description": "Three-letter code for rival driver (e.g. 'LEC')",
                    },
                },
                "required": ["driver1", "driver2"],
            },
        ),
    }
