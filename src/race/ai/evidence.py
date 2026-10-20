"""
Evidence reference generator for Checkpoint 5B.
Transforms validated tool execution outputs into typed EvidenceReference objects.
"""

from typing import List, Dict, Any, Optional
from src.race.ai.schemas import EvidenceReference, EvidenceType


def build_evidence_references(
    tool_name: str,
    tool_args: Dict[str, Any],
    tool_result: Dict[str, Any],
    as_of: float,
) -> List[EvidenceReference]:
    """
    Constructs deterministic EvidenceReference objects from an executed tool call.
    Only called for successful tool outputs.
    """
    if not isinstance(tool_result, dict):
        return []

    # If the tool returned an error or insufficient data flag, skip or handle safely
    if tool_result.get("status") == "ERROR":
        return []

    ev_list: List[EvidenceReference] = []

    if tool_name == "get_race_leaderboard":
        leaderboard = tool_result.get("leaderboard", [])
        top_drivers = [entry.get("driver") for entry in leaderboard[:5] if entry.get("driver")]
        total_drivers = len(leaderboard)
        as_of_round = round(as_of, 1)

        ev = EvidenceReference(
            id=f"ev-leaderboard-{int(as_of)}",
            type="replay",
            label=f"Race Leaderboard (t={as_of_round}s)",
            description=f"Leaderboard state covering {total_drivers} drivers at t={as_of_round}s.",
            driver_codes=top_drivers,
            lap=leaderboard[0].get("lap") if leaderboard else None,
            stint=None,
            as_of=as_of,
            nav_payload={
                "targetTab": "dashboard",
                "as_of": as_of,
            },
        )
        ev_list.append(ev)

    elif tool_name == "compare_driver_laps":
        d_a = tool_args.get("driver_a") or tool_args.get("driver1") or tool_result.get("driver1", "UNK")
        d_b = tool_args.get("driver_b") or tool_args.get("driver2") or tool_result.get("driver2", "UNK")
        lap = tool_args.get("lap") or tool_args.get("lap_number") or tool_result.get("start_lap")
        start_lap = tool_args.get("start_lap") or tool_result.get("start_lap")
        end_lap = tool_args.get("end_lap") or tool_result.get("end_lap")

        lap_label = f"Lap {lap}" if lap is not None else f"Laps {start_lap}-{end_lap}"
        mean_delta = tool_result.get("mean_delta_s")
        delta_str = f"{mean_delta:+.3f}s" if mean_delta is not None else "N/A"

        ev = EvidenceReference(
            id=f"ev-comp-{d_a}-{d_b}-{lap or f'{start_lap}_{end_lap}'}-{int(as_of)}",
            type="comparison",
            label=f"Lap Delta: {d_a} vs {d_b} ({lap_label})",
            description=f"{lap_label} comparison. Reference delta ({d_a} relative to {d_b}): {delta_str}.",
            driver_codes=[d_a, d_b],
            lap=lap or start_lap,
            stint=None,
            as_of=as_of,
            nav_payload={
                "targetTab": "comparison",
                "drivers": [d_a, d_b],
                "lap": lap,
                "start_lap": start_lap,
                "end_lap": end_lap,
            },
        )
        ev_list.append(ev)

    elif tool_name == "get_tyre_stint_degradation":
        driver = tool_args.get("driver", "UNK")
        stint = tool_args.get("stint", 1)
        compound = tool_result.get("compound", "Tyres")
        slope = tool_result.get("slope_s_per_lap")
        slope_str = f"{slope:+.4f}s/lap" if slope is not None else "Insufficient Data"

        ev = EvidenceReference(
            id=f"ev-tyre-{driver}-stint{stint}-{int(as_of)}",
            type="tyres",
            label=f"{driver} Stint {stint} ({compound}) Degradation",
            description=f"Degradation analysis for {driver} stint {stint}: slope {slope_str}.",
            driver_codes=[driver],
            lap=None,
            stint=stint,
            as_of=as_of,
            nav_payload={
                "targetTab": "tyres",
                "driver": driver,
                "stint": stint,
            },
        )
        ev_list.append(ev)

    elif tool_name == "get_pit_strategy_and_waterfall":
        driver = tool_args.get("driver", "UNK")
        stops_count = tool_result.get("total_completed_stops", 0)
        net_loss = tool_result.get("net_pit_loss_s")
        loss_str = f"{net_loss:.1f}s" if net_loss is not None else "0.0s"

        ev = EvidenceReference(
            id=f"ev-pit-{driver}-{int(as_of)}",
            type="pits",
            label=f"{driver} Pit Strategy & Time Loss",
            description=f"Pit strategy waterfall for {driver}: {stops_count} completed stops, net loss {loss_str} (stationary time is an illustrative 2.5s assumption).",
            driver_codes=[driver],
            lap=None,
            stint=None,
            as_of=as_of,
            nav_payload={
                "targetTab": "pits",
                "driver": driver,
            },
        )
        ev_list.append(ev)

    elif tool_name == "get_battle_and_overtake_prediction":
        chaser = tool_args.get("chaser", "UNK")
        leader = tool_args.get("leader", "UNK")
        p_battle = tool_result.get("p_battle")
        p_overtake = tool_result.get("p_overtake")
        p_bat_str = f"{p_battle * 100:.1f}%" if p_battle is not None else "N/A"
        p_ot_str = f"{p_overtake * 100:.1f}%" if p_overtake is not None else "N/A"

        ev = EvidenceReference(
            id=f"ev-battle-{chaser}-{leader}-{int(as_of)}",
            type="battles",
            label=f"Battle Prediction: {chaser} vs {leader}",
            description=f"Estimated battle prob {p_bat_str}, overtake prob {p_ot_str} (probabilistic estimate, not guaranteed).",
            driver_codes=[chaser, leader],
            lap=None,
            stint=None,
            as_of=as_of,
            nav_payload={
                "targetTab": "battles",
                "chaser": chaser,
                "leader": leader,
            },
        )
        ev_list.append(ev)

    return ev_list
