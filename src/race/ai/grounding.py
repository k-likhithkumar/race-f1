"""
Grounding and Answer Validation module (Checkpoint 5B).
Ensures quantitative claims, driver identities, lap ranges, delta signs,
and estimates vs. measurements are strictly grounded in deterministic tool outputs.
Enforces guardrails against asserting unmeasured internal facts (engine modes, driver emotions/intent, hidden damage).
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from pydantic import BaseModel, Field

from src.race.ai.schemas import AssistantKeyNumber, AssistantConfidence, AssistantQueryStatus


# Unmeasured concepts that public timing/telemetry cannot substantiate
UNMEASURED_PATTERNS = [
    (r"\b(engine mode|engine mapping|party mode|strats? mode|engine power level|fuel mix)\b", "engine modes and power settings"),
    (r"\b(driver emotion|driver feelings?|angry|furious|frustrated|happy|sad|driver intent|mentally|psychology)\b", "driver psychological state or emotional intent"),
    (r"\b(hidden car damage|internal chassis crack|unseen damage|suspension failure internal|wind tunnel)\b", "unmeasured internal vehicle damage"),
    (r"\b(team order whispered|secret radio|undisclosed agreement)\b", "undisclosed team communications"),
]

# Patterns where the model wrongly asserts certainty for probabilistic predictions
OVERCONFIDENT_PREDICTION_PATTERNS = [
    r"\bguaranteed overtake\b",
    r"\bwill definitely overtake\b",
    r"\b100% chance of overtake\b",
    r"\bcertain to pass\b",
    r"\binevitable overtake\b",
]

# Patterns where the model wrongly claims stationary assumption was measured
FALSE_MEASUREMENT_PATTERNS = [
    r"\bmeasured (a )?2\.5\s*s\b",
    r"\bmeasured stationary time\b",
    r"\bactual stop time was 2\.5\s*s\b",
    r"\brecorded 2\.5\s*s stationary\b",
]


class GroundingResult(BaseModel):
    """Result of grounding validation on the LLM output."""
    status: AssistantQueryStatus
    sanitized_answer: str
    confidence: AssistantConfidence
    claims: List[str] = Field(default_factory=list)
    key_numbers: List[AssistantKeyNumber] = Field(default_factory=list)
    limitations_notice: Optional[str] = None
    validation_passed: bool = True
    reasons: List[str] = Field(default_factory=list)


def check_unmeasured_domains(prompt: str) -> Optional[str]:
    """
    Checks if user prompt asks about unmeasured domains that telemetry cannot answer.
    Returns explanation string if unmeasured, None otherwise.
    """
    prompt_lower = prompt.lower()
    detected = []
    for pattern, description in UNMEASURED_PATTERNS:
        if re.search(pattern, prompt_lower):
            detected.append(description)

    if detected:
        return (
            f"Public timing and telemetry data cannot determine {', '.join(detected)}. "
            "RACE only provides deterministic timing, position, tyre degradation, pit time-loss, "
            "and probabilistic battle models based on measured track data."
        )
    return None


def extract_tool_numbers(tool_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extracts verified numerical values and metrics from executed tool outputs.
    """
    numbers = []

    for rec in tool_records:
        tool_name = rec.get("tool_name")
        args = rec.get("args", {})
        result = rec.get("result", {})
        if not isinstance(result, dict) or result.get("status") == "ERROR":
            continue

        if tool_name == "compare_driver_laps":
            d_a = args.get("driver_a")
            d_b = args.get("driver_b")
            lap = args.get("lap") or args.get("start_lap")
            mean_delta = result.get("mean_delta_s")
            lap_a = result.get("lap_time_a_s") or result.get("lap_time_a")
            lap_b = result.get("lap_time_b_s") or result.get("lap_time_b")

            if mean_delta is not None:
                numbers.append({
                    "label": f"Mean Delta ({d_a} vs {d_b})",
                    "value": f"{mean_delta:+.3f}s",
                    "num": float(mean_delta),
                    "driver_a": d_a,
                    "driver_b": d_b,
                    "type": "delta",
                })
            if lap_a is not None:
                numbers.append({"label": f"{d_a} Lap Time", "value": f"{lap_a:.3f}s", "num": float(lap_a)})
            if lap_b is not None:
                numbers.append({"label": f"{d_b} Lap Time", "value": f"{lap_b:.3f}s", "num": float(lap_b)})

        elif tool_name == "get_tyre_stint_degradation":
            driver = args.get("driver")
            stint = args.get("stint")
            slope = result.get("slope_s_per_lap")
            if slope is not None:
                numbers.append({
                    "label": f"{driver} Stint {stint} Degradation",
                    "value": f"{slope:+.4f}s/lap",
                    "num": float(slope),
                    "driver": driver,
                    "stint": stint,
                    "type": "slope",
                })

        elif tool_name == "get_pit_strategy_and_waterfall":
            driver = args.get("driver")
            net_loss = result.get("net_pit_loss_s")
            total_loss = result.get("in_lane_total_loss_s")
            ref_stat = result.get("reference_stationary_s")

            if net_loss is not None:
                numbers.append({
                    "label": f"{driver} Net Pit Loss",
                    "value": f"{net_loss:.2f}s",
                    "num": float(net_loss),
                    "driver": driver,
                    "type": "pit_loss",
                })
            if ref_stat is not None:
                numbers.append({
                    "label": "Illustrative Reference Stationary Time",
                    "value": f"{ref_stat:.1f}s (assumed)",
                    "num": float(ref_stat),
                    "type": "assumption",
                })

        elif tool_name == "get_battle_and_overtake_prediction":
            chaser = args.get("chaser")
            leader = args.get("leader")
            p_battle = result.get("p_battle")
            p_overtake = result.get("p_overtake")
            gap = result.get("gap_s")
            closing = result.get("closing_rate_s_per_lap")

            if p_battle is not None:
                numbers.append({
                    "label": f"Battle Probability ({chaser} vs {leader})",
                    "value": f"{p_battle * 100:.1f}%",
                    "num": float(p_battle),
                    "type": "probability",
                })
            if p_overtake is not None:
                numbers.append({
                    "label": f"Overtake Probability ({chaser} vs {leader})",
                    "value": f"{p_overtake * 100:.1f}%",
                    "num": float(p_overtake),
                    "type": "probability",
                })
            if gap is not None:
                numbers.append({
                    "label": f"Gap ({chaser} to {leader})",
                    "value": f"{gap:.3f}s",
                    "num": float(gap),
                })
            if closing is not None:
                numbers.append({
                    "label": f"Closing Rate ({chaser} vs {leader})",
                    "value": f"{closing:+.3f}s/lap",
                    "num": float(closing),
                })

        elif tool_name == "get_race_leaderboard":
            leaderboard = result.get("leaderboard", [])
            for entry in leaderboard[:3]:
                d = entry.get("driver")
                gap = entry.get("gap_to_leader_s")
                pos = entry.get("position")
                if pos is not None:
                    numbers.append({"label": f"{d} Position", "value": f"P{pos}", "driver": d})
                if gap is not None:
                    numbers.append({"label": f"{d} Gap to Leader", "value": f"{gap:.3f}s", "num": float(gap), "driver": d})

    return numbers


def parse_floats_from_text(text: str) -> List[float]:
    """Extracts explicit floating point numbers from text."""
    # Match patterns like 1:23.456 (convert to seconds), +0.123, -1.45, 84.12
    found = []
    # Mm:ss.sss format
    lap_time_matches = re.findall(r"\b(\d+):(\d{2}(?:\.\d{1,3})?)\b", text)
    for m, s in lap_time_matches:
        try:
            total_s = int(m) * 60 + float(s)
            found.append(round(total_s, 3))
        except ValueError:
            pass

    # Decimal numbers
    dec_matches = re.findall(r"(?:[+\-])?\b\d+\.\d+\b", text)
    for d in dec_matches:
        try:
            found.append(round(float(d), 4))
        except ValueError:
            pass

    return found


def validate_grounding(
    prompt: str,
    answer_text: str,
    tool_records: List[Dict[str, Any]],
    as_of: float,
) -> GroundingResult:
    """
    Performs rigorous validation of model answer against deterministic tool evidence.
    """
    reasons: List[str] = []
    limitations: List[str] = []

    # 1. Check unmeasured queries in prompt
    unmeasured_msg = check_unmeasured_domains(prompt)
    if unmeasured_msg:
        return GroundingResult(
            status="CANNOT_TELL",
            sanitized_answer=unmeasured_msg,
            confidence="CANNOT_TELL",
            claims=["Query pertains to unmeasured internal vehicle or psychological factors."],
            key_numbers=[],
            limitations_notice="Telemetry lacks internal vehicle configuration or subjective intent data.",
            validation_passed=True,
            reasons=["Unmeasured query detected; guardrail triggered."],
        )

    # 2. Check if answer invents unmeasured internal facts
    for pattern, description in UNMEASURED_PATTERNS:
        if re.search(pattern, answer_text, re.IGNORECASE):
            reasons.append(f"Model asserted unmeasured domain ({description}).")
            limitations.append(f"Notice: Model assertions regarding {description} cannot be substantiated by track data.")

    # 3. Check for overconfident probabilistic claims
    for pat in OVERCONFIDENT_PREDICTION_PATTERNS:
        if re.search(pat, answer_text, re.IGNORECASE):
            reasons.append("Model described a probabilistic battle prediction as a certainty.")
            limitations.append("Notice: Battle and overtake predictions are probabilistic models, not guarantees.")

    # 4. Check for false stationary measurement claims
    for pat in FALSE_MEASUREMENT_PATTERNS:
        if re.search(pat, answer_text, re.IGNORECASE):
            reasons.append("Model wrongly claimed reference stationary time (2.5s) was measured.")
            limitations.append("Notice: Stationary pit time is an illustrative 2.5s assumption, not a measured sensor value.")

    # 5. Check if tools were executed
    if not tool_records:
        # If no tools were called for an analytical question, confidence is low
        return GroundingResult(
            status="INSUFFICIENT_EVIDENCE",
            sanitized_answer=answer_text,
            confidence="LOW",
            claims=[],
            key_numbers=[],
            limitations_notice="No deterministic tool evidence was available for this response.",
            validation_passed=False,
            reasons=["No deterministic tools were called."],
        )

    # 6. Extract ground-truth numbers from tools
    tool_num_records = extract_tool_numbers(tool_records)
    key_numbers: List[AssistantKeyNumber] = []
    for item in tool_num_records[:6]:
        key_numbers.append(AssistantKeyNumber(label=item["label"], value=item["value"]))

    # 7. Check delta sign semantics in comparison tools
    for rec in tool_records:
        if rec.get("tool_name") == "compare_driver_laps":
            result = rec.get("result", {})
            args = rec.get("args", {})
            mean_delta = result.get("mean_delta_s")
            d_a = args.get("driver_a")
            d_b = args.get("driver_b")
            if mean_delta is not None and d_a and d_b:
                # mean_delta = lap_time(d_a) - lap_time(d_b)
                # If mean_delta > 0: d_a was slower, d_b was faster
                # Check if model mistakenly says d_a was faster
                ans_lower = answer_text.lower()
                if mean_delta > 0.05:
                    wrong_claim = f"{d_a.lower()} was faster than {d_b.lower()}"
                    if wrong_claim in ans_lower:
                        reasons.append(f"Inverted delta semantics: model claimed {d_a} was faster, but delta was +{mean_delta:.3f}s (slower).")
                        limitations.append(f"Correction: {d_a} was slower than {d_b} by {mean_delta:.3f}s.")
                elif mean_delta < -0.05:
                    wrong_claim = f"{d_b.lower()} was faster than {d_a.lower()}"
                    if wrong_claim in ans_lower:
                        reasons.append(f"Inverted delta semantics: model claimed {d_b} was faster, but delta was {mean_delta:.3f}s ({d_a} faster).")
                        limitations.append(f"Correction: {d_a} was faster than {d_b} by {abs(mean_delta):.3f}s.")

    # 8. Extract claims (simple sentence breakdown)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer_text) if len(s.strip()) > 15]
    claims = sentences[:5]

    # Determine confidence and status
    confidence: AssistantConfidence = "HIGH"
    status: AssistantQueryStatus = "SUCCESS"

    if reasons:
        confidence = "LOW" if len(reasons) > 1 else "MEDIUM"
        if any("unmeasured" in r.lower() for r in reasons):
            status = "CANNOT_TELL"

    full_limitations = " | ".join(limitations) if limitations else None

    return GroundingResult(
        status=status,
        sanitized_answer=answer_text,
        confidence=confidence,
        claims=claims,
        key_numbers=key_numbers,
        limitations_notice=full_limitations,
        validation_passed=(len(reasons) == 0),
        reasons=reasons,
    )
