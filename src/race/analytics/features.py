"""
Feature extraction and training dataset builder for Battle & Overtake Probability (M4).

Implements strict point-in-time causal feature extraction:
- Features at lap L / timestamp T strictly consume observations <= T.
- Future observations in [L+1, L+H] are strictly isolated for target label construction.
- Preserves explicit missing-data semantics without fabricating values.
- Separates currently active battles from future battle transitions.
- Separates completed on-track overtakes from pit strategy swaps, retirements,
  and caution period changes.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.race.config import (
    BATTLE_GAP_THRESHOLD_S,
    BATTLE_CLOSING_RATE_THRESHOLD_S,
)
from src.race.analytics.gaps import classify_overtakes, detect_battles

logger = logging.getLogger(__name__)

# Canonical dry compound hierarchy for tier delta calculation
# Positive delta = attacker on softer / grippier compound
COMPOUND_TIERS: Dict[str, int] = {
    "SOFT": 3,
    "MEDIUM": 2,
    "HARD": 1,
}


@dataclass(frozen=True)
class PairFeatures:
    """Causal features extracted at prediction boundary lap L (timestamp T)."""
    race_id: str
    lap_number: int
    attacker: str
    defender: str
    timestamp_s: float
    gap_s: float
    closing_rate_s_per_lap: Optional[float]
    is_closing: bool
    clean_pace_delta_s: Optional[float]
    attacker_recent_pace_s: Optional[float]
    defender_recent_pace_s: Optional[float]
    attacker_clean_laps_stint: int
    defender_clean_laps_stint: int
    data_sufficiency: str  # "HIGH", "MEDIUM", "LOW"
    tyre_age_delta_laps: Optional[int]
    attacker_tyre_age: Optional[int]
    defender_tyre_age: Optional[int]
    attacker_compound: str
    defender_compound: str
    compound_delta_tier: Optional[int]
    is_currently_battling: bool
    current_battle_verdict: Optional[str]
    attacker_stint: int
    defender_stint: int
    laps_remaining: Optional[int]
    track_status_at_lap: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairSample:
    """Complete training/eval sample containing causal features and forward labels."""
    features: PairFeatures
    horizon_laps: int
    battle_next_H_laps: Optional[int]  # 1: yes, 0: no, None: invalid/unreliable
    battle_laps_count_next_H: int
    pass_next_H_laps: Optional[int]    # 1: on-track pass, 0: no pass, None: invalid
    pass_lap_number: Optional[int]
    pass_classification: Optional[str]
    pit_swap_occurred_next_H: bool
    retirement_occurred_next_H: bool
    caution_laps_in_horizon: int
    label_valid: bool
    exclusion_reason: Optional[str]
    is_eval_stride_sample: bool

    def to_dict(self) -> Dict[str, Any]:
        d = self.features.to_dict()
        d.update({
            "horizon_laps": self.horizon_laps,
            "battle_next_H_laps": self.battle_next_H_laps,
            "battle_laps_count_next_H": self.battle_laps_count_next_H,
            "pass_next_H_laps": self.pass_next_H_laps,
            "pass_lap_number": self.pass_lap_number,
            "pass_classification": self.pass_classification,
            "pit_swap_occurred_next_H": self.pit_swap_occurred_next_H,
            "retirement_occurred_next_H": self.retirement_occurred_next_H,
            "caution_laps_in_horizon": self.caution_laps_in_horizon,
            "label_valid": self.label_valid,
            "exclusion_reason": self.exclusion_reason,
            "is_eval_stride_sample": self.is_eval_stride_sample,
        })
        return d


def extract_pair_features(
    laps_df: pd.DataFrame,
    stints_df: Optional[pd.DataFrame],
    attacker: str,
    defender: str,
    as_of_lap: int,
    race_id: str = "unknown",
    total_laps: Optional[int] = None,
) -> Optional[PairFeatures]:
    """
    Extracts causal pair features at as_of_lap strictly using data <= as_of_lap.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Cached laps table. Rows with lap_number > as_of_lap are strictly filtered out.
    stints_df : pd.DataFrame, optional
        Cached stints table.
    attacker : str
        Three-letter code of chasing driver.
    defender : str
        Three-letter code of leading driver.
    as_of_lap : int
        Replay boundary lap.
    race_id : str
        Race identifier.
    total_laps : int, optional
        Scheduled total laps in race.

    Returns
    -------
    PairFeatures or None
        Extracted causal features, or None if cars are not validly on track.
    """
    if laps_df is None or laps_df.empty or as_of_lap < 1:
        return None

    # 1. Enforce strict point-in-time boundary: drop all future laps
    hist_laps = laps_df[laps_df["lap_number"] <= as_of_lap].copy()

    # Get records at current lap
    curr_att = hist_laps[(hist_laps["driver"] == attacker) & (hist_laps["lap_number"] == as_of_lap)]
    curr_def = hist_laps[(hist_laps["driver"] == defender) & (hist_laps["lap_number"] == as_of_lap)]

    if curr_att.empty or curr_def.empty:
        return None

    att_row = curr_att.iloc[0]
    def_row = curr_def.iloc[0]

    # Session time and gap calculation
    t_att = att_row.get("session_time_s")
    t_def = def_row.get("session_time_s")
    if pd.isna(t_att) or pd.isna(t_def):
        return None

    t_att = float(t_att)
    t_def = float(t_def)

    # Gap definition: seconds attacker is behind defender (positive)
    gap_s = round(t_att - t_def, 3)

    # 2. Closing rate from previous lap
    closing_rate: Optional[float] = None
    is_closing = False
    if as_of_lap > 1:
        prev_att = hist_laps[(hist_laps["driver"] == attacker) & (hist_laps["lap_number"] == as_of_lap - 1)]
        prev_def = hist_laps[(hist_laps["driver"] == defender) & (hist_laps["lap_number"] == as_of_lap - 1)]
        if not prev_att.empty and not prev_def.empty:
            pt_att = prev_att.iloc[0].get("session_time_s")
            pt_def = prev_def.iloc[0].get("session_time_s")
            if pd.notna(pt_att) and pd.notna(pt_def):
                prev_gap = round(float(pt_att) - float(pt_def), 3)
                if prev_gap > 0:
                    closing_rate = round(prev_gap - gap_s, 3)
                    is_closing = closing_rate >= BATTLE_CLOSING_RATE_THRESHOLD_S

    # 3. Clean air pace difference over recent clean laps in current stint
    att_stint = int(att_row.get("stint", 1))
    def_stint = int(def_row.get("stint", 1))

    att_stint_laps = hist_laps[(hist_laps["driver"] == attacker) & (hist_laps["stint"] == att_stint)]
    def_stint_laps = hist_laps[(hist_laps["driver"] == defender) & (hist_laps["stint"] == def_stint)]

    # Clean laps in current stint
    att_clean = att_stint_laps[att_stint_laps["is_clean"] == True] if "is_clean" in att_stint_laps.columns else pd.DataFrame()
    def_clean = def_stint_laps[def_stint_laps["is_clean"] == True] if "is_clean" in def_stint_laps.columns else pd.DataFrame()

    att_clean_count = len(att_clean)
    def_clean_count = len(def_clean)

    # Data sufficiency per PDF Page 12: High (5+), Medium (3-4), Low (<3)
    min_clean = min(att_clean_count, def_clean_count)
    if min_clean >= 5:
        data_sufficiency = "HIGH"
    elif min_clean >= 3:
        data_sufficiency = "MEDIUM"
    else:
        data_sufficiency = "LOW"

    # Compute clean air pace over last up to 3 clean laps
    recent_att_clean = att_clean.tail(3)
    recent_def_clean = def_clean.tail(3)

    att_pace: Optional[float] = None
    def_pace: Optional[float] = None
    clean_pace_delta: Optional[float] = None

    if not recent_att_clean.empty and not recent_def_clean.empty:
        att_times = recent_att_clean["lap_time_s"].dropna()
        def_times = recent_def_clean["lap_time_s"].dropna()
        if not att_times.empty and not def_times.empty:
            att_pace = round(float(att_times.mean()), 3)
            def_pace = round(float(def_times.mean()), 3)
            # Positive pace delta = defender lap time > attacker lap time (attacker faster)
            clean_pace_delta = round(def_pace - att_pace, 3)

    # 4. Tyre age and compound
    att_tyre_age = int(att_row["tyre_age_laps"]) if pd.notna(att_row.get("tyre_age_laps")) else None
    def_tyre_age = int(def_row["tyre_age_laps"]) if pd.notna(def_row.get("tyre_age_laps")) else None

    tyre_age_delta: Optional[int] = None
    if att_tyre_age is not None and def_tyre_age is not None:
        # Positive tyre age delta = defender tyres older than attacker tyres (attacker tyre advantage)
        tyre_age_delta = def_tyre_age - att_tyre_age

    att_compound = str(att_row.get("compound", "UNKNOWN")).strip().upper()
    def_compound = str(def_row.get("compound", "UNKNOWN")).strip().upper()

    compound_delta_tier: Optional[int] = None
    if att_compound in COMPOUND_TIERS and def_compound in COMPOUND_TIERS:
        # Positive tier delta = attacker on softer compound than defender
        compound_delta_tier = COMPOUND_TIERS[att_compound] - COMPOUND_TIERS[def_compound]

    # 5. Currently active battle status at lap L
    is_currently_battling = False
    current_battle_verdict: Optional[str] = None
    if 0.0 < gap_s <= BATTLE_GAP_THRESHOLD_S:
        if is_closing:
            is_currently_battling = True
            current_battle_verdict = "ACTIVE_CHASE"
        elif gap_s <= 0.5:
            is_currently_battling = True
            current_battle_verdict = "SUSTAINED_DEFENSE"

    # Laps remaining: strictly based on scheduled total_laps if provided, otherwise None
    laps_remaining = max(0, total_laps - as_of_lap) if total_laps is not None else None
    track_status = str(att_row.get("track_status", "UNKNOWN")).strip()

    return PairFeatures(
        race_id=race_id,
        lap_number=as_of_lap,
        attacker=attacker,
        defender=defender,
        timestamp_s=t_att,
        gap_s=gap_s,
        closing_rate_s_per_lap=closing_rate,
        is_closing=is_closing,
        clean_pace_delta_s=clean_pace_delta,
        attacker_recent_pace_s=att_pace,
        defender_recent_pace_s=def_pace,
        attacker_clean_laps_stint=att_clean_count,
        defender_clean_laps_stint=def_clean_count,
        data_sufficiency=data_sufficiency,
        tyre_age_delta_laps=tyre_age_delta,
        attacker_tyre_age=att_tyre_age,
        defender_tyre_age=def_tyre_age,
        attacker_compound=att_compound,
        defender_compound=def_compound,
        compound_delta_tier=compound_delta_tier,
        is_currently_battling=is_currently_battling,
        current_battle_verdict=current_battle_verdict,
        attacker_stint=att_stint,
        defender_stint=def_stint,
        laps_remaining=laps_remaining,
        track_status_at_lap=track_status,
    )


def _build_pit_laps_map(stints_df: Optional[pd.DataFrame]) -> Dict[str, set]:
    """Extracts in-lap and out-lap numbers for every driver from stints table."""
    pit_laps_map: Dict[str, set] = {}
    if stints_df is not None and not stints_df.empty:
        for _, s_row in stints_df.iterrows():
            drv = s_row.get("driver")
            p_in = s_row.get("pit_in_lap")
            if pd.notna(p_in):
                p_val = int(p_in)
                if drv not in pit_laps_map:
                    pit_laps_map[drv] = set()
                pit_laps_map[drv].add(p_val)
                pit_laps_map[drv].add(p_val + 1)
    return pit_laps_map


def extract_pair_sample(
    laps_df: pd.DataFrame,
    stints_df: Optional[pd.DataFrame],
    attacker: str,
    defender: str,
    as_of_lap: int,
    horizon_laps: int = 5,
    race_id: str = "unknown",
    all_overtakes: Optional[List[Dict[str, Any]]] = None,
    total_laps: Optional[int] = None,
) -> Optional[PairSample]:
    """
    Constructs a complete training/eval sample:
    - Extracts causal features at as_of_lap.
    - Inspects forward window [as_of_lap + 1, as_of_lap + horizon_laps] for labels.
    - Strictly isolates features from future rows.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Complete race laps table.
    stints_df : pd.DataFrame, optional
        Stints table for pit stop checks.
    attacker, defender : str
        Driver codes.
    as_of_lap : int
        Prediction lap.
    horizon_laps : int
        Prediction horizon in laps (default: 5).
    race_id : str
        Race identifier.
    all_overtakes : list of dict, optional
        Pre-computed classified overtakes for the entire race.
    total_laps : int, optional
        Scheduled total laps in race.

    Returns
    -------
    PairSample or None
    """
    # If scheduled total_laps is unknown, it must NOT be inferred from future telemetry.
    # Forward horizon truncation for dataset labeling checks against race end.
    max_telemetry_lap = int(laps_df["lap_number"].max()) if not laps_df.empty else 0
    race_horizon_limit = total_laps if total_laps is not None else max_telemetry_lap

    features = extract_pair_features(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker=attacker,
        defender=defender,
        as_of_lap=as_of_lap,
        race_id=race_id,
        total_laps=total_laps,
    )
    if features is None:
        return None

    # Forward horizon window: [as_of_lap + 1, as_of_lap + horizon_laps]
    start_fwd = as_of_lap + 1
    end_fwd = as_of_lap + horizon_laps

    label_valid = True
    exclusion_reason: Optional[str] = None

    # Check window truncation
    if end_fwd > race_horizon_limit:
        label_valid = False
        exclusion_reason = "TRUNCATED_HORIZON_AT_RACE_END"

    fwd_df = laps_df[(laps_df["lap_number"] >= start_fwd) & (laps_df["lap_number"] <= end_fwd)].copy()

    # Pre-computed overtakes lookup
    if all_overtakes is None:
        all_overtakes = classify_overtakes(laps_df, stints_df)

    pit_laps_map = _build_pit_laps_map(stints_df)

    # 1. Evaluate on-track pass label in horizon
    pass_next_H: Optional[int] = 0
    pass_lap_number: Optional[int] = None
    pass_classification: Optional[str] = None
    pit_swap_occurred = False
    retirement_occurred = False

    for evt in all_overtakes:
        e_lap = evt.get("lap_number", 0)
        if start_fwd <= e_lap <= end_fwd:
            o_drv = evt.get("overtaking_driver")
            d_drv = evt.get("overtaken_driver")

            # Check if attacker passed defender
            if o_drv == attacker and d_drv == defender:
                cls_type = evt.get("classification")
                is_on_track = bool(evt.get("is_on_track_overtake"))

                if is_on_track and cls_type in ["CONFIRMED_ON_TRACK_OVERTAKE", "INFERRED_ON_TRACK_OVERTAKE"]:
                    pass_next_H = 1
                    pass_lap_number = e_lap
                    pass_classification = cls_type
                    break
                elif cls_type == "PIT_STRATEGY_SWAP":
                    pit_swap_occurred = True
                elif cls_type == "RETIREMENT_PASS":
                    retirement_occurred = True

    # 2. Evaluate battle label in horizon:
    # Stays within 1.0s for 2+ laps during [start_fwd, end_fwd] under green flag
    # and excluding pit laps
    battle_laps_count = 0
    caution_laps_count = 0

    for lap in range(start_fwd, min(end_fwd, race_horizon_limit) + 1):
        lap_att = fwd_df[(fwd_df["driver"] == attacker) & (fwd_df["lap_number"] == lap)]
        lap_def = fwd_df[(fwd_df["driver"] == defender) & (fwd_df["lap_number"] == lap)]

        if lap_att.empty or lap_def.empty:
            continue

        ts_att = str(lap_att.iloc[0].get("track_status", "1")).strip()
        ts_def = str(lap_def.iloc[0].get("track_status", "1")).strip()
        if ts_att != "1" or ts_def != "1":
            caution_laps_count += 1
            continue  # Exclude caution laps from active battle count

        # Exclude pit in-lap and out-lap from eligible battle count
        if (attacker in pit_laps_map and lap in pit_laps_map[attacker]) or (defender in pit_laps_map and lap in pit_laps_map[defender]):
            continue

        t1 = lap_att.iloc[0].get("session_time_s")
        t2 = lap_def.iloc[0].get("session_time_s")
        if pd.notna(t1) and pd.notna(t2):
            gap = abs(float(t1) - float(t2))
            if gap <= BATTLE_GAP_THRESHOLD_S:
                battle_laps_count += 1

    battle_next_H: Optional[int] = 1 if battle_laps_count >= 2 else 0

    # Extended caution exclusion: if 3+ laps of 5 are under caution, label is confounded
    if caution_laps_count >= (horizon_laps // 2 + 1):
        label_valid = False
        exclusion_reason = "EXTENDED_CAUTION_IN_HORIZON"

    # Invalidate labels if horizon truncated and event did not occur
    if not label_valid and pass_next_H == 0:
        pass_next_H = None
    if not label_valid and battle_next_H == 0:
        battle_next_H = None

    # Stride 5 evaluation flag for leave-one-race-out test sets
    is_eval_stride = (as_of_lap % 5 == 0)

    return PairSample(
        features=features,
        horizon_laps=horizon_laps,
        battle_next_H_laps=battle_next_H,
        battle_laps_count_next_H=battle_laps_count,
        pass_next_H_laps=pass_next_H,
        pass_lap_number=pass_lap_number,
        pass_classification=pass_classification,
        pit_swap_occurred_next_H=pit_swap_occurred,
        retirement_occurred_next_H=retirement_occurred,
        caution_laps_in_horizon=caution_laps_count,
        label_valid=label_valid,
        exclusion_reason=exclusion_reason,
        is_eval_stride_sample=is_eval_stride,
    )


def build_overtake_dataset(
    races_data: List[Dict[str, Any]],
    candidate_gap_threshold_s: float = 3.0,
    horizon_laps: int = 5,
    min_lap: int = 2,
) -> pd.DataFrame:
    """
    Extracts all candidate pairs across specified races into a structured dataset.

    Parameters
    ----------
    races_data : list of dict
        Each dict must contain:
        - 'race_id': str
        - 'laps': pd.DataFrame
        - 'stints': pd.DataFrame
    candidate_gap_threshold_s : float
        Maximum proximity to qualify as a candidate battle pair (default 3.0s).
    horizon_laps : int
        Prediction horizon (default 5 laps).
    min_lap : int
        Starting lap (default 2 to exclude lap 1 chaos).

    Returns
    -------
    pd.DataFrame
        Dataset of candidate pairs with causal features and forward labels.
    """
    records: List[Dict[str, Any]] = []

    for r_entry in races_data:
        race_id = r_entry.get("race_id", "unknown")
        laps_df = r_entry.get("laps")
        stints_df = r_entry.get("stints")

        if laps_df is None or laps_df.empty:
            continue

        # Precompute overtakes and pit laps for the race
        overtakes = classify_overtakes(laps_df, stints_df)
        pit_laps_map = _build_pit_laps_map(stints_df)
        scheduled_total_laps = r_entry.get("total_laps")
        if scheduled_total_laps is None:
            meta = r_entry.get("metadata")
            if isinstance(meta, dict):
                scheduled_total_laps = meta.get("total_laps")
        if scheduled_total_laps is None and "_" in race_id:
            try:
                parts = race_id.split("_", 1)
                from src.race.data.cache import load_cached_table
                cached_meta = load_cached_table(int(parts[0]), parts[1], "metadata")
                if isinstance(cached_meta, dict):
                    scheduled_total_laps = cached_meta.get("total_laps")
            except Exception:
                pass

        # Iteration bound across completed telemetry laps
        max_available_laps = scheduled_total_laps if scheduled_total_laps is not None else int(laps_df["lap_number"].max())

        for lap in range(min_lap, max_available_laps):
            lap_df = laps_df[laps_df["lap_number"] == lap].dropna(subset=["session_time_s"]).sort_values("session_time_s").reset_index(drop=True)
            if len(lap_df) < 2:
                continue

            # Check consecutive cars on track
            for i in range(len(lap_df) - 1):
                ahead_car = lap_df.iloc[i]
                behind_car = lap_df.iloc[i + 1]

                # Exclude caution periods at lap start (must be strictly green)
                ts_a = str(ahead_car.get("track_status", "1")).strip()
                ts_b = str(behind_car.get("track_status", "1")).strip()
                if ts_a != "1" or ts_b != "1":
                    continue

                d_ahead = ahead_car["driver"]
                d_behind = behind_car["driver"]

                # Exclude pit laps (in-lap and out-lap) from candidate battle pairs
                if (d_ahead in pit_laps_map and lap in pit_laps_map[d_ahead]) or (d_behind in pit_laps_map and lap in pit_laps_map[d_behind]):
                    continue
                if bool(ahead_car.get("is_pit_in")) or bool(ahead_car.get("is_pit_out")):
                    continue
                if bool(behind_car.get("is_pit_in")) or bool(behind_car.get("is_pit_out")):
                    continue

                gap = float(behind_car["session_time_s"]) - float(ahead_car["session_time_s"])
                if 0.0 < gap <= candidate_gap_threshold_s:
                    defender = d_ahead
                    attacker = d_behind

                    sample = extract_pair_sample(
                        laps_df=laps_df,
                        stints_df=stints_df,
                        attacker=attacker,
                        defender=defender,
                        as_of_lap=lap,
                        horizon_laps=horizon_laps,
                        race_id=race_id,
                        all_overtakes=overtakes,
                        total_laps=scheduled_total_laps,
                    )
                    if sample is not None:
                        records.append(sample.to_dict())

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)
