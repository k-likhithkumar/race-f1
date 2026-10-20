"""
Unit tests for Battle & Overtake Feature Extraction and Dataset Builder (Checkpoint 3A).

Validates:
- Strict causal temporal barrier (zero future leakage).
- Missing-data preservation without fabricating values.
- Distinction between currently active battles and future battle labels.
- Distinction between on-track passes and pit strategy / caution swaps.
- Prediction horizon boundaries, truncation, and evaluation stride.
- Real-race dataset construction on cached races.
"""

import pytest
import numpy as np
import pandas as pd

from src.race.analytics.features import (
    PairFeatures,
    PairSample,
    extract_pair_features,
    extract_pair_sample,
    build_overtake_dataset,
    COMPOUND_TIERS,
)
from src.race.data.cache import load_cached_table


@pytest.fixture
def synthetic_race_data():
    """Generates synthetic 15-lap race with controlled gaps, pit stops, and cautions."""
    laps_records = []
    # Drivers: VER (ahead), HAM (chasing)
    # Lap 1-5: Clean green running, gap shrinking from 2.5s down to 0.8s
    # Lap 6: HAM overtakes VER on track
    # Lap 7-10: VER pits on lap 8 (pit swap)
    # Lap 11-15: Caution on lap 12-14
    for lap in range(1, 16):
        # Baseline time
        t_base = 1000.0 + (lap - 1) * 90.0
        
        # VER lap times
        t_ver = t_base
        c_ver = "MEDIUM"
        age_ver = lap
        stint_ver = 1 if lap < 8 else 2
        is_clean_ver = lap not in [1, 8, 12, 13, 14]
        status = "1" if lap not in [12, 13, 14] else "4"
        is_pit_ver = (lap == 8)

        # HAM lap times
        # HAM closes in laps 2-5, passes on lap 6
        if lap == 1:
            t_ham = t_ver + 2.5
        elif lap == 2:
            t_ham = t_ver + 2.0  # closing rate = 0.5s/lap
        elif lap == 3:
            t_ham = t_ver + 1.5  # closing rate = 0.5s/lap
        elif lap == 4:
            t_ham = t_ver + 1.1  # closing rate = 0.4s/lap
        elif lap == 5:
            t_ham = t_ver + 0.6  # closing rate = 0.5s/lap (active battle!)
        elif lap == 6:
            t_ham = t_ver - 0.4  # HAM passes VER on track!
        elif lap == 7:
            t_ham = t_ver - 1.0
        elif lap == 8:
            t_ham = t_ver - 25.0  # VER pits
        else:
            t_ham = t_ver - 22.0

        c_ham = "SOFT"
        age_ham = lap
        stint_ham = 1
        is_clean_ham = lap not in [1, 12, 13, 14]

        laps_records.append({
            "race_id": "synth_gp",
            "lap_number": lap,
            "driver": "VER",
            "lap_time_s": 90.0,
            "session_time_s": t_ver,
            "compound": c_ver,
            "tyre_age_laps": age_ver,
            "stint": stint_ver,
            "position": 1 if t_ver < t_ham else 2,
            "track_status": status,
            "is_clean": is_clean_ver,
            "is_pit_in": is_pit_ver,
            "is_pit_out": False,
        })
        laps_records.append({
            "race_id": "synth_gp",
            "lap_number": lap,
            "driver": "HAM",
            "lap_time_s": 89.5,
            "session_time_s": t_ham,
            "compound": c_ham,
            "tyre_age_laps": age_ham,
            "stint": stint_ham,
            "position": 1 if t_ham < t_ver else 2,
            "track_status": status,
            "is_clean": is_clean_ham,
            "is_pit_in": False,
            "is_pit_out": False,
        })

    laps_df = pd.DataFrame(laps_records)
    stints_df = pd.DataFrame([
        {"driver": "VER", "stint": 1, "compound": "MEDIUM", "start_lap": 1, "end_lap": 8, "pit_in_lap": 8},
        {"driver": "VER", "stint": 2, "compound": "HARD", "start_lap": 9, "end_lap": 15, "pit_in_lap": None},
        {"driver": "HAM", "stint": 1, "compound": "SOFT", "start_lap": 1, "end_lap": 15, "pit_in_lap": None},
    ])
    return laps_df, stints_df


def test_zero_future_leakage(synthetic_race_data):
    """Features extracted at lap L must be completely invariant to truncation of future laps."""
    laps_df, stints_df = synthetic_race_data
    as_of_lap = 4

    # 1. Extract features using complete race table (up to lap 15)
    feat_full = extract_pair_features(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=as_of_lap,
        race_id="synth_gp",
    )
    assert feat_full is not None

    # 2. Extract features using table strictly truncated at as_of_lap
    truncated_laps = laps_df[laps_df["lap_number"] <= as_of_lap].copy()
    feat_trunc = extract_pair_features(
        laps_df=truncated_laps,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=as_of_lap,
        race_id="synth_gp",
    )
    assert feat_trunc is not None

    # Assert all feature properties match exactly
    assert feat_full.gap_s == feat_trunc.gap_s
    assert feat_full.closing_rate_s_per_lap == feat_trunc.closing_rate_s_per_lap
    assert feat_full.clean_pace_delta_s == feat_trunc.clean_pace_delta_s
    assert feat_full.attacker_clean_laps_stint == feat_trunc.attacker_clean_laps_stint
    assert feat_full.tyre_age_delta_laps == feat_trunc.tyre_age_delta_laps
    assert feat_full.compound_delta_tier == feat_trunc.compound_delta_tier
    assert feat_full.is_currently_battling == feat_trunc.is_currently_battling
    assert feat_full.laps_remaining == feat_trunc.laps_remaining


def test_feature_values_correctness(synthetic_race_data):
    """Verify exact arithmetic of gap, closing rate, tyre delta, and compound delta."""
    laps_df, stints_df = synthetic_race_data
    feat = extract_pair_features(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=3,
        race_id="synth_gp",
    )
    assert feat is not None
    # At lap 3: VER session_time = 1180.0, HAM session_time = 1181.5 -> gap = 1.5s
    assert feat.gap_s == 1.5
    # At lap 2 gap was 2.0s -> closing rate = 2.0 - 1.5 = 0.5s/lap
    assert feat.closing_rate_s_per_lap == 0.5
    assert feat.is_closing is True
    # Tyre age: both on lap 3
    assert feat.attacker_tyre_age == 3
    assert feat.defender_tyre_age == 3
    assert feat.tyre_age_delta_laps == 0
    # Compounds: HAM on SOFT (3), VER on MEDIUM (2) -> delta tier = +1
    assert feat.attacker_compound == "SOFT"
    assert feat.defender_compound == "MEDIUM"
    assert feat.compound_delta_tier == 1
    # At lap 3, gap is 1.5s (> 1.0s) so not yet currently battling
    assert feat.is_currently_battling is False


def test_currently_battling_vs_future_battle(synthetic_race_data):
    """Distinguish already active battle at lap L from entering a future battle."""
    laps_df, stints_df = synthetic_race_data

    # At lap 4: gap = 1.1s, closing rate = 0.4s -> not yet in 1.0s battle corridor
    sample_lap4 = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=4,
        horizon_laps=5,
        race_id="synth_gp",
    )
    assert sample_lap4 is not None
    assert sample_lap4.features.is_currently_battling is False
    # But in the next 5 laps (laps 5-9), they enter battle (lap 5 gap = 0.6s, lap 6 pass)
    assert sample_lap4.battle_next_H_laps == 1
    assert sample_lap4.pass_next_H_laps == 1  # pass on lap 6

    # At lap 5: gap = 0.6s, closing rate = 0.5s -> ALREADY currently battling
    sample_lap5 = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=5,
        horizon_laps=5,
        race_id="synth_gp",
    )
    assert sample_lap5 is not None
    assert sample_lap5.features.is_currently_battling is True
    assert sample_lap5.features.current_battle_verdict == "ACTIVE_CHASE"


def test_on_track_overtake_vs_pit_swap(synthetic_race_data):
    """On-track overtakes must be labeled 1, while pit lane swaps must not count as on-track passes."""
    laps_df, stints_df = synthetic_race_data

    # From lap 2, HAM passes VER on track on lap 6 -> pass_next_H_laps = 1
    sample_pass = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=2,
        horizon_laps=5,
        race_id="synth_gp",
    )
    assert sample_pass is not None
    assert sample_pass.pass_next_H_laps == 1
    assert sample_pass.pass_lap_number == 6
    assert sample_pass.pass_classification in ["CONFIRMED_ON_TRACK_OVERTAKE", "INFERRED_ON_TRACK_OVERTAKE"]

    # Now inspect reverse: at lap 7, VER is behind HAM. VER pits on lap 8 (pit swap).
    # VER did NOT pass HAM on track!
    sample_pit = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="VER",
        defender="HAM",
        as_of_lap=7,
        horizon_laps=5,
        race_id="synth_gp",
    )
    assert sample_pit is not None
    assert sample_pit.pass_next_H_laps == 0


def test_missing_data_semantics():
    """Verify that missing compound, tyre age, or pace values are not fabricated."""
    df_missing = pd.DataFrame([
        {
            "race_id": "test",
            "lap_number": 2,
            "driver": "VER",
            "session_time_s": 200.0,
            "lap_time_s": 90.0,
            "compound": "UNKNOWN",
            "tyre_age_laps": None,
            "stint": 1,
            "track_status": "UNKNOWN",
            "is_clean": False,
        },
        {
            "race_id": "test",
            "lap_number": 2,
            "driver": "PER",
            "session_time_s": 201.5,
            "lap_time_s": None,
            "compound": "HARD",
            "tyre_age_laps": 10,
            "stint": 1,
            "track_status": "UNKNOWN",
            "is_clean": False,
        },
    ])
    feat = extract_pair_features(
        laps_df=df_missing,
        stints_df=None,
        attacker="PER",
        defender="VER",
        as_of_lap=2,
    )
    assert feat is not None
    # Unknown compound must preserve None tier delta, never guess
    assert feat.attacker_compound == "HARD"
    assert feat.defender_compound == "UNKNOWN"
    assert feat.compound_delta_tier is None
    # None tyre age must result in None delta
    assert feat.tyre_age_delta_laps is None
    # No clean laps -> pace delta must be None
    assert feat.clean_pace_delta_s is None
    assert feat.data_sufficiency == "LOW"
    # Unknown track status preserved
    assert feat.track_status_at_lap == "UNKNOWN"


def test_horizon_truncation_and_caution_exclusion(synthetic_race_data):
    """Samples near race end or during extended cautions must be marked invalid."""
    laps_df, stints_df = synthetic_race_data

    # Total laps is 15. At lap 12 with H=5 (12+5 = 17 > 15), horizon is truncated
    sample_trunc = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="VER",
        defender="HAM",
        as_of_lap=12,
        horizon_laps=5,
        race_id="synth_gp",
    )
    assert sample_trunc is not None
    assert sample_trunc.label_valid is False
    assert sample_trunc.exclusion_reason == "TRUNCATED_HORIZON_AT_RACE_END"

    # At lap 10 with H=5, laps 11-15 contain 3 caution laps (laps 12, 13, 14)
    # Caution laps count >= 3 out of 5 -> extended caution exclusion
    sample_caution = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="VER",
        defender="HAM",
        as_of_lap=10,
        horizon_laps=5,
        race_id="synth_gp",
    )
    assert sample_caution is not None
    assert sample_caution.label_valid is False
    assert sample_caution.exclusion_reason == "EXTENDED_CAUTION_IN_HORIZON"


def test_real_cached_race_dataset_builder():
    """Extract candidate pair dataset across cached Monza races and verify properties."""
    laps_2022 = load_cached_table(2022, "monza", "laps")
    stints_2022 = load_cached_table(2022, "monza", "stints")
    laps_2023 = load_cached_table(2023, "monza", "laps")
    stints_2023 = load_cached_table(2023, "monza", "stints")

    races = [
        {"race_id": "2022_monza", "laps": laps_2022, "stints": stints_2022},
        {"race_id": "2023_monza", "laps": laps_2023, "stints": stints_2023},
    ]

    df = build_overtake_dataset(races, candidate_gap_threshold_s=3.0, horizon_laps=5)
    assert not df.empty
    assert "gap_s" in df.columns
    assert "closing_rate_s_per_lap" in df.columns
    assert "pass_next_H_laps" in df.columns
    assert "battle_next_H_laps" in df.columns
    assert "is_eval_stride_sample" in df.columns

    # Verify gap bounds for candidate pairs
    assert (df["gap_s"] > 0).all()
    assert (df["gap_s"] <= 3.0).all()

    # Verify pass rate is reasonable
    valid_samples = df[df["label_valid"] == True]
    assert len(valid_samples) > 500
    pos_passes = valid_samples["pass_next_H_laps"].sum()
    pass_rate = pos_passes / len(valid_samples)
    # Historical Monza pass rate for 3s candidate pairs is ~10-20%
    assert 0.05 <= pass_rate <= 0.30

    # Verify stride 5 subset exists and has non-zero size
    stride_samples = valid_samples[valid_samples["is_eval_stride_sample"] == True]
    assert len(stride_samples) > 80


def test_pit_lap_candidate_and_battle_exclusion():
    """Verify that pit in/out laps are excluded from candidate pairs and from forward battle lap counts."""
    # Create 5-lap mini race where VER pits on lap 3
    laps_records = [
        # Lap 2: HAM chasing VER, gap = 0.5s (green running, both on track)
        {"race_id": "pit_test", "lap_number": 2, "driver": "VER", "session_time_s": 200.0, "lap_time_s": 90.0, "stint": 1, "compound": "MEDIUM", "tyre_age_laps": 2, "track_status": "1", "is_clean": True},
        {"race_id": "pit_test", "lap_number": 2, "driver": "HAM", "session_time_s": 200.5, "lap_time_s": 90.0, "stint": 1, "compound": "SOFT", "tyre_age_laps": 2, "track_status": "1", "is_clean": True},
        # Lap 3: VER pits (in-lap). Crossing gap happens to be 0.6s
        {"race_id": "pit_test", "lap_number": 3, "driver": "VER", "session_time_s": 300.0, "lap_time_s": 100.0, "stint": 1, "compound": "MEDIUM", "tyre_age_laps": 3, "track_status": "1", "is_clean": False},
        {"race_id": "pit_test", "lap_number": 3, "driver": "HAM", "session_time_s": 300.6, "lap_time_s": 89.5, "stint": 1, "compound": "SOFT", "tyre_age_laps": 3, "track_status": "1", "is_clean": True},
        # Lap 4: VER out-lap. Crossing gap 0.4s
        {"race_id": "pit_test", "lap_number": 4, "driver": "VER", "session_time_s": 400.0, "lap_time_s": 100.0, "stint": 2, "compound": "HARD", "tyre_age_laps": 1, "track_status": "1", "is_clean": False},
        {"race_id": "pit_test", "lap_number": 4, "driver": "HAM", "session_time_s": 400.4, "lap_time_s": 89.5, "stint": 1, "compound": "SOFT", "tyre_age_laps": 4, "track_status": "1", "is_clean": True},
        # Lap 5: VER back on track, gap 2.5s
        {"race_id": "pit_test", "lap_number": 5, "driver": "VER", "session_time_s": 490.0, "lap_time_s": 90.0, "stint": 2, "compound": "HARD", "tyre_age_laps": 2, "track_status": "1", "is_clean": True},
        {"race_id": "pit_test", "lap_number": 5, "driver": "HAM", "session_time_s": 492.5, "lap_time_s": 90.0, "stint": 1, "compound": "SOFT", "tyre_age_laps": 5, "track_status": "1", "is_clean": True},
    ]
    stints_df = pd.DataFrame([
        {"driver": "VER", "stint": 1, "compound": "MEDIUM", "start_lap": 1, "end_lap": 3, "pit_in_lap": 3},
        {"driver": "VER", "stint": 2, "compound": "HARD", "start_lap": 4, "end_lap": 5, "pit_in_lap": None},
        {"driver": "HAM", "stint": 1, "compound": "SOFT", "start_lap": 1, "end_lap": 5, "pit_in_lap": None},
    ])
    laps_df = pd.DataFrame(laps_records)

    # 1. Verify build_overtake_dataset excludes lap 3 and lap 4 candidate pairs
    ds = build_overtake_dataset([{"race_id": "pit_test", "laps": laps_df, "stints": stints_df}], candidate_gap_threshold_s=3.0, min_lap=2)
    # Only lap 2 (and lap 5) can be candidates. Lap 3 and 4 must be completely excluded
    candidate_laps = set(ds["lap_number"].unique())
    assert 3 not in candidate_laps
    assert 4 not in candidate_laps

    # 2. Verify extract_pair_sample at lap 2 does NOT count pit laps (laps 3 and 4) as battle laps
    sample = extract_pair_sample(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=2,
        horizon_laps=3,
        total_laps=5,
    )
    assert sample is not None
    # In laps 3, 4, 5: lap 3 and 4 were pit laps (excluded), lap 5 gap was 2.5s (> 1.0s)
    # Therefore battle_laps_count_next_H must be 0, not 2!
    assert sample.battle_laps_count_next_H == 0
    assert sample.battle_next_H_laps == 0


def test_future_mutation_invariance_and_unknown_scheduled_laps(synthetic_race_data):
    """
    Requirement 3 Verification:
    1. Features at as_of_lap must be invariant to any mutations, pit changes, position shifts,
       or lap extensions in future telemetry (lap > as_of_lap).
    2. Unknown scheduled lap counts (total_laps=None) must NOT be inferred from future telemetry,
       ensuring laps_remaining is None rather than leaking max(telemetry_laps).
    """
    laps_df, stints_df = synthetic_race_data
    as_of_lap = 3

    # Baseline features with unknown scheduled laps (total_laps=None)
    feat_base = extract_pair_features(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=as_of_lap,
        race_id="synth_gp",
        total_laps=None,
    )
    assert feat_base is not None
    # Crucial causal property: unknown scheduled total_laps must NOT infer 15 from future telemetry
    assert feat_base.laps_remaining is None

    # Baseline features with explicit official scheduled laps (e.g. 53)
    feat_base_sched = extract_pair_features(
        laps_df=laps_df,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=as_of_lap,
        race_id="synth_gp",
        total_laps=53,
    )
    assert feat_base_sched is not None
    assert feat_base_sched.laps_remaining == 50  # 53 - 3

    # Create mutated future telemetry (lap > 3 completely altered)
    mutated_laps = laps_df.copy()
    future_mask = mutated_laps["lap_number"] > as_of_lap
    
    # Invert positions, alter times by hundreds of seconds, change track status, inject pit stops
    mutated_laps.loc[future_mask, "session_time_s"] += 5000.0
    mutated_laps.loc[future_mask, "lap_time_s"] = 150.0
    mutated_laps.loc[future_mask, "position"] = 20
    mutated_laps.loc[future_mask, "track_status"] = "5"  # Red flag
    mutated_laps.loc[future_mask, "is_clean"] = False
    mutated_laps.loc[future_mask, "is_pit_in"] = True

    # Also append 50 extra future laps to test telemetry length expansion
    extra_rows = []
    for l in range(16, 66):
        extra_rows.append({
            "race_id": "synth_gp",
            "lap_number": l,
            "driver": "VER",
            "session_time_s": 6000.0 + l * 90.0,
            "lap_time_s": 90.0,
            "stint": 3,
            "compound": "SOFT",
            "tyre_age_laps": l - 15,
            "position": 1,
            "track_status": "1",
            "is_clean": True,
            "is_pit_in": False,
            "is_pit_out": False,
        })
        extra_rows.append({
            "race_id": "synth_gp",
            "lap_number": l,
            "driver": "HAM",
            "session_time_s": 6000.5 + l * 90.0,
            "lap_time_s": 90.0,
            "stint": 2,
            "compound": "MEDIUM",
            "tyre_age_laps": l - 15,
            "position": 2,
            "track_status": "1",
            "is_clean": True,
            "is_pit_in": False,
            "is_pit_out": False,
        })
    mutated_laps = pd.concat([mutated_laps, pd.DataFrame(extra_rows)], ignore_index=True)

    # Extract features from heavily mutated telemetry
    feat_mutated = extract_pair_features(
        laps_df=mutated_laps,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=as_of_lap,
        race_id="synth_gp",
        total_laps=None,
    )
    assert feat_mutated is not None

    # Every causal feature at as_of_lap=3 MUST be identical to baseline
    assert feat_mutated.gap_s == feat_base.gap_s
    assert feat_mutated.closing_rate_s_per_lap == feat_base.closing_rate_s_per_lap
    assert feat_mutated.is_closing == feat_base.is_closing
    assert feat_mutated.clean_pace_delta_s == feat_base.clean_pace_delta_s
    assert feat_mutated.attacker_recent_pace_s == feat_base.attacker_recent_pace_s
    assert feat_mutated.defender_recent_pace_s == feat_base.defender_recent_pace_s
    assert feat_mutated.attacker_clean_laps_stint == feat_base.attacker_clean_laps_stint
    assert feat_mutated.defender_clean_laps_stint == feat_base.defender_clean_laps_stint
    assert feat_mutated.data_sufficiency == feat_base.data_sufficiency
    assert feat_mutated.tyre_age_delta_laps == feat_base.tyre_age_delta_laps
    assert feat_mutated.compound_delta_tier == feat_base.compound_delta_tier
    assert feat_mutated.is_currently_battling == feat_base.is_currently_battling
    assert feat_mutated.laps_remaining == feat_base.laps_remaining  # Both None

    # Also test with scheduled laps (53): remaining laps is 50 regardless of whether telemetry has 15 or 65 laps
    feat_mutated_sched = extract_pair_features(
        laps_df=mutated_laps,
        stints_df=stints_df,
        attacker="HAM",
        defender="VER",
        as_of_lap=as_of_lap,
        race_id="synth_gp",
        total_laps=53,
    )
    assert feat_mutated_sched.laps_remaining == 50
    assert feat_mutated_sched.laps_remaining == feat_base_sched.laps_remaining

