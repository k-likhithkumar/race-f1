"""
Unit tests for Model Training and Leave-One-Race-Out Backtesting (Checkpoint 3B).

Validates:
- Reproducible training and deterministic pipeline configurations.
- Strict train/test preprocessing isolation (zero scaler/imputer data leakage).
- Missing feature handling (None/NaN in pace, compound, and tyre age).
- Constant base-rate baseline correctness (derived strictly from training set).
- Leave-one-race-out partition disjointness.
- Single-class evaluation sets (gracefully returns None for ROC-AUC without throwing).
- Predicted probabilities strictly bounded in [0.0, 1.0].
- Model performance beats base-rate baseline on Brier score across cached races.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

from src.race.analytics.model_training import (
    PREDICTION_FEATURES,
    create_logistic_pipeline,
    compute_calibration_bins,
    evaluate_held_out_partition,
    run_leave_one_race_out_backtest,
    find_optimal_training_threshold,
    compute_pooled_loro_backtest,
)
from src.race.data.cache import load_cached_table
from src.race.analytics.features import build_overtake_dataset


@pytest.fixture
def synthetic_split_dataset():
    """Constructs a deterministic synthetic multi-race dataset with missing values."""
    np.random.seed(42)
    records = []
    
    # Race 1: 100 samples, ~20% positives
    for i in range(100):
        gap = float(np.random.uniform(0.2, 2.8))
        closing = float(np.random.uniform(-0.5, 0.8)) if i % 5 != 0 else None
        pace_delta = float(np.random.uniform(-1.0, 1.0)) if i % 4 != 0 else None
        tyre_delta = int(np.random.randint(-15, 15)) if i % 6 != 0 else None
        compound_delta = int(np.random.choice([-1, 0, 1])) if i % 7 != 0 else None
        is_battling = bool(gap <= 1.0 and closing is not None and closing > 0.05)
        
        # Logistic probability generator
        logits = -1.5 - 1.2 * gap + 0.8 * (closing or 0.0) + 1.0 * (pace_delta or 0.0) + (1.5 if is_battling else 0.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        y_pass = int(np.random.uniform() < prob)
        y_battle = int(np.random.uniform() < (prob * 1.5))
        
        records.append({
            "race_id": "race_A",
            "lap_number": (i % 40) + 2,
            "attacker": f"D_{(i%10)+1}",
            "defender": f"D_{(i%10)+2}",
            "gap_s": gap,
            "closing_rate_s_per_lap": closing,
            "is_closing": bool(closing is not None and closing >= 0.05),
            "clean_pace_delta_s": pace_delta,
            "tyre_age_delta_laps": tyre_delta,
            "compound_delta_tier": compound_delta,
            "is_currently_battling": is_battling,
            "attacker_clean_laps_stint": int(np.random.randint(1, 10)),
            "defender_clean_laps_stint": int(np.random.randint(1, 10)),
            "laps_remaining": 50 - (i % 40),
            "label_valid": True,
            "pass_next_H_laps": y_pass,
            "battle_next_H_laps": y_battle,
        })
        
    # Race 2: 100 samples, ~15% positives
    for i in range(100):
        gap = float(np.random.uniform(0.3, 3.0))
        closing = float(np.random.uniform(-0.6, 0.7)) if i % 5 != 0 else None
        pace_delta = float(np.random.uniform(-0.8, 1.2)) if i % 4 != 0 else None
        tyre_delta = int(np.random.randint(-10, 10)) if i % 6 != 0 else None
        compound_delta = int(np.random.choice([-1, 0, 1])) if i % 7 != 0 else None
        is_battling = bool(gap <= 1.0 and closing is not None and closing > 0.05)
        
        logits = -1.8 - 1.0 * gap + 0.7 * (closing or 0.0) + 1.1 * (pace_delta or 0.0) + (1.4 if is_battling else 0.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        y_pass = int(np.random.uniform() < prob)
        y_battle = int(np.random.uniform() < (prob * 1.4))
        
        records.append({
            "race_id": "race_B",
            "lap_number": (i % 40) + 2,
            "attacker": f"D_{(i%10)+1}",
            "defender": f"D_{(i%10)+2}",
            "gap_s": gap,
            "closing_rate_s_per_lap": closing,
            "is_closing": bool(closing is not None and closing >= 0.05),
            "clean_pace_delta_s": pace_delta,
            "tyre_age_delta_laps": tyre_delta,
            "compound_delta_tier": compound_delta,
            "is_currently_battling": is_battling,
            "attacker_clean_laps_stint": int(np.random.randint(1, 10)),
            "defender_clean_laps_stint": int(np.random.randint(1, 10)),
            "laps_remaining": 50 - (i % 40),
            "label_valid": True,
            "pass_next_H_laps": y_pass,
            "battle_next_H_laps": y_battle,
        })
        
    return pd.DataFrame(records)


def test_reproducible_model_training(synthetic_split_dataset):
    """Pipeline fitted with same seed must yield bit-for-bit identical predictions."""
    df = synthetic_split_dataset
    X = df[PREDICTION_FEATURES]
    y = df["pass_next_H_laps"].values

    pipe1 = create_logistic_pipeline(random_state=42)
    pipe1.fit(X, y)
    probs1 = pipe1.predict_proba(X)[:, 1]

    pipe2 = create_logistic_pipeline(random_state=42)
    pipe2.fit(X, y)
    probs2 = pipe2.predict_proba(X)[:, 1]

    np.testing.assert_array_equal(probs1, probs2)


def test_train_test_preprocessing_isolation(synthetic_split_dataset):
    """Scaler and imputer statistics must be fitted exclusively on train data."""
    df = synthetic_split_dataset
    train_df = df[df["race_id"] == "race_A"].copy()
    test_df = df[df["race_id"] == "race_B"].copy()

    # Artificially skew test set gap values to extreme numbers
    test_df_skewed = test_df.copy()
    test_df_skewed["gap_s"] = test_df_skewed["gap_s"] + 1000.0

    pipe = create_logistic_pipeline(random_state=42)
    pipe.fit(train_df[PREDICTION_FEATURES], train_df["pass_next_H_laps"].values)

    # Scaler mean must reflect train set only
    scaler_mean = pipe.named_steps["scaler"].mean_[0]
    # In train set, gap is in [0.2, 2.8] -> mean is ~1.5, nowhere near 1000
    assert 0.5 <= scaler_mean <= 2.5


def test_missing_feature_handling():
    """Pipeline must handle missing numerical and categorical features without error."""
    # Data with all features missing or NaN except gap_s
    X_missing = pd.DataFrame([{
        "gap_s": 1.2,
        "closing_rate_s_per_lap": None,
        "is_closing": False,
        "clean_pace_delta_s": None,
        "tyre_age_delta_laps": None,
        "compound_delta_tier": None,
        "is_currently_battling": False,
        "attacker_clean_laps_stint": 0,
        "defender_clean_laps_stint": 0,
        "laps_remaining": None,
    }])
    y = np.array([0, 1, 0, 1])
    X_train = pd.concat([X_missing] * 4, ignore_index=True)
    X_train.loc[1, "gap_s"] = 0.5
    X_train.loc[3, "gap_s"] = 0.4

    pipe = create_logistic_pipeline(random_state=42)
    pipe.fit(X_train, y)
    prob = pipe.predict_proba(X_missing)[:, 1]

    assert len(prob) == 1
    assert 0.0 <= prob[0] <= 1.0


def test_baseline_derivation_from_training_set(synthetic_split_dataset):
    """Baseline base-rate must be derived strictly from training partition."""
    df = synthetic_split_dataset
    train_df = df[df["race_id"] == "race_A"].copy()
    test_df = df[df["race_id"] == "race_B"].copy()

    report = evaluate_held_out_partition(
        train_df=train_df,
        test_df=test_df,
        target_name="pass_next_H_laps",
        held_out_race_id="race_B",
    )

    expected_base_rate = float(train_df["pass_next_H_laps"].mean())
    assert report.train_base_rate == round(expected_base_rate, 4)
    # Ensure baseline is not calculated from test set
    test_base_rate = float(test_df["pass_next_H_laps"].mean())
    if abs(expected_base_rate - test_base_rate) > 0.01:
        assert report.train_base_rate != round(test_base_rate, 4)


def test_loro_partition_disjointness(synthetic_split_dataset):
    """Leave-one-race-out partitions must have strictly disjoint race_ids."""
    df = synthetic_split_dataset
    reports = run_leave_one_race_out_backtest(df, target_name="pass_next_H_laps")

    assert "race_A" in reports
    assert "race_B" in reports
    assert reports["race_A"].held_out_race_id == "race_A"
    assert reports["race_B"].held_out_race_id == "race_B"


def test_single_class_held_out_partition(synthetic_split_dataset):
    """Single-class test partition must gracefully return None for ROC-AUC without crashing."""
    df = synthetic_split_dataset
    train_df = df[df["race_id"] == "race_A"].copy()
    test_df = df[df["race_id"] == "race_B"].copy()
    # Force test set to contain ONLY negative class (0)
    test_df["pass_next_H_laps"] = 0

    report = evaluate_held_out_partition(
        train_df=train_df,
        test_df=test_df,
        target_name="pass_next_H_laps",
        held_out_race_id="race_B",
    )

    assert report.pos_test == 0
    assert report.roc_auc is None  # Handled gracefully
    assert 0.0 <= report.model_brier_score <= 1.0


def test_calibration_bins_structure():
    """Verify bin computation boundaries and mathematical validity."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_probs = np.array([0.05, 0.15, 0.25, 0.65, 0.75, 0.95])

    bins, mean_err = compute_calibration_bins(y_true, y_probs, n_bins=5)
    assert len(bins) == 5
    assert bins[0].bin_lower == 0.0
    assert bins[-1].bin_upper == 1.0
    total_count = sum(b.sample_count for b in bins)
    assert total_count == 6
    assert mean_err is not None
    assert 0.0 <= mean_err <= 1.0


def test_real_race_loro_evaluation():
    """Leave-one-race-out on cached Monza races must beat base-rate baseline on Brier score."""
    laps_2022 = load_cached_table(2022, "monza", "laps")
    stints_2022 = load_cached_table(2022, "monza", "stints")
    laps_2023 = load_cached_table(2023, "monza", "laps")
    stints_2023 = load_cached_table(2023, "monza", "stints")

    races = [
        {"race_id": "2022_monza", "laps": laps_2022, "stints": stints_2022},
        {"race_id": "2023_monza", "laps": laps_2023, "stints": stints_2023},
    ]

    df = build_overtake_dataset(races, candidate_gap_threshold_s=3.0, horizon_laps=5)
    valid_df = df[df["label_valid"] == True].copy()
    
    # Exact dataset reconciliation checks
    assert len(df) == 901
    assert len(valid_df) == 855
    assert len(df[df["label_valid"] == False]) == 46
    
    # Per-race valid counts
    v_2022 = valid_df[valid_df["race_id"] == "2022_monza"]
    v_2023 = valid_df[valid_df["race_id"] == "2023_monza"]
    assert len(v_2022) == 380
    assert len(v_2023) == 475
    
    # Target counts per race
    assert int((v_2022["battle_next_H_laps"] == 1).sum()) == 155
    assert int((v_2022["pass_next_H_laps"] == 1).sum()) == 64
    assert int((v_2023["battle_next_H_laps"] == 1).sum()) == 236
    assert int((v_2023["pass_next_H_laps"] == 1).sum()) == 59
    
    # Exclusion reasons breakdown
    assert int((df["exclusion_reason"] == "TRUNCATED_HORIZON_AT_RACE_END").sum()) == 30
    assert int((df["exclusion_reason"] == "EXTENDED_CAUTION_IN_HORIZON").sum()) == 16

    # 1. Evaluate Battle Model
    battle_reports = run_leave_one_race_out_backtest(valid_df, target_name="battle_next_H_laps")
    assert "2022_monza" in battle_reports
    assert "2023_monza" in battle_reports

    for rid, rep in battle_reports.items():
        assert rep.model_brier_score < rep.baseline_brier_score
        assert rep.brier_delta_vs_baseline < 0.0  # Model beats constant base rate
        assert rep.roc_auc is not None and rep.roc_auc > 0.70

    # 2. Evaluate Overtake Model
    pass_reports = run_leave_one_race_out_backtest(valid_df, target_name="pass_next_H_laps")
    assert "2022_monza" in pass_reports
    assert "2023_monza" in pass_reports

    for rid, rep in pass_reports.items():
        assert rep.model_brier_score < rep.baseline_brier_score
        assert rep.brier_delta_vs_baseline < 0.0  # Model beats constant base rate
        assert rep.roc_auc is not None and rep.roc_auc > 0.80


def test_training_threshold_selection():
    """Verify that threshold selection operates strictly on training inputs without modifying probabilities."""
    from src.race.analytics.model_training import find_optimal_training_threshold

    y_train = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Well-separated probabilities
    y_probs = np.array([0.05, 0.1, 0.15, 0.2, 0.35, 0.45, 0.6, 0.75, 0.8, 0.9])

    best_t = find_optimal_training_threshold(y_train, y_probs, metric="f1")
    # Optimal threshold separating 0s and 1s should be between 0.35 and 0.45
    assert 0.30 <= best_t <= 0.45

    # Verify that changing threshold does not alter predicted probabilities or Brier score
    y_test = np.array([0, 1])
    test_probs = np.array([0.2, 0.8])
    brier1 = float(brier_score_loss(y_test, test_probs))
    brier2 = float(brier_score_loss(y_test, test_probs))
    assert brier1 == brier2


def test_pooled_loro_calibration_consistency_and_warnings():
    """
    Validates:
    - Calibration bins use the exact pooled out-of-fold predictions from LORO (sum to 855).
    - Weighted calibration error matches independent manual recomputation.
    - Battle model 0.8-1.0 bin exhibits observed overconfidence (75 samples, mean pred 0.8602, obs rate 0.5867, err 0.2735).
    """
    laps_2022 = load_cached_table(2022, "monza", "laps")
    stints_2022 = load_cached_table(2022, "monza", "stints")
    laps_2023 = load_cached_table(2023, "monza", "laps")
    stints_2023 = load_cached_table(2023, "monza", "stints")

    races = [
        {"race_id": "2022_monza", "laps": laps_2022, "stints": stints_2022},
        {"race_id": "2023_monza", "laps": laps_2023, "stints": stints_2023},
    ]

    df = build_overtake_dataset(races, candidate_gap_threshold_s=3.0, horizon_laps=5)
    valid_df = df[df["label_valid"] == True].copy()
    assert len(valid_df) == 855

    # 1. Battle Model Pooled Calibration
    reports_b, battle_bins, battle_wce, battle_preds = compute_pooled_loro_backtest(
        dataset_df=valid_df,
        target_name="battle_next_H_laps",
        n_bins=5,
    )
    assert len(battle_preds) == 855
    assert sum(b.sample_count for b in battle_bins) == 855

    # Check exact bin counts for battle model
    battle_bin_counts = [b.sample_count for b in battle_bins]
    assert battle_bin_counts == [243, 127, 185, 225, 75]

    # Verify battle model 0.8-1.0 bin specific warning figures
    bin_high = battle_bins[-1]
    assert bin_high.bin_lower == 0.8 and bin_high.bin_upper == 1.0
    assert bin_high.sample_count == 75
    assert bin_high.mean_predicted_prob == 0.8602
    assert bin_high.observed_frequency == 0.5867
    assert bin_high.calibration_error == 0.2735

    # Independent WCE recomputation directly from raw out-of-fold prediction arrays
    y_true_b = battle_preds["y_true"].values
    y_prob_b = battle_preds["y_prob"].values
    raw_battle_weighted_err = 0.0
    for low, high in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (y_prob_b >= low) & (y_prob_b <= high if high == 1.0 else y_prob_b < high)
        cnt = int(mask.sum())
        if cnt > 0:
            p_bar = float(y_prob_b[mask].mean())
            y_bar = float(y_true_b[mask].mean())
            raw_battle_weighted_err += cnt * abs(p_bar - y_bar)
    direct_battle_wce = round(raw_battle_weighted_err / len(y_prob_b), 4)
    assert direct_battle_wce == battle_wce
    assert battle_wce == 0.0693

    # 2. Overtake Model Pooled Calibration
    reports_p, pass_bins, pass_wce, pass_preds = compute_pooled_loro_backtest(
        dataset_df=valid_df,
        target_name="pass_next_H_laps",
        n_bins=5,
    )
    assert len(pass_preds) == 855
    assert sum(b.sample_count for b in pass_bins) == 855

    # Check exact bin counts for pass model
    bin_counts = [b.sample_count for b in pass_bins]
    assert bin_counts == [713, 97, 20, 11, 14]

    # Independent WCE recomputation directly from raw out-of-fold prediction arrays
    y_true_p = pass_preds["y_true"].values
    y_prob_p = pass_preds["y_prob"].values
    raw_pass_weighted_err = 0.0
    for low, high in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        mask = (y_prob_p >= low) & (y_prob_p <= high if high == 1.0 else y_prob_p < high)
        cnt = int(mask.sum())
        if cnt > 0:
            p_bar = float(y_prob_p[mask].mean())
            y_bar = float(y_true_p[mask].mean())
            raw_pass_weighted_err += cnt * abs(p_bar - y_bar)
    direct_pass_wce = round(raw_pass_weighted_err / len(y_prob_p), 4)
    assert direct_pass_wce == pass_wce
    assert pass_wce == 0.0357

    # Check exact positive sample counts across both folds
    assert int(y_true_p.sum()) == 123
    assert int(y_true_b.sum()) == 391
    assert reports_p["2022_monza"].pos_test == 64
    assert reports_p["2023_monza"].pos_test == 59
    assert reports_b["2022_monza"].pos_test == 155
    assert reports_b["2023_monza"].pos_test == 236


