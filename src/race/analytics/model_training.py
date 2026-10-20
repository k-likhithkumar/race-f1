"""
Model training, evaluation, and leave-one-race-out backtesting for Battle & Overtake Probability (M4).

Implements:
- Two separate regularized logistic regression pipelines (Battle model, Overtake model).
- Strict train/test preprocessing isolation (imputer and scaler fit exclusively on training partition).
- Explicit missing-value handling via median imputation with missing indicators.
- Leave-one-race-out (LORO) backtesting protocol against training-set constant base-rate baselines.
- Comprehensive probabilistic evaluation (Brier score, log-loss, ROC-AUC, calibration bins, confusion matrix).
- Robust edge-case handling for single-class test partitions and insufficient data.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)

# Standard causal feature columns used by both models
PREDICTION_FEATURES: List[str] = [
    "gap_s",
    "closing_rate_s_per_lap",
    "is_closing",
    "clean_pace_delta_s",
    "tyre_age_delta_laps",
    "compound_delta_tier",
    "is_currently_battling",
    "attacker_clean_laps_stint",
    "defender_clean_laps_stint",
    "laps_remaining",
]


@dataclass(frozen=True)
class CalibrationBin:
    """Observed vs predicted calibration statistics for a probability range."""
    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_predicted_prob: Optional[float]
    observed_frequency: Optional[float]
    calibration_error: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelEvaluationReport:
    """Complete evaluation report for a held-out test partition."""
    held_out_race_id: str
    target_name: str
    n_train: int
    n_test: int
    pos_train: int
    pos_test: int
    train_base_rate: float
    baseline_brier_score: float
    baseline_log_loss: float
    model_brier_score: float
    brier_delta_vs_baseline: float  # Negative = model beats baseline
    model_log_loss: float
    roc_auc: Optional[float]
    classification_threshold: float
    precision: float
    recall: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    calibration_bins: List[Dict[str, Any]]
    mean_calibration_error: Optional[float]
    model_coefficients: Dict[str, float]
    model_intercept: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def create_logistic_pipeline(
    c_regularization: float = 1.0,
    random_state: int = 42,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Constructs an isolated preprocessing and estimation pipeline.

    Preprocessing:
    1. SimpleImputer(strategy='median', add_indicator=True): Imputes missing numeric
       features while appending binary indicator columns for missingness.
    2. StandardScaler(): Normalizes features to zero mean and unit variance.
    3. LogisticRegression: L2-penalized deterministic logistic classifier.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=c_regularization,
            random_state=random_state,
            max_iter=max_iter,
            solver="lbfgs",
        )),
    ])


def compute_calibration_bins(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 5,
) -> Tuple[List[CalibrationBin], Optional[float]]:
    """
    Computes preliminary calibration metrics across probability intervals.

    Weighted Calibration Error (WCE) formula:
        WCE = (1 / N) * sum_{b=1}^{B} N_b * |mean_p_b - obs_y_b|
    where:
        - B is the number of bins (default: 5)
        - N is the total sample count: N = sum(N_b)
        - N_b is the sample count in bin b
        - mean_p_b is the mean predicted probability in bin b
        - obs_y_b is the observed empirical event frequency in bin b

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground truth labels (0 or 1).
    y_probs : np.ndarray
        Predicted probabilities in [0.0, 1.0].
    n_bins : int
        Number of equal-width bins between 0.0 and 1.0.

    Returns
    -------
    list of CalibrationBin, float
        Bin statistics and weighted calibration error (WCE).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_probs, bins) - 1
    # Clamp edge case where prob == 1.0 into highest bin
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_records: List[CalibrationBin] = []
    weighted_error_sum = 0.0
    total_samples = len(y_true)

    for b in range(n_bins):
        b_lower = round(float(bins[b]), 2)
        b_upper = round(float(bins[b + 1]), 2)
        mask = (bin_indices == b)
        count = int(np.sum(mask))

        if count > 0:
            mean_pred = round(float(np.mean(y_probs[mask])), 4)
            obs_freq = round(float(np.mean(y_true[mask])), 4)
            err = round(abs(mean_pred - obs_freq), 4)
            weighted_error_sum += err * count
        else:
            mean_pred = None
            obs_freq = None
            err = None

        bin_records.append(CalibrationBin(
            bin_lower=b_lower,
            bin_upper=b_upper,
            sample_count=count,
            mean_predicted_prob=mean_pred,
            observed_frequency=obs_freq,
            calibration_error=err,
        ))

    mean_cal_err = round(weighted_error_sum / total_samples, 4) if total_samples > 0 else None
    return bin_records, mean_cal_err


def find_optimal_training_threshold(
    y_train_true: np.ndarray,
    y_train_probs: np.ndarray,
    metric: str = "f1",
    n_candidates: int = 81,
) -> float:
    """
    Selects a decision threshold strictly using the training partition.

    Never tune a threshold on held-out test data. Changing the classification threshold
    shifts the binary operational trade-off (precision vs. recall), but does NOT alter
    the underlying model probabilities, Brier score, or calibration.

    Parameters
    ----------
    y_train_true : np.ndarray
        Training partition ground truth (0 or 1).
    y_train_probs : np.ndarray
        Training partition predicted probabilities in [0.0, 1.0].
    metric : str
        Target metric to optimize on training partition (default 'f1').
    n_candidates : int
        Number of threshold candidates to evaluate between 0.10 and 0.90.

    Returns
    -------
    float
        Optimal decision threshold for the training partition.
    """
    from sklearn.metrics import f1_score

    thresholds = np.linspace(0.10, 0.90, n_candidates)
    best_threshold = 0.50
    best_score = -1.0

    for t in thresholds:
        preds = (y_train_probs >= t).astype(int)
        if metric == "f1":
            score = float(f1_score(y_train_true, preds, zero_division=0))
        else:
            score = float(f1_score(y_train_true, preds, zero_division=0))

        if score > best_score:
            best_score = score
            best_threshold = round(float(t), 4)

    return best_threshold


def evaluate_held_out_partition(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_name: str,
    held_out_race_id: str,
    feature_cols: List[str] = PREDICTION_FEATURES,
    threshold: float = 0.5,
    c_regularization: float = 1.0,
    random_state: int = 42,
) -> ModelEvaluationReport:
    """
    Trains on train_df and evaluates strictly on test_df with zero preprocessing leakage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data partition (must exclude held_out_race_id).
    test_df : pd.DataFrame
        Test data partition (held-out race).
    target_name : str
        Target column ('battle_next_H_laps' or 'pass_next_H_laps').
    held_out_race_id : str
        Identifier of held-out race.
    feature_cols : list of str
        Features to feed into pipeline.
    threshold : float
        Decision threshold for binary classification metrics (default 0.5).
    c_regularization : float
        L2 regularization parameter C.
    random_state : int
        Deterministic seed.

    Returns
    -------
    ModelEvaluationReport
    """
    if train_df.empty:
        raise ValueError("Training partition is empty.")
    if test_df.empty:
        raise ValueError("Test partition is empty.")

    X_train = train_df[feature_cols].copy()
    y_train = train_df[target_name].astype(int).values

    X_test = test_df[feature_cols].copy()
    y_test = test_df[target_name].astype(int).values

    n_train = len(y_train)
    n_test = len(y_test)
    pos_train = int(np.sum(y_train))
    pos_test = int(np.sum(y_test))

    # Base-rate baseline derived strictly from training partition
    train_base_rate = float(np.mean(y_train))
    baseline_probs = np.full_like(y_test, fill_value=train_base_rate, dtype=float)

    baseline_brier = float(brier_score_loss(y_test, baseline_probs))
    baseline_ll = float(log_loss(y_test, baseline_probs, labels=[0, 1]))

    # Fit pipeline strictly on training data
    pipeline = create_logistic_pipeline(
        c_regularization=c_regularization,
        random_state=random_state,
    )
    pipeline.fit(X_train, y_train)

    # Predict probabilities on test partition
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_preds = (y_probs >= threshold).astype(int)

    # Compute model metrics
    model_brier = float(brier_score_loss(y_test, y_probs))
    brier_delta = float(model_brier - baseline_brier)
    model_ll = float(log_loss(y_test, y_probs, labels=[0, 1]))

    # Single-class guard for ROC-AUC
    if len(np.unique(y_test)) > 1:
        roc_auc = round(float(roc_auc_score(y_test, y_probs)), 4)
    else:
        # Cannot calculate ROC-AUC with only one class present in test set
        roc_auc = None

    prec = float(precision_score(y_test, y_preds, zero_division=0))
    rec = float(recall_score(y_test, y_preds, zero_division=0))

    cm = confusion_matrix(y_test, y_preds, labels=[0, 1])
    tn, fp, fn, tp = [int(val) for val in cm.ravel()]

    cal_bins, mean_cal_err = compute_calibration_bins(y_test, y_probs, n_bins=5)

    # Extract model coefficients
    clf: LogisticRegression = pipeline.named_steps["clf"]
    imputer: SimpleImputer = pipeline.named_steps["imputer"]

    # Map coefficients to feature names including any missing indicators
    base_feats = list(feature_cols)
    if hasattr(imputer, "indicator_") and imputer.indicator_ is not None:
        indicator_indices = imputer.indicator_.features_
        for idx in indicator_indices:
            base_feats.append(f"{feature_cols[idx]}_is_missing")

    coef_dict: Dict[str, float] = {}
    if len(clf.coef_[0]) == len(base_feats):
        for f_name, w in zip(base_feats, clf.coef_[0]):
            coef_dict[f_name] = round(float(w), 4)

    intercept = round(float(clf.intercept_[0]), 4)

    return ModelEvaluationReport(
        held_out_race_id=held_out_race_id,
        target_name=target_name,
        n_train=n_train,
        n_test=n_test,
        pos_train=pos_train,
        pos_test=pos_test,
        train_base_rate=round(train_base_rate, 4),
        baseline_brier_score=round(baseline_brier, 4),
        baseline_log_loss=round(baseline_ll, 4),
        model_brier_score=round(model_brier, 4),
        brier_delta_vs_baseline=round(brier_delta, 4),
        model_log_loss=round(model_ll, 4),
        roc_auc=roc_auc,
        classification_threshold=threshold,
        precision=round(prec, 4),
        recall=round(rec, 4),
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
        calibration_bins=[b.to_dict() for b in cal_bins],
        mean_calibration_error=mean_cal_err,
        model_coefficients=coef_dict,
        model_intercept=intercept,
    )


def run_leave_one_race_out_backtest(
    dataset_df: pd.DataFrame,
    target_name: str,
    feature_cols: List[str] = PREDICTION_FEATURES,
    threshold: float = 0.5,
    c_regularization: float = 1.0,
    random_state: int = 42,
) -> Dict[str, ModelEvaluationReport]:
    """
    Executes Leave-One-Race-Out (LORO) backtesting over all races in dataset_df.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Candidate pairs dataset (must include 'race_id', 'label_valid', target_name, and feature_cols).
    target_name : str
        Target label column ('battle_next_H_laps' or 'pass_next_H_laps').
    feature_cols : list of str
        Features for estimation.
    threshold : float
        Decision threshold.
    c_regularization : float
        Regularization strength.
    random_state : int
        Seed.

    Returns
    -------
    dict
        Mapping of held_out_race_id -> ModelEvaluationReport.
    """
    if dataset_df.empty:
        return {}

    # Filter to strictly valid labels
    valid_df = dataset_df[dataset_df["label_valid"] == True].copy()
    if valid_df.empty:
        return {}

    unique_races = valid_df["race_id"].unique()
    reports: Dict[str, ModelEvaluationReport] = {}

    for race_id in unique_races:
        train_df = valid_df[valid_df["race_id"] != race_id]
        test_df = valid_df[valid_df["race_id"] == race_id]

        if train_df.empty:
            logger.warning(f"Skipping LORO for {race_id}: no other races available for training.")
            continue

        report = evaluate_held_out_partition(
            train_df=train_df,
            test_df=test_df,
            target_name=target_name,
            held_out_race_id=race_id,
            feature_cols=feature_cols,
            threshold=threshold,
            c_regularization=c_regularization,
            random_state=random_state,
        )
        reports[race_id] = report

    return reports


def compute_pooled_loro_backtest(
    dataset_df: pd.DataFrame,
    target_name: str,
    feature_cols: List[str] = PREDICTION_FEATURES,
    threshold: float = 0.5,
    c_regularization: float = 1.0,
    random_state: int = 42,
    n_bins: int = 5,
) -> Tuple[Dict[str, ModelEvaluationReport], List[CalibrationBin], Optional[float], pd.DataFrame]:
    """
    Executes Leave-One-Race-Out (LORO) backtesting and aggregates out-of-fold predictions
    across all held-out partitions into a single pooled calibration evaluation.

    Parameters
    ----------
    dataset_df : pd.DataFrame
        Candidate pairs dataset.
    target_name : str
        Target column ('battle_next_H_laps' or 'pass_next_H_laps').
    feature_cols : list of str
        Features for estimation.
    threshold : float
        Decision threshold.
    c_regularization : float
        Regularization strength C.
    random_state : int
        Seed.
    n_bins : int
        Number of equal-width calibration bins (default 5).

    Returns
    -------
    tuple
        (reports_per_race, pooled_calibration_bins, pooled_wce, pooled_predictions_df)
    """
    valid_df = dataset_df[dataset_df["label_valid"] == True].copy()
    if valid_df.empty:
        return {}, [], None, pd.DataFrame()

    unique_races = valid_df["race_id"].unique()
    reports: Dict[str, ModelEvaluationReport] = {}
    pooled_rows: List[Dict[str, Any]] = []

    for race_id in unique_races:
        train_df = valid_df[valid_df["race_id"] != race_id]
        test_df = valid_df[valid_df["race_id"] == race_id]

        if train_df.empty:
            continue

        report = evaluate_held_out_partition(
            train_df=train_df,
            test_df=test_df,
            target_name=target_name,
            held_out_race_id=race_id,
            feature_cols=feature_cols,
            threshold=threshold,
            c_regularization=c_regularization,
            random_state=random_state,
        )
        reports[race_id] = report

        # Extract out-of-fold predictions for test_df
        pipeline = create_logistic_pipeline(
            c_regularization=c_regularization,
            random_state=random_state,
        )
        pipeline.fit(train_df[feature_cols], train_df[target_name].astype(int).values)
        test_probs = pipeline.predict_proba(test_df[feature_cols])[:, 1]
        test_targets = test_df[target_name].astype(int).values

        for y_t, y_p in zip(test_targets, test_probs):
            pooled_rows.append({
                "race_id": race_id,
                "y_true": int(y_t),
                "y_prob": float(y_p),
            })

    pred_df = pd.DataFrame(pooled_rows)
    if pred_df.empty:
        return reports, [], None, pred_df

    pooled_bins, pooled_wce = compute_calibration_bins(
        y_true=pred_df["y_true"].values,
        y_probs=pred_df["y_prob"].values,
        n_bins=n_bins,
    )
    return reports, pooled_bins, pooled_wce, pred_df

