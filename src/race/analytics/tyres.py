"""
Tyre degradation and performance cliff detection module for RACE.
Estimates fuel-corrected stint pace slopes using ordinary least squares regression
over strictly verified clean laps.
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats

from src.race.config import FUEL_CORRECTION_S_PER_LAP

logger = logging.getLogger(__name__)


def compute_tyre_degradation(
    laps_df: pd.DataFrame,
    driver: str,
    stint: int,
    fuel_correction_s_per_lap: float = FUEL_CORRECTION_S_PER_LAP,
    as_of_lap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Computes fuel-corrected pace degradation slope (s/lap) for a driver's stint
    and detects performance cliff laps.

    The calculated degradation slope is a statistical model estimate, not a direct
    physical measurement of tyre compound wear.

    Parameters
    ----------
    laps_df : pd.DataFrame
        Cached laps DataFrame containing 'driver', 'stint', 'lap_number',
        'lap_time_s', 'is_clean', 'compound', and 'tyre_age_laps'.
    driver : str
        Three-letter driver code (e.g. 'VER', 'LEC').
    stint : int
        Stint index (1, 2, ...).
    fuel_correction_s_per_lap : float
        Pace benefit per lap burned (default: 0.033 s/lap per approved plan).
    as_of_lap : int, optional
        Replay boundary. Laps beyond as_of_lap are strictly excluded.

    Returns
    -------
    dict
        Structured output containing estimated degradation slope, confidence,
        fit metrics, and cliff lap.
    """
    if laps_df is None or laps_df.empty:
        return _empty_degradation_result(driver, stint)

    # Filter to driver and stint
    drv_laps = laps_df[(laps_df["driver"] == driver) & (laps_df["stint"] == stint)].copy()
    if as_of_lap is not None:
        drv_laps = drv_laps[drv_laps["lap_number"] <= as_of_lap]

    if drv_laps.empty:
        return _empty_degradation_result(driver, stint)

    compound = drv_laps["compound"].iloc[0] if "compound" in drv_laps.columns else "UNKNOWN"

    # Strictly filter to clean laps per PDF definition
    clean_laps = drv_laps[drv_laps["is_clean"] == True].dropna(subset=["lap_time_s"]).sort_values("lap_number")
    clean_count = len(clean_laps)

    # If fewer than 4 clean laps, slope is statistically unreliable
    if clean_count < 4:
        return {
            "driver": driver,
            "stint": stint,
            "compound": compound,
            "is_estimate": True,
            "fuel_correction_applied_s_per_lap": fuel_correction_s_per_lap,
            "raw_slope_s_per_lap": None,
            "estimated_deg_rate_s_per_lap": None,
            "clean_laps_count": clean_count,
            "r_squared": None,
            "std_error_s_per_lap": None,
            "p_value": None,
            "cliff_lap": None,
            "confidence": "INSUFFICIENT_DATA",
            "message": f"Stint has only {clean_count} clean laps (minimum 4 required for reliable estimation).",
        }

    # Relative lap index in stint (1-indexed)
    lap_nums = clean_laps["lap_number"].values
    rel_laps = np.arange(1, clean_count + 1)
    raw_times = clean_laps["lap_time_s"].values

    # Apply linear fuel correction: t_corr(l) = t_obs(l) + alpha * (l - 1)
    corr_times = raw_times + fuel_correction_s_per_lap * (rel_laps - 1)

    # Fit OLS regression
    slope_corr, intercept_corr, r_val, p_val, std_err = stats.linregress(rel_laps, corr_times)
    slope_raw, _, _, _, _ = stats.linregress(rel_laps, raw_times)

    # Detect tyre cliff
    # Residuals from linear fit: epsilon = t_corr - (intercept + slope * rel_laps)
    residuals = corr_times - (intercept_corr + slope_corr * rel_laps)
    res_std = np.std(residuals)
    cliff_lap = None

    if res_std > 0 and len(residuals) >= 5:
        # Check for 2 consecutive laps exceeding 2.5 standard deviations above trend
        outliers = np.where(residuals > 2.5 * res_std)[0]
        for i in range(len(outliers) - 1):
            if outliers[i + 1] == outliers[i] + 1:
                cliff_lap = int(lap_nums[outliers[i]])
                break

    # Confidence assessment
    if clean_count >= 6 and (r_val**2 > 0.3 or std_err < 0.05):
        confidence = "HIGH"
    else:
        confidence = "MEDIUM"

    return {
        "driver": driver,
        "stint": stint,
        "compound": compound,
        "is_estimate": True,
        "fuel_correction_applied_s_per_lap": fuel_correction_s_per_lap,
        "raw_slope_s_per_lap": round(float(slope_raw), 4),
        "estimated_deg_rate_s_per_lap": round(float(slope_corr), 4),
        "clean_laps_count": clean_count,
        "r_squared": round(float(r_val**2), 3),
        "std_error_s_per_lap": round(float(std_err), 4),
        "p_value": round(float(p_val), 4),
        "cliff_lap": cliff_lap,
        "confidence": confidence,
    }


def _empty_degradation_result(driver: str, stint: int) -> Dict[str, Any]:
    return {
        "driver": driver,
        "stint": stint,
        "compound": "UNKNOWN",
        "is_estimate": True,
        "fuel_correction_applied_s_per_lap": FUEL_CORRECTION_S_PER_LAP,
        "raw_slope_s_per_lap": None,
        "estimated_deg_rate_s_per_lap": None,
        "clean_laps_count": 0,
        "r_squared": None,
        "std_error_s_per_lap": None,
        "p_value": None,
        "cliff_lap": None,
        "confidence": "INSUFFICIENT_DATA",
    }
