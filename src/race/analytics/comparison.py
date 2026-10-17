"""
Multi-driver lap and telemetry comparison engine for RACE.
Provides unified comparison across drivers, arbitrary lap ranges, stint selections,
and lap-number vs. tyre-age synchronizations.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from src.race.config import FUEL_CORRECTION_S_PER_LAP

logger = logging.getLogger(__name__)


def compare_laps(
    race_data: Dict[str, Any],
    drivers: List[str],
    lap_selection: Union[int, List[int], str, tuple] = "fastest",
    metrics: Optional[List[str]] = None,
    reference_driver: Optional[str] = None,
    sync_mode: str = "lap",
    clean_only: bool = True,
    fuel_corrected: bool = False,
    as_of_lap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    One entry point for the driver comparison screen.

    Parameters
    ----------
    race_data : dict
        Dict containing loaded race tables ('laps', 'stints', 'track_outline').
    drivers : list of str
        List of 3-letter driver codes to compare (e.g. ['VER', 'LEC', 'HAM']).
    lap_selection : int, list of int, str, or tuple
        Selection mode:
        - int: specific single lap number (e.g. 32)
        - list of int: specific lap numbers (e.g. [10, 11, 12])
        - 'fastest': fastest lap for each driver
        - ('stint', stint_num): all laps within a specific stint
        - ('last_clean', N): last N clean laps up to as_of_lap
    metrics : list of str, optional
        Requested metrics: ['lap_time', 'sector_times', 'compound', 'tyre_age'].
    reference_driver : str, optional
        Baseline driver for delta computations. Defaults to first selected driver.
    sync_mode : str
        'lap' (aligned on lap_number) or 'tyre_age' (aligned on tyre_age_laps).
    clean_only : bool
        If True, excludes pit-in, pit-out, caution, and deleted laps (default: True).
    fuel_corrected : bool
        If True, applies linear fuel burn correction to lap times (default: False).
    as_of_lap : int, optional
        Replay boundary. Laps beyond as_of_lap are excluded.

    Returns
    -------
    dict
        Structured output with 'summary_table', 'series_data', and 'chart_spec'.
    """
    if metrics is None:
        metrics = ["lap_time", "sector_times", "compound", "tyre_age"]

    laps_df = race_data.get("laps")
    if laps_df is None or laps_df.empty:
        return _empty_comparison()

    df = laps_df.copy()
    if as_of_lap is not None:
        df = df[df["lap_number"] <= as_of_lap]

    if clean_only:
        df = df[df["is_clean"] == True]

    # Filter to requested drivers
    df = df[df["driver"].isin(drivers)]
    if df.empty:
        return _empty_comparison()

    ref_drv = reference_driver if reference_driver in drivers else drivers[0]

    # Resolve lap selection for each driver
    selected_laps_list = []

    for drv in drivers:
        drv_df = df[df["driver"] == drv].copy()
        if drv_df.empty:
            continue

        if isinstance(lap_selection, int):
            matched = drv_df[drv_df["lap_number"] == lap_selection]
        elif isinstance(lap_selection, list):
            matched = drv_df[drv_df["lap_number"].isin(lap_selection)]
        elif lap_selection == "fastest":
            fastest_idx = drv_df["lap_time_s"].idxmin() if drv_df["lap_time_s"].notna().any() else None
            matched = drv_df.loc[[fastest_idx]] if fastest_idx is not None else pd.DataFrame()
        elif isinstance(lap_selection, tuple) and lap_selection[0] == "stint":
            stint_num = lap_selection[1]
            matched = drv_df[drv_df["stint"] == stint_num]
        elif isinstance(lap_selection, tuple) and lap_selection[0] == "last_clean":
            n = lap_selection[1]
            matched = drv_df.sort_values("lap_number").tail(n)
        else:
            matched = drv_df

        selected_laps_list.append(matched)

    if not selected_laps_list:
        return _empty_comparison()

    comp_df = pd.concat(selected_laps_list, ignore_index=True)

    # Apply fuel correction if requested: t_corr = t_obs + alpha * (lap_num - 1)
    if fuel_corrected:
        comp_df["lap_time_s_raw"] = comp_df["lap_time_s"]
        comp_df["lap_time_s"] = comp_df["lap_time_s_raw"] + FUEL_CORRECTION_S_PER_LAP * (comp_df["lap_number"] - 1)

    # Determine x-axis alignment key
    align_key = "tyre_age_laps" if sync_mode == "tyre_age" else "lap_number"

    # Compute deltas relative to reference driver
    ref_df = comp_df[comp_df["driver"] == ref_drv].set_index(align_key)

    series_data = {}
    for drv in drivers:
        drv_sub = comp_df[comp_df["driver"] == drv].set_index(align_key)
        drv_series = {}

        for key in drv_sub.index:
            lap_t = drv_sub.loc[key, "lap_time_s"] if "lap_time_s" in drv_sub.columns else np.nan
            if isinstance(lap_t, pd.Series):
                lap_t = lap_t.iloc[0]

            delta_vs_ref = np.nan
            if key in ref_df.index:
                ref_t = ref_df.loc[key, "lap_time_s"]
                if isinstance(ref_t, pd.Series):
                    ref_t = ref_t.iloc[0]
                if pd.notna(lap_t) and pd.notna(ref_t):
                    delta_vs_ref = round(float(lap_t - ref_t), 3)

            drv_series[int(key)] = {
                "lap_time_s": round(float(lap_t), 3) if pd.notna(lap_t) else None,
                "delta_to_reference_s": delta_vs_ref,
                "compound": drv_sub.loc[key, "compound"] if "compound" in drv_sub.columns else "UNKNOWN",
                "tyre_age_laps": int(drv_sub.loc[key, "tyre_age_laps"]) if "tyre_age_laps" in drv_sub.columns else None,
                "s1_s": round(float(drv_sub.loc[key, "s1_s"]), 3) if "s1_s" in drv_sub.columns and pd.notna(drv_sub.loc[key, "s1_s"]) else None,
                "s2_s": round(float(drv_sub.loc[key, "s2_s"]), 3) if "s2_s" in drv_sub.columns and pd.notna(drv_sub.loc[key, "s2_s"]) else None,
                "s3_s": round(float(drv_sub.loc[key, "s3_s"]), 3) if "s3_s" in drv_sub.columns and pd.notna(drv_sub.loc[key, "s3_s"]) else None,
            }

        series_data[drv] = drv_series

    # Format summary table
    summary_cols = ["driver", "lap_number", "lap_time_s", "compound", "tyre_age_laps", "stint"]
    if "s1_s" in comp_df.columns:
        summary_cols.extend(["s1_s", "s2_s", "s3_s"])
    avail_cols = [c for c in summary_cols if c in comp_df.columns]
    summary_table = comp_df[avail_cols].sort_values(["driver", align_key]).reset_index(drop=True)

    return {
        "reference_driver": ref_drv,
        "sync_mode": sync_mode,
        "align_key": align_key,
        "clean_only": clean_only,
        "fuel_corrected": fuel_corrected,
        "summary_table": summary_table,
        "series_data": series_data,
        "chart_spec": {
            "x_axis": align_key,
            "x_label": "Tyre Age (Laps)" if sync_mode == "tyre_age" else "Lap Number",
            "y_axis": "lap_time_s",
            "y_label": "Fuel-Corrected Lap Time (s)" if fuel_corrected else "Lap Time (s)",
            "drivers": drivers,
        },
    }


def _empty_comparison() -> Dict[str, Any]:
    return {
        "reference_driver": None,
        "sync_mode": "lap",
        "align_key": "lap_number",
        "clean_only": True,
        "fuel_corrected": False,
        "summary_table": pd.DataFrame(),
        "series_data": {},
        "chart_spec": {},
    }
