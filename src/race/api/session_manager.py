"""
SessionManager for RACE Replay API.
Manages authoritative ReplayAdapter instances and WebSocket subscribers.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any
import asyncio

import numpy as np
import pandas as pd

from src.race.config import PARQUET_CACHE_DIR, FUEL_CORRECTION_S_PER_LAP
from src.race.replay.session import ReplaySession
from src.race.replay.clock import ReplayClock, ClockState
from src.race.replay.adapter import ReplayAdapter
from src.race.replay.state import RaceState, get_state
from src.race.replay.visibility import visible_state
from src.race.api.schemas import (
    RaceSummary,
    ClockStatus,
    ComparisonRequest,
    ComparisonResponse,
    ComparisonLapItem,
    MiniSectorTimingItem,
    TyreDegradationResponse,
    TyreLapPoint,
    StintSummary,
    GapBattleResponse,
    GapLapItem,
    BattleEventItem,
    OvertakeEvidenceItem,
    GapBattleSummary,
    BattleOvertakePredictionResponse,
    PredictionItem,
    SupportingFeatures,
    ModelReliabilityWarning,
    PitStrategyResponse,
    PitStopItem,
    TacticalEvidence,
    TimeLossFactors,
    TimeLossBreakdownItem,
)
from src.race.analytics.telemetry_loader import load_lap_telemetry
from src.race.analytics.mini_sectors import compute_mini_sectors
from src.race.analytics.tyres import compute_tyre_degradation
from src.race.analytics.gaps import compute_gap_history, detect_battles, classify_overtakes
from src.race.analytics.features import extract_pair_features
from src.race.analytics.strategy import analyze_pit_stops, compute_time_loss_breakdown


logger = logging.getLogger(__name__)


class SessionManager:
    """
    Singleton-style manager for cached replay sessions and active adapters.
    Ensures exactly one authoritative ReplayAdapter exists per race_id.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir: Path = cache_dir if cache_dir is not None else PARQUET_CACHE_DIR
        self._sessions: Dict[str, ReplaySession] = {}
        self._adapters: Dict[str, ReplayAdapter] = {}
        self._subscribers: Dict[str, Set[Any]] = {}
        self._tick_tasks: Dict[str, asyncio.Task] = {}
        self._finished_notified: Dict[str, bool] = {}
        self._last_tick_times: Dict[str, float] = {}
        self._mini_sector_cache: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self._overtakes_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    def _get_mini_sectors_for_lap(
        self,
        session: ReplaySession,
        driver: str,
        lap_number: int,
        lap_time_s: float,
    ) -> Dict[str, Any]:
        """
        Loads lap telemetry and computes mini-sectors using compute_mini_sectors.
        Handles missing telemetry, telemetry gaps, and provider metadata.
        Cached in-memory by (race_id, driver, lap_number).
        """
        cache_key = (session.race_id, driver, lap_number)
        if cache_key in self._mini_sector_cache:
            return self._mini_sector_cache[cache_key]

        try:
            tel_df = load_lap_telemetry(session.year, session.gp, driver, lap_number)
        except Exception as exc:
            logger.warning(f"Error loading telemetry for {driver} lap {lap_number} in {session.race_id}: {exc}")
            tel_df = pd.DataFrame()

        if tel_df is None or tel_df.empty:
            res = {
                "telemetry_status": "MISSING",
                "provider": None,
                "precision_tier": None,
                "mini_sectors": [],
            }
            self._mini_sector_cache[cache_key] = res
            return res

        ms_res = compute_mini_sectors(
            telemetry_df=tel_df,
            lap_time_s=lap_time_s,
            track_outline=session.track_outline,
            num_mini_sectors=20,
        )

        raw_status = ms_res.get("status", "VALID")
        if raw_status == "VALID":
            tel_status = "VALID"
        elif raw_status == "UNRELIABLE_DATA_GAP":
            tel_status = "UNRELIABLE_DATA_GAP"
        else:
            tel_status = "MISSING"

        res = {
            "telemetry_status": tel_status,
            "provider": ms_res.get("provider"),
            "precision_tier": ms_res.get("precision_tier"),
            "mini_sectors": ms_res.get("mini_sectors", []),
            "discrepancy_s": ms_res.get("discrepancy_s"),
            "is_accurate": ms_res.get("is_accurate"),
        }
        self._mini_sector_cache[cache_key] = res
        return res


    def normalize_race_id(self, race_id: str) -> str:
        """Converts '2022-monza' or '2022_Monza' to canonical slug '2022_monza'."""
        clean = race_id.strip().lower().replace("-", "_")
        return clean

    def list_available_races(self) -> List[RaceSummary]:
        """
        Discovers all valid cached races by inspecting subdirectories of cache_dir.
        """
        summaries: List[RaceSummary] = []
        if not self.cache_dir.exists():
            return summaries

        for sub in sorted(self.cache_dir.iterdir()):
            if not sub.is_dir():
                continue
            meta_path = sub / "metadata.json"
            if not meta_path.exists():
                continue

            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                year = int(meta["year"])
                gp = str(meta["gp"])
                slug = self.normalize_race_id(f"{year}_{gp}")

                # Load session lazily to obtain exact T_0 and T_finish
                session = self.get_or_load_session(slug)
                summaries.append(
                    RaceSummary(
                        race_id=slug,
                        year=year,
                        gp=gp,
                        official_event_name=meta.get("official_event_name", f"{year} {gp} Grand Prix"),
                        total_laps=session.total_laps,
                        winner=meta.get("winner"),
                        driver_count=int(meta.get("driver_count", len(session.drivers))),
                        T_0=session.T_0,
                        T_finish=session.T_finish,
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to load cached race from {sub}: {e}")

        return summaries

    def _parse_slug(self, normalized_slug: str) -> Tuple[int, str]:
        parts = normalized_slug.split("_", 1)
        if len(parts) != 2 or not parts[0].isdigit():
            raise ValueError(f"Invalid race slug format: '{normalized_slug}'. Expected 'YYYY_gp'.")
        return int(parts[0]), parts[1]

    def get_or_load_session(self, race_id: str) -> ReplaySession:
        """Loads and caches ReplaySession for given race_id."""
        slug = self.normalize_race_id(race_id)
        if slug not in self._sessions:
            year, gp = self._parse_slug(slug)
            try:
                session = ReplaySession(year, gp)
            except FileNotFoundError:
                session = ReplaySession(year, gp.capitalize())
            self._sessions[slug] = session
        return self._sessions[slug]

    def get_track_outline(self, race_id: str) -> Dict[str, Any]:
        """
        Retrieves cached track outline geometry for race_id.
        Raises FileNotFoundError if race is not found or track outline is missing.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)
        if not hasattr(session, "track_outline") or not session.track_outline:
            raise FileNotFoundError(f"Track outline geometry not found for race '{race_id}'.")
        return session.track_outline


    def get_or_create_adapter(self, race_id: str) -> ReplayAdapter:
        """
        Retrieves or initializes the single authoritative ReplayAdapter for race_id.
        """
        slug = self.normalize_race_id(race_id)
        if slug not in self._adapters:
            session = self.get_or_load_session(slug)
            clock = ReplayClock(session)
            adapter = ReplayAdapter(session, clock)
            self._adapters[slug] = adapter
            self._subscribers[slug] = set()
            self._finished_notified[slug] = False
        return self._adapters[slug]

    def get_clock_status(self, race_id: str) -> ClockStatus:
        adapter = self.get_or_create_adapter(race_id)
        clock = adapter.clock
        return ClockStatus(
            session_time_s=round(clock.current_time_s, 3),
            state=clock.state.value,
            speed=clock.speed,
            current_lap=clock.current_lap,
            total_laps=clock.total_laps,
            T_0=round(clock.T_0, 3),
            T_finish=round(clock.T_finish, 3),
        )

    def get_public_state(self, race_id: str, as_of: Optional[float] = None) -> RaceState:
        """
        Retrieves point-in-time RaceState and filters it through public visibility.
        If as_of is provided, queries get_state directly without mutating the clock.
        If as_of is None, queries current clock time of the active adapter.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)
        if as_of is not None:
            raw_state = get_state(session, as_of)
        else:
            adapter = self.get_or_create_adapter(slug)
            raw_state = adapter.poll()

        # Public timing feed: team=None
        return visible_state(raw_state, team=None)

    def get_driver_comparison(self, race_id: str, req: ComparisonRequest) -> ComparisonResponse:
        """
        Executes multi-driver lap comparison with strict per-driver point-in-time
        as_of line-crossing chronology enforcement and fuel correction.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)

        if not req.drivers:
            raise ValueError("At least one driver must be specified for comparison.")

        invalid_drivers = [d for d in req.drivers if d not in session.drivers]
        if invalid_drivers:
            raise ValueError(f"Unknown driver(s): {', '.join(invalid_drivers)}")

        # Resolve reference driver
        if req.reference_driver:
            if req.reference_driver not in session.drivers:
                raise ValueError(f"Reference driver '{req.reference_driver}' is not in the race.")
            ref_drv = req.reference_driver
        else:
            ref_drv = req.drivers[0]

        # Validate lap_mode parameters upfront
        if req.lap_mode == "single":
            if req.lap_number is None:
                raise ValueError("lap_number is required when lap_mode is 'single'")
            if req.lap_number < 1 or req.lap_number > session.total_laps:
                raise ValueError(f"lap_number {req.lap_number} is out of bounds [1, {session.total_laps}]")
        elif req.lap_mode == "range":
            if not req.lap_range or len(req.lap_range) != 2:
                raise ValueError("lap_range must be a list/tuple of two integers [start, end]")
            start_l, end_l = req.lap_range
            if start_l > end_l:
                raise ValueError(f"Invalid lap range: start {start_l} > end {end_l}")
            if start_l < 1 or end_l > session.total_laps:
                raise ValueError(f"lap_range [{start_l}, {end_l}] is out of bounds [1, {session.total_laps}]")
        elif req.lap_mode == "stint":
            if req.stint_number is None or req.stint_number < 1:
                raise ValueError("stint_number (positive integer) is required when lap_mode is 'stint'")
        elif req.lap_mode == "last_clean":
            if req.last_n_clean is not None and req.last_n_clean < 1:
                raise ValueError("last_n_clean must be a positive integer")
        elif req.lap_mode != "fastest":
            raise ValueError(f"Unknown lap_mode: '{req.lap_mode}'")

        if req.sync_mode not in ("lap", "tyre_age"):
            raise ValueError(f"Invalid sync_mode '{req.sync_mode}'. Supported values: 'lap', 'tyre_age'.")

        if req.include_mini_sectors and req.lap_mode not in ("single", "fastest"):
            raise ValueError(
                f"Mini-sector comparison is not supported for lap_mode '{req.lap_mode}'. "
                "Only 'single' and 'fastest' lap modes are supported."
            )

        # Resolve authoritative simulation timestamp t_as_of
        if req.as_of is not None:
            t_as_of = float(req.as_of)
        else:
            if slug in self._adapters:
                t_as_of = float(self._adapters[slug].clock.current_time_s)
            else:
                adapter = self.get_or_create_adapter(slug)
                t_as_of = float(adapter.clock.current_time_s)

        align_key = "tyre_age_laps" if req.sync_mode == "tyre_age" else "lap_number"

        # Pre-race check: no race laps completed before T_0
        if t_as_of < session.T_0:
            return ComparisonResponse(
                race_id=slug,
                status="PRE_RACE",
                reference_driver=ref_drv,
                sync_mode=req.sync_mode,
                align_key=align_key,
                clean_only=req.clean_only,
                fuel_corrected=req.fuel_corrected,
                include_mini_sectors=req.include_mini_sectors,
                as_of_time_s=round(t_as_of, 3),
                total_completed_laps=0,
                laps=[],
            )

        # Post-race clamping: time cannot exceed session finish
        t_as_of_clamped = min(t_as_of, session.T_finish)

        # Step 1: Filter each driver's completed laps at t_as_of_clamped
        all_needed_drivers = list(dict.fromkeys(req.drivers + [ref_drv]))
        eligible_by_driver: Dict[str, pd.DataFrame] = {}

        for d in all_needed_drivers:
            crossing_times = session.driver_lap_crossing_times.get(d, np.array([]))
            lap_numbers = session.driver_lap_numbers.get(d, np.array([]))
            idx = np.searchsorted(crossing_times, t_as_of_clamped, side="right") - 1
            if idx >= 0 and len(lap_numbers) > idx:
                max_lap = int(lap_numbers[idx])
                df_drv = session.driver_laps[d]
                df_drv = df_drv[df_drv["lap_number"] <= max_lap].copy()
            else:
                df_drv = pd.DataFrame()

            if req.clean_only and not df_drv.empty:
                df_drv = df_drv[df_drv["is_clean"] == True]

            eligible_by_driver[d] = df_drv

        # Step 2: Resolve lap selection mode per driver
        selected_by_driver: Dict[str, pd.DataFrame] = {}

        # For single mode with tyre_age sync, establish reference driver's target tyre age and stint
        ref_target_tyre_age = None
        ref_target_stint = None
        if req.lap_mode == "single" and req.sync_mode == "tyre_age":
            ref_eligible = eligible_by_driver.get(ref_drv, pd.DataFrame())
            ref_match = ref_eligible[ref_eligible["lap_number"] == req.lap_number]
            if not ref_match.empty and pd.notna(ref_match.iloc[0].get("tyre_age_laps")):
                ref_target_tyre_age = int(ref_match.iloc[0]["tyre_age_laps"])
                ref_target_stint = int(ref_match.iloc[0]["stint"]) if pd.notna(ref_match.iloc[0].get("stint")) else None

        for d in all_needed_drivers:
            df_drv = eligible_by_driver.get(d, pd.DataFrame())
            if df_drv.empty:
                selected_by_driver[d] = pd.DataFrame()
                continue

            if req.lap_mode == "single":
                if req.sync_mode == "tyre_age":
                    if ref_target_tyre_age is not None:
                        # Match within the same stint life, fallback to head(1)
                        matched = df_drv[df_drv["tyre_age_laps"] == ref_target_tyre_age]
                        if ref_target_stint is not None and not matched.empty:
                            stint_matched = matched[matched["stint"] == ref_target_stint]
                            if not stint_matched.empty:
                                matched = stint_matched
                        matched = matched.head(1)
                    else:
                        matched = pd.DataFrame()
                else:
                    matched = df_drv[df_drv["lap_number"] == req.lap_number]

            elif req.lap_mode == "range":
                start_l, end_l = req.lap_range
                matched = df_drv[(df_drv["lap_number"] >= start_l) & (df_drv["lap_number"] <= end_l)]

            elif req.lap_mode == "fastest":
                valid_laps = df_drv.dropna(subset=["lap_time_s"])
                valid_laps = valid_laps[valid_laps["lap_time_s"] > 0]
                if not valid_laps.empty:
                    fastest_idx = valid_laps["lap_time_s"].idxmin()
                    matched = df_drv.loc[[fastest_idx]]
                else:
                    matched = pd.DataFrame()

            elif req.lap_mode == "stint":
                if req.stint_number is None:
                    raise ValueError("stint_number is required when lap_mode is 'stint'")
                matched = df_drv[df_drv["stint"] == req.stint_number]

            elif req.lap_mode == "last_clean":
                n = req.last_n_clean if req.last_n_clean is not None else 5
                matched = df_drv.sort_values("lap_number").tail(n)

            else:
                raise ValueError(f"Unknown lap_mode: '{req.lap_mode}'")

            selected_by_driver[d] = matched

        non_empty = [df for df in selected_by_driver.values() if not df.empty]
        if not non_empty:
            return ComparisonResponse(
                race_id=slug,
                status="NO_DATA",
                reference_driver=ref_drv,
                sync_mode=req.sync_mode,
                align_key=align_key,
                clean_only=req.clean_only,
                fuel_corrected=req.fuel_corrected,
                include_mini_sectors=req.include_mini_sectors,
                as_of_time_s=round(t_as_of, 3),
                total_completed_laps=0,
                laps=[],
            )

        comp_df = pd.concat(non_empty, ignore_index=True)

        # Step 3: Apply fuel correction if requested
        if req.fuel_corrected:
            comp_df["lap_time_s_raw"] = comp_df["lap_time_s"]
            comp_df["lap_time_s"] = comp_df["lap_time_s_raw"] + FUEL_CORRECTION_S_PER_LAP * (comp_df["lap_number"] - 1)

        # Step 4: Index reference driver times for delta lookups
        # Alignment semantics:
        # - When sync_mode == "lap": indexed by lap_number (int).
        # - When sync_mode == "tyre_age": indexed by composite key (stint, tyre_age_laps).
        #   This ensures repeated tyre-age values from different stints cannot collide
        #   or overwrite each other, preserves race lap numbers, and produces None (null)
        #   when the reference driver has no observation for that (stint, tyre_age).
        ref_sub = comp_df[comp_df["driver"] == ref_drv]
        if req.sync_mode == "tyre_age":
            ref_times_tyre: Dict[Tuple[int, int], float] = {}
            for _, r in ref_sub.iterrows():
                s_val = r.get("stint")
                age_val = r.get("tyre_age_laps")
                t_val = r.get("lap_time_s")
                if pd.notna(s_val) and pd.notna(age_val) and pd.notna(t_val):
                    ref_times_tyre[(int(s_val), int(age_val))] = float(t_val)
        else:
            ref_times_lap: Dict[int, float] = {}
            for _, r in ref_sub.iterrows():
                k = r.get("lap_number")
                t_val = r.get("lap_time_s")
                if pd.notna(k) and pd.notna(t_val):
                    ref_times_lap[int(k)] = float(t_val)

        # Step 4b: Pre-calculate reference driver mini-sectors if mini-sectors requested
        ref_ms_durations: Dict[int, float] = {}
        static_mini_sectors: List[Dict[str, Any]] = []

        if req.include_mini_sectors:
            if session.track_outline and "mini_sectors" in session.track_outline:
                for ms in session.track_outline["mini_sectors"]:
                    static_mini_sectors.append({
                        "mini_sector_id": int(ms["mini_sector_id"]),
                        "sector": int(ms.get("sector", 1)),
                        "start_m": round(float(ms["start_m"]), 2),
                        "end_m": round(float(ms["end_m"]), 2),
                    })
            else:
                total_d = session.total_circuit_distance_m if hasattr(session, "total_circuit_distance_m") else 5000.0
                step = total_d / 20.0
                for i in range(20):
                    static_mini_sectors.append({
                        "mini_sector_id": i + 1,
                        "sector": 1 if i < 7 else (2 if i < 14 else 3),
                        "start_m": round(i * step, 2),
                        "end_m": round((i + 1) * step, 2),
                    })

            # Check if reference driver has an eligible lap observation
            if not ref_sub.empty:
                ref_row = ref_sub.iloc[0]
                ref_lap_num = int(ref_row["lap_number"])
                ref_raw_time = float(ref_row.get("lap_time_s_raw", ref_row["lap_time_s"])) if pd.notna(ref_row.get("lap_time_s_raw", ref_row.get("lap_time_s"))) else np.nan
                ref_ms_data = self._get_mini_sectors_for_lap(session, ref_drv, ref_lap_num, ref_raw_time)
                if ref_ms_data["telemetry_status"] == "VALID":
                    for ms in ref_ms_data["mini_sectors"]:
                        ms_id = int(ms["mini_sector_id"])
                        dur = ms.get("duration_s")
                        if dur is not None and not np.isnan(dur):
                            ref_ms_durations[ms_id] = float(dur)

        # Step 5: Format typed lap items for requested drivers
        lap_items: List[ComparisonLapItem] = []
        sort_keys = ["stint", "tyre_age_laps"] if req.sync_mode == "tyre_age" else ["lap_number"]
        for drv in req.drivers:
            drv_rows = comp_df[comp_df["driver"] == drv]
            if drv_rows.empty:
                continue
            sorted_rows = drv_rows.sort_values(sort_keys)
            for _, r in sorted_rows.iterrows():
                t_val = r.get("lap_time_s")
                delta_s = None
                if req.sync_mode == "tyre_age":
                    s_val = r.get("stint")
                    age_val = r.get("tyre_age_laps")
                    if pd.notna(s_val) and pd.notna(age_val) and pd.notna(t_val):
                        k_pair = (int(s_val), int(age_val))
                        if k_pair in ref_times_tyre:
                            delta_s = round(float(t_val) - ref_times_tyre[k_pair], 3)
                else:
                    k = r.get("lap_number")
                    if pd.notna(k) and int(k) in ref_times_lap and pd.notna(t_val):
                        delta_s = round(float(t_val) - ref_times_lap[int(k)], 3)

                mini_sectors_items: Optional[List[MiniSectorTimingItem]] = None
                drv_tel_status: Optional[str] = None
                drv_provider: Optional[str] = None
                drv_precision_tier: Optional[str] = None

                if req.include_mini_sectors:
                    lap_num = int(r["lap_number"])
                    raw_time = float(r.get("lap_time_s_raw", r["lap_time_s"])) if pd.notna(r.get("lap_time_s_raw", r.get("lap_time_s"))) else np.nan
                    ms_data = self._get_mini_sectors_for_lap(session, str(r["driver"]), lap_num, raw_time)
                    drv_tel_status = ms_data["telemetry_status"]
                    drv_provider = ms_data.get("provider")
                    drv_precision_tier = ms_data.get("precision_tier")

                    mini_sectors_items = []
                    if drv_tel_status == "VALID":
                        computed_by_id = {int(ms["mini_sector_id"]): ms for ms in ms_data["mini_sectors"]}
                        for sms in static_mini_sectors:
                            ms_id = sms["mini_sector_id"]
                            if ms_id in computed_by_id:
                                cms = computed_by_id[ms_id]
                                dur = cms.get("duration_s")
                                dur_val = round(float(dur), 4) if (dur is not None and not np.isnan(dur)) else None
                                spd = cms.get("avg_speed_kmh")
                                spd_val = round(float(spd), 1) if (spd is not None and not np.isnan(spd)) else None

                                delta_val = None
                                if dur_val is not None and ms_id in ref_ms_durations:
                                    if str(r["driver"]) == ref_drv:
                                        delta_val = 0.0
                                    else:
                                        delta_val = round(dur_val - ref_ms_durations[ms_id], 4)

                                mini_sectors_items.append(
                                    MiniSectorTimingItem(
                                        mini_sector_id=ms_id,
                                        sector=int(cms.get("sector", sms["sector"])),
                                        start_m=round(float(cms.get("start_m", sms["start_m"])), 2),
                                        end_m=round(float(cms.get("end_m", sms["end_m"])), 2),
                                        duration_s=dur_val,
                                        delta_to_reference_s=delta_val,
                                        avg_speed_kmh=spd_val,
                                        status="VALID" if dur_val is not None else "MISSING",
                                    )
                                )
                            else:
                                mini_sectors_items.append(
                                    MiniSectorTimingItem(
                                        mini_sector_id=ms_id,
                                        sector=sms["sector"],
                                        start_m=sms["start_m"],
                                        end_m=sms["end_m"],
                                        duration_s=None,
                                        delta_to_reference_s=None,
                                        avg_speed_kmh=None,
                                        status="MISSING",
                                    )
                                )
                    elif drv_tel_status == "UNRELIABLE_DATA_GAP":
                        for sms in static_mini_sectors:
                            mini_sectors_items.append(
                                MiniSectorTimingItem(
                                    mini_sector_id=sms["mini_sector_id"],
                                    sector=sms["sector"],
                                    start_m=sms["start_m"],
                                    end_m=sms["end_m"],
                                    duration_s=None,
                                    delta_to_reference_s=None,
                                    avg_speed_kmh=None,
                                    status="UNRELIABLE_DATA_GAP",
                                )
                            )
                    else:  # MISSING or other
                        for sms in static_mini_sectors:
                            mini_sectors_items.append(
                                MiniSectorTimingItem(
                                    mini_sector_id=sms["mini_sector_id"],
                                    sector=sms["sector"],
                                    start_m=sms["start_m"],
                                    end_m=sms["end_m"],
                                    duration_s=None,
                                    delta_to_reference_s=None,
                                    avg_speed_kmh=None,
                                    status="MISSING",
                                )
                            )

                lap_items.append(
                    ComparisonLapItem(
                        driver=str(r["driver"]),
                        lap_number=int(r["lap_number"]),
                        tyre_age_laps=int(r["tyre_age_laps"]) if pd.notna(r.get("tyre_age_laps")) else None,
                        stint=int(r["stint"]) if pd.notna(r.get("stint")) else None,
                        lap_time_s=round(float(t_val), 3) if pd.notna(t_val) else None,
                        delta_to_reference_s=delta_s,
                        s1_s=round(float(r["s1_s"]), 3) if "s1_s" in r and pd.notna(r.get("s1_s")) else None,
                        s2_s=round(float(r["s2_s"]), 3) if "s2_s" in r and pd.notna(r.get("s2_s")) else None,
                        s3_s=round(float(r["s3_s"]), 3) if "s3_s" in r and pd.notna(r.get("s3_s")) else None,
                        compound=str(r.get("compound", "UNKNOWN")),
                        is_clean=bool(r.get("is_clean", True)),
                        mini_sectors=mini_sectors_items,
                        telemetry_status=drv_tel_status,
                        provider=drv_provider,
                        precision_tier=drv_precision_tier,
                    )
                )

        return ComparisonResponse(
            race_id=slug,
            status="SUCCESS",
            reference_driver=ref_drv,
            sync_mode=req.sync_mode,
            align_key=align_key,
            clean_only=req.clean_only,
            fuel_corrected=req.fuel_corrected,
            include_mini_sectors=req.include_mini_sectors,
            as_of_time_s=round(t_as_of, 3),
            total_completed_laps=len(lap_items),
            laps=lap_items,
        )

    def get_tyre_degradation(
        self,
        race_id: str,
        driver: str,
        stint: Optional[int] = 1,
        as_of: Optional[float] = None,
        fuel_correction_s_per_lap: float = FUEL_CORRECTION_S_PER_LAP,
    ) -> TyreDegradationResponse:
        """
        Computes fuel-corrected pace degradation slope and cliff detection for a driver's stint
        with strict point-in-time as_of replay chronology enforcement.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)

        driver = driver.upper().strip()
        if driver not in session.drivers:
            raise ValueError(f"Driver '{driver}' not found in race '{race_id}'. Available: {session.drivers}")

        stint_idx = int(stint) if stint is not None and stint > 0 else 1

        # Resolve authoritative simulation timestamp t_as_of
        if as_of is not None:
            t_as_of = float(as_of)
        else:
            if slug in self._adapters:
                t_as_of = float(self._adapters[slug].clock.current_time_s)
            else:
                adapter = self.get_or_create_adapter(slug)
                t_as_of = float(adapter.clock.current_time_s)

        # Pre-race check: no laps completed before T_0
        if t_as_of < session.T_0:
            return TyreDegradationResponse(
                race_id=slug,
                driver=driver,
                stint=stint_idx,
                compound="UNKNOWN",
                status="PRE_RACE",
                is_estimate=True,
                fuel_correction_applied_s_per_lap=fuel_correction_s_per_lap,
                fuel_correction_assumption=f"Linear fuel burn compensation of {fuel_correction_s_per_lap:.3f} s/lap.",
                methodology_note="Degradation rate is an ordinary least squares (OLS) linear regression model estimate on clean green-flag laps.",
                degradation_interpretation="Session has not started yet (pre-race).",
                confidence="INSUFFICIENT_DATA",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                available_stints=[],
                laps=[],
                message="No race laps completed as of requested timestamp (pre-race).",
            )

        t_as_of_clamped = min(t_as_of, session.T_finish)

        # Authoritative completed laps for this driver at t_as_of_clamped
        crossing_times = session.driver_lap_crossing_times.get(driver, np.array([]))
        lap_numbers = session.driver_lap_numbers.get(driver, np.array([]))
        idx = np.searchsorted(crossing_times, t_as_of_clamped, side="right") - 1
        if idx >= 0 and len(lap_numbers) > idx:
            max_lap = int(lap_numbers[idx])
        else:
            max_lap = 0

        # Discover all recorded stints for this driver from stints_df (or laps_df fallback)
        stints_list: List[StintSummary] = []
        drv_stints_df = session.stints_df[session.stints_df["driver"] == driver]
        if not drv_stints_df.empty:
            for _, s_row in drv_stints_df.sort_values("stint").iterrows():
                s_num = int(s_row["stint"])
                s_start = int(s_row["start_lap"])
                s_end = int(s_row["end_lap"])
                s_count = int(s_row["lap_count"]) if pd.notna(s_row.get("lap_count")) else (s_end - s_start + 1)
                s_comp = str(s_row.get("compound", "UNKNOWN"))
                stints_list.append(
                    StintSummary(
                        stint=s_num,
                        compound=s_comp,
                        start_lap=s_start,
                        end_lap=s_end,
                        lap_count=s_count,
                        is_started=bool(max_lap >= s_start),
                    )
                )
        else:
            # Fallback from laps_df if stints_df is missing
            drv_laps_all = session.laps_df[session.laps_df["driver"] == driver]
            if not drv_laps_all.empty and "stint" in drv_laps_all.columns:
                for s_num, s_grp in drv_laps_all.groupby("stint"):
                    s_start = int(s_grp["lap_number"].min())
                    s_end = int(s_grp["lap_number"].max())
                    s_comp = str(s_grp["compound"].iloc[0]) if "compound" in s_grp.columns else "UNKNOWN"
                    stints_list.append(
                        StintSummary(
                            stint=int(s_num),
                            compound=s_comp,
                            start_lap=s_start,
                            end_lap=s_end,
                            lap_count=len(s_grp),
                            is_started=bool(max_lap >= s_start),
                        )
                    )

        # Check if the requested stint has started
        target_stint_meta = next((s for s in stints_list if s.stint == stint_idx), None)
        if target_stint_meta and not target_stint_meta.is_started:
            return TyreDegradationResponse(
                race_id=slug,
                driver=driver,
                stint=stint_idx,
                compound=target_stint_meta.compound,
                status="FUTURE_STINT",
                is_estimate=True,
                fuel_correction_applied_s_per_lap=fuel_correction_s_per_lap,
                fuel_correction_assumption=f"Linear fuel burn compensation of {fuel_correction_s_per_lap:.3f} s/lap.",
                methodology_note="Degradation rate is an ordinary least squares (OLS) linear regression model estimate on clean green-flag laps.",
                degradation_interpretation=f"Stint {stint_idx} starts at lap {target_stint_meta.start_lap}, but current replay boundary is lap {max_lap}.",
                confidence="INSUFFICIENT_DATA",
                as_of_lap=max_lap,
                as_of_time_s=round(t_as_of, 3),
                available_stints=stints_list,
                laps=[],
                message=f"Stint {stint_idx} has not started yet as of lap {max_lap}.",
            )

        # Compute tyre degradation over cached laps up to max_lap
        deg_res = compute_tyre_degradation(
            laps_df=session.laps_df,
            driver=driver,
            stint=stint_idx,
            fuel_correction_s_per_lap=fuel_correction_s_per_lap,
            as_of_lap=max_lap if max_lap > 0 else None,
        )

        status = "VALID" if deg_res.get("confidence") in ("HIGH", "MEDIUM") else "INSUFFICIENT_DATA"
        if deg_res.get("clean_laps_count", 0) == 0 and deg_res.get("total_laps_count", 0) == 0:
            status = "NO_DATA"

        # Map laps dicts into TyreLapPoint objects
        lap_points = [TyreLapPoint(**lp) for lp in deg_res.get("laps", [])]

        return TyreDegradationResponse(
            race_id=slug,
            driver=driver,
            stint=stint_idx,
            compound=deg_res.get("compound", "UNKNOWN"),
            status=status,
            is_estimate=True,
            fuel_correction_applied_s_per_lap=deg_res.get("fuel_correction_applied_s_per_lap", fuel_correction_s_per_lap),
            fuel_correction_assumption=deg_res.get("fuel_correction_assumption", ""),
            methodology_note=deg_res.get("methodology_note", ""),
            degradation_interpretation=deg_res.get("degradation_interpretation", ""),
            raw_slope_s_per_lap=deg_res.get("raw_slope_s_per_lap"),
            estimated_deg_rate_s_per_lap=deg_res.get("estimated_deg_rate_s_per_lap"),
            intercept_corr_s=deg_res.get("intercept_corr_s"),
            intercept_raw_s=deg_res.get("intercept_raw_s"),
            clean_laps_count=deg_res.get("clean_laps_count", 0),
            total_laps_count=deg_res.get("total_laps_count", 0),
            r_squared=deg_res.get("r_squared"),
            std_error_s_per_lap=deg_res.get("std_error_s_per_lap"),
            p_value=deg_res.get("p_value"),
            cliff_lap=deg_res.get("cliff_lap"),
            confidence=deg_res.get("confidence", "INSUFFICIENT_DATA"),
            as_of_lap=max_lap,
            as_of_time_s=round(t_as_of, 3),
            available_stints=stints_list,
            laps=lap_points,
            message=deg_res.get("message"),
        )

    def get_gap_and_battle_analysis(
        self,
        race_id: str,
        driver1: str,
        driver2: str,
        as_of: Optional[float] = None,
    ) -> GapBattleResponse:
        """
        Executes head-to-head interval gap tracking, closing rate dynamics,
        and battle/overtake classification with strict as_of point-in-time enforcement.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)

        d1 = driver1.upper().strip()
        d2 = driver2.upper().strip()

        if d1 == d2:
            raise ValueError(f"Driver 1 and Driver 2 must be distinct drivers (received '{d1}' for both).")

        if d1 not in session.drivers:
            raise ValueError(f"Driver '{d1}' not found in race '{race_id}'. Available: {session.drivers}")
        if d2 not in session.drivers:
            raise ValueError(f"Driver '{d2}' not found in race '{race_id}'. Available: {session.drivers}")

        # Resolve authoritative simulation timestamp t_as_of
        if as_of is not None:
            t_as_of = float(as_of)
        else:
            t_as_of = float(session.T_finish)

        # Pre-race check: before T_0
        if t_as_of < session.T_0:
            return GapBattleResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="PRE_RACE",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                total_race_laps=session.total_laps,
                gap_history=[],
                pair_battles=[],
                active_battles_at_as_of=[],
                overtakes_up_to_as_of=[],
                summary=GapBattleSummary(),
                message="No race laps completed as of requested timestamp (pre-race).",
            )

        t_as_of_clamped = min(t_as_of, session.T_finish)

        # Derive each driver's completed laps at t_as_of_clamped
        crossings1 = session.driver_lap_crossing_times.get(d1, np.array([]))
        laps1 = session.driver_lap_numbers.get(d1, np.array([]))
        idx1 = np.searchsorted(crossings1, t_as_of_clamped, side="right") - 1
        max_lap1 = int(laps1[idx1]) if idx1 >= 0 and len(laps1) > idx1 else 0

        crossings2 = session.driver_lap_crossing_times.get(d2, np.array([]))
        laps2 = session.driver_lap_numbers.get(d2, np.array([]))
        idx2 = np.searchsorted(crossings2, t_as_of_clamped, side="right") - 1
        max_lap2 = int(laps2[idx2]) if idx2 >= 0 and len(laps2) > idx2 else 0

        common_boundary_lap = min(max_lap1, max_lap2)
        field_max_lap = max(max_lap1, max_lap2)

        if common_boundary_lap < 1:
            return GapBattleResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="PRE_RACE",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                total_race_laps=session.total_laps,
                gap_history=[],
                pair_battles=[],
                active_battles_at_as_of=[],
                overtakes_up_to_as_of=[],
                summary=GapBattleSummary(),
                message=f"Neither car has completed lap 1 as of simulation time {t_as_of_clamped:.1f}s.",
            )

        # 1. Compute Gap History for the pair up to common_boundary_lap
        gap_df = compute_gap_history(session.laps_df, d1, d2, as_of_lap=common_boundary_lap)
        gap_items: List[GapLapItem] = []
        if not gap_df.empty:
            for _, r in gap_df.iterrows():
                g_val = round(float(r["gap_s"]), 3)
                gap_items.append(
                    GapLapItem(
                        lap_number=int(r["lap_number"]),
                        gap_s=g_val,
                        closing_rate_s_per_lap=round(float(r["closing_rate_s_per_lap"]), 3),
                        leader=str(r["leader"]),
                        chaser=str(r["chaser"]),
                        is_clean_pair=bool(r["is_clean_pair"]),
                        in_battle_window=bool(-1.0 <= g_val <= 1.0),
                    )
                )

        # 2. Compute Overtakes up to field_max_lap with bounded caching to prevent expensive re-analysis
        cache_key = (slug, field_max_lap)
        if cache_key in self._overtakes_cache:
            raw_overtakes = self._overtakes_cache[cache_key]
        else:
            def bounded_tel_loader(drv: str, lap_num: int):
                if lap_num > field_max_lap:
                    return None
                try:
                    return load_lap_telemetry(session.year, session.gp, drv, lap_num)
                except Exception:
                    return None

            raw_overtakes = classify_overtakes(
                session.laps_df,
                session.stints_df,
                as_of_lap=field_max_lap,
                telemetry_loader_func=bounded_tel_loader,
            )
            self._overtakes_cache[cache_key] = raw_overtakes

        overtake_items: List[OvertakeEvidenceItem] = []
        for o in raw_overtakes:
            overtake_items.append(
                OvertakeEvidenceItem(
                    lap_number=int(o["lap_number"]),
                    overtaking_driver=str(o["overtaking_driver"]),
                    overtaken_driver=str(o["overtaken_driver"]),
                    classification=str(o["classification"]),
                    confidence=str(o["confidence"]),
                    is_on_track_overtake=bool(o.get("is_on_track_overtake", False)),
                    crossover_distance_m=round(float(o["crossover_distance_m"]), 1) if o.get("crossover_distance_m") is not None else None,
                    evidence=str(o.get("evidence", "")),
                )
            )

        # 3. Compute Historical Battles between d1 and d2 across laps 2..common_boundary_lap
        pair_battles: List[BattleEventItem] = []
        target_pair = {d1, d2}
        for l in range(2, common_boundary_lap + 1):
            battles_on_lap = detect_battles(session.laps_df, as_of_lap=l, stints_df=session.stints_df)
            for b in battles_on_lap:
                if {b["leader"], b["chaser"]} == target_pair:
                    ot_class = None
                    if b.get("overtake_classification"):
                        raw_ot = b["overtake_classification"]
                        ot_class = OvertakeEvidenceItem(
                            lap_number=int(raw_ot["lap_number"]),
                            overtaking_driver=str(raw_ot["overtaking_driver"]),
                            overtaken_driver=str(raw_ot["overtaken_driver"]),
                            classification=str(raw_ot["classification"]),
                            confidence=str(raw_ot["confidence"]),
                            is_on_track_overtake=bool(raw_ot.get("is_on_track_overtake", False)),
                            crossover_distance_m=round(float(raw_ot["crossover_distance_m"]), 1) if raw_ot.get("crossover_distance_m") is not None else None,
                            evidence=str(raw_ot.get("evidence", "")),
                        )
                    pair_battles.append(
                        BattleEventItem(
                            lap_number=int(b["lap_number"]),
                            leader=str(b["leader"]),
                            chaser=str(b["chaser"]),
                            gap_s=round(float(b["gap_s"]), 3),
                            closing_rate_s_per_lap=round(float(b["closing_rate_s_per_lap"]), 3),
                            is_closing=bool(b.get("is_closing", False)),
                            sustained_pressure=bool(b.get("sustained_pressure", False)),
                            battle_verdict=str(b.get("battle_verdict", "CLOSE_CONTEST")),
                            position_swap_detected=bool(b.get("position_swap_detected", False)),
                            is_overtake_confirmed=bool(b.get("is_overtake_confirmed", False)),
                            overtake_classification=ot_class,
                        )
                    )

        # 4. Compute Active Battles across the entire field strictly at common_boundary_lap
        active_at_as_of: List[BattleEventItem] = []
        raw_active = detect_battles(session.laps_df, as_of_lap=common_boundary_lap, stints_df=session.stints_df)
        for b in raw_active:
            ot_class = None
            if b.get("overtake_classification"):
                raw_ot = b["overtake_classification"]
                ot_class = OvertakeEvidenceItem(
                    lap_number=int(raw_ot["lap_number"]),
                    overtaking_driver=str(raw_ot["overtaking_driver"]),
                    overtaken_driver=str(raw_ot["overtaken_driver"]),
                    classification=str(raw_ot["classification"]),
                    confidence=str(raw_ot["confidence"]),
                    is_on_track_overtake=bool(raw_ot.get("is_on_track_overtake", False)),
                    crossover_distance_m=round(float(raw_ot["crossover_distance_m"]), 1) if raw_ot.get("crossover_distance_m") is not None else None,
                    evidence=str(raw_ot.get("evidence", "")),
                )
            active_at_as_of.append(
                BattleEventItem(
                    lap_number=int(b["lap_number"]),
                    leader=str(b["leader"]),
                    chaser=str(b["chaser"]),
                    gap_s=round(float(b["gap_s"]), 3),
                    closing_rate_s_per_lap=round(float(b["closing_rate_s_per_lap"]), 3),
                    is_closing=bool(b.get("is_closing", False)),
                    sustained_pressure=bool(b.get("sustained_pressure", False)),
                    battle_verdict=str(b.get("battle_verdict", "CLOSE_CONTEST")),
                    position_swap_detected=bool(b.get("position_swap_detected", False)),
                    is_overtake_confirmed=bool(b.get("is_overtake_confirmed", False)),
                    overtake_classification=ot_class,
                )
            )

        # 5. Summary metrics
        total_laps_analyzed = len(gap_items)
        current_gap_s = gap_items[-1].gap_s if gap_items else None
        current_leader = gap_items[-1].leader if gap_items else None

        closest_gap_s = None
        closest_gap_lap = None
        if not gap_df.empty:
            abs_gaps = gap_df["gap_s"].abs()
            min_idx = abs_gaps.idxmin()
            closest_gap_s = round(float(gap_df.loc[min_idx, "gap_s"]), 3)
            closest_gap_lap = int(gap_df.loc[min_idx, "lap_number"])

        laps_in_battle = sum(1 for g in gap_items if g.in_battle_window)
        pair_overtakes = [o for o in raw_overtakes if {o["overtaking_driver"], o["overtaken_driver"]} == target_pair]
        total_overtakes = len(pair_overtakes)
        confirmed_passes = sum(1 for o in pair_overtakes if o.get("classification") == "CONFIRMED_ON_TRACK_OVERTAKE")

        summary = GapBattleSummary(
            total_laps_analyzed=total_laps_analyzed,
            current_gap_s=current_gap_s,
            current_leader=current_leader,
            closest_gap_s=closest_gap_s,
            closest_gap_lap=closest_gap_lap,
            laps_in_battle_window=laps_in_battle,
            total_overtakes_completed=total_overtakes,
            confirmed_on_track_passes=confirmed_passes,
        )

        return GapBattleResponse(
            race_id=slug,
            driver1=d1,
            driver2=d2,
            status="SUCCESS",
            as_of_lap=common_boundary_lap,
            as_of_time_s=round(t_as_of, 3),
            total_race_laps=session.total_laps,
            gap_history=gap_items,
            pair_battles=pair_battles,
            active_battles_at_as_of=active_at_as_of,
            overtakes_up_to_as_of=overtake_items,
            summary=summary,
        )

    def get_battle_and_overtake_prediction(
        self,
        race_id: str,
        driver1: str,
        driver2: str,
        as_of: Optional[float] = None,
        horizon_laps: int = 5,
    ) -> BattleOvertakePredictionResponse:
        """
        Generates point-in-time Battle and Overtake probability predictions
        between driver1 and driver2 at as_of simulation timestamp using trained
        regularized logistic regression pipelines.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)

        d1 = driver1.upper().strip()
        d2 = driver2.upper().strip()

        if d1 == d2:
            raise ValueError(f"Driver 1 and Driver 2 must be distinct drivers (received '{d1}' for both).")

        if d1 not in session.drivers:
            raise ValueError(f"Driver '{d1}' not found in race '{race_id}'. Available: {session.drivers}")
        if d2 not in session.drivers:
            raise ValueError(f"Driver '{d2}' not found in race '{race_id}'. Available: {session.drivers}")

        # Resolve authoritative simulation timestamp t_as_of
        if as_of is not None:
            t_as_of = float(as_of)
        else:
            adapter = self.get_or_create_adapter(slug)
            t_as_of = float(adapter.clock.current_time_s)

        # Standard reliability warnings supported by empirical backtesting results
        reliability_warnings = [
            ModelReliabilityWarning(
                model_name="battle_model",
                warning_type="CALIBRATION_OVERCONFIDENCE",
                message=(
                    "Empirical LORO backtesting indicates significant overconfidence in the [0.8, 1.0) "
                    "probability interval (mean predicted 86.0% vs observed rate 58.7%, absolute error 27.4%). "
                    "High battle probabilities should be interpreted with caution."
                ),
            ),
            ModelReliabilityWarning(
                model_name="overtake_model",
                warning_type="RECALL_LIMITATION",
                message=(
                    "Standard 0.50 decision threshold yields low recall (~20-22%) due to low class base rate (~14%). "
                    "Probabilities below 0.50 do not rule out an overtake; probabilities should be evaluated continuously."
                ),
            ),
            ModelReliabilityWarning(
                model_name="both",
                warning_type="CIRCUIT_SPECIFIC",
                message=(
                    "Models were trained and evaluated exclusively on 2022 and 2023 Italian GP (Monza) data. "
                    "Low-downforce high-speed circuit dynamics may not generalize to high-downforce or street circuits."
                ),
            ),
            ModelReliabilityWarning(
                model_name="both",
                warning_type="UNCERTAINTY_LIMITATION",
                message=(
                    "Predictions are point estimates without calibrated confidence intervals. "
                    "Estimates do not represent certainty."
                ),
            ),
        ]

        # Pre-race check: before T_0
        if t_as_of < session.T_0:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="PRE_RACE",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message="Simulation timestamp is prior to race start (pre-race).",
            )

        t_as_of_clamped = min(t_as_of, session.T_finish)

        # Filter session laps up to t_as_of_clamped strictly to prevent future leaks
        all_laps = session.laps_df
        if all_laps is None or all_laps.empty:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="INSUFFICIENT_DATA",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message="No lap records available in session.",
            )

        hist_laps = all_laps[all_laps["session_time_s"] <= t_as_of_clamped].copy()

        laps_d1 = hist_laps[hist_laps["driver"] == d1]
        laps_d2 = hist_laps[hist_laps["driver"] == d2]

        if laps_d1.empty or laps_d2.empty:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="INSUFFICIENT_DATA",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message=f"Insufficient lap data for one or both drivers ({d1}, {d2}) at requested timestamp.",
            )

        max_l1 = int(laps_d1["lap_number"].max())
        max_l2 = int(laps_d2["lap_number"].max())
        as_of_lap = min(max_l1, max_l2)

        if as_of_lap < 2:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="INSUFFICIENT_DATA",
                as_of_lap=as_of_lap,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message="Fewer than 2 race laps completed at requested timestamp.",
            )

        r1 = laps_d1[laps_d1["lap_number"] == as_of_lap]
        r2 = laps_d2[laps_d2["lap_number"] == as_of_lap]

        if r1.empty or r2.empty:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="INSUFFICIENT_DATA",
                as_of_lap=as_of_lap,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message=f"Missing lap crossing records for drivers at lap {as_of_lap}.",
            )

        t1 = r1.iloc[0].get("session_time_s")
        t2 = r2.iloc[0].get("session_time_s")

        if pd.isna(t1) or pd.isna(t2):
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                status="INSUFFICIENT_DATA",
                as_of_lap=as_of_lap,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message=f"Missing session crossing timestamp at lap {as_of_lap}.",
            )

        t1 = float(t1)
        t2 = float(t2)

        if t1 < t2:
            defender = d1
            attacker = d2
            raw_gap = round(t2 - t1, 3)
        else:
            defender = d2
            attacker = d1
            raw_gap = round(t1 - t2, 3)

        # Proximity window: candidate battle pairs are within 10.0s
        if raw_gap > 10.0:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                attacker=attacker,
                defender=defender,
                status="INSUFFICIENT_DATA",
                as_of_lap=as_of_lap,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message=f"Gap between {attacker} and {defender} is {raw_gap}s (>10.0s proximity window).",
            )

        # Check prediction engine model availability
        from src.race.analytics.prediction_engine import prediction_engine
        if not prediction_engine.is_ready():
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                attacker=attacker,
                defender=defender,
                status="MODEL_UNAVAILABLE",
                as_of_lap=as_of_lap,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="LOW",
                reliability_warnings=reliability_warnings,
                message="Trained prediction pipeline artifacts are unavailable.",
            )

        # Extract causal features strictly bounded at as_of_lap
        strict_laps = all_laps[all_laps["lap_number"] <= as_of_lap].copy()
        stints_df = session.stints_df

        # Scheduled total laps strictly from session metadata if known and > 0, else None
        scheduled_total_laps = session.total_laps if (session.total_laps and session.total_laps > 0) else None

        features = extract_pair_features(
            laps_df=strict_laps,
            stints_df=stints_df,
            attacker=attacker,
            defender=defender,
            as_of_lap=as_of_lap,
            race_id=slug,
            total_laps=scheduled_total_laps,
        )

        if features is None:
            return BattleOvertakePredictionResponse(
                race_id=slug,
                driver1=d1,
                driver2=d2,
                attacker=attacker,
                defender=defender,
                status="INSUFFICIENT_DATA",
                as_of_lap=as_of_lap,
                as_of_time_s=round(t_as_of, 3),
                horizon_laps=horizon_laps,
                data_sufficiency="NONE",
                reliability_warnings=reliability_warnings,
                message="Unable to extract causal pair features at current replay lap.",
            )

        # Predict separate probabilities
        p_battle, p_overtake = prediction_engine.predict_pair(features)

        supporting = SupportingFeatures(
            gap_s=features.gap_s,
            closing_rate_s_per_lap=features.closing_rate_s_per_lap,
            is_closing=features.is_closing,
            clean_pace_delta_s=features.clean_pace_delta_s,
            tyre_age_delta_laps=features.tyre_age_delta_laps,
            compound_delta_tier=features.compound_delta_tier,
            attacker_compound=features.attacker_compound,
            defender_compound=features.defender_compound,
            attacker_clean_laps_stint=features.attacker_clean_laps_stint,
            defender_clean_laps_stint=features.defender_clean_laps_stint,
            is_currently_battling=features.is_currently_battling,
            laps_remaining=features.laps_remaining,
        )

        battle_item = PredictionItem(
            probability=p_battle,
            classification_threshold_0_5=bool(p_battle >= 0.5),
            metric_type="probability_estimate",
            interpretation=(
                f"Estimated {p_battle*100:.1f}% probability that {attacker} and {defender} "
                f"will engage in a sustained battle (gap <= 1.0s for 2+ laps) in the next {horizon_laps} laps."
            ),
        )

        overtake_item = PredictionItem(
            probability=p_overtake,
            classification_threshold_0_5=bool(p_overtake >= 0.5),
            metric_type="probability_estimate",
            interpretation=(
                f"Estimated {p_overtake*100:.1f}% probability that {attacker} will complete an on-track "
                f"overtake on {defender} in the next {horizon_laps} laps."
            ),
        )

        return BattleOvertakePredictionResponse(
            race_id=slug,
            driver1=d1,
            driver2=d2,
            attacker=attacker,
            defender=defender,
            status="SUCCESS",
            as_of_lap=as_of_lap,
            as_of_time_s=round(t_as_of, 3),
            horizon_laps=horizon_laps,
            data_sufficiency=features.data_sufficiency,
            battle_prediction=battle_item,
            overtake_prediction=overtake_item,
            supporting_features=supporting,
            reliability_warnings=reliability_warnings,
        )

    def get_pit_strategy(
        self,
        race_id: str,
        driver: str,
        as_of: Optional[float] = None,
        reference_stationary_s: float = 2.5,
    ) -> PitStrategyResponse:
        """
        Calculates point-in-time pit stop strategy and time-loss decomposition for driver
        at simulation timestamp t_as_of.
        Strictly prevents future leakage using crossing times and lifecycle boundaries.
        """
        slug = self.normalize_race_id(race_id)
        session = self.get_or_load_session(slug)

        if driver not in session.drivers:
            raise ValueError(f"Driver '{driver}' not found in race '{race_id}'. Available: {session.drivers}")

        # Resolve authoritative simulation timestamp t_as_of
        if as_of is not None:
            t_as_of = float(as_of)
        else:
            adapter = self.get_or_create_adapter(slug)
            t_as_of = float(adapter.clock.current_time_s)

        # Check pre-race: no pit stops or race laps before T_0
        if t_as_of < session.T_0:
            return PitStrategyResponse(
                race_id=slug,
                driver=driver,
                status="PRE_RACE",
                as_of_lap=0,
                as_of_time_s=round(t_as_of, 3),
                total_completed_stops=0,
                active_stop=None,
                stops=[],
                time_loss_breakdown=None,
                reference_clean_pace_s=None,
                reference_stationary_s=reference_stationary_s,
                baseline_description="Pre-race: no laps completed",
                accounting_notes=["Simulation is prior to official race start (T_0)."],
                message="Simulation timestamp is prior to race start.",
            )

        # Determine last completed lap for target driver at t_as_of
        crossing_times = session.driver_lap_crossing_times.get(driver, np.array([]))
        lap_numbers = session.driver_lap_numbers.get(driver, np.array([]))
        lap_idx = np.searchsorted(crossing_times, t_as_of, side="right") - 1
        as_of_lap = int(lap_numbers[lap_idx]) if (lap_idx >= 0 and len(lap_numbers) > lap_idx) else 0

        # Perform pit stop analysis with timestamp-level lifecycle
        raw_pits = analyze_pit_stops(
            stints_df=session.stints_df,
            laps_df=session.laps_df,
            driver=driver,
            as_of_lap=as_of_lap if as_of_lap > 0 else None,
            as_of_time_s=t_as_of,
            reference_stationary_s=reference_stationary_s,
        )

        # Perform time-loss decomposition strictly up to t_as_of
        raw_loss = compute_time_loss_breakdown(
            laps_df=session.laps_df,
            stints_df=session.stints_df,
            driver=driver,
            as_of_lap=as_of_lap if as_of_lap > 0 else None,
            as_of_time_s=t_as_of,
        )

        # Build Pydantic models
        parsed_stops: List[PitStopItem] = []
        for s in raw_pits.get("stops", []):
            tac_raw = s.get("tactical_evidence")
            tac_obj = (
                TacticalEvidence(
                    is_heuristic=tac_raw.get("is_heuristic", True),
                    classification=tac_raw.get("classification", "UNKNOWN"),
                    rival_driver=tac_raw.get("rival_driver"),
                    pre_stop_gap_s=tac_raw.get("pre_stop_gap_s"),
                    pre_stop_track_position=tac_raw.get("pre_stop_track_position"),
                    target_pit_lap=tac_raw.get("target_pit_lap", s["pit_in_lap"]),
                    rival_pit_lap=tac_raw.get("rival_pit_lap"),
                    lap_delta=tac_raw.get("lap_delta"),
                    outcome=tac_raw.get("outcome"),
                    evidence_summary=tac_raw.get("evidence_summary", ""),
                )
                if tac_raw
                else None
            )

            parsed_stops.append(
                PitStopItem(
                    stop_number=s["stop_number"],
                    status=s["status"],
                    pit_in_lap=s["pit_in_lap"],
                    out_lap=s["out_lap"],
                    in_lap_time_s=s.get("in_lap_time_s"),
                    out_lap_time_s=s.get("out_lap_time_s"),
                    estimated_total_pit_loss_s=s.get("estimated_total_pit_loss_s"),
                    measured_pit_lane_duration_s=s.get("measured_pit_lane_duration_s"),
                    measured_stationary_s=s.get("measured_stationary_s"),
                    stationary_data_status=s.get("stationary_data_status", "UNAVAILABLE_IN_PROVIDER_FEED"),
                    reference_stationary_s=s.get("reference_stationary_s", reference_stationary_s),
                    stationary_excess_s=s.get("stationary_excess_s"),
                    is_stationary_measured=s.get("is_stationary_measured", False),
                    stationary_metric_type=s.get("stationary_metric_type", "ILLUSTRATIVE_ASSUMPTION"),
                    stationary_assumption_label=s.get("stationary_assumption_label", f"Illustrative benchmark assumption ({reference_stationary_s:.1f}s)"),
                    stationary_disclaimer=s.get("stationary_disclaimer", ""),
                    estimated_pit_lane_transit_s=s.get("estimated_pit_lane_transit_s"),
                    is_transit_loss_estimated=s.get("is_transit_loss_estimated", False),
                    transit_estimation_basis=s.get("transit_estimation_basis", "UNAVAILABLE"),
                    estimated_track_in_out_loss_s=s.get("estimated_track_in_out_loss_s"),
                    track_loss_status=s.get("track_loss_status", "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"),
                    parallel_straight_time_s=s.get("parallel_straight_time_s"),
                    is_safety_car_stop=s.get("is_safety_car_stop", False),
                    tactical_verdict=s.get("tactical_verdict", "UNKNOWN"),
                    tactical_evidence=tac_obj,
                )
            )

        active_stop_raw = raw_pits.get("active_stop")
        active_stop_obj = None
        if active_stop_raw:
            active_stop_obj = next((item for item in parsed_stops if item.stop_number == active_stop_raw["stop_number"]), None)

        time_loss_obj = None
        if raw_loss and raw_loss.get("loss_factors_s"):
            lf = raw_loss["loss_factors_s"]
            time_loss_obj = TimeLossBreakdownItem(
                driver=driver,
                as_of_lap=raw_loss.get("as_of_lap", as_of_lap),
                baseline_scope=raw_loss.get("baseline_scope", "AS_OF_FIELD_MEDIAN"),
                reference_clean_lap_pace_s=raw_loss.get("reference_clean_lap_pace_s", 85.0),
                driver_total_time_s=raw_loss.get("driver_total_time_s", 0.0),
                total_deficit_vs_baseline_s=raw_loss.get("total_deficit_vs_baseline_s", 0.0),
                accounting_identity_verified=raw_loss.get("accounting_identity_verified", True),
                loss_factors_s=TimeLossFactors(
                    caution_and_incidents_s=lf.get("caution_and_incidents_s", 0.0),
                    pit_stop_and_transit_s=lf.get("pit_stop_and_transit_s", 0.0),
                    traffic_s=lf.get("traffic_s", 0.0),
                    tyre_degradation_s=lf.get("tyre_degradation_s", 0.0),
                    unexplained_residual_s=lf.get("unexplained_residual_s", 0.0),
                ),
                attribution_notes=raw_loss.get("attribution_notes", []),
            )

        return PitStrategyResponse(
            race_id=slug,
            driver=driver,
            status="SUCCESS",
            as_of_lap=as_of_lap,
            as_of_time_s=round(t_as_of, 3),
            total_completed_stops=raw_pits.get("total_completed_stops", 0),
            active_stop=active_stop_obj,
            stops=parsed_stops,
            time_loss_breakdown=time_loss_obj,
            reference_clean_pace_s=raw_pits.get("reference_clean_pace_s"),
            reference_stationary_s=reference_stationary_s,
            baseline_description=raw_pits.get("baseline_description", ""),
            accounting_notes=raw_pits.get("accounting_notes", []),
            message=None,
        )

    def add_subscriber(self, race_id: str, ws: Any) -> None:
        """Register a WebSocket client to receive updates for race_id."""
        slug = self.normalize_race_id(race_id)
        self.get_or_create_adapter(slug)
        self._subscribers[slug].add(ws)
        self.ensure_tick_task(slug)

    def remove_subscriber(self, race_id: str, ws: Any) -> None:
        """Unregister a WebSocket client and cancel ticking task if empty."""
        slug = self.normalize_race_id(race_id)
        if slug in self._subscribers:
            self._subscribers[slug].discard(ws)
            if not self._subscribers[slug]:
                # Stop tick task if no subscribers remain
                task = self._tick_tasks.pop(slug, None)
                if task and not task.done():
                    task.cancel()

    def reset_tick_delta(self, race_id: str) -> None:
        """
        Resets monotonic tick timer baseline to now for race_id.
        Prevents stale accumulated sleep delta from jumping simulation time
        immediately after seek, step_lap, or play commands.
        """
        slug = self.normalize_race_id(race_id)
        self._last_tick_times[slug] = time.monotonic()

    def ensure_tick_task(self, race_id: str) -> None:
        """Ensure a single broadcast loop is running for race_id."""
        slug = self.normalize_race_id(race_id)
        task = self._tick_tasks.get(slug)
        if task is None or task.done():
            self._tick_tasks[slug] = asyncio.create_task(self._tick_loop(slug))

    async def _tick_loop(self, slug: str) -> None:
        """
        Transport broadcast loop running at ~10 Hz (nominal 100 ms).
        Advances authoritative ReplayClock via adapter.tick(dt_wall) when PLAYING.
        Never introduces independent simulation drift.
        """
        self._last_tick_times[slug] = time.monotonic()
        try:
            while True:
                await asyncio.sleep(0.1)
                now = time.monotonic()
                last_time = self._last_tick_times.get(slug, now)
                dt_wall = max(0.0, now - last_time)
                self._last_tick_times[slug] = now

                subs = self._subscribers.get(slug, set())
                if not subs:
                    break

                adapter = self.get_or_create_adapter(slug)
                clock = adapter.clock

                if clock.state == ClockState.PLAYING:
                    adapter.tick(dt_wall)
                    await self.broadcast_state(slug, "STATE_UPDATE")

                    if clock.state == ClockState.FINISHED:
                        if not self._finished_notified.get(slug, False):
                            self._finished_notified[slug] = True
                            await self.broadcast_event(
                                slug,
                                {
                                    "type": "PLAYBACK_FINISHED",
                                    "session_time_s": round(clock.current_time_s, 3),
                                    "total_laps": clock.total_laps,
                                },
                            )
                elif clock.state != ClockState.FINISHED:
                    self._finished_notified[slug] = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in tick loop for {slug}: {e}")
        finally:
            self._tick_tasks.pop(slug, None)
            self._last_tick_times.pop(slug, None)

    async def broadcast_state(self, race_id: str, msg_type: str = "STATE_UPDATE") -> None:
        """Broadcast current authoritative state to all connected subscribers."""
        slug = self.normalize_race_id(race_id)
        subs = list(self._subscribers.get(slug, []))
        if not subs:
            return

        adapter = self.get_or_create_adapter(slug)
        clock = adapter.clock
        state = self.get_public_state(slug)

        payload = {
            "type": msg_type,
            "clock": {
                "session_time_s": round(clock.current_time_s, 3),
                "state": clock.state.value,
                "speed": clock.speed,
                "current_lap": clock.current_lap,
                "total_laps": clock.total_laps,
                "T_0": round(clock.T_0, 3),
                "T_finish": round(clock.T_finish, 3),
            },
            "state": state.model_dump(),
        }

        dead = set()
        for ws in subs:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)

        for ws in dead:
            self.remove_subscriber(slug, ws)

        # Notify PLAYBACK_FINISHED once upon reaching finished state
        if clock.state == ClockState.FINISHED:
            if not self._finished_notified.get(slug, False):
                self._finished_notified[slug] = True
                await self.broadcast_event(
                    slug,
                    {
                        "type": "PLAYBACK_FINISHED",
                        "session_time_s": round(clock.current_time_s, 3),
                        "total_laps": clock.total_laps,
                    },
                )
        else:
            self._finished_notified[slug] = False

    async def broadcast_event(self, race_id: str, event_payload: Dict[str, Any]) -> None:
        """Broadcast an arbitrary event payload to all connected subscribers."""
        slug = self.normalize_race_id(race_id)
        subs = list(self._subscribers.get(slug, []))
        dead = set()
        for ws in subs:
            try:
                await ws.send_json(event_payload)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.remove_subscriber(slug, ws)


# Global default instance
session_manager = SessionManager()
