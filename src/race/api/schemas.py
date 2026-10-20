"""
Pydantic schemas for the RACE Replay REST and WebSocket API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class RaceSummary(BaseModel):
    race_id: str
    year: int
    gp: str
    official_event_name: str
    total_laps: int
    winner: Optional[str] = None
    driver_count: int
    T_0: float
    T_finish: float


class ClockStatus(BaseModel):
    session_time_s: float
    state: str
    speed: float
    current_lap: int
    total_laps: int
    T_0: float
    T_finish: float


class TrackPoint(BaseModel):
    dist_m: float
    x: float
    y: float


class TrackSector(BaseModel):
    sector: int
    start_m: float
    end_m: float


class MiniSector(BaseModel):
    mini_sector_id: int
    start_m: float
    end_m: float
    sector: int


class TrackOutline(BaseModel):
    circuit_name: str
    year: int
    total_distance_m: float
    sectors: List[TrackSector]
    mini_sectors: List[MiniSector]
    points: List[TrackPoint]


class MiniSectorTimingItem(BaseModel):
    mini_sector_id: int
    sector: int
    start_m: float
    end_m: float
    duration_s: Optional[float] = None
    delta_to_reference_s: Optional[float] = None
    avg_speed_kmh: Optional[float] = None
    status: str = "VALID"  # 'VALID' | 'MISSING' | 'UNRELIABLE_DATA_GAP'


class ComparisonRequest(BaseModel):
    drivers: List[str]
    lap_mode: str = "single"  # 'single' | 'range' | 'fastest' | 'stint' | 'last_clean'
    lap_number: Optional[int] = None
    lap_range: Optional[List[int]] = None
    stint_number: Optional[int] = None
    last_n_clean: Optional[int] = 5
    reference_driver: Optional[str] = None
    sync_mode: str = "lap"  # 'lap' | 'tyre_age'
    clean_only: bool = True
    fuel_corrected: bool = False
    as_of: Optional[float] = None
    include_mini_sectors: bool = False


class ComparisonLapItem(BaseModel):
    driver: str
    lap_number: int
    tyre_age_laps: Optional[int] = None
    stint: Optional[int] = None
    lap_time_s: Optional[float] = None
    delta_to_reference_s: Optional[float] = None
    s1_s: Optional[float] = None
    s2_s: Optional[float] = None
    s3_s: Optional[float] = None
    compound: str = "UNKNOWN"
    is_clean: bool = True
    mini_sectors: Optional[List[MiniSectorTimingItem]] = None
    telemetry_status: Optional[str] = None  # 'VALID' | 'MISSING' | 'UNRELIABLE_DATA_GAP'
    provider: Optional[str] = None  # 'FastF1' | 'OpenF1'
    precision_tier: Optional[str] = None  # 'HIGH_PRECISION_TRANSPONDER' | 'ESTIMATED_COORDINATE_DERIVED'


class ComparisonResponse(BaseModel):
    race_id: str
    status: str = "SUCCESS"  # 'SUCCESS' | 'PRE_RACE' | 'NO_DATA'
    reference_driver: Optional[str] = None
    sync_mode: str = "lap"
    align_key: str = "lap_number"
    clean_only: bool = True
    fuel_corrected: bool = False
    include_mini_sectors: bool = False
    as_of_time_s: Optional[float] = None
    total_completed_laps: int = 0
    laps: List[ComparisonLapItem] = []


class TyreLapPoint(BaseModel):
    lap_number: int
    tyre_age_laps: Optional[int] = None
    raw_lap_time_s: Optional[float] = None
    fuel_corrected_lap_time_s: Optional[float] = None
    fitted_lap_time_s: Optional[float] = None
    is_clean: bool = True
    is_cliff: bool = False


class StintSummary(BaseModel):
    stint: int
    compound: str
    start_lap: int
    end_lap: int
    lap_count: int
    is_started: bool = True


class TyreDegradationResponse(BaseModel):
    race_id: str
    driver: str
    stint: int
    compound: str
    status: str = "VALID"  # 'VALID' | 'INSUFFICIENT_DATA' | 'FUTURE_STINT' | 'NO_DATA' | 'PRE_RACE'
    is_estimate: bool = True
    fuel_correction_applied_s_per_lap: float = 0.033
    fuel_correction_assumption: str
    methodology_note: str
    degradation_interpretation: str
    raw_slope_s_per_lap: Optional[float] = None
    estimated_deg_rate_s_per_lap: Optional[float] = None
    intercept_corr_s: Optional[float] = None
    intercept_raw_s: Optional[float] = None
    clean_laps_count: int = 0
    total_laps_count: int = 0
    r_squared: Optional[float] = None
    std_error_s_per_lap: Optional[float] = None
    p_value: Optional[float] = None
    cliff_lap: Optional[int] = None
    confidence: str = "INSUFFICIENT_DATA"  # 'HIGH' | 'MEDIUM' | 'INSUFFICIENT_DATA'
    as_of_lap: Optional[int] = None
    as_of_time_s: Optional[float] = None
    available_stints: List[StintSummary] = []
    laps: List[TyreLapPoint] = []
    message: Optional[str] = None


class GapLapItem(BaseModel):
    lap_number: int
    gap_s: float
    closing_rate_s_per_lap: float
    leader: str
    chaser: str
    is_clean_pair: bool
    in_battle_window: bool = False


class OvertakeEvidenceItem(BaseModel):
    lap_number: int
    overtaking_driver: str
    overtaken_driver: str
    classification: str  # 'CONFIRMED_ON_TRACK_OVERTAKE' | 'INFERRED_ON_TRACK_OVERTAKE' | 'PIT_STRATEGY_SWAP' | 'RETIREMENT_PASS' | 'CAUTION_PERIOD_CHANGE' | 'UNKNOWN_DATA_GAP'
    confidence: str  # 'CONFIRMED' | 'INFERRED' | 'UNKNOWN'
    is_on_track_overtake: bool
    crossover_distance_m: Optional[float] = None
    evidence: str


class BattleEventItem(BaseModel):
    lap_number: int
    leader: str
    chaser: str
    gap_s: float
    closing_rate_s_per_lap: float
    is_closing: bool = False
    sustained_pressure: bool = False
    battle_verdict: str  # 'ACTIVE_CHASE' | 'SUSTAINED_DEFENSE' | 'CLOSE_CONTEST' | 'OVERTAKE_COMPLETED'
    position_swap_detected: bool = False
    is_overtake_confirmed: bool = False
    overtake_classification: Optional[OvertakeEvidenceItem] = None


class GapBattleSummary(BaseModel):
    total_laps_analyzed: int = 0
    current_gap_s: Optional[float] = None
    current_leader: Optional[str] = None
    closest_gap_s: Optional[float] = None
    closest_gap_lap: Optional[int] = None
    laps_in_battle_window: int = 0
    total_overtakes_completed: int = 0
    confirmed_on_track_passes: int = 0


class GapBattleResponse(BaseModel):
    race_id: str
    driver1: str
    driver2: str
    status: str = "SUCCESS"  # 'SUCCESS' | 'PRE_RACE' | 'NO_DATA'
    as_of_lap: Optional[int] = None
    as_of_time_s: Optional[float] = None
    total_race_laps: int
    gap_history: List[GapLapItem] = []
    pair_battles: List[BattleEventItem] = []
    active_battles_at_as_of: List[BattleEventItem] = []
    overtakes_up_to_as_of: List[OvertakeEvidenceItem] = []
    summary: GapBattleSummary
    message: Optional[str] = None


class ModelReliabilityWarning(BaseModel):
    model_name: str  # 'battle_model' | 'overtake_model' | 'both'
    warning_type: str  # 'CALIBRATION_OVERCONFIDENCE' | 'CIRCUIT_SPECIFIC' | 'RECALL_LIMITATION' | 'UNCERTAINTY_LIMITATION'
    message: str


class SupportingFeatures(BaseModel):
    gap_s: Optional[float] = None
    closing_rate_s_per_lap: Optional[float] = None
    is_closing: bool = False
    clean_pace_delta_s: Optional[float] = None
    tyre_age_delta_laps: Optional[int] = None
    compound_delta_tier: Optional[int] = None
    attacker_compound: Optional[str] = None
    defender_compound: Optional[str] = None
    attacker_clean_laps_stint: int = 0
    defender_clean_laps_stint: int = 0
    is_currently_battling: bool = False
    laps_remaining: Optional[int] = None


class PredictionItem(BaseModel):
    probability: Optional[float] = None
    classification_threshold_0_5: bool = False
    metric_type: str = "probability_estimate"
    interpretation: str


class BattleOvertakePredictionResponse(BaseModel):
    race_id: str
    driver1: str
    driver2: str
    attacker: Optional[str] = None
    defender: Optional[str] = None
    status: str = "SUCCESS"  # 'SUCCESS' | 'INSUFFICIENT_DATA' | 'PRE_RACE' | 'MODEL_UNAVAILABLE'
    as_of_lap: Optional[int] = None
    as_of_time_s: Optional[float] = None
    horizon_laps: int = 5
    data_sufficiency: str = "LOW"  # 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
    battle_prediction: Optional[PredictionItem] = None
    overtake_prediction: Optional[PredictionItem] = None
    supporting_features: Optional[SupportingFeatures] = None
    reliability_warnings: List[ModelReliabilityWarning] = []
    message: Optional[str] = None


class TacticalEvidence(BaseModel):
    is_heuristic: bool = True
    classification: str = "UNKNOWN"  # 'UNDERCUT_ATTEMPT' | 'OVERCUT_ATTEMPT' | 'SAFETY_CAR_ADVANTAGE' | 'REGULAR_STOP' | 'UNKNOWN'
    rival_driver: Optional[str] = None
    pre_stop_gap_s: Optional[float] = None
    pre_stop_track_position: Optional[int] = None
    target_pit_lap: int
    rival_pit_lap: Optional[int] = None
    lap_delta: Optional[int] = None
    outcome: Optional[str] = None  # 'GAINED_POSITION' | 'RETAINED_POSITION' | 'LOST_POSITION' | 'INSUFFICIENT_DATA'
    evidence_summary: str


class PitStopItem(BaseModel):
    stop_number: int
    status: str  # 'COMPLETED' | 'IN_PROGRESS' | 'OUT_LAP_IN_PROGRESS'
    pit_in_lap: int
    out_lap: int
    in_lap_time_s: Optional[float] = None
    out_lap_time_s: Optional[float] = None
    estimated_total_pit_loss_s: Optional[float] = None
    measured_pit_lane_duration_s: Optional[float] = None
    measured_stationary_s: Optional[float] = None
    stationary_data_status: str = "UNAVAILABLE_IN_PROVIDER_FEED"
    reference_stationary_s: float = 2.5
    stationary_excess_s: Optional[float] = None
    is_stationary_measured: bool = False
    stationary_metric_type: str = "ILLUSTRATIVE_ASSUMPTION"
    stationary_assumption_label: str = "Illustrative benchmark assumption (2.5s)"
    stationary_disclaimer: str = "Stationary wheel change duration is an illustrative 2.5s assumption used for transit estimation, not a measured sensor value."
    estimated_pit_lane_transit_s: Optional[float] = None
    is_transit_loss_estimated: bool = False
    transit_estimation_basis: str = "UNAVAILABLE"
    estimated_track_in_out_loss_s: Optional[float] = None
    track_loss_status: str = "UNAVAILABLE_WITHOUT_PARALLEL_STRAIGHT_BASELINE"
    parallel_straight_time_s: Optional[float] = None
    is_safety_car_stop: bool = False
    tactical_verdict: str = "UNKNOWN"
    tactical_evidence: Optional[TacticalEvidence] = None


class TimeLossFactors(BaseModel):
    caution_and_incidents_s: float = 0.0
    pit_stop_and_transit_s: float = 0.0
    traffic_s: float = 0.0
    tyre_degradation_s: float = 0.0
    unexplained_residual_s: float = 0.0


class TimeLossBreakdownItem(BaseModel):
    driver: str
    as_of_lap: int
    baseline_scope: str
    reference_clean_lap_pace_s: float
    driver_total_time_s: float
    total_deficit_vs_baseline_s: float
    accounting_identity_verified: bool = True
    loss_factors_s: TimeLossFactors
    attribution_notes: List[str] = []


class PitStrategyResponse(BaseModel):
    race_id: str
    driver: str
    status: str = "SUCCESS"  # 'SUCCESS' | 'PRE_RACE' | 'NO_DATA'
    as_of_lap: Optional[int] = None
    as_of_time_s: Optional[float] = None
    total_completed_stops: int = 0
    active_stop: Optional[PitStopItem] = None
    stops: List[PitStopItem] = []
    time_loss_breakdown: Optional[TimeLossBreakdownItem] = None
    reference_clean_pace_s: Optional[float] = None
    reference_stationary_s: float = 2.5
    baseline_description: str
    accounting_notes: List[str] = []
    message: Optional[str] = None







