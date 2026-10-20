/**
 * Type definitions for Gap & Battle Analysis (Milestone 2).
 */

export interface GapLapItem {
  lap_number: number;
  gap_s: number; // Signed gap: positive if driver1 leads, negative if driver2 leads
  closing_rate_s_per_lap: number; // Positive = driver2 catching driver1
  leader: string;
  chaser: string;
  is_clean_pair: boolean;
  in_battle_window: boolean; // True if |gap_s| <= 1.0s
}

export type OvertakeClassification =
  | 'CONFIRMED_ON_TRACK_OVERTAKE'
  | 'INFERRED_ON_TRACK_OVERTAKE'
  | 'PIT_STRATEGY_SWAP'
  | 'RETIREMENT_PASS'
  | 'CAUTION_PERIOD_CHANGE'
  | 'UNKNOWN_DATA_GAP';

export interface OvertakeEvidenceItem {
  lap_number: number;
  overtaking_driver: string;
  overtaken_driver: string;
  classification: OvertakeClassification | string;
  confidence: 'CONFIRMED' | 'INFERRED' | 'UNKNOWN' | string;
  is_on_track_overtake: boolean;
  crossover_distance_m: number | null;
  evidence: string;
}

export type BattleVerdict =
  | 'ACTIVE_CHASE'
  | 'SUSTAINED_DEFENSE'
  | 'CLOSE_CONTEST'
  | 'OVERTAKE_COMPLETED';

export interface BattleEventItem {
  lap_number: number;
  leader: string;
  chaser: string;
  gap_s: number;
  closing_rate_s_per_lap: number;
  is_closing: boolean;
  sustained_pressure: boolean;
  battle_verdict: BattleVerdict | string;
  position_swap_detected: boolean;
  is_overtake_confirmed: boolean;
  overtake_classification?: OvertakeEvidenceItem | null;
}

export interface GapBattleSummary {
  total_laps_analyzed: number;
  current_gap_s: number | null;
  current_leader: string | null;
  closest_gap_s: number | null;
  closest_gap_lap: number | null;
  laps_in_battle_window: number;
  total_overtakes_completed: number;
  confirmed_on_track_passes: number;
}

export interface GapBattleResponse {
  race_id: string;
  driver1: string;
  driver2: string;
  status: 'SUCCESS' | 'PRE_RACE' | 'NO_DATA' | string;
  as_of_lap: number | null;
  as_of_time_s: number | null;
  total_race_laps: number;
  gap_history: GapLapItem[];
  pair_battles: BattleEventItem[];
  active_battles_at_as_of: BattleEventItem[];
  overtakes_up_to_as_of: OvertakeEvidenceItem[];
  summary: GapBattleSummary;
  message?: string | null;
}

export type PredictionStatus =
  | 'SUCCESS'
  | 'INSUFFICIENT_DATA'
  | 'PRE_RACE'
  | 'MODEL_UNAVAILABLE';

export type DataSufficiency = 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE';

export interface ModelReliabilityWarning {
  model_name: 'battle_model' | 'overtake_model' | 'both' | string;
  warning_type:
    | 'CALIBRATION_OVERCONFIDENCE'
    | 'CIRCUIT_SPECIFIC'
    | 'RECALL_LIMITATION'
    | 'UNCERTAINTY_LIMITATION'
    | string;
  message: string;
}

export interface SupportingFeatures {
  gap_s: number | null;
  closing_rate_s_per_lap: number | null;
  is_closing: boolean;
  clean_pace_delta_s: number | null;
  tyre_age_delta_laps: number | null;
  compound_delta_tier: number | null;
  attacker_compound: string | null;
  defender_compound: string | null;
  attacker_clean_laps_stint: number;
  defender_clean_laps_stint: number;
  is_currently_battling: boolean;
  laps_remaining: number | null;
}

export interface PredictionItem {
  probability: number | null;
  classification_threshold_0_5: boolean;
  metric_type: string;
  interpretation: string;
}

export interface BattleOvertakePredictionResponse {
  race_id: string;
  driver1: string;
  driver2: string;
  attacker: string | null;
  defender: string | null;
  status: PredictionStatus | string;
  as_of_lap: number | null;
  as_of_time_s: number | null;
  horizon_laps: number;
  data_sufficiency: DataSufficiency | string;
  battle_prediction: PredictionItem | null;
  overtake_prediction: PredictionItem | null;
  supporting_features: SupportingFeatures | null;
  reliability_warnings: ModelReliabilityWarning[];
  message?: string | null;
}
