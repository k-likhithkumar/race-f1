/**
 * TypeScript types for Checkpoint 4: Pit Strategy & Time Loss Decomposition.
 */

export interface TacticalEvidence {
  is_heuristic: boolean;
  classification: 'UNDERCUT_ATTEMPT' | 'OVERCUT_ATTEMPT' | 'SAFETY_CAR_ADVANTAGE' | 'REGULAR_STOP' | 'UNKNOWN';
  rival_driver?: string | null;
  pre_stop_gap_s?: number | null;
  pre_stop_track_position?: number | null;
  target_pit_lap: number;
  rival_pit_lap?: number | null;
  lap_delta?: number | null;
  outcome?: 'GAINED_POSITION' | 'RETAINED_POSITION' | 'LOST_POSITION' | 'INSUFFICIENT_DATA' | null;
  evidence_summary: string;
}

export interface PitStopItem {
  stop_number: number;
  status: 'COMPLETED' | 'IN_PROGRESS' | 'OUT_LAP_IN_PROGRESS';
  pit_in_lap: number;
  out_lap: number;
  in_lap_time_s?: number | null;
  out_lap_time_s?: number | null;
  estimated_total_pit_loss_s?: number | null;
  measured_pit_lane_duration_s?: number | null;
  measured_stationary_s?: number | null;
  stationary_data_status: string;
  reference_stationary_s: number;
  stationary_excess_s?: number | null;
  is_stationary_measured: boolean;
  stationary_metric_type: string;
  stationary_assumption_label: string;
  stationary_disclaimer: string;
  estimated_pit_lane_transit_s?: number | null;
  is_transit_loss_estimated: boolean;
  transit_estimation_basis: string;
  estimated_track_in_out_loss_s?: number | null;
  track_loss_status: string;
  parallel_straight_time_s?: number | null;
  is_safety_car_stop: boolean;
  tactical_verdict: string;
  tactical_evidence?: TacticalEvidence | null;
}

export interface TimeLossFactors {
  caution_and_incidents_s: number;
  pit_stop_and_transit_s: number;
  traffic_s: number;
  tyre_degradation_s: number;
  unexplained_residual_s: number;
}

export interface TimeLossBreakdownItem {
  driver: string;
  as_of_lap: number;
  baseline_scope: string;
  reference_clean_lap_pace_s: number;
  driver_total_time_s: number;
  total_deficit_vs_baseline_s: number;
  accounting_identity_verified: boolean;
  loss_factors_s: TimeLossFactors;
  attribution_notes: string[];
}

export interface PitStrategyResponse {
  race_id: string;
  driver: string;
  status: 'SUCCESS' | 'PRE_RACE' | 'NO_DATA';
  as_of_lap?: number | null;
  as_of_time_s?: number | null;
  total_completed_stops: number;
  active_stop?: PitStopItem | null;
  stops: PitStopItem[];
  time_loss_breakdown?: TimeLossBreakdownItem | null;
  reference_clean_pace_s?: number | null;
  reference_stationary_s: number;
  baseline_description: string;
  accounting_notes: string[];
  message?: string | null;
}
