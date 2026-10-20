export interface TyreLapPoint {
  lap_number: number;
  tyre_age_laps?: number | null;
  raw_lap_time_s?: number | null;
  fuel_corrected_lap_time_s?: number | null;
  fitted_lap_time_s?: number | null;
  is_clean: boolean;
  is_cliff: boolean;
}

export interface StintSummary {
  stint: number;
  compound: string;
  start_lap: number;
  end_lap: number;
  lap_count: number;
  is_started: boolean;
}

export interface TyreDegradationResponse {
  race_id: string;
  driver: string;
  stint: number;
  compound: string;
  status: 'VALID' | 'INSUFFICIENT_DATA' | 'FUTURE_STINT' | 'NO_DATA' | 'PRE_RACE';
  is_estimate: boolean;
  fuel_correction_applied_s_per_lap: number;
  fuel_correction_assumption: string;
  methodology_note: string;
  degradation_interpretation: string;
  raw_slope_s_per_lap?: number | null;
  estimated_deg_rate_s_per_lap?: number | null;
  intercept_corr_s?: number | null;
  intercept_raw_s?: number | null;
  clean_laps_count: number;
  total_laps_count: number;
  r_squared?: number | null;
  std_error_s_per_lap?: number | null;
  p_value?: number | null;
  cliff_lap?: number | null;
  confidence: 'HIGH' | 'MEDIUM' | 'INSUFFICIENT_DATA';
  as_of_lap?: number | null;
  as_of_time_s?: number | null;
  available_stints: StintSummary[];
  laps: TyreLapPoint[];
  message?: string | null;
}
