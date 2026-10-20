/**
 * TypeScript types for Checkpoint 2B: Driver Comparison UI.
 * Matches backend Pydantic schemas in src/race/api/schemas.py.
 */

export type LapMode = 'single' | 'range' | 'fastest' | 'stint' | 'last_clean';
export type SyncMode = 'lap' | 'tyre_age';
export type CutoffMode = 'replay' | 'full';

export interface MiniSectorTimingItem {
  mini_sector_id: number;
  sector: number;
  start_m: number;
  end_m: number;
  duration_s?: number | null;
  delta_to_reference_s?: number | null;
  avg_speed_kmh?: number | null;
  status: 'VALID' | 'MISSING' | 'UNRELIABLE_DATA_GAP';
}

export interface ComparisonRequest {
  drivers: string[];
  lap_mode?: LapMode;
  lap_number?: number | null;
  lap_range?: [number, number] | null;
  stint_number?: number | null;
  last_n_clean?: number | null;
  reference_driver?: string | null;
  sync_mode?: SyncMode;
  clean_only?: boolean;
  fuel_corrected?: boolean;
  as_of?: number | null;
  include_mini_sectors?: boolean;
}

export interface ComparisonLapItem {
  driver: string;
  lap_number: number;
  tyre_age_laps?: number | null;
  stint?: number | null;
  lap_time_s?: number | null;
  delta_to_reference_s?: number | null;
  s1_s?: number | null;
  s2_s?: number | null;
  s3_s?: number | null;
  compound: string;
  is_clean: boolean;
  mini_sectors?: MiniSectorTimingItem[] | null;
  telemetry_status?: 'VALID' | 'MISSING' | 'UNRELIABLE_DATA_GAP' | null;
  provider?: string | null;
  precision_tier?: string | null;
}

export interface ComparisonResponse {
  race_id: string;
  status: 'SUCCESS' | 'PRE_RACE' | 'NO_DATA';
  reference_driver?: string | null;
  sync_mode: SyncMode;
  align_key: 'lap_number' | 'tyre_age_laps';
  clean_only: boolean;
  fuel_corrected: boolean;
  include_mini_sectors?: boolean;
  as_of_time_s?: number | null;
  total_completed_laps: number;
  laps: ComparisonLapItem[];
}


export interface ComparisonConfigState {
  selectedDrivers: string[];
  referenceDriver: string;
  lapMode: LapMode;
  lapNumber: number;
  startLap: number;
  endLap: number;
  stintNumber: number;
  lastNClean: number;
  syncMode: SyncMode;
  cleanOnly: boolean;
  fuelCorrected: boolean;
  cutoffMode: CutoffMode;
}
