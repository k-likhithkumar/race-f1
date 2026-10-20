export interface RaceSummary {
  race_id: string;
  year: number;
  gp: string;
  official_event_name: string;
  total_laps: number;
  winner?: string;
  driver_count: number;
  T_0: number;
  T_finish: number;
}

export interface ClockStatus {
  session_time_s: number;
  state: 'IDLE' | 'PLAYING' | 'PAUSED' | 'SEEKING' | 'FINISHED';
  speed: number;
  current_lap: number;
  total_laps: number;
  T_0: number;
  T_finish: number;
}

export interface TrackPoint {
  dist_m: number;
  x: number;
  y: number;
}

export interface TrackSector {
  sector: number;
  start_m: number;
  end_m: number;
}

export interface MiniSector {
  mini_sector_id: number;
  start_m: number;
  end_m: number;
  sector: number;
}

export interface TrackOutline {
  circuit_name: string;
  year: number;
  total_distance_m: number;
  sectors: TrackSector[];
  mini_sectors: MiniSector[];
  points: TrackPoint[];
}


export interface DriverState {
  driver: string;
  team: string;
  timing_position: number;
  last_completed_lap: number;
  last_lap_time_s: number | null;
  last_crossing_time_s: number | null;
  timing_gap_to_leader_s: number | null;
  timing_interval_s: number | null;
  recorded_gap_age_s: number | null;
  spatial_position: number | null;
  is_spatial_estimate: boolean;
  spatial_projection_valid: boolean;
  track_distance_m: number | null;
  total_race_distance_m: number | null;
  x: number | null;
  y: number | null;
  estimated_gap_to_leader_s: number | null;
  estimated_interval_s: number | null;
  gap_estimate_status: string;
  status: 'ON_TRACK' | 'OFF_TRACK' | 'STOPPED' | 'IN_PIT' | 'TELEMETRY_MISSING' | 'RETIRED';
  pit_status: 'ON_TRACK' | 'IN_PIT' | 'UNKNOWN';
  is_in_lap: boolean;
  is_out_lap: boolean;
  last_confirmed_compound: string;
  compound: string;
  compound_observation_status: string;
  completed_tyre_age_laps: number;
  active_tyre_lap: number;
  stint: number;
}

export interface RaceState {
  session_time_s: number;
  race_id: string;
  track_status: string;
  weather: {
    air_temp_c: number | null;
    track_temp_c: number | null;
    humidity_pct: number | null;
    rainfall: boolean;
  };
  leaderboard: DriverState[];
  spatial_order: string[];
  active_battles: any[];
  spatial_ordering_valid: boolean;
  spatial_fallback_reason: string | null;
}

export type WSMessage =
  | { type: 'INITIAL_STATE'; clock: ClockStatus; state: RaceState }
  | { type: 'STATE_UPDATE'; clock: ClockStatus; state: RaceState }
  | { type: 'COMMAND_ACK'; action: string; status: string; [key: string]: any }
  | { type: 'ERROR'; code: string; message: string }
  | { type: 'PLAYBACK_FINISHED'; session_time_s: number; total_laps: number };
