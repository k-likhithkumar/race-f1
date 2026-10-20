import React, { useMemo } from 'react';
import { ComparisonResponse, ComparisonLapItem } from '../../types/comparison';
import { DriverOption } from './ComparisonConfigBar';
import { getTeamColor } from '../../constants/teams';
import {
  AlertCircle,
  AlertTriangle,
  Clock,
  Info,
  Loader2,
  ShieldCheck,
  Fuel,
} from 'lucide-react';

export interface ComparisonTableProps {
  data: ComparisonResponse | null;
  isLoading: boolean;
  error: string | null;
  isStale: boolean;
  availableDrivers: DriverOption[];
  onRefresh?: () => void;
}

// Format lap time in M:SS.sss
export function formatLapTime(time_s: number | null | undefined): string {
  if (time_s === null || time_s === undefined || isNaN(time_s) || time_s <= 0) return '—';
  const mins = Math.floor(time_s / 60);
  const secs = (time_s % 60).toFixed(3);
  const paddedSecs = (time_s % 60) < 10 ? `0${secs}` : secs;
  return `${mins}:${paddedSecs}`;
}

// Format sector time in SS.sss
export function formatSectorTime(time_s: number | null | undefined): string {
  if (time_s === null || time_s === undefined || isNaN(time_s) || time_s <= 0) return '—';
  return time_s.toFixed(3);
}

// Format delta with signed seconds to 3 decimals
export function formatDelta(delta_s: number | null | undefined): string {
  if (delta_s === null || delta_s === undefined || isNaN(delta_s)) return '—';
  if (Math.abs(delta_s) < 0.0005) return '0.000s';
  const sign = delta_s > 0 ? '+' : '';
  return `${sign}${delta_s.toFixed(3)}s`;
}

// Determine CSS class for delta coloring
export function getDeltaClass(delta_s: number | null | undefined, isRef: boolean): string {
  if (delta_s === null || delta_s === undefined || isNaN(delta_s)) return 'delta-null';
  if (isRef || Math.abs(delta_s) < 0.0005) return 'delta-ref';
  return delta_s < 0 ? 'delta-faster' : 'delta-slower';
}

// Determine tyre badge class
export function getTyreClass(compound: string | null | undefined): string {
  const norm = (compound || '').toUpperCase();
  switch (norm) {
    case 'SOFT':
      return 'tyre-soft';
    case 'MEDIUM':
      return 'tyre-medium';
    case 'HARD':
      return 'tyre-hard';
    case 'INTERMEDIATE':
      return 'tyre-intermediate';
    case 'WET':
      return 'tyre-wet';
    default:
      return 'tyre-unknown';
  }
}

export const ComparisonTable: React.FC<ComparisonTableProps> = ({
  data,
  isLoading,
  error,
  isStale,
  availableDrivers,
}) => {
  // Map driver codes to team names for coloring
  const driverTeamMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const d of availableDrivers) {
      map.set(d.code, d.team);
    }
    return map;
  }, [availableDrivers]);

  // State A: Error
  if (error) {
    return (
      <div className="comparison-table-card state-card state-error" id="comparison-table-error">
        <AlertCircle size={24} className="state-icon icon-error" />
        <div className="state-text">
          <h3>Comparison Query Failed</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  // State B: Loading
  if (isLoading) {
    return (
      <div className="comparison-table-card state-card state-loading" id="comparison-table-loading">
        <Loader2 size={26} className="spin state-icon icon-loading" />
        <div className="state-text">
          <h3>Querying Comparison API...</h3>
          <p>Retrieving authoritative lap crossing telemetry and calculating baseline deltas.</p>
        </div>
      </div>
    );
  }

  // State C: Initial / Idle (no data yet)
  if (!data) {
    return (
      <div className="comparison-table-card state-card state-idle" id="comparison-table-idle">
        <Info size={24} className="state-icon icon-idle" />
        <div className="state-text">
          <h3>No Comparison Results Yet</h3>
          <p>
            Select drivers and configure lap parameters above, then click <strong>Run Comparison</strong> to query the API.
          </p>
        </div>
      </div>
    );
  }

  // State D: PRE_RACE
  if (data.status === 'PRE_RACE') {
    return (
      <div className="comparison-table-card state-card state-prerace" id="comparison-table-prerace">
        <Clock size={24} className="state-icon icon-warning" />
        <div className="state-text">
          <h3>Pre-Race Simulation Cutoff</h3>
          <p>
            The selected simulation cutoff ({data.as_of_time_s !== null && data.as_of_time_s !== undefined ? `${data.as_of_time_s.toFixed(2)}s` : 'Unknown'}) precedes the race start (<code>T_0</code>). No completed laps exist prior to green flag release.
          </p>
        </div>
      </div>
    );
  }

  // State E: NO_DATA or empty laps array
  if (data.status === 'NO_DATA' || data.laps.length === 0) {
    return (
      <div className="comparison-table-card state-card state-nodata" id="comparison-table-nodata">
        <AlertTriangle size={24} className="state-icon icon-warning" />
        <div className="state-text">
          <h3>No Matching Lap Observations</h3>
          <p>
            No completed laps matched the selected criteria prior to the cutoff ({data.as_of_time_s !== null && data.as_of_time_s !== undefined ? `${data.as_of_time_s.toFixed(2)}s` : 'Full Race'}).
            {data.clean_only && ' Clean Laps Only is active and may have filtered out in-laps, pit exits, or caution periods.'}
          </p>
        </div>
      </div>
    );
  }

  const isTyreAgeSync = data.sync_mode === 'tyre_age';

  return (
    <div className="comparison-table-card" id="comparison-table-container">
      {/* Table Metadata Header Bar */}
      <div className="table-metadata-bar">
        <div className="table-metadata-left">
          <div className="meta-badge meta-race-id mono">{data.race_id}</div>
          <div className="meta-badge meta-ref">
            Reference: <strong className="mono">{data.reference_driver || 'NONE'}</strong>
          </div>
          <div className="meta-badge meta-sync">
            Sync: <strong>{isTyreAgeSync ? 'Tyre Age' : 'Race Lap'}</strong>
          </div>
          <div className="meta-badge meta-cutoff mono">
            Cutoff: {data.as_of_time_s !== null && data.as_of_time_s !== undefined ? `${data.as_of_time_s.toFixed(2)}s` : 'Full Race'}
          </div>
          <div className="meta-badge meta-count mono">
            {data.laps.length} {data.laps.length === 1 ? 'lap record' : 'lap records'}
          </div>
        </div>

        <div className="table-metadata-right">
          {data.clean_only && (
            <span className="modifier-pill modifier-clean" title="Only clean laps included (excludes cautions and in/out laps)">
              <ShieldCheck size={12} /> Clean Only
            </span>
          )}
          {data.fuel_corrected && (
            <span className="modifier-pill modifier-fuel" title="Standard fuel correction (+0.033 s/lap) applied">
              <Fuel size={12} /> Fuel Corrected (+0.033s/lap)
            </span>
          )}
          <span className="status-pill status-ready mono">STATUS: {data.status}</span>
        </div>
      </div>

      {/* Stale Configuration Warning Banner */}
      {isStale && (
        <div className="table-stale-banner" id="table-stale-banner">
          <AlertTriangle size={15} />
          <span>
            Displaying results from previous query. Current configuration has changed — click <strong>Run / Refresh</strong> above to update.
          </span>
        </div>
      )}

      {/* Scrollable Results Table */}
      <div className="table-scroll-wrapper">
        <table className="comparison-table" id="comparison-results-table">
          <thead>
            <tr>
              <th className="th-align-key">{isTyreAgeSync ? 'Tyre Age' : 'Race Lap'}</th>
              <th className="th-driver">Driver</th>
              <th className="th-lap">Actual Lap</th>
              <th className="th-time">Lap Time</th>
              <th className="th-delta">Delta to Ref</th>
              <th className="th-sector">S1</th>
              <th className="th-sector">S2</th>
              <th className="th-sector">S3</th>
              <th className="th-compound">Compound</th>
              <th className="th-age">Age</th>
              <th className="th-stint">Stint</th>
              <th className="th-clean">Status</th>
            </tr>
          </thead>
          <tbody>
            {data.laps.map((item: ComparisonLapItem, idx: number) => {
              const isRef = item.driver === data.reference_driver;
              const teamName = driverTeamMap.get(item.driver);
              const teamColor = getTeamColor(teamName);
              const deltaClass = getDeltaClass(item.delta_to_reference_s, isRef);
              const tyreClass = getTyreClass(item.compound);

              return (
                <tr key={`${item.driver}-${item.lap_number}-${idx}`} className={isRef ? 'ref-driver-row' : ''}>
                  {/* Alignment Key */}
                  <td className="cell-align-key mono">
                    {isTyreAgeSync
                      ? item.tyre_age_laps !== null && item.tyre_age_laps !== undefined
                        ? `Age ${item.tyre_age_laps}`
                        : '—'
                      : `Lap ${item.lap_number}`}
                  </td>

                  {/* Driver with Team Color */}
                  <td className="cell-driver">
                    <div className="driver-cell-inner">
                      <span className="team-indicator" style={{ backgroundColor: teamColor }} />
                      <span className="driver-code mono">{item.driver}</span>
                      {isRef && <span className="ref-badge">REF</span>}
                    </div>
                  </td>

                  {/* Actual Race Lap */}
                  <td className="cell-lap mono">{item.lap_number}</td>

                  {/* Lap Time */}
                  <td className="cell-time mono">{formatLapTime(item.lap_time_s)}</td>

                  {/* Delta to Reference */}
                  <td className={`cell-delta mono ${deltaClass}`}>
                    {formatDelta(item.delta_to_reference_s)}
                  </td>

                  {/* Sector 1 */}
                  <td className="cell-sector mono">{formatSectorTime(item.s1_s)}</td>

                  {/* Sector 2 */}
                  <td className="cell-sector mono">{formatSectorTime(item.s2_s)}</td>

                  {/* Sector 3 */}
                  <td className="cell-sector mono">{formatSectorTime(item.s3_s)}</td>

                  {/* Compound */}
                  <td className="cell-compound">
                    <div className="compound-cell-inner">
                      <span className={`tyre-badge ${tyreClass}`}>
                        {item.compound ? item.compound.charAt(0).toUpperCase() : '?'}
                      </span>
                      <span className="compound-name">{item.compound || 'UNKNOWN'}</span>
                    </div>
                  </td>

                  {/* Tyre Age */}
                  <td className="cell-age mono">
                    {item.tyre_age_laps !== null && item.tyre_age_laps !== undefined
                      ? `${item.tyre_age_laps}L`
                      : '—'}
                  </td>

                  {/* Stint */}
                  <td className="cell-stint mono">
                    {item.stint !== null && item.stint !== undefined ? `S${item.stint}` : '—'}
                  </td>

                  {/* Clean Status */}
                  <td className="cell-clean">
                    {item.is_clean ? (
                      <span className="clean-pill clean-yes">CLEAN</span>
                    ) : (
                      <span className="clean-pill clean-no">NOT CLEAN</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
