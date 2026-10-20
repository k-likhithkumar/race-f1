import React, { useState, useMemo } from 'react';
import {
  ComparisonResponse,
  ComparisonLapItem,
  MiniSectorTimingItem,
} from '../../types/comparison';
import { DriverOption } from './ComparisonConfigBar';
import { getTeamColor } from '../../constants/teams';
import { formatLapTime } from './ComparisonTable';
import {
  Flame,
  AlertCircle,
  AlertTriangle,
  Clock,
  Info,
  Loader2,
  RefreshCw,
  Gauge,
  MapPin,
  Sparkles,
} from 'lucide-react';

export interface ComparisonHeatStripProps {
  data: ComparisonResponse | null;
  isLoading: boolean;
  error: string | null;
  isStale: boolean;
  availableDrivers: DriverOption[];
  currentLapMode?: string;
  onRefresh?: () => void;
}

export interface InspectedMiniSector {
  driver: string;
  ms: MiniSectorTimingItem;
  lapItem: ComparisonLapItem;
}

export const ComparisonHeatStrip: React.FC<ComparisonHeatStripProps> = ({
  data,
  isLoading,
  error,
  isStale,
  availableDrivers,
  currentLapMode = 'single',
  onRefresh,
}) => {
  const [inspectedMs, setInspectedMs] = useState<InspectedMiniSector | null>(null);

  // Driver code to team map (Hook 1)
  const driverTeamMap = useMemo(() => {
    const map: Record<string, string> = {};
    availableDrivers.forEach((d) => {
      map[d.code] = d.team;
    });
    return map;
  }, [availableDrivers]);

  // Reference driver code (Hook 2)
  const refDriver = useMemo(() => {
    return data?.reference_driver || (data?.laps && data.laps.length > 0 ? data.laps[0].driver : '');
  }, [data]);

  // Distinct sector groups from mini-sectors (Hook 3)
  const sectorGroups = useMemo(() => {
    if (!data?.laps || data.laps.length === 0) return { s1: [1, 7], s2: [8, 14], s3: [15, 20] };
    const firstLapWithMs = data.laps.find((l) => l.mini_sectors && l.mini_sectors.length > 0);
    if (!firstLapWithMs || !firstLapWithMs.mini_sectors) {
      return { s1: [1, 7], s2: [8, 14], s3: [15, 20] };
    }
    const s1Ids = firstLapWithMs.mini_sectors.filter((m) => m.sector === 1).map((m) => m.mini_sector_id);
    const s2Ids = firstLapWithMs.mini_sectors.filter((m) => m.sector === 2).map((m) => m.mini_sector_id);
    const s3Ids = firstLapWithMs.mini_sectors.filter((m) => m.sector === 3).map((m) => m.mini_sector_id);
    return {
      s1: s1Ids.length > 0 ? [Math.min(...s1Ids), Math.max(...s1Ids)] : [1, 7],
      s2: s2Ids.length > 0 ? [Math.min(...s2Ids), Math.max(...s2Ids)] : [8, 14],
      s3: s3Ids.length > 0 ? [Math.min(...s3Ids), Math.max(...s3Ids)] : [15, 20],
    };
  }, [data]);

  // Check if current lap mode is supported (Hook 4)
  const isModeSupported = useMemo(() => {
    return currentLapMode === 'single' || currentLapMode === 'fastest';
  }, [currentLapMode]);

  // Check if response has mini-sectors populated (Hook 5)
  const hasMiniSectorData = useMemo(() => {
    if (!data?.laps) return false;
    return data.laps.some((l) => l.mini_sectors && l.mini_sectors.length > 0);
  }, [data]);

  // 1. Loading State
  if (isLoading) {
    return (
      <div className="comparison-card heat-strip-card" id="heat-strip-loading">
        <div className="card-header">
          <div className="card-title-group">
            <Flame className="card-title-icon" size={18} />
            <h3 className="card-title">Mini-Sector Heat Strip</h3>
          </div>
        </div>
        <div className="table-state-container">
          <Loader2 className="spinner-icon" size={32} />
          <h4 className="state-title">Loading Mini-Sector Telemetry...</h4>
          <p className="state-desc">Interpolating distance-based boundary crossings and delta timings.</p>
        </div>
      </div>
    );
  }

  // 2. Error State
  if (error) {
    return (
      <div className="comparison-card heat-strip-card" id="heat-strip-error">
        <div className="card-header">
          <div className="card-title-group">
            <Flame className="card-title-icon" size={18} />
            <h3 className="card-title">Mini-Sector Heat Strip</h3>
          </div>
        </div>
        <div className="table-state-container error-state">
          <AlertCircle className="error-icon" size={36} />
          <h4 className="state-title">Comparison Query Failed</h4>
          <p className="state-desc">{error}</p>
          {onRefresh && (
            <button type="button" className="retry-btn" onClick={onRefresh}>
              <RefreshCw size={14} style={{ marginRight: 6 }} /> Retry Comparison
            </button>
          )}
        </div>
      </div>
    );
  }

  // 3. Unsupported Lap Mode Notice
  if (!isModeSupported) {
    return (
      <div className="comparison-card heat-strip-card" id="heat-strip-unsupported-mode">
        <div className="card-header">
          <div className="card-title-group">
            <Flame className="card-title-icon" size={18} />
            <h3 className="card-title">Mini-Sector Heat Strip</h3>
          </div>
        </div>
        <div className="table-state-container info-state">
          <Info size={36} style={{ color: 'var(--text-accent)' }} />
          <h4 className="state-title">Mode Selection Required</h4>
          <p className="state-desc">
            Mini-sector telemetry comparison is available for <strong>Single Lap</strong> and{' '}
            <strong>Fastest Lap</strong> modes. The current configuration is set to{' '}
            <span className="unsupported-badge">{currentLapMode.toUpperCase()}</span>.
          </p>
          <p className="state-subdesc">
            Please switch to Single Lap or Fastest Lap in the configuration bar above, then click{' '}
            <strong>Run Comparison</strong>.
          </p>
        </div>
      </div>
    );
  }

  // 4. Initial / Idle State
  if (!data) {
    return (
      <div className="comparison-card heat-strip-card" id="heat-strip-idle">
        <div className="card-header">
          <div className="card-title-group">
            <Flame className="card-title-icon" size={18} />
            <h3 className="card-title">Mini-Sector Heat Strip</h3>
          </div>
        </div>
        <div className="table-state-container idle-state">
          <Flame size={40} style={{ color: 'var(--text-muted)' }} />
          <h4 className="state-title">Ready to Compare Mini-Sectors</h4>
          <p className="state-desc">
            Select 2 to 6 drivers and click <strong>Run Comparison</strong> to generate the 20 mini-sector delta
            heat strip.
          </p>
        </div>
      </div>
    );
  }

  // 5. PRE_RACE State
  if (data.status === 'PRE_RACE') {
    return (
      <div className="comparison-card heat-strip-card" id="heat-strip-pre-race">
        <div className="card-header">
          <div className="card-title-group">
            <Flame className="card-title-icon" size={18} />
            <h3 className="card-title">Mini-Sector Heat Strip</h3>
          </div>
        </div>
        <div className="table-state-container pre-race-state">
          <Clock size={36} style={{ color: 'var(--flag-yellow)' }} />
          <h4 className="state-title">Pre-Race Phase</h4>
          <p className="state-desc">
            The current simulation time ({data.as_of_time_s?.toFixed(1)}s) is prior to official race start (T_0).
            Advance playback or switch to Full Race cutoff.
          </p>
        </div>
      </div>
    );
  }

  // 6. NO_DATA State or Missing Mini-Sectors
  if (data.status === 'NO_DATA' || !data.laps || data.laps.length === 0 || !hasMiniSectorData) {
    return (
      <div className="comparison-card heat-strip-card" id="heat-strip-no-data">
        <div className="card-header">
          <div className="card-title-group">
            <Flame className="card-title-icon" size={18} />
            <h3 className="card-title">Mini-Sector Heat Strip</h3>
          </div>
        </div>
        <div className="table-state-container no-data-state">
          <Info size={36} style={{ color: 'var(--text-muted)' }} />
          <h4 className="state-title">No Mini-Sector Telemetry Available</h4>
          <p className="state-desc">
            No completed lap telemetry found for the selected drivers up to the current cutoff point.
          </p>
        </div>
      </div>
    );
  }

  // Helper formatting functions
  const formatCellDelta = (delta: number | null | undefined, isRef: boolean, status: string): string => {
    if (status === 'UNRELIABLE_DATA_GAP') return 'GAP';
    if (status === 'MISSING') return '—';
    if (isRef) return 'REF';
    if (delta === null || delta === undefined) return '—';
    if (Math.abs(delta) < 0.0001) return '0.00';
    const sign = delta > 0 ? '+' : '';
    return `${sign}${delta.toFixed(2)}`;
  };

  const getCellClassName = (
    delta: number | null | undefined,
    isRef: boolean,
    status: string,
    isSelected: boolean
  ): string => {
    const classes = ['heat-strip-cell'];
    if (isSelected) classes.push('is-selected');

    if (status === 'UNRELIABLE_DATA_GAP') {
      classes.push('status-gap');
      return classes.join(' ');
    }
    if (status === 'MISSING') {
      classes.push('status-missing');
      return classes.join(' ');
    }
    if (isRef) {
      classes.push('is-reference-cell');
      return classes.join(' ');
    }
    if (delta === null || delta === undefined) {
      classes.push('status-no-ref');
      return classes.join(' ');
    }
    if (delta < -0.0001) {
      classes.push('delta-faster');
    } else if (delta > 0.0001) {
      classes.push('delta-slower');
    } else {
      classes.push('delta-equal');
    }
    return classes.join(' ');
  };

  return (
    <div className="comparison-card heat-strip-card" id="comparison-heat-strip">
      {/* Card Header */}
      <div className="card-header">
        <div className="card-title-group">
          <Flame className="card-title-icon" size={18} />
          <h3 className="card-title">Mini-Sector Heat Strip (20 Segments)</h3>
          <span className="heat-strip-subtitle">
            Reference: <strong>{refDriver}</strong> • Mode:{' '}
            <strong>{currentLapMode === 'fastest' ? 'Fastest Lap' : `Single Lap`}</strong>
          </span>
        </div>
      </div>

      {/* Stale Configuration Banner */}
      {isStale && (
        <div className="table-stale-banner" id="heat-strip-stale-banner">
          <AlertTriangle size={15} className="stale-banner-icon" />
          <span>Configuration changed. Displaying results from previous execution.</span>
          {onRefresh && (
            <button type="button" className="stale-refresh-btn" onClick={onRefresh}>
              <RefreshCw size={13} style={{ marginRight: 4 }} /> Update
            </button>
          )}
        </div>
      )}

      {/* Color Semantics Legend */}
      <div className="heat-strip-legend" id="heat-strip-legend">
        <span className="legend-label">LEGEND:</span>
        <div className="legend-item">
          <span className="legend-color-box box-faster" />
          <span>Faster than Ref (&lt; 0.0s)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-box box-slower" />
          <span>Slower than Ref (&gt; 0.0s)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-box box-ref" />
          <span>Reference Baseline (0.000s)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-box box-no-ref" />
          <span>No Ref Sector (—)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-box box-gap" />
          <span>Telemetry Gap (&gt; 2.0s)</span>
        </div>
        <div className="legend-item">
          <span className="legend-color-box box-missing" />
          <span>Missing Telemetry</span>
        </div>
      </div>

      {/* Responsive Horizontal Scroll Container */}
      <div className="heat-strip-scroll-container">
        <div className="heat-strip-grid" role="grid" aria-label="Mini-Sector Comparison Grid">
          {/* Top Sector Group Header Row */}
          <div className="grid-row sector-header-row" role="row">
            <div className="grid-cell driver-col-header" role="columnheader">
              SECTOR
            </div>
            <div
              className="grid-cell sector-group-cell sector-1-group"
              style={{ flex: sectorGroups.s1[1] - sectorGroups.s1[0] + 1 }}
              role="columnheader"
            >
              SECTOR 1 (MS {sectorGroups.s1[0]}–{sectorGroups.s1[1]})
            </div>
            <div
              className="grid-cell sector-group-cell sector-2-group"
              style={{ flex: sectorGroups.s2[1] - sectorGroups.s2[0] + 1 }}
              role="columnheader"
            >
              SECTOR 2 (MS {sectorGroups.s2[0]}–{sectorGroups.s2[1]})
            </div>
            <div
              className="grid-cell sector-group-cell sector-3-group"
              style={{ flex: sectorGroups.s3[1] - sectorGroups.s3[0] + 1 }}
              role="columnheader"
            >
              SECTOR 3 (MS {sectorGroups.s3[0]}–{sectorGroups.s3[1]})
            </div>
          </div>

          {/* Mini-Sector Numbers Subheader Row */}
          <div className="grid-row ms-number-header-row" role="row">
            <div className="grid-cell driver-col-header" role="columnheader">
              DRIVER / LAP
            </div>
            {Array.from({ length: 20 }, (_, i) => i + 1).map((msId) => {
              const isSectorDivider =
                msId === sectorGroups.s1[1] || msId === sectorGroups.s2[1];
              return (
                <div
                  key={`ms-hdr-${msId}`}
                  className={`grid-cell ms-col-header ${isSectorDivider ? 'sector-boundary' : ''}`}
                  role="columnheader"
                >
                  {msId}
                </div>
              );
            })}
          </div>

          {/* Driver Rows */}
          {data.laps.map((lapItem) => {
            const isRef = lapItem.driver === refDriver;
            const team = driverTeamMap[lapItem.driver] || 'UNKNOWN';
            const teamColor = getTeamColor(team);
            const miniSectors = lapItem.mini_sectors || [];

            return (
              <div
                key={`heat-strip-driver-${lapItem.driver}`}
                className={`grid-row driver-heat-row ${isRef ? 'is-reference-row' : ''}`}
                role="row"
                id={`heat-strip-row-${lapItem.driver}`}
              >
                {/* Driver Identity Card */}
                <div className="grid-cell driver-meta-cell" role="rowheader">
                  <span className="driver-team-stripe" style={{ backgroundColor: teamColor }} />
                  <div className="driver-meta-info">
                    <div className="driver-line-top">
                      <span className="driver-code">{lapItem.driver}</span>
                      {isRef && <span className="ref-badge">REF</span>}
                      <span className="lap-tag">L{lapItem.lap_number}</span>
                    </div>
                    <div className="driver-line-bottom">
                      <span className="lap-time-text">
                        {formatLapTime(lapItem.lap_time_s)}
                      </span>
                      {lapItem.provider && (
                        <span className="provider-tag">{lapItem.provider}</span>
                      )}
                    </div>
                  </div>
                </div>

                {/* 20 Mini-Sector Cells */}
                {Array.from({ length: 20 }, (_, i) => i + 1).map((msId) => {
                  const ms = miniSectors.find((m) => m.mini_sector_id === msId);
                  const isSectorDivider =
                    msId === sectorGroups.s1[1] || msId === sectorGroups.s2[1];

                  const status = ms ? ms.status : 'MISSING';
                  const delta = ms ? ms.delta_to_reference_s : null;
                  const isSelected =
                    inspectedMs?.driver === lapItem.driver &&
                    inspectedMs?.ms.mini_sector_id === msId;

                  const cellClass = `${getCellClassName(delta, isRef, status, isSelected)}${
                    isSectorDivider ? ' sector-boundary' : ''
                  }`;

                  const accessibleLabel = `${lapItem.driver} Mini-Sector ${msId} (${
                    ms ? `Sector ${ms.sector}` : ''
                  }): ${
                    isRef
                      ? 'Reference Baseline 0.000s'
                      : delta !== null && delta !== undefined
                      ? `${delta > 0 ? '+' : ''}${delta.toFixed(4)}s`
                      : status === 'UNRELIABLE_DATA_GAP'
                      ? 'Telemetry Gap'
                      : 'Missing'
                  }`;

                  return (
                    <div
                      key={`ms-cell-${lapItem.driver}-${msId}`}
                      className={cellClass}
                      role="gridcell"
                      tabIndex={0}
                      data-driver={lapItem.driver}
                      data-ms-id={msId}
                      aria-label={accessibleLabel}
                      onMouseEnter={() => {
                        if (ms) setInspectedMs({ driver: lapItem.driver, ms, lapItem });
                      }}
                      onFocus={() => {
                        if (ms) setInspectedMs({ driver: lapItem.driver, ms, lapItem });
                      }}
                      onClick={() => {
                        if (ms) setInspectedMs({ driver: lapItem.driver, ms, lapItem });
                      }}
                      onKeyDown={(e) => {
                        if ((e.key === 'Enter' || e.key === ' ') && ms) {
                          e.preventDefault();
                          setInspectedMs({ driver: lapItem.driver, ms, lapItem });
                        }
                      }}
                    >
                      <span className="ms-cell-text">
                        {formatCellDelta(delta, isRef, status)}
                      </span>
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>
      </div>

      {/* Accessible Inspection Panel */}
      <div className="heat-strip-inspection-panel" id="heat-strip-cell-details">
        {inspectedMs ? (
          <div className="inspection-content">
            <div className="inspection-header">
              <span
                className="inspection-team-swatch"
                style={{
                  backgroundColor: getTeamColor(driverTeamMap[inspectedMs.driver] || ''),
                }}
              />
              <span className="inspection-driver">{inspectedMs.driver}</span>
              {inspectedMs.driver === refDriver && <span className="ref-badge">REF</span>}
              <span className="inspection-ms-id">Mini-Sector {inspectedMs.ms.mini_sector_id}</span>
              <span className="inspection-sector-tag">Sector {inspectedMs.ms.sector}</span>
            </div>

            <div className="inspection-stats-grid">
              <div className="inspection-stat">
                <span className="stat-label">
                  <Clock size={12} style={{ marginRight: 4 }} /> Delta to Ref
                </span>
                <span
                  className={`stat-value ${
                    inspectedMs.driver === refDriver
                      ? 'delta-zero'
                      : inspectedMs.ms.delta_to_reference_s !== null &&
                        inspectedMs.ms.delta_to_reference_s !== undefined
                      ? inspectedMs.ms.delta_to_reference_s < 0
                        ? 'delta-faster'
                        : 'delta-slower'
                      : ''
                  }`}
                >
                  {inspectedMs.driver === refDriver
                    ? '0.0000s (Reference)'
                    : inspectedMs.ms.delta_to_reference_s !== null &&
                      inspectedMs.ms.delta_to_reference_s !== undefined
                    ? `${inspectedMs.ms.delta_to_reference_s > 0 ? '+' : ''}${inspectedMs.ms.delta_to_reference_s.toFixed(
                        4
                      )}s vs ${refDriver}`
                    : '— (Reference Unavailable)'}
                </span>
              </div>

              <div className="inspection-stat">
                <span className="stat-label">
                  <Sparkles size={12} style={{ marginRight: 4 }} /> Duration
                </span>
                <span className="stat-value">
                  {inspectedMs.ms.duration_s !== null && inspectedMs.ms.duration_s !== undefined
                    ? `${inspectedMs.ms.duration_s.toFixed(4)}s`
                    : '—'}
                </span>
              </div>

              <div className="inspection-stat">
                <span className="stat-label">
                  <Gauge size={12} style={{ marginRight: 4 }} /> Avg Speed
                </span>
                <span className="stat-value">
                  {inspectedMs.ms.avg_speed_kmh !== null &&
                  inspectedMs.ms.avg_speed_kmh !== undefined &&
                  !isNaN(inspectedMs.ms.avg_speed_kmh)
                    ? `${inspectedMs.ms.avg_speed_kmh.toFixed(1)} km/h`
                    : 'Unavailable'}
                </span>
              </div>

              <div className="inspection-stat">
                <span className="stat-label">
                  <MapPin size={12} style={{ marginRight: 4 }} /> Distance Range
                </span>
                <span className="stat-value">
                  {inspectedMs.ms.start_m.toFixed(1)}m → {inspectedMs.ms.end_m.toFixed(1)}m (
                  {(inspectedMs.ms.end_m - inspectedMs.ms.start_m).toFixed(1)}m)
                </span>
              </div>

              <div className="inspection-stat">
                <span className="stat-label">Status</span>
                <span
                  className={`stat-value status-${inspectedMs.ms.status.toLowerCase().replace(/_/g, '-')}`}
                >
                  {inspectedMs.ms.status === 'VALID'
                    ? 'VALID'
                    : inspectedMs.ms.status === 'UNRELIABLE_DATA_GAP'
                    ? 'UNRELIABLE DATA GAP (> 2.0s)'
                    : 'MISSING TELEMETRY'}
                </span>
              </div>

              {inspectedMs.lapItem.precision_tier && (
                <div className="inspection-stat">
                  <span className="stat-label">Precision Tier</span>
                  <span className="stat-value tier-tag">
                    {inspectedMs.lapItem.provider}: {inspectedMs.lapItem.precision_tier}
                  </span>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="inspection-prompt">
            <Info size={15} style={{ marginRight: 6, color: 'var(--text-muted)' }} />
            <span>Hover, focus, or tap any mini-sector cell above to inspect timing, delta, and speed details.</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComparisonHeatStrip;
