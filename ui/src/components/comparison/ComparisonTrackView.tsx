import React, { useState, useMemo } from 'react';
import {
  ComparisonResponse,
  MiniSectorTimingItem,
} from '../../types/comparison';
import { TrackOutline } from '../../types/replay';
import { DriverOption } from './ComparisonConfigBar';
import { CircuitMap, TrackMiniSectorOverlaySegment } from '../CircuitMap';
import { getTeamColor } from '../../constants/teams';
import {
  AlertCircle,
  AlertTriangle,
  Info,
  Loader2,
  RefreshCw,
  Gauge,
  Trophy,
  Zap,
  MapPin,
  Clock,
  Layers,
} from 'lucide-react';

export type TrackOverlayType = 'dominance' | 'delta' | 'speed';

export interface ComparisonTrackViewProps {
  data: ComparisonResponse | null;
  isLoading: boolean;
  error: string | null;
  isStale: boolean;
  availableDrivers: DriverOption[];
  currentLapMode?: string;
  onRefresh?: () => void;
  trackOutline?: TrackOutline | null;
  circuitLoading?: boolean;
  circuitError?: string | null;
}

export const ComparisonTrackView: React.FC<ComparisonTrackViewProps> = ({
  data,
  isLoading,
  error,
  isStale,
  availableDrivers,
  currentLapMode = 'single',
  onRefresh,
  trackOutline,
  circuitLoading = false,
  circuitError = null,
}) => {
  const [activeOverlay, setActiveOverlay] = useState<TrackOverlayType>('dominance');
  const [inspectedDriver, setInspectedDriver] = useState<string>('');
  const [selectedMiniSectorId, setSelectedMiniSectorId] = useState<number | null>(null);
  const [hoveredMiniSectorId, setHoveredMiniSectorId] = useState<number | null>(null);

  // Map driver code to team
  const driverTeamMap = useMemo(() => {
    const map: Record<string, string> = {};
    availableDrivers.forEach((d) => {
      map[d.code] = d.team;
    });
    return map;
  }, [availableDrivers]);

  // Reference driver code
  const refDriver = useMemo(() => {
    return data?.reference_driver || (data?.laps && data.laps.length > 0 ? data.laps[0].driver : '');
  }, [data]);

  // Compared drivers with valid laps
  const comparedDrivers = useMemo(() => {
    if (!data?.laps) return [];
    return data.laps.map((l) => l.driver);
  }, [data]);

  // Initialize inspected driver when data changes if not set or invalid
  const currentInspectedDriver = useMemo(() => {
    if (inspectedDriver && comparedDrivers.includes(inspectedDriver)) {
      return inspectedDriver;
    }
    // Default to first non-reference driver, or reference driver if only 1 driver
    const nonRef = comparedDrivers.find((d) => d !== refDriver);
    return nonRef || refDriver || (comparedDrivers.length > 0 ? comparedDrivers[0] : '');
  }, [inspectedDriver, comparedDrivers, refDriver]);

  // Check if current lap mode is supported for mini-sector analytics
  const isModeSupported = useMemo(() => {
    return currentLapMode === 'single' || currentLapMode === 'fastest';
  }, [currentLapMode]);

  // Check if response has mini-sectors populated
  const hasMiniSectorData = useMemo(() => {
    if (!data?.laps) return false;
    return data.laps.some((l) => l.mini_sectors && l.mini_sectors.length > 0);
  }, [data]);

  // Calculate fastest driver per mini-sector (Dominance Map)
  const dominanceMap = useMemo(() => {
    if (!data?.laps || data.laps.length === 0) return {};
    const result: Record<
      number,
      {
        fastestDriver: string;
        fastestDuration: number;
        fastestTeam: string;
        marginToSecond: number | null;
        secondDriver: string | null;
      }
    > = {};

    for (let msId = 1; msId <= 20; msId++) {
      let fastestD: string | null = null;
      let fastestT = Infinity;
      let secondD: string | null = null;
      let secondT = Infinity;

      for (const lap of data.laps) {
        if (!lap.mini_sectors) continue;
        const msItem = lap.mini_sectors.find((m) => m.mini_sector_id === msId);
        if (
          msItem &&
          msItem.status === 'VALID' &&
          typeof msItem.duration_s === 'number' &&
          msItem.duration_s > 0
        ) {
          if (msItem.duration_s < fastestT) {
            secondT = fastestT;
            secondD = fastestD;
            fastestT = msItem.duration_s;
            fastestD = lap.driver;
          } else if (msItem.duration_s < secondT) {
            secondT = msItem.duration_s;
            secondD = lap.driver;
          }
        }
      }

      if (fastestD !== null && fastestT < Infinity) {
        result[msId] = {
          fastestDriver: fastestD,
          fastestDuration: fastestT,
          fastestTeam: driverTeamMap[fastestD] || 'Unknown',
          marginToSecond: secondT < Infinity ? secondT - fastestT : null,
          secondDriver: secondD,
        };
      }
    }
    return result;
  }, [data, driverTeamMap]);

  // Mini-sector wins tally for Dominance legend
  const dominanceWins = useMemo(() => {
    const tally: Record<string, number> = {};
    comparedDrivers.forEach((d) => {
      tally[d] = 0;
    });
    Object.values(dominanceMap).forEach((dom) => {
      if (tally[dom.fastestDriver] !== undefined) {
        tally[dom.fastestDriver]++;
      } else {
        tally[dom.fastestDriver] = 1;
      }
    });
    return tally;
  }, [dominanceMap, comparedDrivers]);

  // Inspected driver lap item
  const inspectedLapItem = useMemo(() => {
    if (!data?.laps) return null;
    return data.laps.find((l) => l.driver === currentInspectedDriver) || null;
  }, [data, currentInspectedDriver]);

  // Speed min/max across all valid mini-sectors of the inspected driver (or overall)
  const speedRange = useMemo(() => {
    let minSpd = Infinity;
    let maxSpd = -Infinity;
    if (data?.laps) {
      for (const lap of data.laps) {
        if (!lap.mini_sectors) continue;
        for (const ms of lap.mini_sectors) {
          if (ms.status === 'VALID' && typeof ms.avg_speed_kmh === 'number') {
            if (ms.avg_speed_kmh < minSpd) minSpd = ms.avg_speed_kmh;
            if (ms.avg_speed_kmh > maxSpd) maxSpd = ms.avg_speed_kmh;
          }
        }
      }
    }
    if (minSpd === Infinity || maxSpd === -Infinity || minSpd >= maxSpd) {
      return { min: 80, max: 320 };
    }
    return { min: Math.floor(minSpd), max: Math.ceil(maxSpd) };
  }, [data]);

  // Convert speed to vibrant color: Blue (apex ~80) -> Cyan -> Green -> Yellow -> Red (straight ~330)
  const getSpeedColor = (speed: number | null | undefined): string => {
    if (speed === null || speed === undefined || isNaN(speed)) return '#4a5568';
    const min = speedRange.min;
    const max = speedRange.max;
    const ratio = Math.max(0, Math.min(1, (speed - min) / (max - min || 1)));
    // Continuous hue mapping: 195 (Cyan) down to 0 (Red)
    const hue = Math.round(195 - ratio * 195);
    return `hsl(${hue}, 92%, 52%)`;
  };

  // Build the track overlay segments for CircuitMap
  const miniSectorSegments: TrackMiniSectorOverlaySegment[] = useMemo(() => {
    if (!data?.laps || data.laps.length === 0 || !hasMiniSectorData) return [];
    const segments: TrackMiniSectorOverlaySegment[] = [];

    for (let msId = 1; msId <= 20; msId++) {
      let color = '#4a5568'; // default gray for missing
      let label = `Mini-Sector ${msId}`;
      let tooltip = `Mini-Sector ${msId}`;

      if (activeOverlay === 'dominance') {
        const dom = dominanceMap[msId];
        if (dom) {
          color = getTeamColor(dom.fastestTeam);
          label = `MS ${msId}: ${dom.fastestDriver} (${dom.fastestDuration.toFixed(3)}s)`;
          tooltip = `Mini-Sector ${msId}: Fastest ${dom.fastestDriver} (${dom.fastestDuration.toFixed(3)}s)${
            dom.marginToSecond !== null ? ` [gap: +${dom.marginToSecond.toFixed(3)}s vs ${dom.secondDriver}]` : ''
          }`;
        } else {
          color = '#383f50';
          label = `MS ${msId}: No Telemetry`;
          tooltip = `Mini-Sector ${msId}: No valid lap telemetry`;
        }
      } else if (activeOverlay === 'delta') {
        const msItem = inspectedLapItem?.mini_sectors?.find((m) => m.mini_sector_id === msId);
        const isRef = currentInspectedDriver === refDriver;

        if (!msItem || msItem.status === 'MISSING') {
          color = '#4a5568';
          label = `MS ${msId}: Missing Telemetry`;
          tooltip = `Mini-Sector ${msId}: Telemetry missing for ${currentInspectedDriver}`;
        } else if (msItem.status === 'UNRELIABLE_DATA_GAP') {
          color = 'var(--flag-yellow, #ffb800)';
          label = `MS ${msId}: Data Gap`;
          tooltip = `Mini-Sector ${msId}: Data gap detected for ${currentInspectedDriver}`;
        } else if (isRef) {
          color = 'var(--f1-gold, #ffd700)';
          label = `MS ${msId}: REF (${currentInspectedDriver})`;
          tooltip = `Mini-Sector ${msId}: Reference driver baseline (${currentInspectedDriver})`;
        } else if (msItem.delta_to_reference_s !== null && msItem.delta_to_reference_s !== undefined) {
          const delta = msItem.delta_to_reference_s;
          if (delta < -0.0001) {
            color = '#00d26a'; // Faster (green)
            label = `MS ${msId}: ${delta.toFixed(2)}s`;
            tooltip = `Mini-Sector ${msId}: ${currentInspectedDriver} faster by ${Math.abs(delta).toFixed(3)}s vs ${refDriver}`;
          } else if (delta > 0.0001) {
            color = '#ff3b30'; // Slower (red)
            label = `MS ${msId}: +${delta.toFixed(2)}s`;
            tooltip = `Mini-Sector ${msId}: ${currentInspectedDriver} slower by +${delta.toFixed(3)}s vs ${refDriver}`;
          } else {
            color = '#e2e8f0'; // Equal
            label = `MS ${msId}: 0.00s`;
            tooltip = `Mini-Sector ${msId}: Equal timing with ${refDriver}`;
          }
        } else {
          color = '#4a5568';
          label = `MS ${msId}: No Reference`;
          tooltip = `Mini-Sector ${msId}: No delta baseline available`;
        }
      } else if (activeOverlay === 'speed') {
        const msItem = inspectedLapItem?.mini_sectors?.find((m) => m.mini_sector_id === msId);
        if (!msItem || msItem.status !== 'VALID' || typeof msItem.avg_speed_kmh !== 'number') {
          color = '#4a5568';
          label = `MS ${msId}: No Speed`;
          tooltip = `Mini-Sector ${msId}: Speed unavailable for ${currentInspectedDriver}`;
        } else {
          color = getSpeedColor(msItem.avg_speed_kmh);
          label = `MS ${msId}: ${Math.round(msItem.avg_speed_kmh)} km/h`;
          tooltip = `Mini-Sector ${msId} (${currentInspectedDriver}): Avg Speed ${msItem.avg_speed_kmh.toFixed(1)} km/h`;
        }
      }

      segments.push({
        mini_sector_id: msId,
        color,
        label,
        tooltip,
      });
    }

    return segments;
  }, [
    data,
    hasMiniSectorData,
    activeOverlay,
    dominanceMap,
    inspectedLapItem,
    currentInspectedDriver,
    refDriver,
    speedRange,
  ]);

  // Currently inspected mini-sector details
  const selectedSectorDetails = useMemo(() => {
    if (selectedMiniSectorId === null) return null;
    const msId = selectedMiniSectorId;

    // Get boundaries from trackOutline or first lap
    let startM = 0;
    let endM = 0;
    let sectorNum = 1;
    if (trackOutline?.mini_sectors) {
      const msDef = trackOutline.mini_sectors.find((m) => m.mini_sector_id === msId);
      if (msDef) {
        startM = msDef.start_m;
        endM = msDef.end_m;
        sectorNum = msDef.sector;
      }
    } else if (data?.laps) {
      const firstLapWithMs = data.laps.find((l) => l.mini_sectors && l.mini_sectors.length > 0);
      const msDef = firstLapWithMs?.mini_sectors?.find((m: MiniSectorTimingItem) => m.mini_sector_id === msId);
      if (msDef) {
        startM = msDef.start_m;
        endM = msDef.end_m;
        sectorNum = msDef.sector;
      }
    }

    // Drivers breakdown for this mini-sector
    const driversBreakdown: {
      driver: string;
      team: string;
      duration_s: number | null;
      delta_s: number | null;
      avg_speed_kmh: number | null;
      status: string;
      isFastest: boolean;
    }[] = [];

    const dom = dominanceMap[msId];

    if (data?.laps) {
      data.laps.forEach((lap) => {
        const msItem = lap.mini_sectors?.find((m) => m.mini_sector_id === msId);
        driversBreakdown.push({
          driver: lap.driver,
          team: driverTeamMap[lap.driver] || 'Unknown',
          duration_s: msItem && msItem.status === 'VALID' ? msItem.duration_s ?? null : null,
          delta_s: msItem && msItem.status === 'VALID' ? msItem.delta_to_reference_s ?? null : null,
          avg_speed_kmh: msItem && msItem.status === 'VALID' ? msItem.avg_speed_kmh ?? null : null,
          status: msItem?.status || 'MISSING',
          isFastest: dom?.fastestDriver === lap.driver,
        });
      });
    }

    return {
      mini_sector_id: msId,
      sector: sectorNum,
      start_m: Math.round(startM),
      end_m: Math.round(endM),
      length_m: Math.round(endM - startM),
      fastest: dom || null,
      driversBreakdown,
    };
  }, [selectedMiniSectorId, trackOutline, data, dominanceMap, driverTeamMap]);

  // 1. Loading State
  if (isLoading || circuitLoading) {
    return (
      <div className="comparison-card track-view-card" id="track-view-loading">
        <div className="card-header">
          <div className="card-title-group">
            <Layers className="card-title-icon" size={18} />
            <h3 className="card-title">Circuit Track View</h3>
          </div>
        </div>
        <div className="table-state-container">
          <Loader2 className="spinner-icon" size={32} />
          <h4 className="state-title">Loading Track Telemetry & Overlays...</h4>
          <p className="state-desc">Mapping mini-sector boundary metrics onto circuit geometry.</p>
        </div>
      </div>
    );
  }

  // 2. Error State
  if (error || circuitError) {
    return (
      <div className="comparison-card track-view-card" id="track-view-error">
        <div className="card-header">
          <div className="card-title-group">
            <Layers className="card-title-icon" size={18} />
            <h3 className="card-title">Circuit Track View</h3>
          </div>
        </div>
        <div className="table-state-container error-state">
          <AlertCircle className="error-icon" size={36} />
          <h4 className="state-title">Track View Unavailable</h4>
          <p className="state-desc">{error || circuitError}</p>
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
      <div className="comparison-card track-view-card" id="track-view-unsupported-mode">
        <div className="card-header">
          <div className="card-title-group">
            <Layers className="card-title-icon" size={18} />
            <h3 className="card-title">Circuit Track View</h3>
          </div>
        </div>
        <div className="table-state-container unconfigured-state">
          <Info size={36} style={{ color: 'var(--text-muted)' }} />
          <h4 className="state-title">Track Overlays Require Single or Fastest Lap Mode</h4>
          <p className="state-desc">
            Spatial mini-sector overlays are available for <strong>Single Lap</strong> and{' '}
            <strong>Fastest Lap</strong> modes. Multi-lap aggregation (stint, range, last clean) is supported in
            Table and Line views.
          </p>
        </div>
      </div>
    );
  }

  // 4. Idle / Unconfigured State
  if (!data) {
    return (
      <div className="comparison-card track-view-card" id="track-view-idle">
        <div className="card-header">
          <div className="card-title-group">
            <Layers className="card-title-icon" size={18} />
            <h3 className="card-title">Circuit Track View</h3>
          </div>
        </div>
        <div className="table-state-container unconfigured-state">
          <Layers size={36} style={{ color: 'var(--text-muted)' }} />
          <h4 className="state-title">Track View Ready</h4>
          <p className="state-desc">
            Select drivers and click <strong>Run Comparison</strong> to project dominance, deltas, and speed across
            circuit mini-sectors.
          </p>
        </div>
      </div>
    );
  }

  // 5. PRE_RACE State
  if (data.status === 'PRE_RACE') {
    return (
      <div className="comparison-card track-view-card" id="track-view-pre-race">
        <div className="card-header">
          <div className="card-title-group">
            <Layers className="card-title-icon" size={18} />
            <h3 className="card-title">Circuit Track View</h3>
          </div>
        </div>
        <div className="table-state-container pre-race-state">
          <Clock size={36} style={{ color: 'var(--flag-yellow)' }} />
          <h4 className="state-title">Pre-Race Session Time</h4>
          <p className="state-desc">
            The replay clock is positioned before race start. Advance replay playback or select Full Race cutoff to
            view track overlays.
          </p>
        </div>
      </div>
    );
  }

  // 6. NO_DATA State or Missing Mini-Sectors
  if (data.status === 'NO_DATA' || !data.laps || data.laps.length === 0 || !hasMiniSectorData) {
    return (
      <div className="comparison-card track-view-card" id="track-view-no-data">
        <div className="card-header">
          <div className="card-title-group">
            <Layers className="card-title-icon" size={18} />
            <h3 className="card-title">Circuit Track View</h3>
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

  return (
    <div className="comparison-card track-view-card" id="comparison-track-view">
      {/* Card Header & Controls */}
      <div className="card-header track-view-header">
        <div className="card-title-group">
          <Layers className="card-title-icon" size={18} />
          <h3 className="card-title">Track View (20 Mini-Sectors)</h3>
          <span className="heat-strip-subtitle">
            Reference: <strong>{refDriver}</strong> • Mode:{' '}
            <strong>{currentLapMode === 'fastest' ? 'Fastest Lap' : `Single Lap`}</strong>
          </span>
        </div>

        {/* Overlay Mode Switcher */}
        <div className="track-overlay-controls" role="toolbar" aria-label="Track overlay selection">
          <div className="track-overlay-toggle-group">
            <button
              type="button"
              id="track-overlay-dominance"
              className={`track-overlay-btn${activeOverlay === 'dominance' ? ' active' : ''}`}
              onClick={() => setActiveOverlay('dominance')}
              title="Mini-Sector Dominance (Fastest driver per mini-sector)"
            >
              <Trophy size={14} style={{ marginRight: 6 }} /> Dominance
            </button>
            <button
              type="button"
              id="track-overlay-delta"
              className={`track-overlay-btn${activeOverlay === 'delta' ? ' active' : ''}`}
              onClick={() => setActiveOverlay('delta')}
              title="Delta to Reference Driver"
            >
              <Zap size={14} style={{ marginRight: 6 }} /> Delta
            </button>
            <button
              type="button"
              id="track-overlay-speed"
              className={`track-overlay-btn${activeOverlay === 'speed' ? ' active' : ''}`}
              onClick={() => setActiveOverlay('speed')}
              title="Average Speed Heatmap"
            >
              <Gauge size={14} style={{ marginRight: 6 }} /> Speed
            </button>
          </div>

          {/* Driver Selector Dropdown (Active for Delta & Speed overlays) */}
          {(activeOverlay === 'delta' || activeOverlay === 'speed') && (
            <div className="track-driver-selector-wrapper">
              <label htmlFor="track-driver-select" className="track-driver-label">
                Driver:
              </label>
              <select
                id="track-driver-select"
                className="track-driver-select"
                value={currentInspectedDriver}
                onChange={(e) => setInspectedDriver(e.target.value)}
              >
                {comparedDrivers.map((d) => (
                  <option key={d} value={d}>
                    {d} {d === refDriver ? '(Ref)' : ''}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Stale Configuration Banner */}
      {isStale && (
        <div className="table-stale-banner" id="track-view-stale-banner">
          <AlertTriangle size={15} className="stale-banner-icon" />
          <span>Configuration changed. Displaying results from previous execution.</span>
          {onRefresh && (
            <button type="button" className="stale-refresh-btn" onClick={onRefresh}>
              <RefreshCw size={13} style={{ marginRight: 4 }} /> Update
            </button>
          )}
        </div>
      )}

      {/* Interactive Track Area */}
      <div className="track-view-main-layout">
        <div className="track-map-wrapper">
          <CircuitMap
            id="comparison-circuit-map"
            className="comparison-circuit-map-view"
            trackOutline={trackOutline || null}
            drivers={[]}
            miniSectorSegments={miniSectorSegments}
            selectedMiniSectorId={selectedMiniSectorId}
            onSelectMiniSector={(id) => setSelectedMiniSectorId(id === selectedMiniSectorId ? null : id)}
            hoveredMiniSectorId={hoveredMiniSectorId}
            onHoverMiniSector={setHoveredMiniSectorId}
            hideDriverMarkers={true}
            hideDefaultLegend={true}
            hideHeader={false}
          />
        </div>

        {/* Track Overlay Legend */}
        <div className="track-overlay-legend" id="track-overlay-legend">
          {activeOverlay === 'dominance' && (
            <div className="dominance-legend-grid">
              <div className="legend-title">Fastest Driver per Mini-Sector:</div>
              <div className="dominance-drivers-list">
                {comparedDrivers.map((d) => {
                  const wins = dominanceWins[d] || 0;
                  const teamColor = getTeamColor(driverTeamMap[d] || 'Unknown');
                  return (
                    <div key={d} className="dominance-legend-item">
                      <span className="legend-color-swatch" style={{ background: teamColor }}></span>
                      <strong className="driver-code">{d}</strong>
                      <span className="wins-count mono">{wins} / 20</span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {activeOverlay === 'delta' && (
            <div className="delta-legend-row">
              <div className="legend-title">Delta vs {refDriver}:</div>
              <div className="delta-swatches">
                <div className="legend-item">
                  <span className="legend-color-swatch delta-faster-swatch"></span>
                  <span>Faster (&lt; -0.00s)</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color-swatch is-ref-swatch"></span>
                  <span>Reference ({refDriver})</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color-swatch delta-slower-swatch"></span>
                  <span>Slower (&gt; +0.00s)</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color-swatch status-gap-swatch"></span>
                  <span>Data Gap</span>
                </div>
                <div className="legend-item">
                  <span className="legend-color-swatch status-missing-swatch"></span>
                  <span>Missing</span>
                </div>
              </div>
            </div>
          )}

          {activeOverlay === 'speed' && (
            <div className="speed-legend-row">
              <div className="legend-title">Speed Spectrum ({currentInspectedDriver}):</div>
              <div className="speed-bar-container">
                <span className="speed-label mono">{speedRange.min} km/h (Apex)</span>
                <div className="speed-gradient-bar"></div>
                <span className="speed-label mono">{speedRange.max} km/h (Top Speed)</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Selected Section Details Panel */}
      <div className="track-section-details-panel" id="track-section-details">
        {selectedSectorDetails ? (
          <div className="section-details-content">
            <div className="section-details-header">
              <div className="section-title-group">
                <MapPin size={16} className="section-icon" />
                <h4 className="section-title">
                  Mini-Sector #{selectedSectorDetails.mini_sector_id} (Sector {selectedSectorDetails.sector})
                </h4>
                <span className="section-distance mono">
                  {selectedSectorDetails.start_m}m – {selectedSectorDetails.end_m}m ({selectedSectorDetails.length_m}m)
                </span>
              </div>

              {selectedSectorDetails.fastest && (
                <div className="section-winner-badge">
                  <Trophy size={14} style={{ marginRight: 6, color: 'var(--f1-gold)' }} />
                  <span>
                    Fastest: <strong>{selectedSectorDetails.fastest.fastestDriver}</strong> (
                    {selectedSectorDetails.fastest.fastestDuration.toFixed(3)}s)
                  </span>
                  {selectedSectorDetails.fastest.marginToSecond !== null && (
                    <span className="winner-margin mono">
                      +
                      {selectedSectorDetails.fastest.marginToSecond.toFixed(3)}s vs{' '}
                      {selectedSectorDetails.fastest.secondDriver}
                    </span>
                  )}
                </div>
              )}
            </div>

            {/* Drivers Breakdown Table */}
            <div className="section-drivers-table-wrapper">
              <table className="section-drivers-table">
                <thead>
                  <tr>
                    <th>Driver</th>
                    <th>Time</th>
                    <th>Delta vs {refDriver}</th>
                    <th>Avg Speed</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedSectorDetails.driversBreakdown.map((row) => {
                    const isRef = row.driver === refDriver;
                    const deltaStr =
                      row.status === 'UNRELIABLE_DATA_GAP'
                        ? 'GAP'
                        : row.status === 'MISSING'
                        ? '—'
                        : isRef
                        ? 'REF'
                        : row.delta_s !== null
                        ? `${row.delta_s > 0 ? '+' : ''}${row.delta_s.toFixed(2)}s`
                        : '—';

                    const deltaColorClass =
                      row.status === 'UNRELIABLE_DATA_GAP'
                        ? 'delta-gap'
                        : isRef
                        ? 'delta-ref'
                        : row.delta_s !== null && row.delta_s < -0.0001
                        ? 'delta-faster-text'
                        : row.delta_s !== null && row.delta_s > 0.0001
                        ? 'delta-slower-text'
                        : '';

                    return (
                      <tr key={row.driver} className={row.isFastest ? 'fastest-row' : ''}>
                        <td>
                          <div className="driver-cell-badge">
                            <span
                              className="driver-swatch"
                              style={{ background: getTeamColor(row.team) }}
                            ></span>
                            <span className="driver-name-code">{row.driver}</span>
                            {row.isFastest && <span className="fastest-pill">FASTEST</span>}
                          </div>
                        </td>
                        <td className="mono">
                          {row.duration_s !== null ? `${row.duration_s.toFixed(3)}s` : '—'}
                        </td>
                        <td className={`mono ${deltaColorClass}`}>{deltaStr}</td>
                        <td className="mono">
                          {row.avg_speed_kmh !== null ? `${row.avg_speed_kmh.toFixed(1)} km/h` : '—'}
                        </td>
                        <td>
                          <span className={`status-pill status-${row.status.toLowerCase()}`}>
                            {row.status}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="section-details-empty">
            <Info size={16} style={{ marginRight: 8, color: 'var(--text-muted)' }} />
            <span>Select a mini-sector on the track to inspect timing, deltas, and speed telemetry.</span>
          </div>
        )}
      </div>
    </div>
  );
};
