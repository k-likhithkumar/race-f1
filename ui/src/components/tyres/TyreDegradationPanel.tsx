import React, { useState, useMemo } from 'react';
import { TyreDegradationResponse, TyreLapPoint } from '../../types/tyres';
import { DriverOption } from '../comparison/ComparisonConfigBar';
import { getTeamColor } from '../../constants/teams';
import {
  AlertTriangle,
  RotateCw,
  Info,
  Clock,
  CheckCircle2,
  AlertCircle,
  TrendingUp,
  Activity,
  Layers,
} from 'lucide-react';

interface TyreDegradationPanelProps {
  data: TyreDegradationResponse | null;
  isLoading: boolean;
  error: string | null;
  selectedDriver: string;
  onSelectDriver: (driver: string) => void;
  selectedStint: number;
  onSelectStint: (stint: number) => void;
  syncWithReplay: boolean;
  onToggleSync: (sync: boolean) => void;
  onRefresh: () => void;
  availableDrivers: DriverOption[];
  currentReplayLap?: number;
  currentReplayTimeS?: number;
}

const COMPOUND_COLORS: Record<string, string> = {
  SOFT: '#FF3B30',
  MEDIUM: '#FFCC00',
  HARD: '#FFFFFF',
  INTERMEDIATE: '#34C759',
  WET: '#007AFF',
  UNKNOWN: '#8C96A8',
};

const formatSecondsToLapTime = (sec: number | null | undefined): string => {
  if (sec === null || sec === undefined || isNaN(sec)) return '--:--.---';
  const mins = Math.floor(sec / 60);
  const remainingSecs = sec % 60;
  return `${mins}:${remainingSecs.toFixed(3).padStart(6, '0')}`;
};

export const TyreDegradationPanel: React.FC<TyreDegradationPanelProps> = ({
  data,
  isLoading,
  error,
  selectedDriver,
  onSelectDriver,
  selectedStint,
  onSelectStint,
  syncWithReplay,
  onToggleSync,
  onRefresh,
  availableDrivers,
  currentReplayLap,
  currentReplayTimeS,
}) => {
  const [hoveredLap, setHoveredLap] = useState<TyreLapPoint | null>(null);

  const compoundColor = useMemo(() => {
    const comp = data?.compound?.toUpperCase() || 'UNKNOWN';
    return COMPOUND_COLORS[comp] || COMPOUND_COLORS.UNKNOWN;
  }, [data?.compound]);

  const driverTeam = useMemo(() => {
    const drv = availableDrivers.find((d) => d.code === selectedDriver);
    return drv?.team;
  }, [availableDrivers, selectedDriver]);

  const teamColor = useMemo(() => getTeamColor(driverTeam), [driverTeam]);

  // Chart layout parameters
  const chartWidth = 860;
  const chartHeight = 340;
  const padding = { top: 30, right: 40, bottom: 45, left: 75 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Compute scales for SVG chart
  const { laps, yMin, yMax, getX, getY } = useMemo(() => {
    const validLaps = data?.laps || [];
    if (validLaps.length === 0) {
      return {
        laps: [],
        yMin: 80,
        yMax: 100,
        getX: (_lap: number) => 0,
        getY: (_time: number) => 0,
      };
    }

    const lapNumbers = validLaps.map((l) => l.lap_number);
    const xMinVal = Math.min(...lapNumbers);
    const xMaxVal = Math.max(...lapNumbers);

    // Collect all valid times for Y domain
    const allTimes: number[] = [];
    validLaps.forEach((l) => {
      if (l.raw_lap_time_s) allTimes.push(l.raw_lap_time_s);
      if (l.fuel_corrected_lap_time_s) allTimes.push(l.fuel_corrected_lap_time_s);
      if (l.fitted_lap_time_s) allTimes.push(l.fitted_lap_time_s);
    });

    const rawYMin = allTimes.length > 0 ? Math.min(...allTimes) : 80;
    const rawYMax = allTimes.length > 0 ? Math.max(...allTimes) : 100;
    const ySpan = Math.max(rawYMax - rawYMin, 1.5);
    const yMinVal = rawYMin - ySpan * 0.1;
    const yMaxVal = rawYMax + ySpan * 0.1;

    const xSpan = Math.max(xMaxVal - xMinVal, 1);
    const getXCoord = (lap: number) =>
      padding.left + ((lap - xMinVal) / xSpan) * innerWidth;

    const getYCoord = (timeVal: number) =>
      padding.top + innerHeight - ((timeVal - yMinVal) / (yMaxVal - yMinVal)) * innerHeight;

    return {
      laps: validLaps,
      xMin: xMinVal,
      xMax: xMaxVal,
      yMin: yMinVal,
      yMax: yMaxVal,
      getX: getXCoord,
      getY: getYCoord,
    };
  }, [data?.laps, innerWidth, innerHeight, padding.left, padding.top]);

  // Y-axis tick values (5 ticks)
  const yTicks = useMemo(() => {
    const ticks: number[] = [];
    const step = (yMax - yMin) / 4;
    for (let i = 0; i <= 4; i++) {
      ticks.push(yMin + step * i);
    }
    return ticks;
  }, [yMin, yMax]);

  return (
    <div className="tyre-degradation-workspace" id="tyre-degradation-panel">
      {/* Header & Controls Bar */}
      <div className="tyre-panel-controls-bar">
        <div className="tyre-controls-left">
          <div className="tyre-control-group">
            <label htmlFor="select-tyre-driver" className="tyre-control-label">
              Driver
            </label>
            <select
              id="select-tyre-driver"
              className="tyre-driver-select"
              value={selectedDriver}
              onChange={(e) => onSelectDriver(e.target.value)}
            >
              {availableDrivers.map((d) => (
                <option key={d.code} value={d.code}>
                  {d.code} {d.team ? `(${d.team})` : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Stint selection pills */}
          <div className="tyre-control-group">
            <span className="tyre-control-label">Stint</span>
            <div className="tyre-stint-pills" role="radiogroup" aria-label="Stint selection">
              {(data?.available_stints || []).map((s) => {
                const sCompColor = COMPOUND_COLORS[s.compound?.toUpperCase()] || '#8C96A8';
                const isSelected = s.stint === selectedStint;
                return (
                  <button
                    key={s.stint}
                    type="button"
                    id={`btn-stint-${s.stint}`}
                    className={`stint-pill-btn ${isSelected ? 'active' : ''} ${!s.is_started ? 'unstarted' : ''}`}
                    onClick={() => onSelectStint(s.stint)}
                    title={`Stint ${s.stint}: ${s.compound} (Laps ${s.start_lap}–${s.end_lap})`}
                  >
                    <span
                      className="stint-compound-dot"
                      style={{ backgroundColor: sCompColor }}
                    />
                    Stint {s.stint}
                    <span className="stint-compound-tag">{s.compound}</span>
                  </button>
                );
              })}
              {(!data?.available_stints || data.available_stints.length === 0) && (
                <span className="text-muted" style={{ fontSize: '12px' }}>
                  Stint 1 (default)
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="tyre-controls-right">
          {/* Replay Synchronisation Toggle */}
          <div className="tyre-sync-toggle-group">
            <button
              type="button"
              id="btn-toggle-tyre-sync"
              className={`tyre-sync-btn ${syncWithReplay ? 'active' : ''}`}
              onClick={() => onToggleSync(!syncWithReplay)}
              title={
                syncWithReplay
                  ? `Synced with replay clock at ${currentReplayTimeS ? `${currentReplayTimeS.toFixed(1)}s` : ''} (Lap ${currentReplayLap || 1})`
                  : 'Currently showing entire stint'
              }
            >
              <Clock size={14} />
              {syncWithReplay ? `Synced (L${currentReplayLap || 1})` : 'Full Stint'}
            </button>
          </div>

          <button
            type="button"
            id="btn-refresh-tyres"
            className="tyre-refresh-btn"
            onClick={onRefresh}
            disabled={isLoading}
            title="Refresh tyre degradation data"
          >
            <RotateCw size={14} className={isLoading ? 'spin-icon' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="tyre-alert-banner error" id="tyre-error-banner">
          <AlertCircle size={18} />
          <div className="tyre-alert-content">
            <strong>Error loading tyre data:</strong> {error}
          </div>
          <button type="button" className="retry-btn" onClick={onRefresh}>
            Retry
          </button>
        </div>
      )}

      {/* Pre-Race / Unstarted Notice */}
      {data?.status === 'PRE_RACE' && (
        <div className="tyre-alert-banner info" id="tyre-pre-race-banner">
          <Info size={18} />
          <div className="tyre-alert-content">
            <strong>Pre-Race Session:</strong> No race laps have been completed at this timestamp.
            Advance the replay clock to view stint tyre degradation.
          </div>
        </div>
      )}

      {/* Future Stint Notice */}
      {data?.status === 'FUTURE_STINT' && (
        <div className="tyre-alert-banner warning" id="tyre-future-stint-banner">
          <AlertTriangle size={18} />
          <div className="tyre-alert-content">
            <strong>Stint Not Started:</strong> {data.message}
          </div>
        </div>
      )}

      {/* Cliff Detection Alert Banner */}
      {data?.cliff_lap && (
        <div className="tyre-alert-banner cliff-alert" id="tyre-cliff-banner">
          <AlertTriangle size={20} className="cliff-icon" />
          <div className="tyre-alert-content">
            <strong>⚠️ Performance Cliff Warning:</strong> A sharp drop-off was detected at{' '}
            <span className="cliff-highlight">Lap {data.cliff_lap}</span>. Lap times exceeded the
            expected degradation trend line by more than 2.5 standard deviations for consecutive laps.
          </div>
        </div>
      )}

      {/* KPI Cards Row */}
      <div className="tyre-kpi-grid" id="tyre-kpi-cards">
        {/* Estimated Degradation Rate */}
        <div className="tyre-kpi-card" id="card-deg-rate">
          <div className="kpi-header">
            <span className="kpi-label">Estimated Deg Rate</span>
            <TrendingUp size={16} className="kpi-icon" />
          </div>
          <div className="kpi-value-row">
            {data?.estimated_deg_rate_s_per_lap !== null &&
            data?.estimated_deg_rate_s_per_lap !== undefined ? (
              <span
                className={`kpi-value ${
                  data.estimated_deg_rate_s_per_lap > 0 ? 'deg-positive' : 'deg-negative'
                }`}
              >
                {data.estimated_deg_rate_s_per_lap > 0 ? '+' : ''}
                {data.estimated_deg_rate_s_per_lap.toFixed(3)}
                <span className="kpi-unit">s/lap</span>
              </span>
            ) : (
              <span className="kpi-value muted">--</span>
            )}
          </div>
          <div className="kpi-subtitle">Fuel-corrected regression slope</div>
        </div>

        {/* Raw Lap Trend */}
        <div className="tyre-kpi-card" id="card-raw-slope">
          <div className="kpi-header">
            <span className="kpi-label">Observed Pace Slope</span>
            <Activity size={16} className="kpi-icon" />
          </div>
          <div className="kpi-value-row">
            {data?.raw_slope_s_per_lap !== null && data?.raw_slope_s_per_lap !== undefined ? (
              <span className="kpi-value">
                {data.raw_slope_s_per_lap > 0 ? '+' : ''}
                {data.raw_slope_s_per_lap.toFixed(3)}
                <span className="kpi-unit">s/lap</span>
              </span>
            ) : (
              <span className="kpi-value muted">--</span>
            )}
          </div>
          <div className="kpi-subtitle">Uncorrected raw lap progression</div>
        </div>

        {/* Compound & Stint Age */}
        <div className="tyre-kpi-card" id="card-compound-age">
          <div className="kpi-header">
            <span className="kpi-label">Compound & Age</span>
            <Layers size={16} className="kpi-icon" />
          </div>
          <div className="kpi-value-row">
            <div className="compound-badge" style={{ borderColor: compoundColor }}>
              <span className="compound-dot" style={{ backgroundColor: compoundColor }} />
              <span className="compound-name" style={{ color: compoundColor }}>
                {data?.compound || 'UNKNOWN'}
              </span>
            </div>
            <span className="stint-laps-count">
              {data?.total_laps_count || 0} <span className="kpi-unit">laps</span>
            </span>
          </div>
          <div className="kpi-subtitle">
            Stint {selectedStint} | {selectedDriver} ({driverTeam || 'F1'})
          </div>
        </div>

        {/* Clean Laps & Data Confidence */}
        <div className="tyre-kpi-card" id="card-confidence">
          <div className="kpi-header">
            <span className="kpi-label">Data Confidence</span>
            {data?.confidence === 'HIGH' ? (
              <CheckCircle2 size={16} className="text-green" />
            ) : (
              <Info size={16} className="text-amber" />
            )}
          </div>
          <div className="kpi-value-row">
            <span
              className={`confidence-badge confidence-${(
                data?.confidence || 'INSUFFICIENT_DATA'
              ).toLowerCase()}`}
            >
              {data?.confidence || 'INSUFFICIENT'}
            </span>
            <span className="clean-laps-stat">
              {data?.clean_laps_count || 0} / {data?.total_laps_count || 0} clean
            </span>
          </div>
          <div className="kpi-subtitle">
            {data?.r_squared !== null && data?.r_squared !== undefined
              ? `R² = ${data.r_squared.toFixed(2)} | SE = ±${(data.std_error_s_per_lap || 0).toFixed(3)}s`
              : 'Minimum 4 clean laps required'}
          </div>
        </div>
      </div>

      {/* SVG Interactive Chart Section */}
      <div className="tyre-chart-container" id="tyre-chart-wrapper">
        <div className="tyre-chart-header">
          <div className="chart-title-group">
            <h3 className="tyre-chart-title">
              Fuel-Corrected Lap-Time Degradation Curve
            </h3>
            <span className="chart-subtitle">
              Measured clean laps vs. Ordinary Least Squares (OLS) fuel-corrected trend line
            </span>
          </div>

          <div className="chart-legend" id="tyre-chart-legend">
            <div className="legend-item">
              <span className="legend-dot raw-dot" />
              <span>Observed Lap Time</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot corrected-dot" />
              <span>Fuel-Corrected Time</span>
            </div>
            <div className="legend-item">
              <span className="legend-line trend-line" />
              <span>Fitted Degradation Slope</span>
            </div>
            {data?.cliff_lap && (
              <div className="legend-item">
                <span className="legend-dot cliff-marker" />
                <span>Performance Cliff</span>
              </div>
            )}
          </div>
        </div>

        {isLoading ? (
          <div className="tyre-chart-loading" id="tyre-chart-loading">
            <RotateCw size={24} className="spin-icon" />
            <span>Computing stint degradation regression...</span>
          </div>
        ) : laps.length === 0 ? (
          <div className="tyre-chart-empty" id="tyre-chart-empty">
            <Info size={28} />
            <p>No lap data available for this stint selection.</p>
          </div>
        ) : (
          <div className="svg-responsive-wrapper">
            <svg
              className="tyre-svg-chart"
              viewBox={`0 0 ${chartWidth} ${chartHeight}`}
              preserveAspectRatio="xMidYMid meet"
              role="img"
              aria-label="Tyre degradation trend chart"
            >
              {/* Horizontal Gridlines & Y-axis labels */}
              {yTicks.map((val, idx) => {
                const yPos = getY(val);
                return (
                  <g key={`ytick-${idx}`} className="chart-grid-line">
                    <line
                      x1={padding.left}
                      y1={yPos}
                      x2={chartWidth - padding.right}
                      y2={yPos}
                      stroke="rgba(255, 255, 255, 0.08)"
                      strokeDasharray="4 4"
                    />
                    <text
                      x={padding.left - 10}
                      y={yPos + 4}
                      textAnchor="end"
                      fill="var(--text-muted, #8c96a8)"
                      fontSize="11px"
                      fontFamily="var(--font-mono, monospace)"
                    >
                      {formatSecondsToLapTime(val)}
                    </text>
                  </g>
                );
              })}

              {/* X-axis ticks & labels */}
              {laps.map((l) => {
                const xPos = getX(l.lap_number);
                return (
                  <g key={`xtick-${l.lap_number}`} className="chart-x-tick">
                    <line
                      x1={xPos}
                      y1={padding.top}
                      x2={xPos}
                      y2={padding.top + innerHeight}
                      stroke="rgba(255, 255, 255, 0.04)"
                    />
                    <text
                      x={xPos}
                      y={padding.top + innerHeight + 20}
                      textAnchor="middle"
                      fill="var(--text-muted, #8c96a8)"
                      fontSize="11px"
                      fontFamily="var(--font-mono, monospace)"
                    >
                      L{l.lap_number}
                    </text>
                  </g>
                );
              })}

              {/* Axis Titles */}
              <text
                x={padding.left + innerWidth / 2}
                y={chartHeight - 10}
                textAnchor="middle"
                fill="var(--text-muted, #8c96a8)"
                fontSize="12px"
                fontWeight="500"
              >
                Lap Number (Stint Progression)
              </text>

              <text
                x={-(padding.top + innerHeight / 2)}
                y={20}
                transform="rotate(-90)"
                textAnchor="middle"
                fill="var(--text-muted, #8c96a8)"
                fontSize="12px"
                fontWeight="500"
              >
                Lap Time
              </text>

              {/* Connecting line for Raw Observed Times */}
              {laps.length > 1 && (
                <path
                  d={laps
                    .filter((l) => l.raw_lap_time_s !== null && l.raw_lap_time_s !== undefined)
                    .map((l, i) => `${i === 0 ? 'M' : 'L'} ${getX(l.lap_number)} ${getY(l.raw_lap_time_s!)}`)
                    .join(' ')}
                  fill="none"
                  stroke="rgba(255, 255, 255, 0.25)"
                  strokeWidth="1.5"
                  strokeDasharray="3 3"
                />
              )}

              {/* Fitted Regression Line (Fuel-Corrected Degradation Slope) */}
              {data?.clean_laps_count && data.clean_laps_count >= 4 && (
                <path
                  d={laps
                    .filter((l) => l.fitted_lap_time_s !== null && l.fitted_lap_time_s !== undefined)
                    .map((l, i) => `${i === 0 ? 'M' : 'L'} ${getX(l.lap_number)} ${getY(l.fitted_lap_time_s!)}`)
                    .join(' ')}
                  fill="none"
                  stroke={teamColor || '#1E41FF'}
                  strokeWidth="2.5"
                />
              )}

              {/* Cliff Lap Vertical Beacon */}
              {data?.cliff_lap && (
                <g className="cliff-marker-group">
                  <line
                    x1={getX(data.cliff_lap)}
                    y1={padding.top}
                    x2={getX(data.cliff_lap)}
                    y2={padding.top + innerHeight}
                    stroke="#FF3B30"
                    strokeWidth="2"
                    strokeDasharray="4 2"
                  />
                  <rect
                    x={getX(data.cliff_lap) - 36}
                    y={padding.top - 20}
                    width="72"
                    height="18"
                    rx="4"
                    fill="#FF3B30"
                  />
                  <text
                    x={getX(data.cliff_lap)}
                    y={padding.top - 7}
                    textAnchor="middle"
                    fill="#FFFFFF"
                    fontSize="10px"
                    fontWeight="700"
                  >
                    CLIFF LAP
                  </text>
                </g>
              )}

              {/* Data Points */}
              {laps.map((l) => {
                const xPos = getX(l.lap_number);
                const isHovered = hoveredLap?.lap_number === l.lap_number;

                return (
                  <g key={`points-${l.lap_number}`} className="lap-points-group">
                    {/* Raw Observed Time Circle */}
                    {l.raw_lap_time_s !== null && l.raw_lap_time_s !== undefined && (
                      <circle
                        cx={xPos}
                        cy={getY(l.raw_lap_time_s)}
                        r={isHovered ? 6 : l.is_clean ? 4 : 3}
                        fill={l.is_clean ? '#FFFFFF' : 'transparent'}
                        stroke={l.is_clean ? '#A0AEC0' : '#718096'}
                        strokeWidth={l.is_clean ? 1.5 : 1}
                        className="raw-point"
                        onMouseEnter={() => setHoveredLap(l)}
                        onMouseLeave={() => setHoveredLap(null)}
                      />
                    )}

                    {/* Fuel-Corrected Time Point */}
                    {l.fuel_corrected_lap_time_s !== null &&
                      l.fuel_corrected_lap_time_s !== undefined && (
                        <rect
                          x={xPos - (isHovered ? 5 : 3.5)}
                          y={getY(l.fuel_corrected_lap_time_s) - (isHovered ? 5 : 3.5)}
                          width={isHovered ? 10 : 7}
                          height={isHovered ? 10 : 7}
                          fill={l.is_cliff ? '#FF3B30' : compoundColor}
                          stroke="#1A202C"
                          strokeWidth="1"
                          className="corrected-point"
                          onMouseEnter={() => setHoveredLap(l)}
                          onMouseLeave={() => setHoveredLap(null)}
                        />
                      )}
                  </g>
                );
              })}
            </svg>

            {/* Hover Tooltip Box */}
            {hoveredLap && (
              <div
                className="tyre-tooltip-card"
                style={{
                  left: `${Math.min(
                    Math.max(getX(hoveredLap.lap_number), 120),
                    chartWidth - 140
                  )}px`,
                  top: `${Math.min(
                    getY(hoveredLap.raw_lap_time_s || yMin) - 30,
                    chartHeight - 140
                  )}px`,
                }}
              >
                <div className="tooltip-title">
                  Lap {hoveredLap.lap_number}
                  {hoveredLap.tyre_age_laps !== null && (
                    <span className="tooltip-age">
                      (Tyre Age: {hoveredLap.tyre_age_laps} laps)
                    </span>
                  )}
                </div>
                <div className="tooltip-row">
                  <span className="t-label">Raw Observed:</span>
                  <span className="t-val">{formatSecondsToLapTime(hoveredLap.raw_lap_time_s)}</span>
                </div>
                {hoveredLap.fuel_corrected_lap_time_s !== null && (
                  <div className="tooltip-row">
                    <span className="t-label">Fuel-Corrected:</span>
                    <span className="t-val highlight">
                      {formatSecondsToLapTime(hoveredLap.fuel_corrected_lap_time_s)}
                    </span>
                  </div>
                )}
                {hoveredLap.fitted_lap_time_s !== null && (
                  <div className="tooltip-row">
                    <span className="t-label">Model Trend:</span>
                    <span className="t-val">
                      {formatSecondsToLapTime(hoveredLap.fitted_lap_time_s)}
                    </span>
                  </div>
                )}
                <div className="tooltip-status">
                  {hoveredLap.is_clean ? (
                    <span className="status-clean">✓ Verified Clean Lap</span>
                  ) : (
                    <span className="status-dirty">⚠️ Non-clean / Flagged Lap</span>
                  )}
                  {hoveredLap.is_cliff && (
                    <span className="status-cliff"> • CLIFF DETECTED</span>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Assumptions & Methodology Explanation Note */}
      <div className="tyre-methodology-card" id="tyre-assumptions-card">
        <div className="methodology-header">
          <Info size={16} />
          <h4>Model Assumptions & Analysis Context</h4>
        </div>
        <div className="methodology-grid">
          <div className="methodology-item">
            <strong>Fuel Correction Baseline</strong>
            <p>
              {data?.fuel_correction_assumption ||
                'Linear fuel burn compensation applied (nominal 0.033 s/lap baseline).'}
            </p>
          </div>
          <div className="methodology-item">
            <strong>Statistical Nature of Degradation</strong>
            <p>
              {data?.methodology_note ||
                'Estimated degradation rate is derived via Ordinary Least Squares (OLS) regression over strictly verified green-flag clean laps. It represents pace decay, not a direct physical tyre compound sensor measurement.'}
            </p>
          </div>
          <div className="methodology-item">
            <strong>Engineering Interpretation</strong>
            <p>
              {data?.degradation_interpretation ||
                'Observing pace degradation helps identify compound degradation and potential tyre cliff windows.'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
