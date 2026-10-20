import React, { useState, useMemo, useCallback } from 'react';
import { ComparisonResponse, ComparisonLapItem } from '../../types/comparison';
import { DriverOption } from './ComparisonConfigBar';
import { getTeamColor } from '../../constants/teams';
import { formatLapTime, formatDelta, getTyreClass } from './ComparisonTable';
import {
  TrendingUp,
  AlertCircle,
  AlertTriangle,
  Clock,
  Info,
  Loader2,
  ShieldCheck,
  Fuel,
  Minus,
} from 'lucide-react';

export interface ComparisonChartProps {
  data: ComparisonResponse | null;
  isLoading: boolean;
  error: string | null;
  isStale: boolean;
  availableDrivers: DriverOption[];
}

export type ChartMode = 'lap_time' | 'delta';

export interface PlottedPoint {
  driver: string;
  lapNumber: number;
  val: number;
  x: number;
  y: number;
  item: ComparisonLapItem;
  key: string;
}

export interface DriverSeries {
  driver: string;
  team: string;
  color: string;
  isRef: boolean;
  pathSegments: string[];
  points: PlottedPoint[];
}

export const ComparisonChart: React.FC<ComparisonChartProps> = ({
  data,
  isLoading,
  error,
  isStale,
  availableDrivers,
}) => {
  const [chartMode, setChartMode] = useState<ChartMode>('lap_time');
  const [selectedPoint, setSelectedPoint] = useState<PlottedPoint | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<PlottedPoint | null>(null);
  const [focusedDriver, setFocusedDriver] = useState<string | null>(null);

  // Map driver codes to team names (Hook 1)
  const driverTeamMap = useMemo(() => {
    const map = new Map<string, string>();
    for (const d of availableDrivers) {
      map.set(d.code, d.team);
    }
    return map;
  }, [availableDrivers]);

  // Safe reference to laps array (Hook 2)
  const laps = useMemo(() => data?.laps || [], [data?.laps]);

  // Dimensions & coordinate system constants
  const svgWidth = 840;
  const svgHeight = 360;
  const marginLeft = 72;
  const marginRight = 36;
  const marginTop = 30;
  const marginBottom = 45;
  const plotWidth = svgWidth - marginLeft - marginRight;
  const plotHeight = svgHeight - marginTop - marginBottom;

  // Extract all valid lap numbers for X-axis domain (Hook 3)
  const allLapNumbers = useMemo(() => {
    const lapsSet = new Set<number>();
    for (const item of laps) {
      if (typeof item.lap_number === 'number' && !isNaN(item.lap_number) && item.lap_number > 0) {
        lapsSet.add(item.lap_number);
      }
    }
    return Array.from(lapsSet).sort((a, b) => a - b);
  }, [laps]);

  const minLap = allLapNumbers.length > 0 ? allLapNumbers[0] : 1;
  const maxLap = allLapNumbers.length > 0 ? allLapNumbers[allLapNumbers.length - 1] : 1;
  const isSingleLap = minLap === maxLap;

  // X Coordinate scaling function (strictly using actual item.lap_number) (Hook 4)
  const getX = useCallback(
    (lap: number): number => {
      if (isSingleLap) {
        return marginLeft + plotWidth / 2;
      }
      const ratio = (lap - minLap) / (maxLap - minLap || 1);
      return marginLeft + ratio * plotWidth;
    },
    [minLap, maxLap, isSingleLap, marginLeft, plotWidth]
  );

  // Group laps by driver while preserving original driver ordering (Hook 5)
  const driversOrder = useMemo(() => {
    const ordered: string[] = [];
    for (const item of laps) {
      if (!ordered.includes(item.driver)) {
        ordered.push(item.driver);
      }
    }
    return ordered;
  }, [laps]);

  // Group and sort laps by driver (Hook 6)
  const lapsByDriver = useMemo(() => {
    const map = new Map<string, ComparisonLapItem[]>();
    for (const d of driversOrder) {
      map.set(d, []);
    }
    for (const item of laps) {
      const list = map.get(item.driver);
      if (list) {
        list.push(item);
      }
    }
    for (const [, list] of map.entries()) {
      list.sort((a, b) => {
        if (a.lap_number !== b.lap_number) return a.lap_number - b.lap_number;
        if ((a.stint ?? 0) !== (b.stint ?? 0)) return (a.stint ?? 0) - (b.stint ?? 0);
        return (a.tyre_age_laps ?? 0) - (b.tyre_age_laps ?? 0);
      });
    }
    return map;
  }, [laps, driversOrder]);

  // Compute Y domain for Mode A: Lap Time (Hook 7)
  const { minTime, maxTime, hasValidTimes } = useMemo(() => {
    const validTimes: number[] = [];
    for (const item of laps) {
      if (typeof item.lap_time_s === 'number' && !isNaN(item.lap_time_s) && item.lap_time_s > 0) {
        validTimes.push(item.lap_time_s);
      }
    }
    if (validTimes.length === 0) {
      return { minTime: 80, maxTime: 90, hasValidTimes: false };
    }
    let min = Math.min(...validTimes);
    let max = Math.max(...validTimes);
    if (min === max) {
      min -= 1.0;
      max += 1.0;
    } else {
      const pad = (max - min) * 0.08;
      min = Math.max(0.1, min - pad);
      max = max + pad;
    }
    return { minTime: min, maxTime: max, hasValidTimes: true };
  }, [laps]);

  // Compute Y domain for Mode B: Delta to Reference (Hook 8)
  const { minDelta, maxDelta, hasValidDeltas } = useMemo(() => {
    const validDeltas: number[] = [];
    for (const item of laps) {
      if (typeof item.delta_to_reference_s === 'number' && !isNaN(item.delta_to_reference_s)) {
        validDeltas.push(item.delta_to_reference_s);
      }
    }
    if (validDeltas.length === 0) {
      return { minDelta: -1.0, maxDelta: 1.0, hasValidDeltas: false };
    }
    let min = Math.min(0, ...validDeltas);
    let max = Math.max(0, ...validDeltas);
    if (min === max) {
      min -= 0.5;
      max += 0.5;
    } else {
      const pad = Math.max(0.1, (max - min) * 0.1);
      min = min - pad;
      max = max + pad;
    }
    return { minDelta: min, maxDelta: max, hasValidDeltas: true };
  }, [laps]);

  // Y Coordinate scaling function (Hook 9)
  // Mode A: Lower time (faster) appears HIGHER on chart -> y = marginTop at minTime
  // Mode B: Standard Cartesian with zero baseline
  const getY = useCallback(
    (val: number): number => {
      if (chartMode === 'lap_time') {
        const ratio = (val - minTime) / (maxTime - minTime || 1);
        return marginTop + ratio * plotHeight;
      } else {
        const ratio = (val - minDelta) / (maxDelta - minDelta || 1);
        return marginTop + plotHeight - ratio * plotHeight;
      }
    },
    [chartMode, minTime, maxTime, minDelta, maxDelta, marginTop, plotHeight]
  );

  // Zero-reference line Y coordinate in delta mode (Hook 10)
  const yZero = useMemo(() => {
    if (chartMode !== 'delta') return null;
    return getY(0);
  }, [chartMode, getY]);

  // Generate Plotted Series & Discontinuous Path Segments (Hook 11)
  const seriesData: DriverSeries[] = useMemo(() => {
    return driversOrder.map((drv) => {
      const items = lapsByDriver.get(drv) || [];
      const team = driverTeamMap.get(drv) || '';
      const color = getTeamColor(team);
      const isRef = drv === data?.reference_driver;

      const points: PlottedPoint[] = [];
      const pathSegments: string[] = [];

      let currentSegment: { x: number; y: number }[] = [];
      let lastLap: number | null = null;

      items.forEach((item, idx) => {
        const val = chartMode === 'lap_time' ? item.lap_time_s : item.delta_to_reference_s;

        // Skip missing or null values
        if (typeof val !== 'number' || isNaN(val) || (chartMode === 'lap_time' && val <= 0)) {
          if (currentSegment.length > 1) {
            pathSegments.push(
              currentSegment
                .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
                .join(' ')
            );
          }
          currentSegment = [];
          lastLap = null;
          return;
        }

        const x = getX(item.lap_number);
        const y = getY(val);

        const pointKey = `${drv}-${item.lap_number}-${item.stint ?? 's'}-${item.tyre_age_laps ?? 'a'}-${idx}`;
        points.push({
          driver: drv,
          lapNumber: item.lap_number,
          val,
          x,
          y,
          item,
          key: pointKey,
        });

        // Path continuity check
        if (lastLap !== null && item.lap_number === lastLap + 1) {
          currentSegment.push({ x, y });
        } else {
          if (currentSegment.length > 1) {
            pathSegments.push(
              currentSegment
                .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
                .join(' ')
            );
          }
          currentSegment = [{ x, y }];
        }
        lastLap = item.lap_number;
      });

      if (currentSegment.length > 1) {
        pathSegments.push(
          currentSegment
            .map((pt, i) => `${i === 0 ? 'M' : 'L'} ${pt.x.toFixed(2)} ${pt.y.toFixed(2)}`)
            .join(' ')
        );
      }

      return {
        driver: drv,
        team,
        color,
        isRef,
        pathSegments,
        points,
      };
    });
  }, [driversOrder, lapsByDriver, driverTeamMap, data?.reference_driver, chartMode, getX, getY]);

  // Generate X-Axis Ticks (Hook 12)
  const xTicks = useMemo(() => {
    if (allLapNumbers.length === 0) return [];
    if (isSingleLap) return [minLap];

    const span = maxLap - minLap;
    if (span <= 12) {
      return allLapNumbers;
    }

    let step = 1;
    if (span > 50) step = 10;
    else if (span > 25) step = 5;
    else if (span > 12) step = 2;

    const ticks: number[] = [];
    for (let l = minLap; l <= maxLap; l += step) {
      ticks.push(l);
    }
    if (ticks[ticks.length - 1] !== maxLap) {
      ticks.push(maxLap);
    }
    return ticks;
  }, [allLapNumbers, minLap, maxLap, isSingleLap]);

  // Generate Y-Axis Ticks (Hook 13)
  const yTicks = useMemo(() => {
    const ticks: { val: number; y: number; label: string }[] = [];
    const count = 5;

    if (chartMode === 'lap_time') {
      if (!hasValidTimes) return [];
      const step = (maxTime - minTime) / (count - 1);
      for (let i = 0; i < count; i++) {
        const val = minTime + i * step;
        ticks.push({
          val,
          y: getY(val),
          label: formatLapTime(val),
        });
      }
    } else {
      if (!hasValidDeltas) return [];
      const step = (maxDelta - minDelta) / (count - 1);
      for (let i = 0; i < count; i++) {
        const val = minDelta + i * step;
        ticks.push({
          val,
          y: getY(val),
          label: formatDelta(val),
        });
      }
      if (!ticks.some((t) => Math.abs(t.val) < 0.05)) {
        ticks.push({
          val: 0,
          y: getY(0),
          label: '0.000s',
        });
      }
      ticks.sort((a, b) => a.val - b.val);
    }
    return ticks;
  }, [chartMode, hasValidTimes, hasValidDeltas, minTime, maxTime, minDelta, maxDelta, getY]);

  // =========================================================================
  // CONDITIONAL STATE RENDERING (All hooks called unconditionally above)
  // =========================================================================

  // Handle State: Error
  if (error) {
    return (
      <div className="comparison-chart-card state-card state-error" id="comparison-chart-error">
        <AlertCircle size={24} className="state-icon icon-error" />
        <div className="state-text">
          <h3>Pace Chart Unavailable</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  // Handle State: Loading
  if (isLoading) {
    return (
      <div className="comparison-chart-card state-card state-loading" id="comparison-chart-loading">
        <Loader2 size={26} className="spin state-icon icon-loading" />
        <div className="state-text">
          <h3>Rendering Pace Comparison...</h3>
          <p>Mapping lap telemetry across drivers and scaling axes.</p>
        </div>
      </div>
    );
  }

  // Handle State: Initial / Idle
  if (!data) {
    return (
      <div className="comparison-chart-card state-card state-idle" id="comparison-chart-idle">
        <Info size={24} className="state-icon icon-idle" />
        <div className="state-text">
          <h3>Pace Chart Ready</h3>
          <p>
            Configure driver comparison parameters above and click <strong>Run Comparison</strong> to generate the pace chart and results table.
          </p>
        </div>
      </div>
    );
  }

  // Handle State: PRE_RACE
  if (data.status === 'PRE_RACE') {
    return (
      <div className="comparison-chart-card state-card state-prerace" id="comparison-chart-prerace">
        <Clock size={24} className="state-icon icon-warning" />
        <div className="state-text">
          <h3>Pre-Race Simulation Cutoff</h3>
          <p>
            The selected simulation cutoff precedes the green flag. No completed race laps exist for charting prior to race start.
          </p>
        </div>
      </div>
    );
  }

  // Handle State: NO_DATA or empty laps array
  if (data.status === 'NO_DATA' || data.laps.length === 0) {
    return (
      <div className="comparison-chart-card state-card state-nodata" id="comparison-chart-nodata">
        <AlertTriangle size={24} className="state-icon icon-warning" />
        <div className="state-text">
          <h3>No Chartable Observations</h3>
          <p>
            No completed laps matched the selected criteria prior to the cutoff.
            {data.clean_only && ' Clean Laps Only may have filtered out available laps.'}
          </p>
        </div>
      </div>
    );
  }

  const activePoint = hoveredPoint || selectedPoint;

  return (
    <div className="comparison-chart-card" id="comparison-pace-chart">
      {/* Chart Header Bar */}
      <div className="chart-header-bar">
        <div className="chart-header-left">
          <div className="chart-title-group">
            <TrendingUp size={18} className="chart-title-icon" />
            <h3 className="chart-title">
              {chartMode === 'lap_time' ? 'Lap-Time Comparison' : 'Delta-to-Reference Pace'}
            </h3>
          </div>

          <div className="chart-meta-tags">
            <span className="chart-badge badge-race mono">{data.race_id}</span>
            <span className="chart-badge badge-sync">
              Sync: <strong>{data.sync_mode === 'tyre_age' ? 'Tyre Age' : 'Race Lap'}</strong>
            </span>
            <span className="chart-badge badge-ref">
              Ref: <strong className="mono">{data.reference_driver || 'NONE'}</strong>
            </span>
            {data.fuel_corrected && (
              <span className="chart-badge badge-fuel" title="Fuel correction applied by backend (+0.033s/lap)">
                <Fuel size={12} /> Fuel Corrected
              </span>
            )}
            {data.clean_only && (
              <span className="chart-badge badge-clean">
                <ShieldCheck size={12} /> Clean Only
              </span>
            )}
          </div>
        </div>

        {/* Mode Selector */}
        <div className="chart-mode-selector" role="group" aria-label="Pace Chart Display Mode">
          <button
            type="button"
            id="chart-mode-lap-time"
            className={`chart-mode-btn${chartMode === 'lap_time' ? ' active' : ''}`}
            onClick={() => setChartMode('lap_time')}
            aria-pressed={chartMode === 'lap_time'}
          >
            Lap Time
          </button>
          <button
            type="button"
            id="chart-mode-delta"
            className={`chart-mode-btn${chartMode === 'delta' ? ' active' : ''}`}
            onClick={() => setChartMode('delta')}
            aria-pressed={chartMode === 'delta'}
          >
            Delta to Ref
          </button>
        </div>
      </div>

      {/* Stale Configuration Warning Banner */}
      {isStale && (
        <div className="table-stale-banner" id="chart-stale-banner">
          <AlertTriangle size={15} />
          <span>
            Chart shows previous query results. Configuration has changed — click <strong>Run Comparison</strong> above to update.
          </span>
        </div>
      )}

      {/* Legend */}
      <div className="chart-legend" id="chart-legend">
        <span className="legend-label">Drivers:</span>
        {seriesData.map((s) => {
          const isDimmed = focusedDriver !== null && focusedDriver !== s.driver;
          return (
            <button
              key={s.driver}
              type="button"
              className={`legend-item ${isDimmed ? 'dimmed' : ''} ${s.isRef ? 'is-ref' : ''}`}
              onClick={() => setFocusedDriver(focusedDriver === s.driver ? null : s.driver)}
              title={`Click to isolate ${s.driver}`}
            >
              <span className="legend-swatch" style={{ backgroundColor: s.color }} />
              <span className="legend-driver-code mono">{s.driver}</span>
              {s.isRef && <span className="legend-ref-tag">REF</span>}
            </button>
          );
        })}
        {focusedDriver && (
          <button
            type="button"
            className="legend-reset-btn"
            onClick={() => setFocusedDriver(null)}
          >
            Reset Filter
          </button>
        )}
      </div>

      {/* SVG Pace Chart Container */}
      <div className="chart-svg-container">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="comparison-chart-svg"
          role="img"
          aria-label={
            chartMode === 'lap_time'
              ? 'Lap-Time Comparison pace chart across race laps'
              : 'Delta-to-Reference pace chart across race laps'
          }
        >
          <defs>
            <clipPath id="chart-plot-clip">
              <rect x={marginLeft} y={marginTop} width={plotWidth} height={plotHeight} />
            </clipPath>
          </defs>

          {/* Background Plot Area */}
          <rect
            x={marginLeft}
            y={marginTop}
            width={plotWidth}
            height={plotHeight}
            className="chart-plot-bg"
          />

          {/* Y Grid Lines and Labels */}
          {yTicks.map((tick, i) => (
            <g key={`y-tick-${i}`} className="chart-grid-group">
              <line
                x1={marginLeft}
                x2={marginLeft + plotWidth}
                y1={tick.y}
                y2={tick.y}
                className="chart-grid-line"
              />
              <text
                x={marginLeft - 10}
                y={tick.y + 4}
                className="chart-tick-label y-tick mono"
                textAnchor="end"
              >
                {tick.label}
              </text>
            </g>
          ))}

          {/* Zero-Reference Line for Delta Mode */}
          {chartMode === 'delta' && yZero !== null && (
            <g className="chart-zero-line-group">
              <line
                x1={marginLeft}
                x2={marginLeft + plotWidth}
                y1={yZero}
                y2={yZero}
                className="chart-zero-line"
                id="chart-zero-reference-line"
              />
              <text
                x={marginLeft + plotWidth - 6}
                y={yZero - 6}
                className="chart-zero-line-label mono"
                textAnchor="end"
              >
                0.000s (BASELINE)
              </text>
            </g>
          )}

          {/* X Grid Lines and Labels (Actual Race Laps) */}
          {xTicks.map((lap) => {
            const x = getX(lap);
            return (
              <g key={`x-tick-${lap}`} className="chart-grid-group">
                <line
                  x1={x}
                  x2={x}
                  y1={marginTop}
                  y2={marginTop + plotHeight}
                  className="chart-grid-line"
                />
                <text
                  x={x}
                  y={marginTop + plotHeight + 18}
                  className="chart-tick-label x-tick mono"
                  textAnchor="middle"
                >
                  {lap}
                </text>
              </g>
            );
          })}

          {/* Axis Titles */}
          <text
            x={marginLeft + plotWidth / 2}
            y={marginTop + plotHeight + 38}
            className="chart-axis-title x-axis-title"
            textAnchor="middle"
          >
            Actual Race Lap
          </text>

          <text
            x={-(marginTop + plotHeight / 2)}
            y={18}
            transform="rotate(-90)"
            className="chart-axis-title y-axis-title"
            textAnchor="middle"
          >
            {chartMode === 'lap_time' ? 'Lap Time (Faster ▲)' : 'Delta to Reference (s)'}
          </text>

          {/* Series Paths clipped to plot area */}
          <g clipPath="url(#chart-plot-clip)">
            {seriesData.map((s) => {
              const isDimmed = focusedDriver !== null && focusedDriver !== s.driver;
              return (
                <g
                  key={`series-${s.driver}`}
                  className={`driver-series ${isDimmed ? 'series-dimmed' : ''}`}
                >
                  {s.pathSegments.map((d, sIdx) => (
                    <path
                      key={`seg-${sIdx}`}
                      d={d}
                      fill="none"
                      stroke={s.color}
                      strokeWidth={s.isRef ? 3 : 2}
                      strokeDasharray={s.isRef && chartMode === 'delta' ? '5 3' : undefined}
                      className="chart-line"
                    />
                  ))}
                </g>
              );
            })}
          </g>

          {/* Data Points (Rendered at exact actual race lap coordinates) */}
          <g className="chart-points-layer">
            {seriesData.map((s) => {
              const isDimmed = focusedDriver !== null && focusedDriver !== s.driver;
              return s.points.map((pt) => {
                const isSelected = selectedPoint?.key === pt.key;
                const isHovered = hoveredPoint?.key === pt.key;
                const isActive = isSelected || isHovered;

                return (
                  <circle
                    key={pt.key}
                    cx={pt.x}
                    cy={pt.y}
                    r={isActive ? 6 : s.isRef ? 4.5 : 3.5}
                    fill={isActive ? 'var(--text-primary)' : s.color}
                    stroke={isActive ? s.color : 'var(--bg-base)'}
                    strokeWidth={isActive ? 2.5 : 1.5}
                    className={`chart-point ${isActive ? 'is-active' : ''} ${isDimmed ? 'point-dimmed' : ''}`}
                    tabIndex={0}
                    role="button"
                    aria-label={`Driver ${pt.driver}, Lap ${pt.lapNumber}: ${
                      chartMode === 'lap_time'
                        ? formatLapTime(pt.val)
                        : formatDelta(pt.val)
                    }`}
                    onMouseEnter={() => setHoveredPoint(pt)}
                    onMouseLeave={() => setHoveredPoint(null)}
                    onClick={() => setSelectedPoint(pt)}
                    onFocus={() => setSelectedPoint(pt)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault();
                        setSelectedPoint(pt);
                      }
                    }}
                  />
                );
              });
            })}
          </g>
        </svg>
      </div>

      {/* Selected Observation Inspection Details Bar */}
      <div className="chart-inspection-panel" id="chart-point-details">
        {activePoint ? (
          <div className="point-details-row">
            <div className="details-driver">
              <span
                className="team-indicator"
                style={{
                  backgroundColor: getTeamColor(driverTeamMap.get(activePoint.driver)),
                }}
              />
              <strong className="mono">{activePoint.driver}</strong>
              {activePoint.item.driver === data.reference_driver && (
                <span className="ref-badge">REF</span>
              )}
            </div>

            <div className="details-stat">
              <span className="stat-label">Race Lap:</span>
              <span className="stat-value mono">{activePoint.lapNumber}</span>
            </div>

            <div className="details-stat">
              <span className="stat-label">Lap Time:</span>
              <span className="stat-value mono">{formatLapTime(activePoint.item.lap_time_s)}</span>
            </div>

            <div className="details-stat">
              <span className="stat-label">Delta to Ref:</span>
              <span
                className={`stat-value mono ${
                  activePoint.item.delta_to_reference_s !== null &&
                  activePoint.item.delta_to_reference_s !== undefined
                    ? activePoint.item.delta_to_reference_s < 0
                      ? 'delta-faster'
                      : activePoint.item.delta_to_reference_s > 0
                      ? 'delta-slower'
                      : 'delta-ref'
                    : 'delta-null'
                }`}
              >
                {formatDelta(activePoint.item.delta_to_reference_s)}
              </span>
            </div>

            <div className="details-stat">
              <span className="stat-label">Compound:</span>
              <span className={`tyre-badge ${getTyreClass(activePoint.item.compound)}`}>
                {activePoint.item.compound ? activePoint.item.compound.charAt(0).toUpperCase() : '?'}
              </span>
              <span className="stat-value compound-text">{activePoint.item.compound || 'UNKNOWN'}</span>
            </div>

            <div className="details-stat">
              <span className="stat-label">Tyre Age:</span>
              <span className="stat-value mono">
                {activePoint.item.tyre_age_laps !== null && activePoint.item.tyre_age_laps !== undefined
                  ? `${activePoint.item.tyre_age_laps}L`
                  : '—'}
              </span>
            </div>

            <div className="details-stat">
              <span className="stat-label">Stint:</span>
              <span className="stat-value mono">
                {activePoint.item.stint !== null && activePoint.item.stint !== undefined
                  ? `S${activePoint.item.stint}`
                  : '—'}
              </span>
            </div>

            <div className="details-stat">
              <span className="stat-label">Status:</span>
              <span
                className={`clean-pill ${
                  activePoint.item.is_clean ? 'clean-yes' : 'clean-no'
                }`}
              >
                {activePoint.item.is_clean ? 'CLEAN' : 'NOT CLEAN'}
              </span>
            </div>

            {selectedPoint && (
              <button
                type="button"
                className="details-close-btn"
                onClick={() => setSelectedPoint(null)}
                aria-label="Clear selected point"
              >
                <Minus size={14} />
              </button>
            )}
          </div>
        ) : (
          <div className="point-details-placeholder">
            <span>Hover or focus on any plotted point to inspect exact lap time, deltas, compound, and stint telemetry.</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ComparisonChart;
