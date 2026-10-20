import React, { useState, useMemo } from 'react';
import {
  Swords,
  Clock,
  RotateCcw,
  ArrowRightLeft,
  Shield,
  Zap,
  Info,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import { useGapBattleAnalysis } from '../../hooks/useGapBattleAnalysis';
import { useBattleOvertakePrediction } from '../../hooks/useBattleOvertakePrediction';
import { BattleOvertakePredictionCard } from './BattleOvertakePredictionCard';
import type { GapLapItem } from '../../types/battles';
import { getTeamColor } from '../../constants/teams';

interface GapBattlePanelProps {
  raceId: string;
  availableDrivers: string[];
  driverTeamMap?: Record<string, string>;
  currentSessionTimeS?: number;
  totalRaceLaps?: number;
  initialDriver1?: string;
  initialDriver2?: string;
}

export const GapBattlePanel: React.FC<GapBattlePanelProps> = ({
  raceId,
  availableDrivers,
  driverTeamMap = {},
  currentSessionTimeS,
  totalRaceLaps = 53,
  initialDriver1,
  initialDriver2,
}) => {
  // Driver selection defaults: VER and LEC or first two available drivers
  const [driver1, setDriver1] = useState<string>(
    initialDriver1 || (availableDrivers.includes('VER') ? 'VER' : availableDrivers[0] || 'VER')
  );
  const [driver2, setDriver2] = useState<string>(
    initialDriver2 || (availableDrivers.includes('LEC')
      ? 'LEC'
      : availableDrivers.find((d) => d !== driver1) || 'LEC')
  );

  const prevInitD1 = React.useRef(initialDriver1);
  const prevInitD2 = React.useRef(initialDriver2);

  // Update drivers only if evidence navigation supplies a new target
  React.useEffect(() => {
    if (initialDriver1 && initialDriver1 !== prevInitD1.current && availableDrivers.includes(initialDriver1)) {
      prevInitD1.current = initialDriver1;
      setDriver1(initialDriver1);
    }
  }, [initialDriver1, availableDrivers]);

  React.useEffect(() => {
    if (initialDriver2 && initialDriver2 !== prevInitD2.current && availableDrivers.includes(initialDriver2)) {
      prevInitD2.current = initialDriver2;
      setDriver2(initialDriver2);
    }
  }, [initialDriver2, availableDrivers]);

  // Sync mode: sync with live replay clock vs full race analysis
  const [syncWithReplay, setSyncWithReplay] = useState<boolean>(false);

  // Filter for Overtake / Battle feed: 'all' | 'on_track' | 'pit_swaps' | 'caution'
  const [eventFilter, setEventFilter] = useState<'all' | 'on_track' | 'pit_swaps' | 'caution'>('all');

  // Tooltip hover state on chart
  const [hoveredLap, setHoveredLap] = useState<GapLapItem | null>(null);

  const { data, isLoading, error, refetch } = useGapBattleAnalysis({
    raceId,
    driver1,
    driver2,
    asOfTimeS: currentSessionTimeS,
    syncWithReplay,
  });

  const {
    data: predictionData,
    isLoading: predictionLoading,
    error: predictionError,
    refetch: refetchPrediction,
  } = useBattleOvertakePrediction({
    raceId,
    driver1,
    driver2,
    asOfTimeS: currentSessionTimeS,
    asOfLap: data?.as_of_lap,
    syncWithReplay,
    horizonLaps: 5,
  });

  const handleRefreshAll = () => {
    refetch();
    refetchPrediction();
  };

  // Swap drivers
  const handleSwapDrivers = () => {
    setDriver1(driver2);
    setDriver2(driver1);
  };

  const d1Color = getTeamColor(driverTeamMap[driver1]);
  const d2Color = getTeamColor(driverTeamMap[driver2]);

  // Filtered overtakes
  const filteredOvertakes = useMemo(() => {
    if (!data || !data.overtakes_up_to_as_of) return [];
    if (eventFilter === 'all') return data.overtakes_up_to_as_of;
    if (eventFilter === 'on_track') {
      return data.overtakes_up_to_as_of.filter(
        (o) =>
          o.classification === 'CONFIRMED_ON_TRACK_OVERTAKE' ||
          o.classification === 'INFERRED_ON_TRACK_OVERTAKE'
      );
    }
    if (eventFilter === 'pit_swaps') {
      return data.overtakes_up_to_as_of.filter((o) => o.classification === 'PIT_STRATEGY_SWAP');
    }
    if (eventFilter === 'caution') {
      return data.overtakes_up_to_as_of.filter(
        (o) =>
          o.classification === 'CAUTION_PERIOD_CHANGE' ||
          o.classification === 'RETIREMENT_PASS' ||
          o.classification === 'UNKNOWN_DATA_GAP'
      );
    }
    return data.overtakes_up_to_as_of;
  }, [data, eventFilter]);

  // Pair specific overtakes
  const pairOvertakes = useMemo(() => {
    if (!data || !data.overtakes_up_to_as_of) return [];
    const pair = new Set([driver1, driver2]);
    return data.overtakes_up_to_as_of.filter(
      (o) => pair.has(o.overtaking_driver) && pair.has(o.overtaken_driver)
    );
  }, [data, driver1, driver2]);

  // Build SVG Chart Geometry
  const chartConfig = useMemo(() => {
    if (!data || !data.gap_history || data.gap_history.length === 0) {
      return null;
    }
    const history = data.gap_history;
    const maxLaps = Math.max(totalRaceLaps, ...history.map((h) => h.lap_number));

    // Calculate gap bounds
    const gaps = history.map((h) => h.gap_s);
    let maxGap = Math.max(2.0, ...gaps);
    let minGap = Math.min(-2.0, ...gaps);
    // Add 10% padding
    const absBound = Math.max(Math.abs(maxGap), Math.abs(minGap), 3.0) * 1.15;
    const yMin = -absBound;
    const yMax = absBound;

    const width = 800;
    const height = 320;
    const padding = { top: 30, right: 35, bottom: 40, left: 60 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const scaleX = (lap: number) => padding.left + ((lap - 1) / Math.max(maxLaps - 1, 1)) * chartW;
    const scaleY = (gap: number) => padding.top + chartH / 2 - (gap / (yMax - yMin)) * chartH;

    const zeroY = scaleY(0);
    const pos1sY = scaleY(1.0);
    const neg1sY = scaleY(-1.0);

    const points = history.map((h) => ({
      ...h,
      x: scaleX(h.lap_number),
      y: scaleY(h.gap_s),
    }));

    // SVG path string
    const pathD = points.reduce((acc, pt, idx) => {
      return idx === 0 ? `M ${pt.x.toFixed(1)},${pt.y.toFixed(1)}` : `${acc} L ${pt.x.toFixed(1)},${pt.y.toFixed(1)}`;
    }, '');

    // Map overtakes to points
    const overtakeGlyphs = pairOvertakes.map((ot) => {
      const match = points.find((p) => p.lap_number === ot.lap_number);
      return {
        ...ot,
        x: match ? match.x : scaleX(ot.lap_number),
        y: match ? match.y : zeroY,
      };
    });

    return {
      width,
      height,
      padding,
      zeroY,
      pos1sY,
      neg1sY,
      points,
      pathD,
      overtakeGlyphs,
      yMin,
      yMax,
      maxLaps,
    };
  }, [data, totalRaceLaps, pairOvertakes]);

  // Helper for classification badge
  const renderBadge = (classification: string, _confidence?: string) => {
    switch (classification) {
      case 'CONFIRMED_ON_TRACK_OVERTAKE':
        return (
          <span className="badge badge-confirmed" title="Validated via distance-aligned telemetry crossover">
            <CheckCircle2 size={12} className="mr-1" /> Confirmed Pass
          </span>
        );
      case 'INFERRED_ON_TRACK_OVERTAKE':
        return (
          <span className="badge badge-inferred" title="Position swap on green flag with no pit activity">
            <Zap size={12} className="mr-1" /> Inferred Pass
          </span>
        );
      case 'PIT_STRATEGY_SWAP':
        return (
          <span className="badge badge-pit" title="One or both cars in pit lane during passing window">
            <Clock size={12} className="mr-1" /> Pit Strategy
          </span>
        );
      case 'RETIREMENT_PASS':
        return (
          <span className="badge badge-retirement" title="Car retired or stopped running">
            <AlertCircle size={12} className="mr-1" /> Retirement
          </span>
        );
      case 'CAUTION_PERIOD_CHANGE':
        return (
          <span className="badge badge-caution" title="Order change occurred during safety car or VSC">
            <Shield size={12} className="mr-1" /> Caution Period
          </span>
        );
      default:
        return (
          <span className="badge badge-unknown" title="Telemetry data missing or telemetry gap > 2.0s">
            <HelpCircle size={12} className="mr-1" /> Unknown Gap
          </span>
        );
    }
  };

  return (
    <div className="gap-battle-container">
      {/* 1. Header & Configuration Bar */}
      <div className="battle-header-card">
        <div className="battle-header-title">
          <Swords className="battle-title-icon" size={22} />
          <div>
            <h2>Gap & Battle Analysis</h2>
            <p className="battle-subtitle">
              Head-to-head interval tracking, closing dynamics, and telemetry-validated overtakes
            </p>
          </div>
        </div>

        <div className="battle-controls-row">
          {/* Driver 1 Selector */}
          <div className="driver-picker-group">
            <label htmlFor="select-driver1">Driver 1 (Lead Baseline):</label>
            <div className="driver-select-wrapper">
              <span className="team-indicator" style={{ backgroundColor: d1Color }} />
              <select
                id="select-driver1"
                value={driver1}
                onChange={(e) => setDriver1(e.target.value)}
                className="f1-select"
              >
                {availableDrivers.map((d) => (
                  <option key={d} value={d} disabled={d === driver2}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Swap Button */}
          <button
            id="btn-swap-drivers"
            onClick={handleSwapDrivers}
            className="f1-btn btn-swap"
            title="Swap Driver 1 and Driver 2 (inverts signed gap)"
          >
            <ArrowRightLeft size={16} />
          </button>

          {/* Driver 2 Selector */}
          <div className="driver-picker-group">
            <label htmlFor="select-driver2">Driver 2 (Chaser):</label>
            <div className="driver-select-wrapper">
              <span className="team-indicator" style={{ backgroundColor: d2Color }} />
              <select
                id="select-driver2"
                value={driver2}
                onChange={(e) => setDriver2(e.target.value)}
                className="f1-select"
              >
                {availableDrivers.map((d) => (
                  <option key={d} value={d} disabled={d === driver1}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Replay Sync Toggle */}
          <div className="sync-toggle-group">
            <label className="checkbox-label" htmlFor="toggle-replay-sync">
              <input
                id="toggle-replay-sync"
                type="checkbox"
                checked={syncWithReplay}
                onChange={(e) => setSyncWithReplay(e.target.checked)}
              />
              <span>Sync with Replay Clock</span>
            </label>
            {syncWithReplay && currentSessionTimeS !== undefined && (
              <span className="clock-badge">
                <Clock size={12} className="mr-1" />
                {currentSessionTimeS.toFixed(1)}s
              </span>
            )}
          </div>

          {/* Refetch Button */}
          <button id="btn-refresh-battles" onClick={handleRefreshAll} className="f1-btn btn-refresh" title="Refresh analysis">
            <RotateCcw size={16} />
          </button>
        </div>
      </div>

      {/* Error / Loading / Pre-Race States */}
      {isLoading && (
        <div className="battle-state-banner loading">
          <Clock className="animate-spin mr-2" size={20} /> Loading gap and battle intelligence...
        </div>
      )}
      {error && (
        <div className="battle-state-banner error">
          <AlertCircle className="mr-2" size={20} /> {error}
        </div>
      )}
      {data && data.status === 'PRE_RACE' && (
        <>
          <div className="battle-state-banner info">
            <Info className="mr-2" size={20} /> Pre-Race: Simulation timestamp is before official race start. Advance the replay to observe lap gaps.
          </div>
          <BattleOvertakePredictionCard
            prediction={predictionData}
            isLoading={predictionLoading}
            error={predictionError}
            driver1={driver1}
            driver2={driver2}
            driverTeamMap={driverTeamMap}
            onRetry={refetchPrediction}
          />
        </>
      )}

      {data && data.status === 'SUCCESS' && (
        <>
          {/* 2. Key Performance Indicators (KPIs) */}
          <div className="battle-kpi-grid">
            {/* Current Gap */}
            <div className="kpi-card" id="kpi-current-gap">
              <div className="kpi-label">Current Gap ({driver1} vs {driver2})</div>
              <div className="kpi-value-row">
                <span
                  className="kpi-value"
                  style={{
                    color:
                      (data.summary.current_gap_s || 0) >= 0 ? d1Color : d2Color,
                  }}
                >
                  {data.summary.current_gap_s !== null
                    ? `${data.summary.current_gap_s > 0 ? '+' : ''}${data.summary.current_gap_s.toFixed(3)}s`
                    : 'N/A'}
                </span>
                {data.summary.current_leader && (
                  <span className="kpi-sub-badge">
                    Leader: <strong>{data.summary.current_leader}</strong>
                  </span>
                )}
              </div>
              <div className="kpi-caption">
                {data.summary.current_gap_s !== null && Math.abs(data.summary.current_gap_s) <= 1.0 ? (
                  <span className="text-warning">⚡ Within 1.0s Battle Window</span>
                ) : (
                  <span>Positive = {driver1} ahead; Negative = {driver2} ahead</span>
                )}
              </div>
            </div>

            {/* Closing Rate */}
            <div className="kpi-card" id="kpi-closing-rate">
              <div className="kpi-label">Latest Closing Dynamics</div>
              <div className="kpi-value-row">
                {data.gap_history.length > 0 ? (
                  (() => {
                    const latest = data.gap_history[data.gap_history.length - 1];
                    const rate = latest.closing_rate_s_per_lap;
                    const isClosing = rate > 0.05;
                    const isFalling = rate < -0.05;
                    return (
                      <>
                        <span className="kpi-value">
                          {isClosing && <TrendingUp size={22} className="text-success mr-1" />}
                          {isFalling && <TrendingDown size={22} className="text-danger mr-1" />}
                          {Math.abs(rate).toFixed(3)} s/lap
                        </span>
                        <span className="kpi-sub-badge">
                          {isClosing
                            ? `${latest.chaser} gaining`
                            : isFalling
                            ? `${latest.leader} pulling away`
                            : 'Stabilized'}
                        </span>
                      </>
                    );
                  })()
                ) : (
                  <span className="kpi-value">N/A</span>
                )}
              </div>
              <div className="kpi-caption">Rate of delta convergence between cars</div>
            </div>

            {/* Battle Window Laps */}
            <div className="kpi-card" id="kpi-battle-laps">
              <div className="kpi-label">Laps In Battle Window (≤ 1.0s)</div>
              <div className="kpi-value-row">
                <span className="kpi-value text-accent">{data.summary.laps_in_battle_window}</span>
                <span className="kpi-sub-badge">of {data.summary.total_laps_analyzed} laps</span>
              </div>
              <div className="kpi-caption">
                {(
                  (data.summary.laps_in_battle_window /
                    Math.max(data.summary.total_laps_analyzed, 1)) *
                  100
                ).toFixed(1)}
                % of completed race distance
              </div>
            </div>

            {/* Completed Overtakes */}
            <div className="kpi-card" id="kpi-overtakes">
              <div className="kpi-label">Total Field Passes (to as_of)</div>
              <div className="kpi-value-row">
                <span className="kpi-value text-gold">{data.summary.total_overtakes_completed}</span>
                <span className="kpi-sub-badge">
                  {data.summary.confirmed_on_track_passes} Confirmed On-Track
                </span>
              </div>
              <div className="kpi-caption">
                Pair passes: <strong>{pairOvertakes.length}</strong> ({driver1} ↔ {driver2})
              </div>
            </div>
          </div>

          {/* 2.5 Machine Learning Battle & Overtake Prediction */}
          <BattleOvertakePredictionCard
            prediction={predictionData}
            isLoading={predictionLoading}
            error={predictionError}
            driver1={driver1}
            driver2={driver2}
            driverTeamMap={driverTeamMap}
            onRetry={refetchPrediction}
          />

          {/* 3. Interactive SVG Gap Chart */}
          <div className="battle-chart-card">
            <div className="chart-header">
              <div className="chart-title">
                <h3>Signed Interval Gap Progression (Laps 1 – {data.as_of_lap || totalRaceLaps})</h3>
                <span className="chart-hint">Hover data points for lap telemetry insights</span>
              </div>

              {/* Chart Legend */}
              <div className="chart-legend">
                <div className="legend-item">
                  <span className="legend-line" style={{ backgroundColor: 'var(--brand-primary, #e10600)' }} />
                  <span>Signed Gap</span>
                </div>
                <div className="legend-item">
                  <span className="legend-corridor" />
                  <span>±1.0s Battle Zone</span>
                </div>
                <div className="legend-item">
                  <span className="legend-point marker-pass" />
                  <span>Confirmed Overtake</span>
                </div>
                <div className="legend-item">
                  <span className="legend-point marker-pit" />
                  <span>Pit Swap</span>
                </div>
              </div>
            </div>

            {chartConfig ? (
              <div className="svg-wrapper" id="gap-chart-svg-container">
                <svg
                  viewBox={`0 0 ${chartConfig.width} ${chartConfig.height}`}
                  className="gap-battle-svg"
                  preserveAspectRatio="xMidYMid meet"
                >
                  <defs>
                    <linearGradient id="battleBandGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#ffb800" stopOpacity="0.12" />
                      <stop offset="50%" stopColor="#ffb800" stopOpacity="0.04" />
                      <stop offset="100%" stopColor="#ffb800" stopOpacity="0.12" />
                    </linearGradient>
                  </defs>

                  {/* Battle Corridor Band (±1.0s) */}
                  <rect
                    x={chartConfig.padding.left}
                    y={chartConfig.pos1sY}
                    width={chartConfig.width - chartConfig.padding.left - chartConfig.padding.right}
                    height={Math.max(chartConfig.neg1sY - chartConfig.pos1sY, 0)}
                    fill="url(#battleBandGrad)"
                  />
                  {/* +1.0s Boundary Line */}
                  <line
                    x1={chartConfig.padding.left}
                    y1={chartConfig.pos1sY}
                    x2={chartConfig.width - chartConfig.padding.right}
                    y2={chartConfig.pos1sY}
                    stroke="#ffb800"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                    strokeOpacity="0.5"
                  />
                  <text
                    x={chartConfig.width - chartConfig.padding.right + 4}
                    y={chartConfig.pos1sY + 3}
                    fill="#ffb800"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    +1.0s
                  </text>

                  {/* -1.0s Boundary Line */}
                  <line
                    x1={chartConfig.padding.left}
                    y1={chartConfig.neg1sY}
                    x2={chartConfig.width - chartConfig.padding.right}
                    y2={chartConfig.neg1sY}
                    stroke="#ffb800"
                    strokeDasharray="4 4"
                    strokeWidth="1"
                    strokeOpacity="0.5"
                  />
                  <text
                    x={chartConfig.width - chartConfig.padding.right + 4}
                    y={chartConfig.neg1sY + 3}
                    fill="#ffb800"
                    fontSize="10"
                    fontFamily="monospace"
                  >
                    -1.0s
                  </text>

                  {/* Zero Line (Lead Swap Baseline) */}
                  <line
                    x1={chartConfig.padding.left}
                    y1={chartConfig.zeroY}
                    x2={chartConfig.width - chartConfig.padding.right}
                    y2={chartConfig.zeroY}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                    strokeDasharray="2 2"
                    strokeOpacity="0.7"
                  />
                  <text
                    x={chartConfig.padding.left - 30}
                    y={chartConfig.zeroY + 4}
                    fill="#ffffff"
                    fontSize="11"
                    fontFamily="monospace"
                  >
                    0.0s
                  </text>

                  {/* Driver 1 Territory Label */}
                  <text
                    x={chartConfig.padding.left + 10}
                    y={chartConfig.padding.top + 15}
                    fill={d1Color}
                    fontSize="11"
                    fontWeight="600"
                    fontFamily="sans-serif"
                  >
                    ▲ {driver1} Leading
                  </text>

                  {/* Driver 2 Territory Label */}
                  <text
                    x={chartConfig.padding.left + 10}
                    y={chartConfig.height - chartConfig.padding.bottom - 8}
                    fill={d2Color}
                    fontSize="11"
                    fontWeight="600"
                    fontFamily="sans-serif"
                  >
                    ▼ {driver2} Leading
                  </text>

                  {/* Signed Gap Path */}
                  <path
                    d={chartConfig.pathD}
                    fill="none"
                    stroke="var(--brand-primary, #e10600)"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />

                  {/* Data Point Nodes */}
                  {chartConfig.points.map((pt) => {
                    const isHovered = hoveredLap?.lap_number === pt.lap_number;
                    const inWindow = pt.in_battle_window;
                    return (
                      <g key={pt.lap_number}>
                        <circle
                          cx={pt.x}
                          cy={pt.y}
                          r={isHovered ? 6 : inWindow ? 4 : 2.5}
                          fill={isHovered ? '#ffffff' : inWindow ? '#ffb800' : 'var(--brand-primary, #e10600)'}
                          stroke="#11151c"
                          strokeWidth="1.5"
                          onMouseEnter={() => setHoveredLap(pt)}
                          onMouseLeave={() => setHoveredLap(null)}
                          style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                        />
                      </g>
                    );
                  })}

                  {/* Overtake Glyphs */}
                  {chartConfig.overtakeGlyphs.map((glyph, idx) => {
                    const isConfirmed = glyph.classification === 'CONFIRMED_ON_TRACK_OVERTAKE';
                    const isPit = glyph.classification === 'PIT_STRATEGY_SWAP';
                    return (
                      <g key={idx} transform={`translate(${glyph.x}, ${glyph.y})`}>
                        {isConfirmed ? (
                          <polygon
                            points="0,-8 7,0 0,8 -7,0"
                            fill="#00d26a"
                            stroke="#ffffff"
                            strokeWidth="1.5"
                            className="overtake-glyph-confirmed"
                          >
                            <title>
                              Lap {glyph.lap_number}: {glyph.overtaking_driver} passed {glyph.overtaken_driver} (Confirmed)
                            </title>
                          </polygon>
                        ) : isPit ? (
                          <rect
                            x="-5"
                            y="-5"
                            width="10"
                            height="10"
                            fill="#ff8700"
                            stroke="#ffffff"
                            strokeWidth="1"
                            className="overtake-glyph-pit"
                          >
                            <title>
                              Lap {glyph.lap_number}: Pit Strategy Swap ({glyph.overtaking_driver} passed {glyph.overtaken_driver})
                            </title>
                          </rect>
                        ) : (
                          <circle
                            cx="0"
                            cy="0"
                            r="5"
                            fill="#00b4d8"
                            stroke="#ffffff"
                            strokeWidth="1"
                          >
                            <title>
                              Lap {glyph.lap_number}: {glyph.overtaking_driver} passed {glyph.overtaken_driver} ({glyph.classification})
                            </title>
                          </circle>
                        )}
                      </g>
                    );
                  })}

                  {/* X-Axis Laps */}
                  {Array.from(
                    { length: Math.ceil(chartConfig.maxLaps / 5) + 1 },
                    (_, i) => i * 5
                  ).map((lapNum) => {
                    if (lapNum === 0 || lapNum > chartConfig.maxLaps) return null;
                    const xPos =
                      chartConfig.padding.left +
                      ((lapNum - 1) / Math.max(chartConfig.maxLaps - 1, 1)) *
                        (chartConfig.width - chartConfig.padding.left - chartConfig.padding.right);
                    return (
                      <g key={lapNum}>
                        <line
                          x1={xPos}
                          y1={chartConfig.height - chartConfig.padding.bottom}
                          x2={xPos}
                          y2={chartConfig.height - chartConfig.padding.bottom + 5}
                          stroke="#718096"
                          strokeWidth="1"
                        />
                        <text
                          x={xPos}
                          y={chartConfig.height - chartConfig.padding.bottom + 18}
                          fill="#a0aec0"
                          fontSize="10"
                          textAnchor="middle"
                          fontFamily="monospace"
                        >
                          L{lapNum}
                        </text>
                      </g>
                    );
                  })}
                </svg>

                {/* Floating Tooltip */}
                {hoveredLap && (
                  <div
                    className="gap-chart-tooltip"
                    style={{
                      left: `${chartConfig.points.find((p) => p.lap_number === hoveredLap.lap_number)?.x || 0}px`,
                      top: `${Math.max(
                        20,
                        (chartConfig.points.find((p) => p.lap_number === hoveredLap.lap_number)?.y || 0) - 75
                      )}px`,
                    }}
                  >
                    <div className="tooltip-title">Lap {hoveredLap.lap_number}</div>
                    <div className="tooltip-row">
                      <span>Gap:</span>
                      <strong className={hoveredLap.gap_s >= 0 ? 'text-primary' : 'text-accent'}>
                        {hoveredLap.gap_s > 0 ? '+' : ''}
                        {hoveredLap.gap_s.toFixed(3)}s
                      </strong>
                    </div>
                    <div className="tooltip-row">
                      <span>Leader:</span>
                      <strong>{hoveredLap.leader}</strong>
                    </div>
                    <div className="tooltip-row">
                      <span>Closing Rate:</span>
                      <span>{hoveredLap.closing_rate_s_per_lap.toFixed(3)} s/lap</span>
                    </div>
                    {hoveredLap.in_battle_window && (
                      <div className="tooltip-badge">⚡ Inside Battle Window (≤ 1.0s)</div>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="chart-empty-state">No gap data points available to render.</div>
            )}
          </div>

          {/* 4. Active Battles at as_of Section */}
          <div className="active-battles-card">
            <div className="section-header">
              <div className="section-title">
                <Zap size={18} className="text-warning mr-2" />
                <h3>Active Battles Across Field at as_of (Lap {data.as_of_lap || 'N/A'})</h3>
              </div>
              <span className="section-tag">
                {data.active_battles_at_as_of.length} Concurrent Battle(s)
              </span>
            </div>

            {data.active_battles_at_as_of.length > 0 ? (
              <div className="active-battles-grid">
                {data.active_battles_at_as_of.map((b, idx) => {
                  const isPair =
                    (b.leader === driver1 && b.chaser === driver2) ||
                    (b.leader === driver2 && b.chaser === driver1);
                  return (
                    <div
                      key={idx}
                      className={`active-battle-item ${isPair ? 'active-battle-highlight' : ''}`}
                    >
                      <div className="battle-drivers">
                        <span className="leader-tag">{b.leader}</span>
                        <span className="vs-tag">vs</span>
                        <span className="chaser-tag">{b.chaser}</span>
                        {isPair && <span className="selected-pair-badge">Selected Pair</span>}
                      </div>

                      <div className="battle-metrics">
                        <div className="metric">
                          <span className="m-label">Gap:</span>
                          <span className="m-val mono">{b.gap_s.toFixed(3)}s</span>
                        </div>
                        <div className="metric">
                          <span className="m-label">Closing:</span>
                          <span className="m-val mono">
                            {b.closing_rate_s_per_lap > 0 ? '+' : ''}
                            {b.closing_rate_s_per_lap.toFixed(3)} s/lap
                          </span>
                        </div>
                      </div>

                      <div className="battle-verdict-row">
                        <span className={`verdict-pill ${b.battle_verdict.toLowerCase()}`}>
                          {b.battle_verdict.replace('_', ' ')}
                        </span>
                        {b.sustained_pressure && (
                          <span className="pressure-pill" title="Under 0.7s pressure maintained">
                            Sustained Pressure
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="no-active-battles">
                No cars within the 1.0s green-flag battle threshold at this simulation moment.
              </div>
            )}
          </div>

          {/* 5. Historical Position Changes & Overtakes Timeline */}
          <div className="overtakes-timeline-card">
            <div className="section-header">
              <div className="section-title">
                <Shield size={18} className="text-accent mr-2" />
                <h3>Historical Position Changes & Overtakes Log</h3>
              </div>

              {/* Filter Tabs */}
              <div className="filter-pill-group">
                <button
                  className={`filter-pill ${eventFilter === 'all' ? 'active' : ''}`}
                  onClick={() => setEventFilter('all')}
                >
                  All Events ({data.overtakes_up_to_as_of.length})
                </button>
                <button
                  className={`filter-pill ${eventFilter === 'on_track' ? 'active' : ''}`}
                  onClick={() => setEventFilter('on_track')}
                >
                  On-Track Passes ({data.summary.confirmed_on_track_passes})
                </button>
                <button
                  className={`filter-pill ${eventFilter === 'pit_swaps' ? 'active' : ''}`}
                  onClick={() => setEventFilter('pit_swaps')}
                >
                  Pit Swaps
                </button>
                <button
                  className={`filter-pill ${eventFilter === 'caution' ? 'active' : ''}`}
                  onClick={() => setEventFilter('caution')}
                >
                  Caution / Other
                </button>
              </div>
            </div>

            {filteredOvertakes.length > 0 ? (
              <div className="overtakes-table-wrapper">
                <table className="f1-table overtakes-table">
                  <thead>
                    <tr>
                      <th>Lap</th>
                      <th>Overtaking Driver</th>
                      <th>Overtaken Driver</th>
                      <th>Classification</th>
                      <th>Telemetry Distance</th>
                      <th>Evidence / Explanation</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredOvertakes.map((o, idx) => {
                      const isSelectedPair =
                        (o.overtaking_driver === driver1 && o.overtaken_driver === driver2) ||
                        (o.overtaking_driver === driver2 && o.overtaken_driver === driver1);
                      return (
                        <tr key={idx} className={isSelectedPair ? 'selected-pair-row' : ''}>
                          <td className="mono bold">L{o.lap_number}</td>
                          <td>
                            <span className="driver-pill">
                              <span
                                className="team-indicator"
                                style={{
                                  backgroundColor: getTeamColor(driverTeamMap[o.overtaking_driver]),
                                }}
                              />
                              {o.overtaking_driver}
                            </span>
                          </td>
                          <td>
                            <span className="driver-pill">
                              <span
                                className="team-indicator"
                                style={{
                                  backgroundColor: getTeamColor(driverTeamMap[o.overtaken_driver]),
                                }}
                              />
                              {o.overtaken_driver}
                            </span>
                          </td>
                          <td>{renderBadge(o.classification, o.confidence)}</td>
                          <td className="mono text-muted">
                            {o.crossover_distance_m !== null
                              ? `${o.crossover_distance_m.toFixed(0)}m`
                              : '—'}
                          </td>
                          <td className="evidence-text">{o.evidence}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="no-events-state">
                No events matching filter "{eventFilter}" recorded up to the current simulation time.
              </div>
            )}
          </div>

          {/* 6. Scientific Methodology Note */}
          <div className="methodology-note-card">
            <Info size={16} className="text-muted mr-2 flex-shrink-0" />
            <div>
              <strong>RACE Gap & Battle Analytics Rules:</strong> Battles are defined as proximity ≤ 1.0s under green flag conditions (track status 1) outside pit cycles. Proximity alone is classified as an <em>ACTIVE_CHASE</em> or <em>SUSTAINED_DEFENSE</em> and is strictly <strong>never</strong> counted as an overtake. Position changes are categorized as <em>CONFIRMED_ON_TRACK_OVERTAKE</em> only when physical distance-aligned telemetry verifies an on-track crossover without exceeding the 2.0s telemetry gap limit.
            </div>
          </div>
        </>
      )}
    </div>
  );
};
