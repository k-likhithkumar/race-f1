import React, { useMemo } from 'react';
import { TrackOutline, DriverState } from '../types/replay';
import { getTeamColor } from '../constants/teams';
import { MapPin, Compass, AlertCircle, Loader2 } from 'lucide-react';

export interface TrackMiniSectorOverlaySegment {
  mini_sector_id: number;
  color: string;
  label?: string;
  tooltip?: string;
}

export interface CircuitMapProps {
  trackOutline: TrackOutline | null;
  drivers?: DriverState[];
  loading?: boolean;
  error?: string | null;
  id?: string;
  className?: string;
  miniSectorSegments?: TrackMiniSectorOverlaySegment[];
  selectedMiniSectorId?: number | null;
  onSelectMiniSector?: (msId: number) => void;
  hoveredMiniSectorId?: number | null;
  onHoverMiniSector?: (msId: number | null) => void;
  hideDriverMarkers?: boolean;
  hideDefaultLegend?: boolean;
  hideHeader?: boolean;
}

export const CircuitMap: React.FC<CircuitMapProps> = ({
  trackOutline,
  drivers = [],
  loading = false,
  error = null,
  id,
  className,
  miniSectorSegments,
  selectedMiniSectorId,
  onSelectMiniSector,
  hoveredMiniSectorId,
  onHoverMiniSector,
  hideDriverMarkers = false,
  hideDefaultLegend = false,
  hideHeader = false,
}) => {
  const mapData = useMemo(() => {
    if (!trackOutline || !trackOutline.points || trackOutline.points.length === 0) {
      return null;
    }

    const pts = trackOutline.points;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;

    // SVG Y-inversion: in telemetry, +Y is North. In SVG, +Y is Down.
    // Setting svgY = -p.y ensures North is Up and clockwise racing is preserved.
    const pathCommands: string[] = [];

    for (let i = 0; i < pts.length; i++) {
      const p = pts[i];
      const sx = p.x;
      const sy = -p.y;

      if (sx < minX) minX = sx;
      if (sx > maxX) maxX = sx;
      if (sy < minY) minY = sy;
      if (sy > maxY) maxY = sy;

      pathCommands.push(`${i === 0 ? 'M' : 'L'} ${sx.toFixed(2)} ${sy.toFixed(2)}`);
    }

    const width = maxX - minX;
    const height = maxY - minY;
    // Add 10% padding around bounding box for clean breathing room
    const pad = Math.max(width, height) * 0.1;

    const viewBox = `${(minX - pad).toFixed(2)} ${(minY - pad).toFixed(2)} ${(width + 2 * pad).toFixed(2)} ${(height + 2 * pad).toFixed(2)}`;
    const pathString = `${pathCommands.join(' ')} Z`;

    // Normalizing scale factor for driver markers so they render with identical
    // visual pixel size across 2022 (decimeters) and 2023 (meters) units
    const maxDim = Math.max(width, height) + 2 * pad;
    const markerScale = maxDim / 650.0;

    // Start / Finish Line perpendicular calculation at point 0
    const p0 = pts[0];
    const p1 = pts[1] || pts[0];
    const sx0 = p0.x;
    const sy0 = -p0.y;
    const sx1 = p1.x;
    const sy1 = -p1.y;

    const dx = sx1 - sx0;
    const dy = sy1 - sy0;
    const dist = Math.hypot(dx, dy) || 1.0;
    const tx = dx / dist;
    const ty = dy / dist;
    // Normal vector perpendicular to track direction
    const nx = -ty;
    const ny = tx;

    // Scale perpendicular line length proportionally with circuit units
    const lineLen = pad * 0.05;
    const sfLine = {
      x1: sx0 + nx * lineLen,
      y1: sy0 + ny * lineLen,
      x2: sx0 - nx * lineLen,
      y2: sy0 - ny * lineLen,
    };

    // Locate sector boundary points along the track outline
    const sectorMarkers: { sector: number; x: number; y: number }[] = [];
    if (trackOutline.sectors && trackOutline.total_distance_m > 0) {
      for (const sec of trackOutline.sectors) {
        if (sec.start_m > 10.0 && sec.start_m < trackOutline.total_distance_m - 10.0) {
          // Find point closest to sector start_m
          let bestPt = pts[0];
          let minDiff = Infinity;
          for (const pt of pts) {
            const diff = Math.abs(pt.dist_m - sec.start_m);
            if (diff < minDiff) {
              minDiff = diff;
              bestPt = pt;
            }
          }
          sectorMarkers.push({
            sector: sec.sector,
            x: bestPt.x,
            y: -bestPt.y,
          });
        }
      }
    }

    // Generate individual SVG path segments for each mini-sector (1..20)
    const miniSectorPaths: Record<number, string> = {};
    if (trackOutline.mini_sectors && trackOutline.mini_sectors.length > 0) {
      for (const ms of trackOutline.mini_sectors) {
        const startDist = ms.start_m;
        const endDist = ms.end_m;

        let startIdx = 0;
        while (startIdx < pts.length - 1 && pts[startIdx + 1].dist_m <= startDist) {
          startIdx++;
        }

        let endIdx = startIdx;
        while (endIdx < pts.length - 1 && pts[endIdx].dist_m < endDist) {
          endIdx++;
        }

        const segPts = pts.slice(startIdx, endIdx + 1);
        if (segPts.length > 0) {
          const cmds: string[] = [];
          for (let i = 0; i < segPts.length; i++) {
            const p = segPts[i];
            cmds.push(`${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${(-p.y).toFixed(2)}`);
          }
          if (ms.mini_sector_id === trackOutline.mini_sectors.length && pts[0]) {
            cmds.push(`L ${pts[0].x.toFixed(2)} ${(-pts[0].y).toFixed(2)}`);
          }
          miniSectorPaths[ms.mini_sector_id] = cmds.join(' ');
        }
      }
    }

    return {
      viewBox,
      pathString,
      startPoint: { x: sx0, y: sy0 },
      sfLine,
      sectorMarkers,
      markerScale,
      trackLengthKm: (trackOutline.total_distance_m / 1000).toFixed(3),
      miniSectorPaths,
    };
  }, [trackOutline]);

  // Filter valid drivers with verified coordinates and active on-track states
  const visibleDrivers = useMemo(() => {
    if (!drivers || drivers.length === 0) return [];
    return drivers
      .filter((d) => {
        if (d.x === null || d.y === null || typeof d.x !== 'number' || typeof d.y !== 'number') return false;
        if (isNaN(d.x) || isNaN(d.y)) return false;
        // Never fabricate or show positions for retired or telemetry-missing cars
        if (d.status === 'RETIRED' || d.status === 'TELEMETRY_MISSING') return false;
        return true;
      })
      // Sort descending by timing_position so P1 (leader) is rendered last and appears on top
      .sort((a, b) => b.timing_position - a.timing_position);
  }, [drivers]);

  const retiredCount = useMemo(() => {
    if (!drivers) return 0;
    return drivers.filter((d) => d.status === 'RETIRED').length;
  }, [drivers]);

  if (loading) {
    return (
      <div className="circuit-map-card">
        <div className="circuit-map-loading">
          <Loader2 className="animate-spin" size={28} />
          <span>Loading circuit geometry...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="circuit-map-card">
        <div className="circuit-map-error">
          <AlertCircle size={24} />
          <div>
            <strong>Circuit Geometry Unavailable</strong>
            <p>{error}</p>
          </div>
        </div>
      </div>
    );
  }

  if (!trackOutline || !mapData) {
    return (
      <div className="circuit-map-card">
        <div className="circuit-map-empty">
          <MapPin size={24} />
          <span>No circuit outline available for this race.</span>
        </div>
      </div>
    );
  }

  return (
    <div className={`circuit-map-card ${className || ''}`} id={id || 'circuit-map-container'}>
      {/* Circuit Map Header */}
      {!hideHeader && (
        <div className="circuit-map-header">
          <div className="circuit-title-area">
            <h2 className="circuit-name" title={trackOutline.circuit_name}>
              {trackOutline.circuit_name}
            </h2>
            <div className="circuit-badges">
              <span className="circuit-pill">{trackOutline.year}</span>
              <span className="circuit-pill mono">{mapData.trackLengthKm} km</span>
              <span className="circuit-pill">
                {trackOutline.sectors.length} Sectors
              </span>
              <span className="circuit-pill mono" id="cars-on-track-count">
                {visibleDrivers.length} / {drivers.length || 20} On Track
              </span>
              {retiredCount > 0 && (
                <span className="circuit-pill" style={{ color: 'var(--text-muted)' }}>
                  {retiredCount} RET
                </span>
              )}
            </div>
          </div>

          <div className="circuit-compass" title="Orientation: North is Up (+Y inverted)">
            <Compass size={14} />
            <span>N</span>
          </div>
        </div>
      )}

      {/* SVG Circuit Canvas */}
      <div className="circuit-svg-wrapper">
        <svg
          className="circuit-svg"
          viewBox={mapData.viewBox}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label={`Circuit map of ${trackOutline.circuit_name}`}
        >
          <defs>
            {/* Subtle glow filter */}
            <filter id="track-glow" x="-20%" y="-20%" width="140%" height="140%">
              <feGaussianBlur stdDeviation="3" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Ambient Glow Underlay */}
          <path
            d={mapData.pathString}
            className="track-ambient-glow"
            fill="none"
            vectorEffect="non-scaling-stroke"
            style={{ pointerEvents: 'none' }}
          />

          {/* Track Roadbed (Outer asphalt border) */}
          <path
            d={mapData.pathString}
            className="track-roadbed"
            fill="none"
            vectorEffect="non-scaling-stroke"
            style={{ pointerEvents: 'none' }}
          />

          {/* Track Surface (Inner asphalt) */}
          <path
            d={mapData.pathString}
            className="track-surface"
            fill="none"
            vectorEffect="non-scaling-stroke"
            style={{ pointerEvents: 'none' }}
          />

          {/* Track Centerline / Racing Line */}
          <path
            d={mapData.pathString}
            className="track-centerline"
            fill="none"
            vectorEffect="non-scaling-stroke"
            style={{ pointerEvents: 'none' }}
          />

          {/* Mini-Sector Overlay Layer */}
          {miniSectorSegments && miniSectorSegments.length > 0 && (
            <g className="track-mini-sectors-layer" role="group" aria-label="Mini-Sector Track Overlays">
              {miniSectorSegments.map((seg) => {
                const pathD = mapData.miniSectorPaths[seg.mini_sector_id];
                if (!pathD) return null;
                const isSelected = selectedMiniSectorId === seg.mini_sector_id;
                const isHovered = hoveredMiniSectorId === seg.mini_sector_id;
                return (
                  <path
                    key={`ms-path-${seg.mini_sector_id}`}
                    id={`track-minisector-${seg.mini_sector_id}`}
                    d={pathD}
                    fill="none"
                    stroke={seg.color}
                    strokeWidth={isSelected ? 10 : isHovered ? 8 : 6}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    opacity={isSelected ? 1.0 : isHovered ? 0.95 : 0.85}
                    vectorEffect="non-scaling-stroke"
                    className={`track-mini-sector-segment ${isSelected ? 'is-selected' : ''} ${isHovered ? 'is-hovered' : ''}`}
                    role="button"
                    tabIndex={0}
                    aria-label={seg.label || `Mini-Sector ${seg.mini_sector_id}`}
                    onClick={() => onSelectMiniSector && onSelectMiniSector(seg.mini_sector_id)}
                    onMouseEnter={() => onHoverMiniSector && onHoverMiniSector(seg.mini_sector_id)}
                    onMouseLeave={() => onHoverMiniSector && onHoverMiniSector(null)}
                    onKeyDown={(e) => {
                      if ((e.key === 'Enter' || e.key === ' ') && onSelectMiniSector) {
                        e.preventDefault();
                        onSelectMiniSector(seg.mini_sector_id);
                      }
                    }}
                    style={{
                      cursor: onSelectMiniSector ? 'pointer' : 'default',
                      transition: 'stroke-width 0.15s ease, opacity 0.15s ease',
                    }}
                  >
                    <title>{seg.tooltip || `Mini-Sector ${seg.mini_sector_id}`}</title>
                  </path>
                );
              })}
            </g>
          )}

          {/* Sector Boundary Ticks */}
          {mapData.sectorMarkers.map((sm) => (
            <g key={`sec-${sm.sector}`}>
              <circle
                cx={sm.x}
                cy={sm.y}
                r={4}
                className="sector-marker-dot"
                vectorEffect="non-scaling-stroke"
              />
            </g>
          ))}

          {/* Start / Finish Line */}
          <g className="start-finish-group">
            {/* White baseline */}
            <line
              x1={mapData.sfLine.x1}
              y1={mapData.sfLine.y1}
              x2={mapData.sfLine.x2}
              y2={mapData.sfLine.y2}
              className="sf-line-base"
              vectorEffect="non-scaling-stroke"
            />
            {/* Red dashed overlay */}
            <line
              x1={mapData.sfLine.x1}
              y1={mapData.sfLine.y1}
              x2={mapData.sfLine.x2}
              y2={mapData.sfLine.y2}
              className="sf-line-dash"
              vectorEffect="non-scaling-stroke"
            />
            {/* Start/finish beacon dot */}
            <circle
              cx={mapData.startPoint.x}
              cy={mapData.startPoint.y}
              r={3.5}
              className="sf-beacon"
              vectorEffect="non-scaling-stroke"
            />
          </g>

          {/* Moving Driver Markers Layer */}
          {!hideDriverMarkers && (
            <g className="driver-markers-layer" role="region" aria-label="Live Driver Track Positions">
              {visibleDrivers.map((d) => {
                const sx = d.x!;
                const sy = -d.y!;
                const teamColor = getTeamColor(d.team);
                const isLeader = d.timing_position === 1;
                const isStopped = d.status === 'STOPPED';
                const isOffTrack = d.status === 'OFF_TRACK';
                const isInPit = d.status === 'IN_PIT' || d.pit_status === 'IN_PIT';

                return (
                  <g
                    key={d.driver}
                    id={`driver-marker-${d.driver}`}
                    className={`driver-marker ${isLeader ? 'is-leader' : ''} ${isStopped ? 'is-stopped' : ''}`}
                    transform={`translate(${sx.toFixed(2)}, ${sy.toFixed(2)}) scale(${mapData.markerScale.toFixed(4)})`}
                    data-driver={d.driver}
                    data-pos={d.timing_position}
                    data-status={d.status}
                  >
                    <title>{`${d.driver} (P${d.timing_position}) - ${d.team} | Lap ${d.last_completed_lap} | Status: ${d.status}`}</title>

                    {/* Leader Gold Halo */}
                    {isLeader && (
                      <circle
                        r={13.5}
                        fill="none"
                        stroke="var(--f1-gold, #ffd700)"
                        strokeWidth={1.5}
                        strokeDasharray="3 2"
                        className="driver-leader-halo"
                      />
                    )}

                    {/* Driver Marker Circle */}
                    <circle
                      r={10}
                      fill={teamColor}
                      stroke={isStopped ? 'var(--flag-red, #ff3b30)' : '#ffffff'}
                      strokeWidth={isStopped ? 2 : 1.5}
                      strokeDasharray={isOffTrack ? '3 2' : undefined}
                      className="driver-marker-circle"
                    />

                    {/* Driver Code Text */}
                    <text
                      y={3.2}
                      textAnchor="middle"
                      fill="#ffffff"
                      stroke="#0a0c10"
                      strokeWidth={2.2}
                      paintOrder="stroke"
                      fontSize={7.5}
                      fontWeight="800"
                      fontFamily="var(--font-mono)"
                      className="driver-marker-code"
                    >
                      {d.driver}
                    </text>

                    {/* Status Indicator Badges */}
                    {isStopped && (
                      <g transform="translate(7, -7)">
                        <circle r={3.5} fill="var(--flag-red, #ff3b30)" stroke="#ffffff" strokeWidth={0.8} />
                        <text y={1.2} textAnchor="middle" fill="#ffffff" fontSize={4.5} fontWeight="900">!</text>
                      </g>
                    )}

                    {isInPit && !isStopped && (
                      <g transform="translate(0, -11)">
                        <rect x={-8} y={-4} width={16} height={7} rx={2} fill="var(--flag-yellow, #ffb800)" stroke="#10141d" strokeWidth={0.5} />
                        <text y={1.2} textAnchor="middle" fill="#10141d" fontSize={4.5} fontWeight="900" fontFamily="var(--font-mono)">PIT</text>
                      </g>
                    )}
                  </g>
                );
              })}
            </g>
          )}
        </svg>
      </div>

      {/* Circuit Map Footer */}
      {!hideDefaultLegend && (
        <div className="circuit-map-footer">
          <div className="map-legend">
            <div className="legend-item">
              <span className="legend-line sf-legend"></span>
              <span>Start / Finish</span>
            </div>
            <div className="legend-item">
              <span className="legend-line track-legend"></span>
              <span>Track Centerline</span>
            </div>
            <div className="legend-item">
              <span className="legend-dot driver-legend"></span>
              <span>Driver Marker</span>
            </div>
          </div>
          <div className="projection-note">Synchronized • Replay Clock</div>
        </div>
      )}
    </div>
  );
};
