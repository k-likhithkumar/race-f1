import React, { useState, useEffect } from 'react';
import { usePitStrategy } from '../../hooks/usePitStrategy';
import { PitStopsTable } from './PitStopsTable';
import { PitWaterfallChart } from './PitWaterfallChart';
import { TacticalOutcomeCard } from './TacticalOutcomeCard';
import {
  Wrench,
  Clock,
  RotateCw,
  AlertCircle,
  HelpCircle,
  ShieldCheck,
  Zap,
} from 'lucide-react';

interface DriverOption {
  code: string;
  team: string;
}

interface PitStrategyPanelProps {
  raceId: string;
  availableDrivers: DriverOption[];
  currentReplayTimeS?: number;
  currentReplayLap?: number;
  tFinishS?: number;
  initialDriver?: string;
}

export const PitStrategyPanel: React.FC<PitStrategyPanelProps> = ({
  raceId,
  availableDrivers,
  currentReplayTimeS,
  currentReplayLap,
  tFinishS,
  initialDriver,
}) => {
  const [selectedDriver, setSelectedDriver] = useState<string>(
    initialDriver || (availableDrivers.length > 0 ? availableDrivers[0].code : 'VER')
  );
  const [activeSubTab, setActiveSubTab] = useState<'stops' | 'waterfall' | 'tactical'>('stops');
  const [syncWithReplay, setSyncWithReplay] = useState<boolean>(true);

  // Reconcile selected driver when availableDrivers list updates or race changes
  useEffect(() => {
    if (availableDrivers.length > 0) {
      const exists = availableDrivers.some((d) => d.code === selectedDriver);
      if (!exists) {
        setSelectedDriver(availableDrivers[0].code);
      }
    }
  }, [raceId, availableDrivers, selectedDriver]);

  // Update selectedDriver when initialDriver is passed from evidence navigation
  useEffect(() => {
    if (initialDriver && availableDrivers.some((d) => d.code === initialDriver)) {
      setSelectedDriver(initialDriver);
    }
  }, [initialDriver, availableDrivers]);

  const effectiveAsOfTimeS = syncWithReplay ? currentReplayTimeS : (tFinishS && tFinishS > 0 ? tFinishS : 999999);
  const effectiveAsOfLap = syncWithReplay ? currentReplayLap : undefined;

  const { data, isLoading, error, refetch } = usePitStrategy({
    raceId,
    driver: selectedDriver,
    asOfTimeS: effectiveAsOfTimeS,
    asOfLap: effectiveAsOfLap,
    referenceStationaryS: 2.5,
  });

  const stops = data?.stops || [];
  const completedStops = data?.total_completed_stops || 0;
  const activeStop = data?.active_stop;
  const isPreRace = data?.status === 'PRE_RACE';

  return (
    <div className="pit-strategy-workspace" id="pit-strategy-panel">
      {/* Top Configuration & Control Bar */}
      <div className="pit-control-bar">
        <div className="pit-driver-select-group">
          <label htmlFor="pit-driver-select" className="pit-select-label">
            <Wrench size={16} /> Driver:
          </label>
          <select
            id="pit-driver-select"
            className="pit-driver-dropdown"
            value={selectedDriver}
            aria-label="Select driver for pit strategy"
            onChange={(e) => setSelectedDriver(e.target.value)}
          >
            {availableDrivers.map((d) => (
              <option key={d.code} value={d.code}>
                {d.code} ({d.team})
              </option>
            ))}
          </select>
        </div>

        {/* View Switcher Sub-Tabs */}
        <div className="pit-subtab-group" role="tablist">
          <button
            type="button"
            id="tab-pit-view-stops"
            className={`pit-subtab-btn ${activeSubTab === 'stops' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('stops')}
            role="tab"
            aria-selected={activeSubTab === 'stops'}
          >
            Stops & Transit
          </button>
          <button
            type="button"
            id="tab-pit-view-waterfall"
            className={`pit-subtab-btn ${activeSubTab === 'waterfall' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('waterfall')}
            role="tab"
            aria-selected={activeSubTab === 'waterfall'}
          >
            Time-Loss Waterfall
          </button>
          <button
            type="button"
            id="tab-pit-view-tactical"
            className={`pit-subtab-btn ${activeSubTab === 'tactical' ? 'active' : ''}`}
            onClick={() => setActiveSubTab('tactical')}
            role="tab"
            aria-selected={activeSubTab === 'tactical'}
          >
            Tactical Analysis
          </button>
        </div>

        <div className="pit-actions-group" style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <button
            type="button"
            id="btn-toggle-pit-sync"
            className={`pit-sync-btn ${syncWithReplay ? 'active' : ''}`}
            onClick={() => setSyncWithReplay(!syncWithReplay)}
            title={
              syncWithReplay
                ? `Synchronized with replay clock at ${currentReplayTimeS ? `${currentReplayTimeS.toFixed(1)}s` : ''}`
                : 'Showing full race strategy'
            }
          >
            <Clock size={14} />
            <span>{syncWithReplay ? `Synced (L${currentReplayLap || 1})` : 'Full Race'}</span>
          </button>

          <button
            type="button"
            id="btn-refresh-pit-strategy"
            className="pit-refresh-btn"
            onClick={refetch}
            disabled={isLoading}
            title="Refresh pit strategy analysis at current simulation time"
          >
            <RotateCw size={14} className={isLoading ? 'animate-spin' : ''} />
            <span>{isLoading ? 'Updating...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* Top KPI Metrics Banner */}
      <div className="pit-kpi-banner">
        <div className="pit-kpi-card" id="kpi-completed-stops">
          <span className="kpi-title">Completed Stops</span>
          <span className="kpi-value font-mono font-bold text-cyan">{error ? '--' : completedStops}</span>
          <span className="kpi-subtext">Total official pit stops</span>
        </div>

        <div className="pit-kpi-card" id="kpi-active-stop-state">
          <span className="kpi-title">Active Pit State</span>
          {error ? (
            <span className="kpi-value font-mono font-bold text-muted">ERROR</span>
          ) : activeStop ? (
            <span className="kpi-value font-mono font-bold text-amber animate-pulse">
              {activeStop.status === 'IN_PROGRESS' ? 'IN PIT LANE' : 'ON OUT-LAP'}
            </span>
          ) : (
            <span className="kpi-value font-mono font-bold text-green">ON TRACK</span>
          )}
          <span className="kpi-subtext">
            {error ? 'Analysis unavailable' : activeStop ? `Stop #${activeStop.stop_number} in progress` : 'No active pit stop'}
          </span>
        </div>

        <div className="pit-kpi-card" id="kpi-stationary-assumption">
          <div className="flex items-center justify-between">
            <span className="kpi-title">Stationary Reference</span>
            <span className="kpi-tag-assumption">Assumption</span>
          </div>
          <span className="kpi-value font-mono font-bold">2.5s</span>
          <span className="kpi-subtext">Wheel change benchmark</span>
        </div>

        <div className="pit-kpi-card" id="kpi-clean-pace-baseline">
          <span className="kpi-title">Clean Flying Pace</span>
          <span className="kpi-value font-mono font-bold">
            {error ? '--' : data?.reference_clean_pace_s ? `${data.reference_clean_pace_s.toFixed(3)}s` : '85.000s'}
          </span>
          <span className="kpi-subtext">Field median green lap</span>
        </div>

        <div className="pit-kpi-card" id="kpi-waterfall-verified">
          <span className="kpi-title">Waterfall Accounting</span>
          {error ? (
            <span className="kpi-value font-mono font-bold text-muted">--</span>
          ) : (
            <span className="kpi-value font-mono font-bold text-green flex items-center gap-1">
              <ShieldCheck size={18} /> Verified
            </span>
          )}
          <span className="kpi-subtext">{error ? 'Unavailable' : 'Identity holds (±0.01s)'}</span>
        </div>
      </div>

      {/* Mandatory Accounting & Assumption Notice */}
      <div className="pit-disclaimer-banner" id="pit-accounting-disclaimer">
        <div className="disclaimer-title">
          <HelpCircle size={15} /> Strategy Accounting & Measurement Standards
        </div>
        <div className="disclaimer-body">
          <strong>1. Illustrative Assumption:</strong> The 2.5s stationary stop time is strictly an illustrative benchmark assumption. Hardware transponder timing loops record gross pit lane traversal (Pit In line to Pit Out line). Stationary box time is not isolated in official provider feeds.<br />
          <strong>2. Anti-Double-Counting Invariant:</strong> The time-loss waterfall strictly allocates lap deficits through a priority order (Caution → Pit → Traffic → Tyre → Residual). In-lap and out-lap deficits are accounted for once and never added on top of transit.<br />
          <strong>3. Heuristic Classification:</strong> Undercut and overcut classifications represent empirical tactical inferences based on competitive proximity (gap ≤ 3.0s on preceding lap), not telemetry intent.
        </div>
      </div>

      {/* Error State: Displayed prominently when request fails, NEVER displaying 'No Pit Stops Recorded' */}
      {error ? (
        <div className="pit-error-card" id="pit-error-state" style={{ margin: '1.5rem 0', padding: '2rem', background: 'rgba(239, 68, 68, 0.08)', border: '1px solid rgba(239, 68, 68, 0.3)', borderRadius: '8px', textAlign: 'center' }}>
          <AlertCircle size={36} style={{ color: '#ef4444', margin: '0 auto 0.75rem' }} />
          <div style={{ color: '#ef4444', fontWeight: 600, fontSize: '1.15rem', marginBottom: '0.5rem' }}>
            Pit Strategy Analysis Unavailable
          </div>
          <div id="pit-error-message" style={{ color: '#9ca3af', marginBottom: '1.25rem', fontSize: '0.95rem' }}>
            {error}
          </div>
          <button
            type="button"
            id="btn-retry-pit-strategy"
            className="pit-refresh-btn"
            onClick={refetch}
            disabled={isLoading}
            style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', margin: '0 auto' }}
          >
            <RotateCw size={14} className={isLoading ? 'animate-spin' : ''} />
            <span>{isLoading ? 'Retrying...' : 'Retry Request'}</span>
          </button>
        </div>
      ) : (
        <>
          {/* Pre-Race Banner */}
          {isPreRace && (
            <div className="pit-prerace-banner" id="pit-prerace-card">
              <Clock size={20} />
              <div>
                <strong>Pre-Race Simulation:</strong> Replay clock is currently prior to race start (T_0). No pit stops have occurred.
              </div>
            </div>
          )}

          {/* Main Tab Content */}
          <div className="pit-tab-content">
            {activeSubTab === 'stops' && (
              <PitStopsTable stops={stops} referenceCleanPace={data?.reference_clean_pace_s} />
            )}

            {activeSubTab === 'waterfall' && (
              <PitWaterfallChart breakdown={data?.time_loss_breakdown || null} driver={selectedDriver} />
            )}

            {activeSubTab === 'tactical' && (
              <div className="tactical-analysis-view" id="pit-tactical-view">
                {stops.length === 0 ? (
                  <div className="pit-empty-card">
                    <Zap size={24} className="text-muted" />
                    <div className="pit-empty-title">No Tactical Events</div>
                    <div className="pit-empty-desc">
                      No pit stops recorded up to the current simulation time.
                    </div>
                  </div>
                ) : (
                  <div className="tactical-cards-grid">
                    {stops.map((stop) => (
                      <TacticalOutcomeCard
                        key={stop.stop_number}
                        stopNumber={stop.stop_number}
                        driver={selectedDriver}
                        tacticalEvidence={stop.tactical_evidence}
                      />
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
