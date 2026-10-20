import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { RaceSummary, TrackOutline } from './types/replay';
import { ComparisonConfigState, ComparisonRequest } from './types/comparison';
import { useReplaySocket } from './hooks/useReplaySocket';
import { useDriverComparison } from './hooks/useDriverComparison';
import { HeaderBar } from './components/HeaderBar';
import { PlaybackBar } from './components/PlaybackBar';
import { TimingTower } from './components/TimingTower';
import { CircuitMap } from './components/CircuitMap';
import { ComparisonConfigBar, DriverOption } from './components/comparison/ComparisonConfigBar';
import { ComparisonTable } from './components/comparison/ComparisonTable';
import { ComparisonChart } from './components/comparison/ComparisonChart';
import { ComparisonHeatStrip } from './components/comparison/ComparisonHeatStrip';
import { ComparisonTrackView } from './components/comparison/ComparisonTrackView';
import { TyreDegradationPanel } from './components/tyres/TyreDegradationPanel';
import { useTyreDegradation } from './hooks/useTyreDegradation';
import { GapBattlePanel } from './components/battles/GapBattlePanel';
import { PitStrategyPanel } from './components/pits/PitStrategyPanel';
import { AIAssistantDrawer } from './components/assistant/AIAssistantDrawer';
import { useAIAssistant } from './hooks/useAIAssistant';
import { EvidenceReference } from './types/assistant';
import { AlertCircle, WifiOff } from 'lucide-react';

export const App: React.FC = () => {
  const [races, setRaces] = useState<RaceSummary[]>([]);
  const [selectedRaceId, setSelectedRaceId] = useState<string>('2022_monza');
  const [loading, setLoading] = useState<boolean>(true);
  const [apiError, setApiError] = useState<string | null>(null);

  // AI Strategy Assistant drawer state
  const [isAssistantOpen, setIsAssistantOpen] = useState<boolean>(false);

  // View navigation state: Replay Dashboard vs Driver Comparison vs Tyre Degradation vs Gap & Battles vs Pit Strategy
  const [activeView, setActiveView] = useState<'dashboard' | 'comparison' | 'tyres' | 'battles' | 'pits'>('dashboard');

  // Comparison sub-view navigation state: 'all' | 'track' | 'heat_strip' | 'line' | 'table'
  const [activeCompView, setActiveCompView] = useState<'all' | 'track' | 'heat_strip' | 'line' | 'table'>('all');

  // Tyre Degradation Analysis state
  const [tyreDriver, setTyreDriver] = useState<string>('VER');
  const [tyreStint, setTyreStint] = useState<number>(1);
  const [tyreSyncWithReplay, setTyreSyncWithReplay] = useState<boolean>(false);

  // Driver selection states for deep linking / evidence navigation
  const [pitDriver, setPitDriver] = useState<string>('VER');
  const [battleDriver1, setBattleDriver1] = useState<string>('VER');
  const [battleDriver2, setBattleDriver2] = useState<string>('LEC');

  // Track outline geometry state
  const [trackOutline, setTrackOutline] = useState<TrackOutline | null>(null);
  const [trackLoading, setTrackLoading] = useState<boolean>(false);
  const [trackError, setTrackError] = useState<string | null>(null);

  // Comparison Configuration State (persistent across tab switching)
  const [comparisonConfig, setComparisonConfig] = useState<ComparisonConfigState>({
    selectedDrivers: ['VER', 'LEC'],
    referenceDriver: 'VER',
    lapMode: 'single',
    lapNumber: 1,
    startLap: 1,
    endLap: 10,
    stintNumber: 1,
    lastNClean: 5,
    syncMode: 'lap',
    cleanOnly: true,
    fuelCorrected: false,
    cutoffMode: 'replay',
  });

  // Dedicated Driver Comparison API hook
  const {
    data: comparisonData,
    isLoading: comparisonLoading,
    error: comparisonError,
    lastExecutedConfig,
    executeComparison,
    clear: clearComparison,
  } = useDriverComparison();

  // Clear comparison data and abort pending requests when race selection changes
  useEffect(() => {
    clearComparison();
  }, [selectedRaceId, clearComparison]);

  // Fetch available cached races
  useEffect(() => {
    fetch('/api/races')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: RaceSummary[]) => {
        setRaces(data);
        if (data.length > 0 && !data.some((r) => r.race_id === selectedRaceId)) {
          setSelectedRaceId(data[0].race_id);
        }
        setLoading(false);
      })
      .catch((err) => {
        setApiError(`Failed to fetch race catalog: ${err.message}`);
        setLoading(false);
      });
  }, []);

  // Fetch track outline whenever selectedRaceId changes
  useEffect(() => {
    if (!selectedRaceId) return;
    setTrackLoading(true);
    setTrackError(null);
    fetch(`/api/races/${selectedRaceId}/track`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: TrackOutline) => {
        setTrackOutline(data);
        setTrackLoading(false);
      })
      .catch((err) => {
        setTrackError(err.message || 'Failed to load track geometry');
        setTrackOutline(null);
        setTrackLoading(false);
      });
  }, [selectedRaceId]);

  const {
    clock,
    raceState,
    isConnected,
    error: wsError,
    play,
    pause,
    seek,
    stepLap,
    setSpeed,
  } = useReplaySocket(selectedRaceId);

  // Derive real available drivers from the active race replay state
  const availableDrivers: DriverOption[] = useMemo(() => {
    return (raceState?.leaderboard || []).map((d) => ({
      code: d.driver,
      team: d.team,
    }));
  }, [raceState?.leaderboard]);

  const driverCodes = useMemo(() => availableDrivers.map((d) => d.code), [availableDrivers]);
  const driverTeamMap = useMemo(() => {
    const map: Record<string, string> = {};
    availableDrivers.forEach((d) => {
      map[d.code] = d.team;
    });
    return map;
  }, [availableDrivers]);

  // Reconcile driver selection when race changes or available drivers list updates
  useEffect(() => {
    if (availableDrivers.length === 0) return;
    const availableCodes = new Set(availableDrivers.map((d) => d.code));
    setComparisonConfig((prev) => {
      const validSelected = prev.selectedDrivers.filter((d) => availableCodes.has(d));
      let nextSelected = validSelected;
      if (nextSelected.length < 2 && availableDrivers.length >= 2) {
        // Fallback to top 2 available drivers in the active race
        nextSelected = [availableDrivers[0].code, availableDrivers[1].code];
      }
      let nextRef = prev.referenceDriver;
      if (!nextSelected.includes(nextRef)) {
        nextRef = nextSelected.length > 0 ? nextSelected[0] : '';
      }
      if (
        nextSelected.length === prev.selectedDrivers.length &&
        nextSelected.every((val, idx) => val === prev.selectedDrivers[idx]) &&
        nextRef === prev.referenceDriver
      ) {
        return prev;
      }
      return {
        ...prev,
        selectedDrivers: nextSelected,
        referenceDriver: nextRef,
      };
    });

    if (availableDrivers.length > 0 && !availableDrivers.some((d) => d.code === tyreDriver)) {
      setTyreDriver(availableDrivers[0].code);
    }
  }, [availableDrivers, tyreDriver]);

  const currentRaceSummary = races.find((r) => r.race_id === selectedRaceId);
  const totalLaps = clock?.total_laps || currentRaceSummary?.total_laps || 53;
  const currentReplayTimeS = clock?.session_time_s ?? currentRaceSummary?.T_0 ?? 0;
  const currentReplayLap = clock?.current_lap ?? 1;
  const tFinishS = clock?.T_finish ?? currentRaceSummary?.T_finish ?? 0;

  // Dedicated Tyre Degradation API hook
  const {
    data: tyreData,
    isLoading: tyreLoading,
    error: tyreError,
    refetch: refetchTyreData,
  } = useTyreDegradation({
    raceId: selectedRaceId,
    driver: tyreDriver,
    stint: tyreStint,
    asOfTimeS: currentReplayTimeS,
    syncWithReplay: tyreSyncWithReplay,
  });

  // Execute comparison API query on explicit user trigger
  const handleRunComparison = useCallback(() => {
    const shouldIncludeMiniSectors =
      comparisonConfig.lapMode === 'single' || comparisonConfig.lapMode === 'fastest';

    const req: ComparisonRequest = {
      drivers: comparisonConfig.selectedDrivers,
      lap_mode: comparisonConfig.lapMode,
      lap_number: comparisonConfig.lapMode === 'single' ? comparisonConfig.lapNumber : undefined,
      lap_range: comparisonConfig.lapMode === 'range' ? [comparisonConfig.startLap, comparisonConfig.endLap] : undefined,
      stint_number: comparisonConfig.lapMode === 'stint' ? comparisonConfig.stintNumber : undefined,
      last_n_clean: comparisonConfig.lapMode === 'last_clean' ? comparisonConfig.lastNClean : undefined,
      reference_driver: comparisonConfig.referenceDriver,
      sync_mode: comparisonConfig.syncMode,
      clean_only: comparisonConfig.cleanOnly,
      fuel_corrected: comparisonConfig.fuelCorrected,
      as_of: comparisonConfig.cutoffMode === 'replay' ? currentReplayTimeS : tFinishS,
      include_mini_sectors: shouldIncludeMiniSectors,
    };
    executeComparison(selectedRaceId, req, comparisonConfig);
  }, [comparisonConfig, currentReplayTimeS, tFinishS, selectedRaceId, executeComparison]);

  // Detect if current comparison configuration has changed since last execution
  const isComparisonStale = useMemo(() => {
    if (!lastExecutedConfig) return false;
    if (comparisonConfig.selectedDrivers.length !== lastExecutedConfig.selectedDrivers.length) return true;
    for (let i = 0; i < comparisonConfig.selectedDrivers.length; i++) {
      if (comparisonConfig.selectedDrivers[i] !== lastExecutedConfig.selectedDrivers[i]) return true;
    }
    return (
      comparisonConfig.referenceDriver !== lastExecutedConfig.referenceDriver ||
      comparisonConfig.lapMode !== lastExecutedConfig.lapMode ||
      comparisonConfig.lapNumber !== lastExecutedConfig.lapNumber ||
      comparisonConfig.startLap !== lastExecutedConfig.startLap ||
      comparisonConfig.endLap !== lastExecutedConfig.endLap ||
      comparisonConfig.stintNumber !== lastExecutedConfig.stintNumber ||
      comparisonConfig.lastNClean !== lastExecutedConfig.lastNClean ||
      comparisonConfig.syncMode !== lastExecutedConfig.syncMode ||
      comparisonConfig.cleanOnly !== lastExecutedConfig.cleanOnly ||
      comparisonConfig.fuelCorrected !== lastExecutedConfig.fuelCorrected ||
      comparisonConfig.cutoffMode !== lastExecutedConfig.cutoffMode
    );
  }, [comparisonConfig, lastExecutedConfig]);

  // Dedicated AI Strategy Assistant Hook
  const {
    statusData: assistantStatus,
    statusLoading: assistantStatusLoading,
    statusError: assistantStatusError,
    turns: assistantTurns,
    isSubmitting: assistantSubmitting,
    queryError: assistantQueryError,
    submitQuery: submitAssistantQuery,
    clearConversation: clearAssistantConversation,
  } = useAIAssistant({
    raceId: selectedRaceId,
    currentReplayTimeS,
  });

  // Handle deep-link navigation from Assistant Evidence Chips
  const handleNavigateEvidence = useCallback(
    (ev: EvidenceReference) => {
      const payload = ev.nav_payload || {};
      const targetTab = payload.targetTab || ev.type;

      if (targetTab === 'dashboard' || ev.type === 'replay') {
        setActiveView('dashboard');
        const asOfTime = typeof payload.as_of === 'number' ? payload.as_of : ev.as_of;
        if (typeof asOfTime === 'number' && !isNaN(asOfTime)) {
          seek(asOfTime);
        }
      } else if (targetTab === 'comparison' || ev.type === 'comparison') {
        setActiveView('comparison');
        const dList: string[] =
          payload.drivers && Array.isArray(payload.drivers) && payload.drivers.length >= 2
            ? payload.drivers
            : ev.driver_codes && ev.driver_codes.length >= 2
            ? ev.driver_codes
            : ['VER', 'LEC'];
        const lapNum = payload.lap ?? ev.lap;
        const sLap = payload.start_lap;
        const eLap = payload.end_lap;
        const mode: 'single' | 'range' = sLap && eLap ? 'range' : 'single';

        const nextConfig: ComparisonConfigState = {
          selectedDrivers: dList,
          referenceDriver: dList[0] || 'VER',
          lapMode: mode,
          lapNumber: lapNum || 1,
          startLap: sLap || 1,
          endLap: eLap || 10,
          stintNumber: 1,
          lastNClean: 5,
          syncMode: 'lap',
          cleanOnly: true,
          fuelCorrected: false,
          cutoffMode: 'replay',
        };
        setComparisonConfig(nextConfig);

        const req: ComparisonRequest = {
          drivers: dList,
          lap_mode: mode,
          lap_number: mode === 'single' ? (lapNum || 1) : undefined,
          lap_range: mode === 'range' ? [sLap || 1, eLap || 10] : undefined,
          reference_driver: dList[0] || 'VER',
          sync_mode: 'lap',
          clean_only: true,
          fuel_corrected: false,
          as_of: currentReplayTimeS,
          include_mini_sectors: mode === 'single',
        };
        executeComparison(selectedRaceId, req, nextConfig);
      } else if (targetTab === 'tyres' || ev.type === 'tyres') {
        setActiveView('tyres');
        const driver = payload.driver || (ev.driver_codes && ev.driver_codes[0]);
        const stint = payload.stint ?? ev.stint ?? 1;
        if (driver) setTyreDriver(driver);
        setTyreStint(stint);
      } else if (targetTab === 'pits' || ev.type === 'pits') {
        setActiveView('pits');
        const driver = payload.driver || (ev.driver_codes && ev.driver_codes[0]);
        if (driver) setPitDriver(driver);
      } else if (targetTab === 'battles' || ev.type === 'battles') {
        setActiveView('battles');
        const chaser = payload.chaser || (ev.driver_codes && ev.driver_codes[0]);
        const leader = payload.leader || (ev.driver_codes && ev.driver_codes[1]);
        if (chaser) setBattleDriver1(chaser);
        if (leader) setBattleDriver2(leader);
      }
    },
    [seek, currentReplayTimeS, executeComparison, selectedRaceId]
  );

  return (
    <div className="replay-app">
      <HeaderBar
        races={races}
        selectedRaceId={selectedRaceId}
        onSelectRace={(id) => setSelectedRaceId(id)}
        clock={clock}
        raceState={raceState}
        isAssistantOpen={isAssistantOpen}
        onToggleAssistant={() => setIsAssistantOpen((prev) => !prev)}
        assistantAvailable={assistantStatus?.available}
      />

      <PlaybackBar
        clock={clock}
        onPlay={play}
        onPause={pause}
        onSeek={seek}
        onStepLap={stepLap}
        onSetSpeed={setSpeed}
      />

      <main className="main-content">
        {/* View Navigation Tabs */}
        <div className="view-navigation-bar" role="tablist" id="main-view-navigation">
          <button
            type="button"
            id="tab-replay-dashboard"
            className={`view-tab-btn ${activeView === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveView('dashboard')}
            role="tab"
            aria-selected={activeView === 'dashboard'}
          >
            Replay Dashboard
          </button>
          <button
            type="button"
            id="tab-driver-comparison"
            className={`view-tab-btn ${activeView === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveView('comparison')}
            role="tab"
            aria-selected={activeView === 'comparison'}
          >
            Driver Comparison
          </button>
          <button
            type="button"
            id="tab-tyre-degradation"
            className={`view-tab-btn ${activeView === 'tyres' ? 'active' : ''}`}
            onClick={() => setActiveView('tyres')}
            role="tab"
            aria-selected={activeView === 'tyres'}
          >
            Tyre Degradation
          </button>
          <button
            type="button"
            id="tab-gap-battles"
            className={`view-tab-btn ${activeView === 'battles' ? 'active' : ''}`}
            onClick={() => setActiveView('battles')}
            role="tab"
            aria-selected={activeView === 'battles'}
          >
            Gap & Battles
          </button>
          <button
            type="button"
            id="tab-pit-strategy"
            className={`view-tab-btn ${activeView === 'pits' ? 'active' : ''}`}
            onClick={() => setActiveView('pits')}
            role="tab"
            aria-selected={activeView === 'pits'}
          >
            Pit Strategy
          </button>
        </div>

        {!isConnected && (
          <div
            id="ws-disconnected-banner"
            style={{
              background: 'rgba(255, 184, 0, 0.1)',
              border: '1px solid rgba(255, 184, 0, 0.3)',
              color: 'var(--flag-yellow)',
              padding: '10px 16px',
              borderRadius: '6px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '13px',
              fontWeight: 600,
            }}
          >
            <WifiOff size={16} /> Connecting to Replay WebSocket feed...
          </div>
        )}

        {(apiError || wsError) && (
          <div
            id="api-error-banner"
            style={{
              background: 'rgba(255, 59, 48, 0.15)',
              border: '1px solid rgba(255, 59, 48, 0.4)',
              color: 'var(--flag-red)',
              padding: '10px 16px',
              borderRadius: '6px',
              marginBottom: '16px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              fontSize: '13px',
            }}
          >
            <AlertCircle size={16} /> {apiError || wsError}
          </div>
        )}

        {loading ? (
          <div style={{ padding: '40px', textAlign: 'center', color: 'var(--text-muted)' }}>
            Loading race replay session...
          </div>
        ) : activeView === 'dashboard' ? (
          <div className="replay-dashboard-grid" id="replay-dashboard-view">
            <TimingTower
              leaderboard={raceState?.leaderboard || []}
              activeBattles={raceState?.active_battles || []}
            />
            <CircuitMap
              trackOutline={trackOutline}
              drivers={raceState?.leaderboard || []}
              loading={trackLoading}
              error={trackError}
            />
          </div>
        ) : activeView === 'comparison' ? (
          <div className="comparison-workspace-view" id="driver-comparison-view">
            <ComparisonConfigBar
              config={comparisonConfig}
              onChange={setComparisonConfig}
              availableDrivers={availableDrivers}
              totalLaps={totalLaps}
              currentReplayTimeS={currentReplayTimeS}
              currentReplayLap={currentReplayLap}
              tFinishS={tFinishS}
              onRunComparison={handleRunComparison}
              isLoading={comparisonLoading}
              error={comparisonError}
              lastExecutedConfig={lastExecutedConfig}
              lastResponse={comparisonData}
            />

            {/* Comparison Visualization Sub-Navigation */}
            <div className="comparison-view-tabs" role="tablist" id="comparison-view-navigation">
              <button
                type="button"
                id="tab-comp-view-all"
                className={`comp-view-tab-btn ${activeCompView === 'all' ? 'active' : ''}`}
                onClick={() => setActiveCompView('all')}
                role="tab"
                aria-selected={activeCompView === 'all'}
              >
                All Views
              </button>
              <button
                type="button"
                id="tab-comp-view-track"
                className={`comp-view-tab-btn ${activeCompView === 'track' ? 'active' : ''}`}
                onClick={() => setActiveCompView('track')}
                role="tab"
                aria-selected={activeCompView === 'track'}
              >
                Track
              </button>
              <button
                type="button"
                id="tab-comp-view-heat-strip"
                className={`comp-view-tab-btn ${activeCompView === 'heat_strip' ? 'active' : ''}`}
                onClick={() => setActiveCompView('heat_strip')}
                role="tab"
                aria-selected={activeCompView === 'heat_strip'}
              >
                Heat Strip
              </button>
              <button
                type="button"
                id="tab-comp-view-line"
                className={`comp-view-tab-btn ${activeCompView === 'line' ? 'active' : ''}`}
                onClick={() => setActiveCompView('line')}
                role="tab"
                aria-selected={activeCompView === 'line'}
              >
                Line (Pace Chart)
              </button>
              <button
                type="button"
                id="tab-comp-view-table"
                className={`comp-view-tab-btn ${activeCompView === 'table' ? 'active' : ''}`}
                onClick={() => setActiveCompView('table')}
                role="tab"
                aria-selected={activeCompView === 'table'}
              >
                Table
              </button>
            </div>

            {(activeCompView === 'all' || activeCompView === 'track') && (
              <ComparisonTrackView
                data={comparisonData}
                isLoading={comparisonLoading}
                error={comparisonError}
                isStale={isComparisonStale}
                availableDrivers={availableDrivers}
                currentLapMode={comparisonConfig.lapMode}
                onRefresh={handleRunComparison}
                trackOutline={trackOutline}
                circuitLoading={trackLoading}
                circuitError={trackError}
              />
            )}

            {(activeCompView === 'all' || activeCompView === 'heat_strip') && (
              <ComparisonHeatStrip
                data={comparisonData}
                isLoading={comparisonLoading}
                error={comparisonError}
                isStale={isComparisonStale}
                availableDrivers={availableDrivers}
                currentLapMode={comparisonConfig.lapMode}
                onRefresh={handleRunComparison}
              />
            )}

            {(activeCompView === 'all' || activeCompView === 'line') && (
              <ComparisonChart
                data={comparisonData}
                isLoading={comparisonLoading}
                error={comparisonError}
                isStale={isComparisonStale}
                availableDrivers={availableDrivers}
              />
            )}

            {(activeCompView === 'all' || activeCompView === 'table') && (
              <ComparisonTable
                data={comparisonData}
                isLoading={comparisonLoading}
                error={comparisonError}
                isStale={isComparisonStale}
                availableDrivers={availableDrivers}
                onRefresh={handleRunComparison}
              />
            )}
          </div>
        ) : activeView === 'tyres' ? (
          <TyreDegradationPanel
            data={tyreData}
            isLoading={tyreLoading}
            error={tyreError}
            selectedDriver={tyreDriver}
            onSelectDriver={setTyreDriver}
            selectedStint={tyreStint}
            onSelectStint={setTyreStint}
            syncWithReplay={tyreSyncWithReplay}
            onToggleSync={setTyreSyncWithReplay}
            onRefresh={refetchTyreData}
            availableDrivers={availableDrivers}
            currentReplayLap={currentReplayLap}
            currentReplayTimeS={currentReplayTimeS}
          />
        ) : activeView === 'pits' ? (
          <PitStrategyPanel
            raceId={selectedRaceId}
            availableDrivers={availableDrivers}
            currentReplayTimeS={currentReplayTimeS}
            currentReplayLap={currentReplayLap}
            tFinishS={tFinishS}
            initialDriver={pitDriver || availableDrivers[0]?.code || 'VER'}
          />
        ) : (
          <GapBattlePanel
            raceId={selectedRaceId}
            availableDrivers={driverCodes.length > 0 ? driverCodes : ['VER', 'LEC', 'HAM', 'RUS', 'SAI', 'PER']}
            driverTeamMap={driverTeamMap}
            currentSessionTimeS={currentReplayTimeS}
            totalRaceLaps={totalLaps}
            initialDriver1={battleDriver1}
            initialDriver2={battleDriver2}
          />
        )}
      </main>

      <AIAssistantDrawer
        isOpen={isAssistantOpen}
        onClose={() => setIsAssistantOpen(false)}
        statusData={assistantStatus}
        statusLoading={assistantStatusLoading}
        statusError={assistantStatusError}
        turns={assistantTurns}
        isSubmitting={assistantSubmitting}
        queryError={assistantQueryError}
        onSubmitQuery={submitAssistantQuery}
        onClearConversation={clearAssistantConversation}
        onNavigateEvidence={handleNavigateEvidence}
        currentReplayTimeS={currentReplayTimeS}
        currentReplayLap={currentReplayLap}
      />
    </div>
  );
};

export default App;
