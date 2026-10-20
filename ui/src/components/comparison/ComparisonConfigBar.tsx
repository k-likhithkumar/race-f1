import React from 'react';
import {
  ComparisonConfigState,
  ComparisonRequest,
  ComparisonResponse,
} from '../../types/comparison';
import { getTeamColor } from '../../constants/teams';
import {
  Clock,
  X,
  AlertTriangle,
  ShieldCheck,
  Fuel,
  Users,
  Play,
  RefreshCw,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';

export interface DriverOption {
  code: string;
  team: string;
}

interface ComparisonConfigBarProps {
  config: ComparisonConfigState;
  onChange: (newConfig: ComparisonConfigState) => void;
  availableDrivers: DriverOption[];
  totalLaps: number;
  currentReplayTimeS: number;
  currentReplayLap: number;
  tFinishS: number;
  onRunComparison: () => void;
  isLoading: boolean;
  error: string | null;
  lastExecutedConfig: ComparisonConfigState | null;
  lastResponse: ComparisonResponse | null;
}

export const ComparisonConfigBar: React.FC<ComparisonConfigBarProps> = ({
  config,
  onChange,
  availableDrivers,
  totalLaps,
  currentReplayTimeS,
  currentReplayLap,
  tFinishS,
  onRunComparison,
  isLoading,
  error,
  lastExecutedConfig,
  lastResponse,
}) => {
  const {
    selectedDrivers,
    referenceDriver,
    lapMode,
    lapNumber,
    startLap,
    endLap,
    stintNumber,
    lastNClean,
    syncMode,
    cleanOnly,
    fuelCorrected,
    cutoffMode,
  } = config;

  // Validation rules
  const validationErrors: string[] = [];
  if (selectedDrivers.length < 2) {
    validationErrors.push('Select at least 2 drivers to compare (maximum 6).');
  }
  if (selectedDrivers.length > 6) {
    validationErrors.push('A maximum of 6 drivers can be selected.');
  }
  if (!selectedDrivers.includes(referenceDriver)) {
    validationErrors.push('The reference driver must be one of the selected drivers.');
  }
  if (lapMode === 'single' && (lapNumber < 1 || lapNumber > totalLaps)) {
    validationErrors.push(`Lap number must be between 1 and ${totalLaps}.`);
  }
  if (lapMode === 'range') {
    if (startLap < 1 || endLap > totalLaps) {
      validationErrors.push(`Lap range must be within [1, ${totalLaps}].`);
    }
    if (startLap > endLap) {
      validationErrors.push(`Start lap (${startLap}) cannot exceed end lap (${endLap}).`);
    }
  }
  if (lapMode === 'stint' && stintNumber < 1) {
    validationErrors.push('Stint number must be at least 1.');
  }
  if (lapMode === 'last_clean' && lastNClean < 1) {
    validationErrors.push('Last clean laps count must be at least 1.');
  }

  // Detect whether current draft config differs from the last executed request
  const isConfigDirty = (() => {
    if (!lastExecutedConfig) return false;
    if (selectedDrivers.length !== lastExecutedConfig.selectedDrivers.length) return true;
    for (let i = 0; i < selectedDrivers.length; i++) {
      if (selectedDrivers[i] !== lastExecutedConfig.selectedDrivers[i]) return true;
    }
    return (
      referenceDriver !== lastExecutedConfig.referenceDriver ||
      lapMode !== lastExecutedConfig.lapMode ||
      lapNumber !== lastExecutedConfig.lapNumber ||
      startLap !== lastExecutedConfig.startLap ||
      endLap !== lastExecutedConfig.endLap ||
      stintNumber !== lastExecutedConfig.stintNumber ||
      lastNClean !== lastExecutedConfig.lastNClean ||
      syncMode !== lastExecutedConfig.syncMode ||
      cleanOnly !== lastExecutedConfig.cleanOnly ||
      fuelCorrected !== lastExecutedConfig.fuelCorrected ||
      cutoffMode !== lastExecutedConfig.cutoffMode
    );
  })();

  // Driver addition
  const handleAddDriver = (driverCode: string) => {
    if (!driverCode) return;
    if (selectedDrivers.includes(driverCode)) return;
    if (selectedDrivers.length >= 6) return;

    const nextDrivers = [...selectedDrivers, driverCode];
    const nextRef = referenceDriver || nextDrivers[0];
    onChange({
      ...config,
      selectedDrivers: nextDrivers,
      referenceDriver: nextRef,
    });
  };

  // Driver removal
  const handleRemoveDriver = (driverCode: string) => {
    const nextDrivers = selectedDrivers.filter((d) => d !== driverCode);
    let nextRef = referenceDriver;
    if (referenceDriver === driverCode) {
      nextRef = nextDrivers.length > 0 ? nextDrivers[0] : '';
    }
    onChange({
      ...config,
      selectedDrivers: nextDrivers,
      referenceDriver: nextRef,
    });
  };

  // Reference driver selection
  const handleSetReference = (driverCode: string) => {
    if (selectedDrivers.includes(driverCode)) {
      onChange({ ...config, referenceDriver: driverCode });
    }
  };

  // Derived unselected drivers for dropdown
  const unselectedDrivers = availableDrivers.filter(
    (d) => !selectedDrivers.includes(d.code)
  );

  // Formulated typed comparison request for inspectability
  const formulatedRequest: ComparisonRequest = {
    drivers: selectedDrivers,
    lap_mode: lapMode,
    lap_number: lapMode === 'single' ? lapNumber : undefined,
    lap_range: lapMode === 'range' ? [startLap, endLap] : undefined,
    stint_number: lapMode === 'stint' ? stintNumber : undefined,
    last_n_clean: lapMode === 'last_clean' ? lastNClean : undefined,
    reference_driver: referenceDriver,
    sync_mode: syncMode,
    clean_only: cleanOnly,
    fuel_corrected: fuelCorrected,
    as_of: cutoffMode === 'replay' ? currentReplayTimeS : tFinishS,
  };

  return (
    <div className="comparison-config-container" id="driver-comparison-config-bar">
      <div className="config-header">
        <div className="config-header-left">
          <div className="config-title">
            <Users size={18} className="text-red" />
            <h2>Driver Comparison Configuration</h2>
          </div>
          <span className="config-subtitle">
            Configure multi-driver telemetry and lap comparison with authoritative replay barrier enforcement.
          </span>
        </div>

        <div className="config-header-actions">
          <button
            type="button"
            id="btn-run-comparison"
            className="btn-run-comparison"
            disabled={validationErrors.length > 0 || isLoading}
            onClick={onRunComparison}
            title={
              validationErrors.length > 0
                ? 'Fix configuration errors to run comparison'
                : isLoading
                ? 'Comparison query in progress...'
                : 'Execute comparison request against active race session'
            }
          >
            {isLoading ? (
              <>
                <Loader2 size={15} className="spin" />
                <span>Querying API...</span>
              </>
            ) : lastResponse ? (
              <>
                <RefreshCw size={14} />
                <span>Run / Refresh</span>
              </>
            ) : (
              <>
                <Play size={14} fill="currentColor" />
                <span>Run Comparison</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Execution Error Banner */}
      {error && (
        <div className="config-error-alert" id="comparison-error-alert">
          <AlertCircle size={16} />
          <div>
            <strong>Comparison Query Failed:</strong> {error}
          </div>
        </div>
      )}

      {/* Stale Configuration Warning Banner */}
      {isConfigDirty && lastResponse && (
        <div className="config-stale-alert" id="comparison-stale-alert">
          <AlertTriangle size={15} />
          <span>
            Configuration has changed since last comparison query. Click <strong>Run / Refresh</strong> to fetch updated results.
          </span>
        </div>
      )}

      {/* Up-to-date Status Indicator */}
      {!isConfigDirty && lastResponse && !isLoading && !error && (
        <div className="config-success-banner" id="comparison-success-banner">
          <CheckCircle2 size={15} />
          <span>
            Comparison results up to date ({lastResponse.laps.length} laps, status: {lastResponse.status}, cutoff: {lastResponse.as_of_time_s !== null && lastResponse.as_of_time_s !== undefined ? `${lastResponse.as_of_time_s.toFixed(2)}s` : 'Full Race'}).
          </span>
        </div>
      )}

      {/* Validation Banner if invalid */}
      {validationErrors.length > 0 && (
        <div className="config-validation-alert" id="comparison-validation-alert">
          <AlertTriangle size={16} />
          <div>
            {validationErrors.map((err, idx) => (
              <div key={idx} className="validation-item">
                {err}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="config-grid">
        {/* SECTION 1: Driver Selection */}
        <div className="config-section" id="section-driver-selection">
          <label className="section-label">
            1. Drivers ({selectedDrivers.length}/6 selected)
          </label>
          <div className="driver-chips-row">
            {selectedDrivers.map((code) => {
              const driverInfo = availableDrivers.find((d) => d.code === code);
              const teamColor = getTeamColor(driverInfo?.team);
              const isReference = code === referenceDriver;
              return (
                <div
                  key={code}
                  className={`driver-chip ${isReference ? 'is-reference' : ''}`}
                  style={{ borderLeftColor: teamColor }}
                  title={`${driverInfo?.team || 'Unknown team'}${isReference ? ' (Reference Baseline)' : ''}`}
                >
                  <span className="driver-code mono">{code}</span>
                  {isReference && <span className="ref-badge">REF</span>}
                  <button
                    type="button"
                    className="chip-remove-btn"
                    onClick={() => handleRemoveDriver(code)}
                    title={`Remove ${code}`}
                    aria-label={`Remove ${code}`}
                  >
                    <X size={12} />
                  </button>
                </div>
              );
            })}

            {/* Dropdown to add more drivers */}
            {selectedDrivers.length < 6 && unselectedDrivers.length > 0 && (
              <div className="add-driver-dropdown-wrapper">
                <select
                  id="add-driver-select"
                  className="config-select-compact"
                  value=""
                  onChange={(e) => handleAddDriver(e.target.value)}
                  aria-label="Add driver to comparison"
                >
                  <option value="" disabled>
                    + Add Driver
                  </option>
                  {unselectedDrivers.map((d) => (
                    <option key={d.code} value={d.code}>
                      {d.code} — {d.team}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {/* Reference Driver Selector */}
          {selectedDrivers.length > 0 && (
            <div className="reference-driver-row">
              <span className="sublabel">Reference Driver (Baseline):</span>
              <select
                id="reference-driver-select"
                className="config-select-compact mono"
                value={referenceDriver}
                onChange={(e) => handleSetReference(e.target.value)}
                aria-label="Choose baseline reference driver"
              >
                {selectedDrivers.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>

        {/* SECTION 2: Lap Selection Mode */}
        <div className="config-section" id="section-lap-mode">
          <label className="section-label">2. Lap Selection Mode</label>
          <div className="mode-pills-group" role="radiogroup" aria-label="Lap selection mode">
            {(
              [
                ['single', 'Single Lap'],
                ['range', 'Lap Range'],
                ['fastest', 'Fastest Lap'],
                ['stint', 'Stint'],
                ['last_clean', 'Last N Clean'],
              ] as const
            ).map(([modeKey, label]) => (
              <button
                key={modeKey}
                type="button"
                className={`mode-pill ${lapMode === modeKey ? 'active' : ''}`}
                onClick={() => onChange({ ...config, lapMode: modeKey })}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Contextual Mode Controls */}
          <div className="contextual-controls-panel">
            {lapMode === 'single' && (
              <div className="input-field-inline">
                <label htmlFor="input-lap-number">Lap Number:</label>
                <input
                  id="input-lap-number"
                  type="number"
                  className="config-input-number mono"
                  min={1}
                  max={totalLaps}
                  value={lapNumber}
                  onChange={(e) =>
                    onChange({ ...config, lapNumber: parseInt(e.target.value, 10) || 1 })
                  }
                />
                <span className="field-hint">of {totalLaps} laps</span>
              </div>
            )}

            {lapMode === 'range' && (
              <div className="input-fields-row">
                <div className="input-field-inline">
                  <label htmlFor="input-range-start">Start Lap:</label>
                  <input
                    id="input-range-start"
                    type="number"
                    className="config-input-number mono"
                    min={1}
                    max={totalLaps}
                    value={startLap}
                    onChange={(e) =>
                      onChange({ ...config, startLap: parseInt(e.target.value, 10) || 1 })
                    }
                  />
                </div>
                <div className="input-field-inline">
                  <label htmlFor="input-range-end">End Lap:</label>
                  <input
                    id="input-range-end"
                    type="number"
                    className="config-input-number mono"
                    min={1}
                    max={totalLaps}
                    value={endLap}
                    onChange={(e) =>
                      onChange({ ...config, endLap: parseInt(e.target.value, 10) || 1 })
                    }
                  />
                </div>
                <span className="field-hint">Bounds: [1, {totalLaps}]</span>
              </div>
            )}

            {lapMode === 'fastest' && (
              <div className="mode-description-hint">
                Compares the single fastest valid lap completed by each selected driver prior to the cutoff.
              </div>
            )}

            {lapMode === 'stint' && (
              <div className="input-field-inline">
                <label htmlFor="input-stint-number">Stint Number:</label>
                <input
                  id="input-stint-number"
                  type="number"
                  className="config-input-number mono"
                  min={1}
                  max={10}
                  value={stintNumber}
                  onChange={(e) =>
                    onChange({ ...config, stintNumber: parseInt(e.target.value, 10) || 1 })
                  }
                />
                <span className="field-hint">1-indexed race stint</span>
              </div>
            )}

            {lapMode === 'last_clean' && (
              <div className="input-field-inline">
                <label htmlFor="input-last-n-clean">Laps (N):</label>
                <input
                  id="input-last-n-clean"
                  type="number"
                  className="config-input-number mono"
                  min={1}
                  max={30}
                  value={lastNClean}
                  onChange={(e) =>
                    onChange({ ...config, lastNClean: parseInt(e.target.value, 10) || 1 })
                  }
                />
                <span className="field-hint">most recent clean laps</span>
              </div>
            )}
          </div>
        </div>

        {/* SECTION 3: Synchronization Axis */}
        <div className="config-section" id="section-sync-mode">
          <label className="section-label">3. Synchronization Axis</label>
          <div className="mode-pills-group" role="radiogroup" aria-label="Synchronization axis">
            <button
              type="button"
              className={`mode-pill ${syncMode === 'lap' ? 'active' : ''}`}
              onClick={() => onChange({ ...config, syncMode: 'lap' })}
            >
              Race Lap Number
            </button>
            <button
              type="button"
              className={`mode-pill ${syncMode === 'tyre_age' ? 'active' : ''}`}
              onClick={() => onChange({ ...config, syncMode: 'tyre_age' })}
            >
              Tyre Age (Degradation)
            </button>
          </div>
          <div className="mode-description-hint">
            {syncMode === 'lap'
              ? 'Alignd by authoritative race lap number. Missing reference observations yield null deltas.'
              : 'Aligned by tyre age within matching stint lifecycles. Original race lap numbers are strictly preserved.'}
          </div>
        </div>

        {/* SECTION 4: Modifiers & Point-in-Time Cutoff */}
        <div className="config-section" id="section-modifiers-cutoff">
          <label className="section-label">4. Analysis Modifiers & Point-in-Time Barrier</label>
          <div className="modifiers-checkboxes-row">
            <label className="checkbox-label" title="Exclude in/out laps, cautions, safety car, and deleted laps">
              <input
                type="checkbox"
                checked={cleanOnly}
                onChange={(e) => onChange({ ...config, cleanOnly: e.target.checked })}
              />
              <ShieldCheck size={14} className="icon-subtle" />
              <span>Clean Laps Only</span>
            </label>

            <label className="checkbox-label" title="Apply standard linear fuel burn pace benefit (+0.033 s/lap)">
              <input
                type="checkbox"
                checked={fuelCorrected}
                onChange={(e) => onChange({ ...config, fuelCorrected: e.target.checked })}
              />
              <Fuel size={14} className="icon-subtle" />
              <span>Fuel Corrected (+0.033s/lap)</span>
            </label>
          </div>

          <div className="cutoff-selector-row">
            <span className="sublabel">Analysis Boundary:</span>
            <div className="mode-pills-group" role="radiogroup" aria-label="Point-in-time analysis boundary">
              <button
                type="button"
                className={`mode-pill compact ${cutoffMode === 'replay' ? 'active' : ''}`}
                onClick={() => onChange({ ...config, cutoffMode: 'replay' })}
                title="Strictly bound by the active replay clock (prevents future lap leakage)"
              >
                <Clock size={12} style={{ marginRight: '4px' }} />
                At Replay Time (Lap {currentReplayLap})
              </button>
              <button
                type="button"
                className={`mode-pill compact ${cutoffMode === 'full' ? 'active' : ''}`}
                onClick={() => onChange({ ...config, cutoffMode: 'full' })}
                title="Analyze across all completed Grand Prix laps up to race finish"
              >
                Full Race
              </button>
            </div>
          </div>
          <div className="field-hint" style={{ marginTop: '6px' }}>
            {cutoffMode === 'replay'
              ? `Effective cutoff: t_as_of = ${currentReplayTimeS.toFixed(2)}s (Leader Lap ${currentReplayLap})`
              : `Effective cutoff: Full Race (T_finish = ${tFinishS.toFixed(2)}s)`}
          </div>
        </div>
      </div>

      {/* Formulated Request Summary */}
      <div className="formulated-summary-card">
        <div className="summary-header">
          <span className="summary-title mono">Comparison Request Summary</span>
          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {isConfigDirty && lastResponse && (
              <span className="status-pill status-modified" id="pill-config-modified">
                CONFIG MODIFIED
              </span>
            )}
            <span
              className={`status-pill ${
                validationErrors.length === 0 ? 'status-ready' : 'status-pending'
              }`}
            >
              {validationErrors.length === 0 ? 'VALID & READY' : 'CONFIGURATION INCOMPLETE'}
            </span>
          </div>
        </div>
        <div className="summary-code mono">
          POST /api/races/.../compare — drivers: [{formulatedRequest.drivers.join(', ')}], ref:{' '}
          {formulatedRequest.reference_driver || 'NONE'}, mode: {formulatedRequest.lap_mode}, sync:{' '}
          {formulatedRequest.sync_mode}, as_of:{' '}
          {cutoffMode === 'replay'
            ? `${currentReplayTimeS.toFixed(2)}s (Active Replay)`
            : `${tFinishS.toFixed(2)}s (Full Race)`}
        </div>
      </div>
    </div>
  );
};
