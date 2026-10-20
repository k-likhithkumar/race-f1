import React from 'react';
import {
  Zap,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  Info,
  Clock,
  RotateCcw,
  CheckCircle2,
} from 'lucide-react';
import type {
  BattleOvertakePredictionResponse,
  ModelReliabilityWarning,
} from '../../types/battles';
import { getTeamColor } from '../../constants/teams';

interface BattleOvertakePredictionCardProps {
  prediction: BattleOvertakePredictionResponse | null;
  isLoading: boolean;
  error: string | null;
  driver1: string;
  driver2: string;
  driverTeamMap?: Record<string, string>;
  onRetry?: () => void;
}

export const BattleOvertakePredictionCard: React.FC<BattleOvertakePredictionCardProps> = ({
  prediction,
  isLoading,
  error,
  driver1,
  driver2,
  driverTeamMap = {},
  onRetry,
}) => {
  const d1Color = getTeamColor(driverTeamMap[driver1]);
  const d2Color = getTeamColor(driverTeamMap[driver2]);

  // 1. Error State
  if (error) {
    return (
      <div className="prediction-card-container state-error" id="prediction-card-error">
        <div className="prediction-state-content">
          <AlertCircle className="state-icon text-danger" size={24} />
          <div className="state-text">
            <h4>Prediction Request Error</h4>
            <p>{error}</p>
          </div>
          {onRetry && (
            <button
              id="btn-retry-prediction"
              onClick={onRetry}
              className="f1-btn btn-retry"
              title="Retry prediction query"
            >
              <RotateCcw size={14} className="mr-1" /> Retry
            </button>
          )}
        </div>
      </div>
    );
  }

  // 2. Loading State (when no prior data exists)
  if (isLoading && !prediction) {
    return (
      <div className="prediction-card-container state-loading" id="prediction-card-loading">
        <div className="prediction-state-content">
          <Clock className="state-icon animate-spin text-accent" size={24} />
          <div className="state-text">
            <h4>Computing Machine Learning Forecast...</h4>
            <p>Evaluating point-in-time closing dynamics and tyre features for {driver1} vs {driver2}</p>
          </div>
        </div>
      </div>
    );
  }

  // 3. Pre-Race State
  if (prediction && prediction.status === 'PRE_RACE') {
    return (
      <div className="prediction-card-container state-info" id="prediction-card-pre-race">
        <div className="prediction-state-content">
          <Info className="state-icon text-info" size={24} />
          <div className="state-text">
            <h4>Pre-Race Standby</h4>
            <p>
              {prediction.message ||
                'Simulation timestamp is prior to the official race start (T0). Advance replay into racing laps to compute battle & overtake probabilities.'}
            </p>
          </div>
        </div>
      </div>
    );
  }

  // 4. Model Unavailable State
  if (prediction && prediction.status === 'MODEL_UNAVAILABLE') {
    return (
      <div className="prediction-card-container state-warning" id="prediction-card-unavailable">
        <div className="prediction-state-content">
          <AlertCircle className="state-icon text-warning" size={24} />
          <div className="state-text">
            <h4>Prediction Models Unavailable</h4>
            <p>
              {prediction.message ||
                'Pre-trained battle and overtake logistic regression pipelines are offline or unavailable.'}
            </p>
          </div>
          {onRetry && (
            <button onClick={onRetry} className="f1-btn btn-retry">
              <RotateCcw size={14} className="mr-1" /> Check Models
            </button>
          )}
        </div>
      </div>
    );
  }

  // 5. Insufficient Data State
  if (prediction && prediction.status === 'INSUFFICIENT_DATA') {
    const features = prediction.supporting_features;
    return (
      <div className="prediction-card-container state-insufficient" id="prediction-card-insufficient">
        <div className="prediction-header">
          <div className="prediction-title-group">
            <Zap size={18} className="text-warning mr-2" />
            <span className="prediction-title">5-Lap Horizon Machine Learning Forecast</span>
            <span className="badge badge-insufficient ml-2">Insufficient Data</span>
          </div>
          <div className="prediction-meta">
            <span>Pair: {driver1} ↔ {driver2}</span>
            {prediction.as_of_lap && <span> | Lap {prediction.as_of_lap}</span>}
          </div>
        </div>
        <div className="insufficient-message-row">
          <Info size={16} className="text-muted mr-2 flex-shrink-0" />
          <p>
            {prediction.message ||
              'Prediction requires a proximity gap ≤ 10.0s and at least 2 completed race laps for both drivers.'}
          </p>
        </div>
        {features && features.gap_s !== null && (
          <div className="insufficient-features-row">
            <span className="feat-chip">
              Current Gap: <strong>{features.gap_s.toFixed(3)}s</strong>
            </span>
            {features.laps_remaining !== null && (
              <span className="feat-chip">
                Laps Remaining: <strong>{features.laps_remaining}</strong>
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  // 6. No Data / Idle fallback
  if (!prediction || prediction.status !== 'SUCCESS') {
    return null;
  }

  const {
    attacker,
    defender,
    as_of_lap,
    horizon_laps,
    battle_prediction,
    overtake_prediction,
    supporting_features,
    reliability_warnings,
  } = prediction;

  const attackerColor = attacker ? getTeamColor(driverTeamMap[attacker]) : d1Color;
  const defenderColor = defender ? getTeamColor(driverTeamMap[defender]) : d2Color;

  const battleProb = battle_prediction?.probability ?? null;
  const overtakeProb = overtake_prediction?.probability ?? null;

  return (
    <div className="prediction-card-container" id="battle-overtake-prediction-card">
      {/* A. Header Bar */}
      <div className="prediction-header">
        <div className="prediction-title-group">
          <Zap size={18} className="text-gold mr-2" />
          <h3 className="prediction-title">
            5-Lap Horizon Battle & Overtake ML Forecast
          </h3>
          <span className="horizon-pill">Horizon: +{horizon_laps} Laps</span>
          {isLoading && (
            <span className="updating-pill">
              <Clock size={11} className="animate-spin mr-1" /> Updating
            </span>
          )}
        </div>

        {/* Attacker vs Defender Identities */}
        <div className="driver-roles-group">
          <div className="role-chip attacker-role" id="role-attacker">
            <span className="role-label">ATTACKER</span>
            <span className="role-driver">
              <span className="team-indicator" style={{ backgroundColor: attackerColor }} />
              <strong>{attacker || 'N/A'}</strong>
            </span>
          </div>
          <span className="role-vs">vs</span>
          <div className="role-chip defender-role" id="role-defender">
            <span className="role-label">DEFENDER</span>
            <span className="role-driver">
              <span className="team-indicator" style={{ backgroundColor: defenderColor }} />
              <strong>{defender || 'N/A'}</strong>
            </span>
          </div>
          {as_of_lap && (
            <span className="as-of-lap-tag">As of Lap {as_of_lap}</span>
          )}
        </div>
      </div>

      {/* B. Dual Probability Gauges (Strictly Separate) */}
      <div className="prediction-gauges-grid">
        {/* Battle Probability Gauge */}
        <div className="probability-gauge-card battle-gauge" id="gauge-battle-probability">
          <div className="gauge-header">
            <div className="gauge-title-row">
              <span className="gauge-name">Future Battle Probability</span>
              <span className="gauge-condition">Proximity ≤ 1.0s within next {horizon_laps} laps</span>
            </div>
            <div className="gauge-value-display">
              <span className="gauge-pct mono">
                {battleProb !== null ? `${(battleProb * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Progress Bar Container */}
          <div className="gauge-track">
            <div
              className="gauge-fill gauge-fill-battle"
              style={{ width: `${Math.min(Math.max((battleProb || 0) * 100, 0), 100)}%` }}
            />
            {/* 50% Decision Threshold Line */}
            <div className="threshold-marker" style={{ left: '50%' }} title="0.50 Decision Threshold">
              <span className="threshold-line" />
              <span className="threshold-label">0.50</span>
            </div>
          </div>

          <div className="gauge-footer">
            <span
              className={`threshold-verdict ${
                battle_prediction?.classification_threshold_0_5 ? 'verdict-positive' : 'verdict-negative'
              }`}
            >
              {battle_prediction?.classification_threshold_0_5 ? (
                <>
                  <CheckCircle2 size={12} className="mr-1" /> Battle Predicted (≥ 0.50)
                </>
              ) : (
                'Under 0.50 Threshold'
              )}
            </span>
            <span className="gauge-interpretation">
              {battle_prediction?.interpretation || 'Probability estimate for 1.0s engagement.'}
            </span>
          </div>
        </div>

        {/* On-Track Overtake Probability Gauge */}
        <div className="probability-gauge-card overtake-gauge" id="gauge-overtake-probability">
          <div className="gauge-header">
            <div className="gauge-title-row">
              <span className="gauge-name">On-Track Overtake Probability</span>
              <span className="gauge-condition">Lead swap pass within next {horizon_laps} laps</span>
            </div>
            <div className="gauge-value-display">
              <span className="gauge-pct mono">
                {overtakeProb !== null ? `${(overtakeProb * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
          </div>

          {/* Progress Bar Container */}
          <div className="gauge-track">
            <div
              className="gauge-fill gauge-fill-overtake"
              style={{ width: `${Math.min(Math.max((overtakeProb || 0) * 100, 0), 100)}%` }}
            />
            {/* 50% Decision Threshold Line */}
            <div className="threshold-marker" style={{ left: '50%' }} title="0.50 Decision Threshold">
              <span className="threshold-line" />
              <span className="threshold-label">0.50</span>
            </div>
          </div>

          <div className="gauge-footer">
            <span
              className={`threshold-verdict ${
                overtake_prediction?.classification_threshold_0_5 ? 'verdict-positive' : 'verdict-negative'
              }`}
            >
              {overtake_prediction?.classification_threshold_0_5 ? (
                <>
                  <CheckCircle2 size={12} className="mr-1" /> Pass Predicted (≥ 0.50)
                </>
              ) : (
                'Under 0.50 Threshold'
              )}
            </span>
            <span className="gauge-interpretation">
              {overtake_prediction?.interpretation || 'Excludes pit swaps and caution passes.'}
            </span>
          </div>
        </div>
      </div>

      {/* C. Supporting Features Bar (State at as_of) */}
      {supporting_features && (
        <div className="prediction-features-panel" id="prediction-supporting-features">
          <div className="features-title">
            <span>Supporting Causal Features (as of Lap {as_of_lap || 'N/A'}):</span>
          </div>
          <div className="features-chips-grid">
            {/* Current Gap */}
            <div className="feat-chip-item">
              <span className="feat-chip-label">Current Gap:</span>
              <span className="feat-chip-val mono">
                {supporting_features.gap_s !== null ? `${supporting_features.gap_s.toFixed(3)}s` : 'N/A'}
              </span>
            </div>

            {/* Closing Rate */}
            <div className="feat-chip-item">
              <span className="feat-chip-label">Closing Rate:</span>
              <span className="feat-chip-val mono">
                {supporting_features.closing_rate_s_per_lap !== null ? (
                  <>
                    {supporting_features.is_closing ? (
                      <TrendingUp size={13} className="text-success inline-icon" />
                    ) : (
                      <TrendingDown size={13} className="text-danger inline-icon" />
                    )}
                    {Math.abs(supporting_features.closing_rate_s_per_lap).toFixed(3)} s/lap
                  </>
                ) : (
                  'N/A'
                )}
              </span>
            </div>

            {/* Clean Pace Difference */}
            <div className="feat-chip-item">
              <span className="feat-chip-label">Clean-Pace Delta:</span>
              <span className="feat-chip-val mono">
                {supporting_features.clean_pace_delta_s !== null ? (
                  <>
                    {supporting_features.clean_pace_delta_s < 0 ? (
                      <span className="text-success">
                        {Math.abs(supporting_features.clean_pace_delta_s).toFixed(3)}s faster
                      </span>
                    ) : (
                      <span className="text-danger">
                        +{supporting_features.clean_pace_delta_s.toFixed(3)}s slower
                      </span>
                    )}
                  </>
                ) : (
                  'Insufficient clean laps'
                )}
              </span>
            </div>

            {/* Tyre-Age Difference */}
            <div className="feat-chip-item">
              <span className="feat-chip-label">Tyre-Age Delta:</span>
              <span className="feat-chip-val mono">
                {supporting_features.tyre_age_delta_laps !== null ? (
                  <>
                    {supporting_features.tyre_age_delta_laps < 0 ? (
                      <span className="text-success">
                        {Math.abs(supporting_features.tyre_age_delta_laps)} laps fresher
                      </span>
                    ) : supporting_features.tyre_age_delta_laps > 0 ? (
                      <span className="text-warning">
                        +{supporting_features.tyre_age_delta_laps} laps older
                      </span>
                    ) : (
                      'Equal tyre age'
                    )}
                  </>
                ) : (
                  'Unknown'
                )}
              </span>
            </div>

            {/* Compounds */}
            {(supporting_features.attacker_compound || supporting_features.defender_compound) && (
              <div className="feat-chip-item">
                <span className="feat-chip-label">Compounds:</span>
                <span className="feat-chip-val">
                  {supporting_features.attacker_compound || 'UNK'} vs{' '}
                  {supporting_features.defender_compound || 'UNK'}
                </span>
              </div>
            )}

            {/* Laps Remaining */}
            {supporting_features.laps_remaining !== null && (
              <div className="feat-chip-item">
                <span className="feat-chip-label">Laps Remaining:</span>
                <span className="feat-chip-val mono">{supporting_features.laps_remaining}</span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* D. Model Reliability & Calibration Warnings */}
      {reliability_warnings.length > 0 && (
        <div className="prediction-warnings-container" id="prediction-reliability-warnings">
          {reliability_warnings.map((w: ModelReliabilityWarning, idx: number) => {
            const isOverconfidence = w.warning_type === 'CALIBRATION_OVERCONFIDENCE';
            return (
              <div
                key={idx}
                className={`warning-item ${isOverconfidence ? 'warning-overconfidence' : 'warning-general'}`}
              >
                <AlertCircle size={15} className="warning-icon flex-shrink-0" />
                <div className="warning-body">
                  <strong className="warning-tag">
                    {isOverconfidence ? 'Calibration Limitation' : w.warning_type}:
                  </strong>{' '}
                  <span>{w.message}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* E. Scientific Reliability Disclaimer */}
      <div className="prediction-disclaimer-note">
        <Info size={14} className="text-muted mr-2 flex-shrink-0" />
        <span>
          <strong>Scientific Methodology:</strong> Trained using causal regularized logistic regression evaluated via
          leave-one-race-out backtesting on Italian Grand Prix data. Probabilities are point-in-time statistical estimates;
          no confidence intervals are fabricated or implied. Battle and overtake events are evaluated as separate targets.
        </span>
      </div>
    </div>
  );
};
