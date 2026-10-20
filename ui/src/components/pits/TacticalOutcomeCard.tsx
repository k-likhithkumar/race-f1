import React from 'react';
import { TacticalEvidence } from '../../types/pits';
import { Crosshair, Shield, AlertCircle, TrendingUp, HelpCircle } from 'lucide-react';

interface TacticalOutcomeCardProps {
  tacticalEvidence?: TacticalEvidence | null;
  stopNumber: number;
  driver: string;
}

export const TacticalOutcomeCard: React.FC<TacticalOutcomeCardProps> = ({
  tacticalEvidence,
  stopNumber,
  driver,
}) => {
  if (!tacticalEvidence) {
    return null;
  }

  const { classification, rival_driver, pre_stop_gap_s, lap_delta, outcome, evidence_summary } =
    tacticalEvidence;

  const isUndercut = classification === 'UNDERCUT_ATTEMPT';
  const isOvercut = classification === 'OVERCUT_ATTEMPT';
  const isSafetyCar = classification === 'SAFETY_CAR_ADVANTAGE';
  const isRegular = classification === 'REGULAR_STOP';
  const isUnknown = classification === 'UNKNOWN';

  return (
    <div className="tactical-evidence-card" id={`tactical-evidence-stop-${stopNumber}`}>
      <div className="tactical-card-header">
        <div className="flex items-center gap-2">
          {isUndercut && <TrendingUp size={16} className="text-purple" />}
          {isOvercut && <Crosshair size={16} className="text-orange" />}
          {isSafetyCar && <Shield size={16} className="text-yellow" />}
          {isRegular && <CheckBadgeIcon size={16} />}
          {isUnknown && <HelpCircle size={16} className="text-muted" />}
          <span className="tactical-title">{driver} Stop #{stopNumber} Tactical Classification</span>
        </div>

        <span className="heuristic-tag">Heuristic Inference</span>
      </div>

      <div className="tactical-main-verdict">
        <span className={`verdict-badge ${classification.toLowerCase()}`}>
          {classification.replace(/_/g, ' ')}
        </span>
        {outcome && outcome !== 'INSUFFICIENT_DATA' && (
          <span className={`outcome-badge ${outcome === 'GAINED_POSITION' ? 'gained' : 'lost'}`}>
            {outcome.replace(/_/g, ' ')}
          </span>
        )}
      </div>

      {/* Supporting Evidence Metrics */}
      {(rival_driver || pre_stop_gap_s !== null || lap_delta !== null) && (
        <div className="tactical-evidence-grid">
          {rival_driver && (
            <div className="evidence-item">
              <span className="evidence-label">Immediate Rival</span>
              <span className="evidence-val font-mono font-bold text-cyan">{rival_driver}</span>
            </div>
          )}

          {pre_stop_gap_s !== null && pre_stop_gap_s !== undefined && (
            <div className="evidence-item">
              <span className="evidence-label">Pre-Stop Gap</span>
              <span className="evidence-val font-mono">{pre_stop_gap_s.toFixed(2)}s</span>
            </div>
          )}

          {lap_delta !== null && lap_delta !== undefined && (
            <div className="evidence-item">
              <span className="evidence-label">Pit Lap Delta</span>
              <span className="evidence-val font-mono">
                {lap_delta > 0 ? `+${lap_delta} laps` : `${lap_delta} laps`}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Grounded Summary Text */}
      <div className="tactical-summary-text font-mono">
        {evidence_summary || 'No detailed evidence recorded.'}
      </div>

      {/* Uncertainty & Limitations Disclaimer */}
      <div className="tactical-disclaimer">
        <AlertCircle size={12} className="inline mr-1" />
        <strong>Limitation Notice:</strong> Undercut and overcut classifications are heuristic inferences
        derived from pre-stop track proximity (gap ≤ 3.0s) and relative pit timing. They do not constitute
        direct telemetry evidence of team radio strategy intent.
      </div>
    </div>
  );
};

const CheckBadgeIcon: React.FC<{ size: number }> = ({ size }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
    <polyline points="22 4 12 14.01 9 11.01" />
  </svg>
);
