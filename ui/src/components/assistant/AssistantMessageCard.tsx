import React from 'react';
import { AssistantTurn, EvidenceReference } from '../../types/assistant';
import { EvidenceChip } from './EvidenceChip';
import { KeyNumberBadge } from './KeyNumberBadge';
import {
  Sparkles,
  User,
  AlertTriangle,
  Info,
  ShieldCheck,
  RotateCw,
  Cpu,
} from 'lucide-react';

interface AssistantMessageCardProps {
  turn: AssistantTurn;
  onNavigateEvidence: (evidence: EvidenceReference) => void;
  onRetry?: () => void;
}

export const AssistantMessageCard: React.FC<AssistantMessageCardProps> = ({
  turn,
  onNavigateEvidence,
  onRetry,
}) => {
  const isUser = turn.role === 'user';

  if (isUser) {
    return (
      <div className="assistant-turn user-turn" id={`turn-${turn.id}`}>
        <div className="turn-meta">
          <div className="turn-author">
            <User size={13} />
            <span>You</span>
          </div>
          <div className="turn-time mono">
            {turn.asOf !== undefined && (
              <span className="turn-asof">t = {turn.asOf.toFixed(1)}s</span>
            )}
            <span>{turn.timestamp}</span>
          </div>
        </div>
        <div className="user-message-bubble">
          {turn.content}
        </div>
      </div>
    );
  }

  // Assistant turn
  const resp = turn.response;
  const confidence = resp?.confidence;
  const status = resp?.status;

  const getConfidenceBadge = () => {
    if (!confidence) return null;
    switch (confidence) {
      case 'HIGH':
        return <span className="confidence-pill conf-high"><ShieldCheck size={11} /> High Grounding</span>;
      case 'MEDIUM':
        return <span className="confidence-pill conf-med"><Info size={11} /> Medium Grounding</span>;
      case 'LOW':
        return <span className="confidence-pill conf-low"><AlertTriangle size={11} /> Low Grounding</span>;
      case 'CANNOT_TELL':
        return <span className="confidence-pill conf-cannot-tell"><Info size={11} /> Cannot Tell</span>;
      case 'UNAVAILABLE':
        return <span className="confidence-pill conf-unavailable"><AlertTriangle size={11} /> Unavailable</span>;
      default:
        return null;
    }
  };

  const getStatusBadge = () => {
    if (!status || status === 'SUCCESS') return null;
    return (
      <span className={`status-pill status-pill-${status.toLowerCase()}`}>
        {status.replace(/_/g, ' ')}
      </span>
    );
  };

  return (
    <div className="assistant-turn bot-turn" id={`turn-${turn.id}`}>
      <div className="turn-meta">
        <div className="turn-author">
          <Sparkles size={13} className="bot-sparkle-icon" />
          <span>Strategy Assistant</span>
          {getConfidenceBadge()}
          {getStatusBadge()}
        </div>
        <div className="turn-time mono">
          {resp?.as_of !== undefined && (
            <span className="turn-asof">t = {resp.as_of.toFixed(1)}s</span>
          )}
          <span>{turn.timestamp}</span>
        </div>
      </div>

      <div className="bot-message-card">
        {turn.isLoading ? (
          <div className="bot-loading-indicator" aria-live="polite">
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-dot" />
            <span className="loading-text">Analyzing deterministic race telemetry...</span>
          </div>
        ) : (
          <>
            {/* Answer Text */}
            <div className="bot-answer-text">
              {turn.content || resp?.answer || 'No answer generated.'}
            </div>

            {/* Key Numbers Grid */}
            {resp?.key_numbers && resp.key_numbers.length > 0 && (
              <div className="key-numbers-section">
                <div className="section-label">Key Metrics</div>
                <div className="key-numbers-grid">
                  {resp.key_numbers.map((kn, idx) => (
                    <KeyNumberBadge key={`${kn.label}-${idx}`} keyNumber={kn} />
                  ))}
                </div>
              </div>
            )}

            {/* Grounding Evidence References */}
            {resp?.evidence && resp.evidence.length > 0 && (
              <div className="evidence-section">
                <div className="section-label">
                  <ShieldCheck size={12} style={{ color: 'var(--flag-green)' }} />
                  <span>Verified Deterministic Evidence</span>
                </div>
                <div className="evidence-chips-wrap">
                  {resp.evidence.map((ev) => (
                    <EvidenceChip
                      key={ev.id}
                      evidence={ev}
                      onNavigate={onNavigateEvidence}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Limitations and Uncertainty Notices */}
            {resp?.limitations_notice && (
              <div className="limitations-alert" role="note">
                <AlertTriangle size={14} className="limitations-icon" />
                <div className="limitations-content">
                  <span className="limitations-title">Reliability Notice:</span>{' '}
                  <span className="limitations-text">{resp.limitations_notice}</span>
                </div>
              </div>
            )}

            {/* Tool Execution Tags */}
            {resp?.tool_calls_executed && resp.tool_calls_executed.length > 0 && (
              <div className="tool-calls-meta">
                <Cpu size={11} />
                <span>Deterministic tools:</span>
                {resp.tool_calls_executed.map((tc, idx) => (
                  <span key={idx} className="tool-tag mono">{tc}</span>
                ))}
              </div>
            )}

            {/* Turn Error State */}
            {turn.error && (
              <div className="turn-error-alert" role="alert">
                <AlertTriangle size={14} />
                <span>{turn.error}</span>
                {onRetry && (
                  <button type="button" className="retry-btn" onClick={onRetry}>
                    <RotateCw size={12} /> Retry
                  </button>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};
