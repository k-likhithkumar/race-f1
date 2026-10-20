import React, { useState, useEffect, useRef } from 'react';
import {
  AssistantStatusResponse,
  AssistantTurn,
  EvidenceReference,
} from '../../types/assistant';
import { AssistantMessageCard } from './AssistantMessageCard';
import {
  Sparkles,
  X,
  Send,
  Trash2,
  AlertCircle,
  Clock,
  ArrowRight,
} from 'lucide-react';

interface AIAssistantDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  statusData: AssistantStatusResponse | null;
  statusLoading: boolean;
  statusError: string | null;
  turns: AssistantTurn[];
  isSubmitting: boolean;
  queryError: string | null;
  onSubmitQuery: (prompt: string, overrideAsOf?: number) => Promise<void>;
  onClearConversation: () => void;
  onNavigateEvidence: (evidence: EvidenceReference) => void;
  currentReplayTimeS: number;
  currentReplayLap: number;
}

const SUGGESTED_PROMPTS = [
  {
    title: 'Race Leaderboard',
    prompt: 'Who is leading the race as of now?',
    desc: 'Point-in-time order, gaps, and track status',
  },
  {
    title: 'Driver Pace Comparison',
    prompt: 'Compare lap times between VER and LEC.',
    desc: 'Lap deltas, telemetry reference pace',
  },
  {
    title: 'Tyre Degradation',
    prompt: "What is VER's tyre degradation on stint 1?",
    desc: 'Linear wear slope and fit quality',
  },
  {
    title: 'Pit Stop Waterfall',
    prompt: 'How much time did ALB lose during pit stops?',
    desc: 'Stationary assumption vs measured pit delta',
  },
  {
    title: 'Battle Prediction',
    prompt: 'What are the battle and overtake chances between VER and SAI?',
    desc: '5-lap probabilistic battle estimate',
  },
];

export const AIAssistantDrawer: React.FC<AIAssistantDrawerProps> = ({
  isOpen,
  onClose,
  statusData,
  statusLoading,
  statusError,
  turns,
  isSubmitting,
  queryError,
  onSubmitQuery,
  onClearConversation,
  onNavigateEvidence,
  currentReplayTimeS,
  currentReplayLap,
}) => {
  const [inputPrompt, setInputPrompt] = useState<string>('');
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  // Auto-focus input when opened
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 100);
    }
  }, [isOpen]);

  // Close on ESC key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Scroll to bottom when turns change
  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [turns, isOpen]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = inputPrompt.trim();
    if (!trimmed || isSubmitting) return;

    onSubmitQuery(trimmed);
    setInputPrompt('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSelectSuggested = (prompt: string) => {
    setInputPrompt(prompt);
    inputRef.current?.focus();
  };

  if (!isOpen) return null;

  const isAvailable = statusData?.available === true;
  const modelName = statusData?.model || (statusLoading ? 'Checking...' : 'Unavailable');

  return (
    <div
      className="assistant-drawer-overlay"
      id="assistant-drawer-overlay"
      onClick={onClose}
    >
      <div
        className="assistant-drawer"
        id="assistant-drawer-container"
        role="dialog"
        aria-label="AI Strategy Assistant"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Drawer Header */}
        <div className="assistant-drawer-header">
          <div className="assistant-header-title">
            <div className="bot-icon-badge">
              <Sparkles size={16} />
            </div>
            <div>
              <div className="title-row">
                <h3>AI Strategy Assistant</h3>
                <span className={`status-pill ${isAvailable ? 'status-green' : statusData?.status_reason === 'quota_exhausted' ? 'status-orange' : statusData?.status_reason === 'provider_error' || statusData?.status_reason === 'invalid_api_key' ? 'status-red' : 'status-yellow'}`} id="assistant-status-pill">
                  {statusLoading
                    ? 'Checking...'
                    : isAvailable
                    ? 'Online'
                    : statusData?.status_reason === 'quota_exhausted'
                    ? 'Quota Exhausted'
                    : statusData?.status_reason === 'provider_error'
                    ? 'Provider Error'
                    : statusData?.status_reason === 'invalid_api_key'
                    ? 'Auth Failed'
                    : 'API Key Missing'}
                </span>
              </div>
              <span className="model-id mono text-muted">Model: {modelName}</span>
            </div>
          </div>

          <div className="assistant-header-actions">
            {turns.length > 0 && (
              <button
                type="button"
                className="header-action-btn"
                id="btn-clear-chat"
                onClick={onClearConversation}
                title="Clear conversation history"
                aria-label="Clear conversation history"
              >
                <Trash2 size={15} />
              </button>
            )}
            <button
              type="button"
              className="header-action-btn btn-close"
              id="btn-close-assistant"
              onClick={onClose}
              title="Close Assistant (Esc)"
              aria-label="Close Assistant"
            >
              <X size={16} />
            </button>
          </div>
        </div>

        {/* Missing API Key / Unavailable Banner */}
        {(statusError || (!isAvailable && !statusLoading)) && (
          <div className="assistant-unavailable-banner" id="assistant-key-notice">
            <AlertCircle size={15} className="banner-icon" />
            <div className="banner-text">
              {statusError ? (
                <><strong>Status Check Error:</strong> {statusError}</>
              ) : statusData?.status_reason === 'quota_exhausted' ? (
                <><strong>API Quota Exhausted:</strong> {statusData?.message || 'Daily free-tier request quota limit reached (HTTP 429). Deterministic analytics remain fully active.'}</>
              ) : statusData?.status_reason === 'invalid_api_key' ? (
                <><strong>Authentication Failed:</strong> {statusData?.message || 'The configured API key was rejected by the provider. Please check your credentials.'}</>
              ) : statusData?.status_reason === 'provider_error' ? (
                <><strong>Provider Initialization Error:</strong> {statusData?.message || 'Could not initialize AI provider. Verify network connectivity and model settings.'}</>
              ) : (
                <><strong>{statusData?.provider === 'nvidia' ? 'NVIDIA_API_KEY Not Configured' : 'GEMINI_API_KEY Not Configured'}:</strong> {statusData?.message || 'External LLM queries require an API key in .env. Responses will return structured availability notices.'}</>
              )}
            </div>
          </div>
        )}

        {/* Message Feed / Suggested Prompts */}
        <div className="assistant-messages-feed" id="assistant-messages-feed">
          {turns.length === 0 ? (
            <div className="assistant-empty-state">
              <div className="empty-state-hero">
                <div className="empty-sparkle-circle">
                  <Sparkles size={24} />
                </div>
                <h4>Grounded F1 Race Intelligence</h4>
                <p>
                  Ask strategy questions anchored to verified deterministic analytics. Claims are bounded by the authoritative replay cutoff.
                </p>
              </div>

              <div className="suggested-prompts-section">
                <span className="suggested-label">Suggested Inquiries:</span>
                <div className="suggested-prompts-grid">
                  {SUGGESTED_PROMPTS.map((sp, idx) => (
                    <button
                      key={idx}
                      type="button"
                      className="suggested-prompt-card"
                      id={`suggested-prompt-${idx}`}
                      onClick={() => handleSelectSuggested(sp.prompt)}
                    >
                      <div className="suggested-card-header">
                        <span className="suggested-card-title">{sp.title}</span>
                        <ArrowRight size={13} className="suggested-arrow" />
                      </div>
                      <span className="suggested-card-prompt">"{sp.prompt}"</span>
                      <span className="suggested-card-desc text-muted">{sp.desc}</span>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            turns.map((turn) => (
              <AssistantMessageCard
                key={turn.id}
                turn={turn}
                onNavigateEvidence={onNavigateEvidence}
                onRetry={turn.error ? () => onSubmitQuery(turn.content) : undefined}
              />
            ))
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div className="assistant-input-container">
          {queryError && (
            <div className="assistant-query-error-bar" role="alert">
              <AlertCircle size={13} />
              <span>{queryError}</span>
            </div>
          )}

          <div className="input-meta-bar">
            <div className="asof-context-pill mono" title="Authoritative replay timestamp used for this inquiry">
              <Clock size={11} />
              <span>Cutoff: t = {currentReplayTimeS.toFixed(1)}s (Lap {currentReplayLap})</span>
            </div>
            <div className="char-count text-muted mono">
              {inputPrompt.length}/1000
            </div>
          </div>

          <form onSubmit={handleSubmit} className="assistant-input-form">
            <textarea
              ref={inputRef}
              id="assistant-prompt-input"
              className="assistant-textarea"
              placeholder="Ask an F1 strategy question (e.g. tyre wear, pit loss, gap battles)..."
              value={inputPrompt}
              maxLength={1000}
              onChange={(e) => setInputPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={2}
              disabled={isSubmitting}
              aria-label="Assistant inquiry prompt"
            />
            <button
              type="submit"
              id="assistant-submit-btn"
              className="assistant-send-btn"
              disabled={!inputPrompt.trim() || isSubmitting}
              title="Submit Inquiry (Enter)"
              aria-label="Submit inquiry"
            >
              <Send size={15} />
            </button>
          </form>
          <div className="input-keyboard-hint text-muted">
            Press <strong>Enter</strong> to send, <strong>Shift+Enter</strong> for newline
          </div>
        </div>
      </div>
    </div>
  );
};
