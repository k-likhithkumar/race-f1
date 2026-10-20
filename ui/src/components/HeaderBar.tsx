import React from 'react';
import { RaceSummary, ClockStatus, RaceState } from '../types/replay';
import { Flag, CloudRain, Thermometer, ShieldAlert, Sparkles } from 'lucide-react';

interface HeaderBarProps {
  races: RaceSummary[];
  selectedRaceId: string;
  onSelectRace: (raceId: string) => void;
  clock: ClockStatus | null;
  raceState: RaceState | null;
  isAssistantOpen?: boolean;
  onToggleAssistant?: () => void;
  assistantAvailable?: boolean;
}

export const HeaderBar: React.FC<HeaderBarProps> = ({
  races,
  selectedRaceId,
  onSelectRace,
  clock,
  raceState,
  isAssistantOpen,
  onToggleAssistant,
  assistantAvailable,
}) => {
  const trackStatus = raceState?.track_status || 'UNKNOWN';

  const getStatusBadge = () => {
    switch (trackStatus) {
      case '1':
        return <span className="track-status-pill status-green"><Flag size={14} /> Track Clear (Green)</span>;
      case '2':
        return <span className="track-status-pill status-yellow"><Flag size={14} /> Yellow Flag</span>;
      case '4':
        return <span className="track-status-pill status-sc"><ShieldAlert size={14} /> Safety Car</span>;
      case '5':
        return <span className="track-status-pill status-red"><Flag size={14} /> Red Flag</span>;
      case '6':
        return <span className="track-status-pill status-sc"><ShieldAlert size={14} /> VSC Deployed</span>;
      case '7':
        return <span className="track-status-pill status-yellow"><Flag size={14} /> VSC Ending</span>;
      case 'PRE_RACE':
        return <span className="track-status-pill status-unknown">Pre-Race</span>;
      default:
        return <span className="track-status-pill status-unknown">Status: {trackStatus}</span>;
    }
  };

  const weather = raceState?.weather;

  return (
    <header className="header-bar" id="f1-header-bar">
      <div className="header-left">
        <span className="brand-badge">RACE F1</span>
        <div className="race-selector">
          <select
            id="race-select-dropdown"
            value={selectedRaceId}
            onChange={(e) => onSelectRace(e.target.value)}
          >
            {races.map((r) => (
              <option key={r.race_id} value={r.race_id}>
                {r.year} {r.gp} - {r.official_event_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="header-center">
        <div className="lap-pill" id="leader-lap-pill">
          <span>LAP</span>
          <span className="current mono">{clock ? clock.current_lap : 1}</span>
          <span className="text-muted">/ {clock ? clock.total_laps : 53}</span>
        </div>

        <div id="track-status-container">
          {getStatusBadge()}
        </div>
      </div>

      <div className="header-right">
        {weather && (
          <div className="weather-badge mono" id="weather-badge">
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
              <Thermometer size={13} />
              {weather.air_temp_c !== null ? `${weather.air_temp_c.toFixed(1)}°C` : '-'}
            </span>
            <span style={{ marginLeft: '8px', color: 'var(--text-muted)' }}>
              Track: {weather.track_temp_c !== null ? `${weather.track_temp_c.toFixed(1)}°C` : '-'}
            </span>
            {weather.rainfall && (
              <span style={{ marginLeft: '8px', color: '#0091ff', display: 'inline-flex', alignItems: 'center', gap: '2px' }}>
                <CloudRain size={13} /> Wet
              </span>
            )}
          </div>
        )}

        {onToggleAssistant && (
          <button
            type="button"
            id="btn-toggle-assistant"
            className={`assistant-toggle-btn ${isAssistantOpen ? 'active' : ''}`}
            onClick={onToggleAssistant}
            title="Toggle AI Strategy Assistant"
            aria-label="Toggle AI Strategy Assistant"
          >
            <Sparkles size={14} className="sparkle-icon" />
            <span>AI Assistant</span>
            {assistantAvailable !== undefined && (
              <span
                className={`assistant-status-dot ${assistantAvailable ? 'dot-online' : 'dot-offline'}`}
                title={assistantAvailable ? 'Gemini API Connected' : 'API Key Missing'}
              />
            )}
          </button>
        )}
      </div>
    </header>
  );
};
