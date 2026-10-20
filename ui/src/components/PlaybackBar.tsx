import React from 'react';
import { ClockStatus } from '../types/replay';
import { Play, Pause, SkipBack, SkipForward } from 'lucide-react';
import { Scrubber } from './Scrubber';

interface PlaybackBarProps {
  clock: ClockStatus | null;
  onPlay: () => void;
  onPause: () => void;
  onSeek: (targetTimeS: number) => void;
  onStepLap: (direction: 1 | -1) => void;
  onSetSpeed: (speed: number) => void;
}

const SPEED_PRESETS = [0.5, 1.0, 2.0, 5.0, 10.0];

export const PlaybackBar: React.FC<PlaybackBarProps> = ({
  clock,
  onPlay,
  onPause,
  onSeek,
  onStepLap,
  onSetSpeed,
}) => {
  const isPlaying = clock?.state === 'PLAYING';
  const isFinished = clock?.state === 'FINISHED';

  const formatElapsed = (sec: number, t0: number) => {
    const elapsed = Math.max(0, sec - t0);
    const h = Math.floor(elapsed / 3600);
    const m = Math.floor((elapsed % 3600) / 60);
    const s = Math.floor(elapsed % 60);
    const ms = Math.floor((elapsed % 1) * 1000);

    const hStr = h > 0 ? `${h.toString().padStart(2, '0')}:` : '';
    const mStr = m.toString().padStart(2, '0');
    const sStr = s.toString().padStart(2, '0');
    const msStr = ms.toString().padStart(3, '0');

    return `${hStr}${mStr}:${sStr}.${msStr}`;
  };

  return (
    <div className="playback-bar-container" id="playback-bar">
      <Scrubber clock={clock} onSeek={onSeek} onPlay={onPlay} onPause={onPause} />

      <div className="playback-controls-row">
        <div className="controls-left">
          <button
            id="btn-play-pause"
            className={`btn-control ${isPlaying ? 'btn-primary' : ''}`}
            onClick={isPlaying ? onPause : onPlay}
            disabled={isFinished}
            title={isPlaying ? 'Pause Replay' : 'Start Replay'}
          >
            {isPlaying ? <Pause size={16} /> : <Play size={16} />}
            <span>{isPlaying ? 'PAUSE' : 'PLAY'}</span>
          </button>

          <button
            id="btn-step-lap-prev"
            className="btn-control"
            onClick={() => onStepLap(-1)}
            title="Step to Previous Lap Start"
          >
            <SkipBack size={15} />
            <span>-1 LAP</span>
          </button>

          <button
            id="btn-step-lap-next"
            className="btn-control"
            onClick={() => onStepLap(1)}
            disabled={isFinished}
            title="Step to Next Lap Start"
          >
            <span>+1 LAP</span>
            <SkipForward size={15} />
          </button>
        </div>

        <div className="controls-center">
          <div className="timecode-display" id="timecode-display">
            <span>{clock ? formatElapsed(clock.session_time_s, clock.T_0) : '00:00.000'}</span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginLeft: '8px' }}>
              ({clock ? clock.session_time_s.toFixed(2) : '0.00'}s)
            </span>
          </div>
        </div>

        <div className="controls-right" id="speed-controls">
          <span style={{ fontSize: '11px', color: 'var(--text-muted)', marginRight: '4px' }}>SPEED:</span>
          {SPEED_PRESETS.map((spd) => (
            <button
              key={spd}
              id={`btn-speed-${spd}x`}
              className={`speed-btn ${clock?.speed === spd ? 'active' : ''}`}
              onClick={() => onSetSpeed(spd)}
            >
              {spd}x
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
