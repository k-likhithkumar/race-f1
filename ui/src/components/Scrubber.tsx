import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ClockStatus } from '../types/replay';

interface ScrubberProps {
  clock: ClockStatus | null;
  onSeek: (targetTimeS: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
}

export const Scrubber: React.FC<ScrubberProps> = ({ clock, onSeek, onPlay, onPause }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [localTime, setLocalTime] = useState<number | null>(null);

  const isDraggingRef = useRef(false);
  const localTimeRef = useRef<number | null>(null);
  const wasPlayingRef = useRef(false);
  const sliderRef = useRef<HTMLInputElement | null>(null);

  const handleCommitDrag = useCallback((pointerId?: number) => {
    if (!isDraggingRef.current) return;
    isDraggingRef.current = false;
    setIsDragging(false);

    if (pointerId !== undefined && sliderRef.current) {
      try {
        sliderRef.current.releasePointerCapture(pointerId);
      } catch (_) {}
    }

    const targetTime = localTimeRef.current;
    localTimeRef.current = null;
    setLocalTime(null);

    if (targetTime !== null && !isNaN(targetTime)) {
      onSeek(targetTime);
    }

    if (wasPlayingRef.current) {
      wasPlayingRef.current = false;
      if (onPlay) {
        onPlay();
      }
    }
  }, [onSeek, onPlay]);

  useEffect(() => {
    const handleWindowPointerUp = () => {
      if (isDraggingRef.current) {
        handleCommitDrag();
      }
    };
    window.addEventListener('pointerup', handleWindowPointerUp);
    window.addEventListener('pointercancel', handleWindowPointerUp);
    return () => {
      window.removeEventListener('pointerup', handleWindowPointerUp);
      window.removeEventListener('pointercancel', handleWindowPointerUp);
    };
  }, [handleCommitDrag]);

  if (!clock) return null;

  const { session_time_s, T_0, T_finish } = clock;
  const isCurrentlyDragging = isDragging || isDraggingRef.current;
  const displayTime = isCurrentlyDragging && localTime !== null ? localTime : session_time_s;
  const progressPct = Math.max(0, Math.min(100, ((displayTime - T_0) / (T_finish - T_0)) * 100));

  const handlePointerDown = (e: React.PointerEvent<HTMLInputElement>) => {
    try {
      e.currentTarget.setPointerCapture(e.pointerId);
    } catch (_) {}

    isDraggingRef.current = true;
    wasPlayingRef.current = clock.state === 'PLAYING';

    if (wasPlayingRef.current && onPause) {
      onPause();
    }

    const val = parseFloat(e.currentTarget.value);
    localTimeRef.current = val;
    setLocalTime(val);
    setIsDragging(true);
  };

  const handlePointerUp = (e: React.PointerEvent<HTMLInputElement>) => {
    handleCommitDrag(e.pointerId);
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    if (isDraggingRef.current) {
      localTimeRef.current = val;
      setLocalTime(val);
    } else {
      onSeek(val);
    }
  };

  const formatSeconds = (sec: number) => {
    const totalSec = Math.max(0, Math.floor(sec - T_0));
    const h = Math.floor(totalSec / 3600);
    const m = Math.floor((totalSec % 3600) / 60);
    const s = totalSec % 60;
    return `${h > 0 ? `${h}:` : ''}${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="scrubber-row" id="replay-scrubber-row">
      <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
        00:00 (Start)
      </span>
      <input
        ref={sliderRef}
        type="range"
        id="timeline-scrubber"
        className="scrubber-slider"
        min={T_0}
        max={T_finish}
        step={0.5}
        value={displayTime}
        onChange={handleInputChange}
        onInput={handleInputChange}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerUp}
        style={{
          background: `linear-gradient(to right, var(--f1-red) 0%, var(--f1-red) ${progressPct}%, var(--border-subtle) ${progressPct}%, var(--border-subtle) 100%)`
        }}
      />
      <span className="mono" style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
        {formatSeconds(T_finish)} (Finish)
      </span>
    </div>
  );
};
