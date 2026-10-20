import { useEffect, useRef, useState, useCallback } from 'react';
import { ClockStatus, RaceState, WSMessage } from '../types/replay';

interface UseReplaySocketReturn {
  clock: ClockStatus | null;
  raceState: RaceState | null;
  isConnected: boolean;
  error: string | null;
  play: () => void;
  pause: () => void;
  seek: (targetTimeS: number) => void;
  stepLap: (direction: 1 | -1) => void;
  setSpeed: (speed: number) => void;
}

export function useReplaySocket(raceId: string): UseReplaySocketReturn {
  const [clock, setClock] = useState<ClockStatus | null>(null);
  const [raceState, setRaceState] = useState<RaceState | null>(null);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);

  const connect = useCallback(() => {
    if (!raceId) return;

    // Determine ws url
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    // Use host (proxied by Vite to backend) or direct 8000
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws/replay/${raceId}`;

    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      setError(null);
    };

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);

        if (msg.type === 'INITIAL_STATE' || msg.type === 'STATE_UPDATE') {
          setClock(msg.clock);
          setRaceState(msg.state);
        } else if (msg.type === 'COMMAND_ACK') {
          // Command acknowledged
        } else if (msg.type === 'ERROR') {
          setError(`[${msg.code}] ${msg.message}`);
        } else if (msg.type === 'PLAYBACK_FINISHED') {
          setClock((prev) => (prev ? { ...prev, state: 'FINISHED' } : null));
        }
      } catch (err) {
        console.error('Failed to parse WebSocket message:', err);
      }
    };

    ws.onerror = () => {
      setError('WebSocket connection error.');
    };

    ws.onclose = () => {
      setIsConnected(false);
      socketRef.current = null;
      // Reconnect after 2 seconds
      reconnectTimeoutRef.current = window.setTimeout(() => {
        connect();
      }, 2000);
    };
  }, [raceId]);

  useEffect(() => {
    // Clear previous race state immediately upon switching races
    setClock(null);
    setRaceState(null);
    setError(null);
    setIsConnected(false);

    connect();

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      if (socketRef.current) {
        const ws = socketRef.current;
        socketRef.current = null;
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        ws.close();
      }
    };
  }, [connect]);

  const sendCommand = useCallback((payload: Record<string, any>) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(payload));
    }
  }, []);

  const play = useCallback(() => {
    sendCommand({ action: 'PLAY' });
  }, [sendCommand]);

  const pause = useCallback(() => {
    sendCommand({ action: 'PAUSE' });
  }, [sendCommand]);

  const seek = useCallback(
    (targetTimeS: number) => {
      sendCommand({ action: 'SEEK', target_time_s: targetTimeS });
    },
    [sendCommand]
  );

  const stepLap = useCallback(
    (direction: 1 | -1) => {
      sendCommand({ action: 'STEP_LAP', direction });
    },
    [sendCommand]
  );

  const setSpeed = useCallback(
    (speed: number) => {
      sendCommand({ action: 'SET_SPEED', speed });
    },
    [sendCommand]
  );

  return {
    clock,
    raceState,
    isConnected,
    error,
    play,
    pause,
    seek,
    stepLap,
    setSpeed,
  };
}
