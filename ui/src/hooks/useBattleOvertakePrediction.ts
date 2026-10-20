import { useState, useEffect, useCallback, useRef } from 'react';
import type { BattleOvertakePredictionResponse } from '../types/battles';

interface UseBattleOvertakePredictionParams {
  raceId: string;
  driver1: string;
  driver2: string;
  asOfTimeS?: number;
  asOfLap?: number | null;
  syncWithReplay: boolean;
  horizonLaps?: number;
}

interface UseBattleOvertakePredictionReturn {
  data: BattleOvertakePredictionResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

export const useBattleOvertakePrediction = ({
  raceId,
  driver1,
  driver2,
  asOfTimeS,
  asOfLap,
  syncWithReplay,
  horizonLaps = 5,
}: UseBattleOvertakePredictionParams): UseBattleOvertakePredictionReturn => {
  const [data, setData] = useState<BattleOvertakePredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const lastFetchedKeyRef = useRef<string>('');
  const lastLapRef = useRef<number | null | undefined>(undefined);

  const fetchPrediction = useCallback(
    (force: boolean = false) => {
      const d1 = driver1.toUpperCase().trim();
      const d2 = driver2.toUpperCase().trim();

      if (!raceId || !d1 || !d2) {
        setData(null);
        setError(null);
        setIsLoading(false);
        return;
      }

      if (d1 === d2) {
        setError('Driver 1 and Driver 2 must be distinct drivers.');
        setData(null);
        setIsLoading(false);
        return;
      }

      // Build target request URL
      const params = new URLSearchParams();
      params.set('driver1', d1);
      params.set('driver2', d2);
      params.set('horizon_laps', String(horizonLaps));

      let effectiveAsOf: string | null = null;
      if (syncWithReplay && asOfTimeS !== undefined) {
        // Round to 1 decimal place to prevent microsecond jitter duplicate requests
        effectiveAsOf = asOfTimeS.toFixed(1);
        params.set('as_of', effectiveAsOf);
      }

      const requestKey = `${raceId}:${d1}:${d2}:${syncWithReplay}:${effectiveAsOf || 'full'}:${horizonLaps}`;

      // Avoid duplicate requests unless forced (e.g. manual refetch)
      if (!force && requestKey === lastFetchedKeyRef.current && data !== null) {
        return;
      }

      // Cancel previous in-flight request to guard against race conditions and stale responses
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;
      lastFetchedKeyRef.current = requestKey;

      setIsLoading(true);
      setError(null);

      fetch(`/api/races/${encodeURIComponent(raceId)}/predictions?${params.toString()}`, {
        signal: controller.signal,
      })
        .then(async (res) => {
          if (!res.ok) {
            const errBody = await res.json().catch(() => ({}));
            throw new Error(errBody.detail || `HTTP ${res.status}: Failed to fetch prediction.`);
          }
          return res.json() as Promise<BattleOvertakePredictionResponse>;
        })
        .then((json) => {
          setData(json);
          setIsLoading(false);
        })
        .catch((err: Error) => {
          if (err.name === 'AbortError') {
            // Cancelled intentionally due to newer parameters
            return;
          }
          setError(err.message || 'Failed to generate battle & overtake prediction.');
          setIsLoading(false);
        });
    },
    [raceId, driver1, driver2, syncWithReplay, asOfTimeS, horizonLaps, data]
  );

  // Clear state immediately whenever driver pair or race changes to prevent stale data display
  useEffect(() => {
    setData(null);
    setError(null);
    lastFetchedKeyRef.current = '';
  }, [raceId, driver1, driver2]);

  // Trigger fetch on parameter changes:
  // - Driver pair or race changes
  // - Replay sync mode toggle
  // - Completed lap change during replay sync (preferring lap boundaries over every tick)
  useEffect(() => {
    const lapChanged = asOfLap !== lastLapRef.current;
    lastLapRef.current = asOfLap;

    // If syncWithReplay is off, or lap changed, or first run, fetch prediction
    fetchPrediction(lapChanged);

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [raceId, driver1, driver2, syncWithReplay, asOfLap, fetchPrediction]);

  const handleRefetch = useCallback(() => {
    fetchPrediction(true);
  }, [fetchPrediction]);

  return {
    data,
    isLoading,
    error,
    refetch: handleRefetch,
  };
};
