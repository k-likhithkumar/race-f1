import { useState, useEffect, useRef, useCallback } from 'react';
import { PitStrategyResponse } from '../types/pits';

interface UsePitStrategyOptions {
  raceId: string;
  driver: string;
  asOfTimeS?: number;
  asOfLap?: number;
  referenceStationaryS?: number;
}

interface UsePitStrategyReturn {
  data: PitStrategyResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

export function usePitStrategy({
  raceId,
  driver,
  asOfTimeS,
  asOfLap,
  referenceStationaryS = 2.5,
}: UsePitStrategyOptions): UsePitStrategyReturn {
  const [data, setData] = useState<PitStrategyResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Track last fetched lap and driver to avoid redundant network queries on every 100ms clock tick
  const lastFetchedKeyRef = useRef<string>('');
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchPitStrategy = useCallback(async (forced: boolean = false) => {
    if (!raceId || !driver) {
      setData(null);
      return;
    }

    const currentKey = `${raceId}_${driver}_t${asOfTimeS ?? 'none'}_lap${asOfLap ?? 'none'}_ref${referenceStationaryS}`;
    if (!forced && currentKey === lastFetchedKeyRef.current) {
      return;
    }

    // Cancel pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setIsLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams({
        driver,
        reference_stationary: String(referenceStationaryS),
      });
      if (asOfTimeS !== undefined && asOfTimeS !== null) {
        params.append('as_of', String(asOfTimeS));
      }

      const res = await fetch(`/api/races/${encodeURIComponent(raceId)}/pits?${params.toString()}`, {
        signal: abortController.signal,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `HTTP ${res.status}: Failed to fetch pit strategy`);
      }

      const json: PitStrategyResponse = await res.json();
      setData(json);
      lastFetchedKeyRef.current = currentKey;
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        setError(err.message || 'Error loading pit strategy analysis');
        setData(null);
      }
    } finally {
      setIsLoading(false);
    }
  }, [raceId, driver, asOfTimeS, asOfLap, referenceStationaryS]);

  // Trigger fetch when driver or lap changes
  useEffect(() => {
    fetchPitStrategy(false);

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchPitStrategy]);

  const refetch = useCallback(() => {
    fetchPitStrategy(true);
  }, [fetchPitStrategy]);

  return {
    data,
    isLoading,
    error,
    refetch,
  };
}
