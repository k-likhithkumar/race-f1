import { useState, useEffect, useCallback, useRef } from 'react';
import { GapBattleResponse } from '../types/battles';

interface UseGapBattleAnalysisParams {
  raceId: string;
  driver1: string;
  driver2: string;
  asOfTimeS?: number;
  syncWithReplay: boolean;
}

interface UseGapBattleAnalysisReturn {
  data: GapBattleResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

export const useGapBattleAnalysis = ({
  raceId,
  driver1,
  driver2,
  asOfTimeS,
  syncWithReplay,
}: UseGapBattleAnalysisParams): UseGapBattleAnalysisReturn => {
  const [data, setData] = useState<GapBattleResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchAnalysis = useCallback(() => {
    if (!raceId || !driver1 || !driver2) {
      setData(null);
      return;
    }

    if (driver1.toUpperCase().trim() === driver2.toUpperCase().trim()) {
      setError('Driver 1 and Driver 2 must be distinct drivers.');
      setData(null);
      return;
    }

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    setIsLoading(true);
    setError(null);

    const params = new URLSearchParams();
    params.set('driver1', driver1.toUpperCase().trim());
    params.set('driver2', driver2.toUpperCase().trim());
    if (syncWithReplay && asOfTimeS !== undefined) {
      params.set('as_of', String(asOfTimeS));
    }

    fetch(`/api/races/${encodeURIComponent(raceId)}/battles?${params.toString()}`, {
      signal: controller.signal,
    })
      .then(async (res) => {
        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(errBody.detail || `HTTP ${res.status}: Failed to fetch gap & battle analysis.`);
        }
        return res.json() as Promise<GapBattleResponse>;
      })
      .then((json) => {
        setData(json);
        setIsLoading(false);
      })
      .catch((err: Error) => {
        if (err.name === 'AbortError') {
          return;
        }
        setError(err.message || 'Failed to load gap & battle data.');
        setIsLoading(false);
      });
  }, [raceId, driver1, driver2, asOfTimeS, syncWithReplay]);

  useEffect(() => {
    fetchAnalysis();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchAnalysis]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchAnalysis,
  };
};
