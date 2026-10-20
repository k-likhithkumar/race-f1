import { useState, useEffect, useCallback, useRef } from 'react';
import { TyreDegradationResponse } from '../types/tyres';

interface UseTyreDegradationParams {
  raceId: string;
  driver: string;
  stint: number;
  asOfTimeS?: number;
  syncWithReplay: boolean;
  fuelCorrection?: number;
}

interface UseTyreDegradationReturn {
  data: TyreDegradationResponse | null;
  isLoading: boolean;
  error: string | null;
  refetch: () => void;
}

export const useTyreDegradation = ({
  raceId,
  driver,
  stint,
  asOfTimeS,
  syncWithReplay,
  fuelCorrection = 0.033,
}: UseTyreDegradationParams): UseTyreDegradationReturn => {
  const [data, setData] = useState<TyreDegradationResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const fetchDegradation = useCallback(() => {
    if (!raceId || !driver) {
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
    params.set('driver', driver);
    params.set('stint', String(stint || 1));
    params.set('fuel_correction', String(fuelCorrection));
    if (syncWithReplay && asOfTimeS !== undefined) {
      params.set('as_of', String(asOfTimeS));
    } else {
      // Full stint view (effectively race finish cutoff)
      params.set('as_of', '1000000');
    }

    fetch(`/api/races/${encodeURIComponent(raceId)}/tyres?${params.toString()}`, {
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) {
          return res.json().then((errData) => {
            throw new Error(errData.detail || `Server error (HTTP ${res.status})`);
          }).catch((parseErr) => {
            throw new Error(parseErr.message || `HTTP ${res.status}`);
          });
        }
        return res.json();
      })
      .then((responseData: TyreDegradationResponse) => {
        setData(responseData);
        setIsLoading(false);
      })
      .catch((err) => {
        if (err.name === 'AbortError') return;
        setError(err.message || 'Failed to fetch tyre degradation data');
        setIsLoading(false);
      });
  }, [raceId, driver, stint, asOfTimeS, syncWithReplay, fuelCorrection]);

  useEffect(() => {
    fetchDegradation();
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchDegradation]);

  return {
    data,
    isLoading,
    error,
    refetch: fetchDegradation,
  };
};
