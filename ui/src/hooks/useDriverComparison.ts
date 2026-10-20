import { useState, useRef, useCallback, useEffect } from 'react';
import {
  ComparisonRequest,
  ComparisonResponse,
  ComparisonConfigState,
} from '../types/comparison';

export interface UseDriverComparisonReturn {
  data: ComparisonResponse | null;
  isLoading: boolean;
  error: string | null;
  lastExecutedRequest: ComparisonRequest | null;
  lastExecutedConfig: ComparisonConfigState | null;
  executeComparison: (
    raceId: string,
    request: ComparisonRequest,
    configSnapshot: ComparisonConfigState
  ) => Promise<void>;
  clear: () => void;
}

export function useDriverComparison(): UseDriverComparisonReturn {
  const [data, setData] = useState<ComparisonResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [lastExecutedRequest, setLastExecutedRequest] = useState<ComparisonRequest | null>(null);
  const [lastExecutedConfig, setLastExecutedConfig] = useState<ComparisonConfigState | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef<number>(0);

  const clear = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    requestIdRef.current += 1;
    setData(null);
    setIsLoading(false);
    setError(null);
    setLastExecutedRequest(null);
    setLastExecutedConfig(null);
  }, []);

  const executeComparison = useCallback(
    async (
      raceId: string,
      request: ComparisonRequest,
      configSnapshot: ComparisonConfigState
    ) => {
      // Abort any prior in-flight request
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      const controller = new AbortController();
      abortControllerRef.current = controller;

      const currentReqId = ++requestIdRef.current;
      setIsLoading(true);
      setError(null);

      try {
        const response = await fetch(`/api/races/${encodeURIComponent(raceId)}/compare`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(request),
          signal: controller.signal,
        });

        // Ignore stale responses if a newer request was dispatched
        if (currentReqId !== requestIdRef.current) {
          return;
        }

        if (!response.ok) {
          let errorMessage = `HTTP ${response.status}: Failed to compare drivers`;
          try {
            const errData = await response.json();
            if (typeof errData?.detail === 'string') {
              errorMessage = errData.detail;
            } else if (Array.isArray(errData?.detail)) {
              errorMessage = errData.detail
                .map((d: { msg?: string }) => d.msg || JSON.stringify(d))
                .join('; ');
            }
          } catch {
            // Non-JSON response body fallback
          }
          throw new Error(errorMessage);
        }

        const json = (await response.json()) as ComparisonResponse;

        // Double check request ID before committing to state
        if (currentReqId === requestIdRef.current) {
          setData(json);
          setLastExecutedRequest(request);
          setLastExecutedConfig({ ...configSnapshot });
          setIsLoading(false);
        }
      } catch (err: unknown) {
        // Suppress abort errors from cancelled requests
        if (err instanceof DOMException && err.name === 'AbortError') {
          return;
        }
        if (currentReqId === requestIdRef.current) {
          const message = err instanceof Error ? err.message : 'Unknown error during comparison';
          setError(message);
          setIsLoading(false);
        }
      }
    },
    []
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    data,
    isLoading,
    error,
    lastExecutedRequest,
    lastExecutedConfig,
    executeComparison,
    clear,
  };
}
