import { useState, useEffect, useCallback, useRef } from 'react';
import {
  AssistantStatusResponse,
  AssistantQueryRequest,
  AssistantQueryResponse,
  AssistantTurn,
  AssistantChatMessage,
} from '../types/assistant';

interface UseAIAssistantProps {
  raceId: string;
  currentReplayTimeS?: number;
}

export function useAIAssistant({ raceId, currentReplayTimeS }: UseAIAssistantProps) {
  const [statusData, setStatusData] = useState<AssistantStatusResponse | null>(null);
  const [statusLoading, setStatusLoading] = useState<boolean>(true);
  const [statusError, setStatusError] = useState<string | null>(null);

  const [turns, setTurns] = useState<AssistantTurn[]>([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [queryError, setQueryError] = useState<string | null>(null);

  // Store last submitted prompt for retry functionality
  const lastPromptRef = useRef<{ prompt: string; asOf?: number } | null>(null);

  // Fetch status on raceId change
  const fetchStatus = useCallback(async () => {
    if (!raceId) return;
    setStatusLoading(true);
    setStatusError(null);
    try {
      const res = await fetch(`/api/races/${raceId}/assistant/status`);
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}: Failed to retrieve assistant status`);
      }
      const data: AssistantStatusResponse = await res.json();
      setStatusData(data);
    } catch (err: any) {
      setStatusError(err.message || 'Failed to check assistant availability');
      setStatusData(null);
    } finally {
      setStatusLoading(false);
    }
  }, [raceId]);

  useEffect(() => {
    fetchStatus();
    // Clear conversation when switching to a different race
    setTurns([]);
    setQueryError(null);
    lastPromptRef.current = null;
  }, [fetchStatus]);

  // Submit query to /api/races/{race_id}/assistant/query
  const submitQuery = useCallback(
    async (rawPrompt: string, overrideAsOf?: number) => {
      const prompt = rawPrompt.trim();
      if (!prompt || isSubmitting) return;

      const asOf = overrideAsOf !== undefined ? overrideAsOf : currentReplayTimeS;
      lastPromptRef.current = { prompt, asOf };
      setIsSubmitting(true);
      setQueryError(null);

      const timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
      const userTurnId = `user-${Date.now()}`;
      const assistantTurnId = `assistant-${Date.now()}`;

      // Build conversation history from completed turns (max 10 turns)
      const conversationHistory: AssistantChatMessage[] = [];
      turns.slice(-10).forEach((t) => {
        if (t.role === 'user' && t.content) {
          conversationHistory.push({ role: 'user', content: t.content });
        } else if (t.role === 'assistant' && t.response?.answer) {
          conversationHistory.push({ role: 'assistant', content: t.response.answer });
        }
      });

      // Optimistically append user message and loading assistant bubble
      const userTurn: AssistantTurn = {
        id: userTurnId,
        role: 'user',
        content: prompt,
        timestamp,
        asOf,
      };

      const assistantPlaceholder: AssistantTurn = {
        id: assistantTurnId,
        role: 'assistant',
        content: '',
        timestamp,
        asOf,
        isLoading: true,
      };

      setTurns((prev) => [...prev, userTurn, assistantPlaceholder]);

      try {
        const payload: AssistantQueryRequest = {
          prompt,
          as_of: asOf,
          conversation_history: conversationHistory.length > 0 ? conversationHistory : undefined,
        };

        const res = await fetch(`/api/races/${raceId}/assistant/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => null);
          const detail = errBody?.detail || `Server returned HTTP ${res.status}`;
          throw new Error(detail);
        }

        const data: AssistantQueryResponse = await res.json();

        // Update assistant turn with successful response
        setTurns((prev) =>
          prev.map((t) =>
            t.id === assistantTurnId
              ? {
                  ...t,
                  content: data.answer,
                  response: data,
                  isLoading: false,
                  error: data.status === 'ERROR' ? (data.limitations_notice || 'Query resulted in an error.') : null,
                }
              : t
          )
        );
      } catch (err: any) {
        const errMsg = err.message || 'An unexpected error occurred while communicating with the assistant.';
        setQueryError(errMsg);
        setTurns((prev) =>
          prev.map((t) =>
            t.id === assistantTurnId
              ? {
                  ...t,
                  content: 'Unable to complete request.',
                  error: errMsg,
                  isLoading: false,
                }
              : t
          )
        );
      } finally {
        setIsSubmitting(false);
      }
    },
    [isSubmitting, currentReplayTimeS, turns, raceId]
  );

  const retryLastQuery = useCallback(() => {
    if (lastPromptRef.current && !isSubmitting) {
      submitQuery(lastPromptRef.current.prompt, lastPromptRef.current.asOf);
    }
  }, [submitQuery, isSubmitting]);

  const clearConversation = useCallback(() => {
    setTurns([]);
    setQueryError(null);
    lastPromptRef.current = null;
  }, []);

  return {
    statusData,
    statusLoading,
    statusError,
    turns,
    isSubmitting,
    queryError,
    submitQuery,
    retryLastQuery,
    clearConversation,
    refetchStatus: fetchStatus,
  };
}
