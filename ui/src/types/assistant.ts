/**
 * Type definitions for AI Strategy Assistant (Checkpoint 5C).
 * Matches backend Pydantic schemas in src/race/ai/schemas.py.
 */

export type EvidenceType = 'replay' | 'comparison' | 'tyres' | 'pits' | 'battles';

export type AssistantConfidence = 'HIGH' | 'MEDIUM' | 'LOW' | 'CANNOT_TELL' | 'UNAVAILABLE';

export type AssistantQueryStatus = 'SUCCESS' | 'CANNOT_TELL' | 'UNAVAILABLE' | 'INSUFFICIENT_EVIDENCE' | 'ERROR';

export interface EvidenceReference {
  id: string;
  type: EvidenceType;
  label: string;
  description: string;
  driver_codes: string[];
  lap?: number | null;
  stint?: number | null;
  as_of: number;
  nav_payload: Record<string, any>;
}

export interface AssistantKeyNumber {
  label: string;
  value: string;
  unit?: string | null;
}

export interface AssistantStatusResponse {
  available: boolean;
  provider: string;
  model: string;
  message: string;
  tools_registered: string[];
  status_reason?: 'available' | 'missing_api_key' | 'invalid_api_key' | 'provider_error' | 'unexpected_error' | 'quota_exhausted' | null;
}

export interface AssistantChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface AssistantQueryRequest {
  prompt: string;
  as_of?: number;
  conversation_history?: AssistantChatMessage[];
}

export interface AssistantQueryResponse {
  status: AssistantQueryStatus;
  answer: string;
  claims: string[];
  key_numbers: AssistantKeyNumber[];
  evidence: EvidenceReference[];
  confidence: AssistantConfidence;
  limitations_notice?: string | null;
  tool_calls_executed: string[];
  as_of: number;
  error_code?: string | null;
}

export interface AssistantTurn {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  asOf?: number;
  response?: AssistantQueryResponse;
  error?: string | null;
  isLoading?: boolean;
}
