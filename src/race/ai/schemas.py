"""
Pydantic schemas for the AI Strategy Assistant module (Checkpoint 5).
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field


EvidenceType = Literal['replay', 'comparison', 'tyres', 'pits', 'battles']
AssistantConfidence = Literal['HIGH', 'MEDIUM', 'LOW', 'CANNOT_TELL', 'UNAVAILABLE']
AssistantQueryStatus = Literal['SUCCESS', 'CANNOT_TELL', 'UNAVAILABLE', 'INSUFFICIENT_EVIDENCE', 'ERROR']


class EvidenceReference(BaseModel):
    """
    Structured, typed evidence reference linking an analytical claim
    to deterministic data and front-end navigation payloads.
    """
    id: str = Field(..., description="Unique stable ID for this evidence reference")
    type: EvidenceType = Field(..., description="Category of the evidence")
    label: str = Field(..., description="Human-readable chip title (e.g. 'Lap 15 Delta: VER vs LEC')")
    description: str = Field(..., description="Descriptive explanation of the underlying evidence")
    driver_codes: List[str] = Field(default_factory=list, description="Associated driver codes")
    lap: Optional[int] = Field(None, description="Relevant lap number if applicable")
    stint: Optional[int] = Field(None, description="Relevant stint number if applicable")
    as_of: float = Field(..., description="Point-in-time replay cutoff timestamp in seconds")
    nav_payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Navigation payload for front-end deep-linking",
    )


class AssistantKeyNumber(BaseModel):
    """Specific quantitative metric reported in answer."""
    label: str = Field(..., description="Metric description (e.g. 'Closing Rate')")
    value: str = Field(..., description="Formatted value string (e.g. '+0.15s/lap')")
    unit: Optional[str] = Field(None, description="Unit if separated (e.g. 's')")


class AssistantStatusResponse(BaseModel):
    """System availability and diagnostic status of AI Strategy Assistant."""
    available: bool = Field(..., description="Whether the assistant can currently answer queries")
    provider: str = Field(..., description="Active LLM provider identifier ('gemini' or 'mock')")
    model: str = Field(..., description="Configured model identifier")
    message: str = Field(..., description="Human-readable status or setup instruction")
    tools_registered: List[str] = Field(..., description="List of available deterministic tools")
    status_reason: Optional[Literal[
        'available',
        'missing_api_key',
        'invalid_api_key',
        'provider_error',
        'unexpected_error',
        'quota_exhausted',
    ]] = Field(None, description="Diagnostic status classification code")


class AssistantChatMessage(BaseModel):
    """A single conversation turn in chat history."""
    role: Literal['user', 'assistant'] = Field(..., description="Sender role")
    content: str = Field(..., max_length=1000, description="Message text content")


class AssistantQueryRequest(BaseModel):
    """Incoming user query to the AI Strategy Assistant."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="User's natural language question")
    as_of: Optional[float] = Field(
        None,
        description="Optional requested simulation cutoff timestamp in seconds. Validated by server.",
    )
    conversation_history: Optional[List[AssistantChatMessage]] = Field(
        default=None,
        description="Optional preceding conversation turns (max 10)",
    )


class AssistantQueryResponse(BaseModel):
    """Structured response from the AI Strategy Assistant."""
    status: AssistantQueryStatus = Field(..., description="Execution status")
    answer: str = Field(..., description="Synthesized natural language answer")
    claims: List[str] = Field(default_factory=list, description="Atomic factual claims")
    key_numbers: List[AssistantKeyNumber] = Field(default_factory=list, description="Key numerical metrics")
    evidence: List[EvidenceReference] = Field(default_factory=list, description="Verified evidence references")
    confidence: AssistantConfidence = Field(..., description="Evaluated confidence rating")
    limitations_notice: Optional[str] = Field(None, description="Caveats, assumptions, or unmeasured data notices")
    tool_calls_executed: List[str] = Field(default_factory=list, description="Names of tools called during query")
    as_of: float = Field(..., description="Authoritative simulation cutoff timestamp used")
    error_code: Optional[str] = Field(None, description="Error code if query failed")
