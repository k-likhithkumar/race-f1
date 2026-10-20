"""
RACE AI Strategy Assistant package (Checkpoint 5B).
"""

from src.race.ai.schemas import (
    EvidenceReference,
    AssistantConfidence,
    AssistantQueryStatus,
    AssistantKeyNumber,
    AssistantStatusResponse,
    AssistantChatMessage,
    AssistantQueryRequest,
    AssistantQueryResponse,
)
from src.race.ai.provider import LLMProvider, GeminiLLMProvider, MockLLMProvider
from src.race.ai.tools import ToolRegistry
from src.race.ai.evidence import build_evidence_references
from src.race.ai.grounding import validate_grounding, check_unmeasured_domains
from src.race.ai.orchestrator import (
    AIStrategyOrchestrator,
    set_global_llm_provider,
    get_active_llm_provider,
)

__all__ = [
    "EvidenceReference",
    "AssistantConfidence",
    "AssistantQueryStatus",
    "AssistantKeyNumber",
    "AssistantStatusResponse",
    "AssistantChatMessage",
    "AssistantQueryRequest",
    "AssistantQueryResponse",
    "LLMProvider",
    "GeminiLLMProvider",
    "MockLLMProvider",
    "ToolRegistry",
    "build_evidence_references",
    "validate_grounding",
    "check_unmeasured_domains",
    "AIStrategyOrchestrator",
    "set_global_llm_provider",
    "get_active_llm_provider",
]
