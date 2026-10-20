"""
Bounded AI Strategy Assistant Orchestrator (Checkpoint 5B).
Orchestrates the bounded tool-calling loop (max 3 turns), timeout enforcement,
deterministic tool dispatching, grounding verification, and typed response synthesis.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Set, Tuple

from src.race import config
from src.race.config import (
    AI_MAX_TOOL_TURNS,
    AI_MAX_PROMPT_CHARS,
    AI_MAX_HISTORY_MESSAGES,
)
from src.race.ai.schemas import (
    AssistantQueryRequest,
    AssistantQueryResponse,
    AssistantConfidence,
    EvidenceReference,
)
from src.race.ai.provider import (
    LLMProvider,
    GeminiLLMProvider,
    NvidiaLLMProvider,
    MockLLMProvider,
    QuotaExhaustedError,
)
from src.race.ai.tools import ToolRegistry
from src.race.ai.evidence import build_evidence_references
from src.race.ai.grounding import validate_grounding, check_unmeasured_domains

logger = logging.getLogger(__name__)


SYSTEM_INSTRUCTION = """You are the RACE F1 AI Strategy Assistant.
You provide clear, accurate race intelligence and strategy analysis grounded strictly in deterministic data.
CRITICAL RULES:
1. You MUST NOT calculate lap times, race positions, gaps, tyre degradation rates, pit losses, or probabilities yourself.
2. Call the available tools to obtain verified numerical evidence. Once tool results are provided, synthesize your final response directly from the returned data without re-calling the same tools.
3. Keep your answers concise, objective, and directly grounded in the tool results.
4. Reference driver convention: In compare_driver_laps, mean_delta_s is (lap_time_a - lap_time_b). A positive delta means driver A is SLOWER than driver B.
5. In pit strategy, the reference stationary time (2.5s) is an illustrative assumption, not a measured sensor value. Never describe it as measured.
6. Battle predictions and overtake probabilities are statistical models, not guarantees. Never state that an overtake is guaranteed or certain.
7. If asked about driver emotions, intent, engine power modes, or internal chassis damage, explain that track timing and telemetry data cannot measure these.
"""


_active_provider: Optional[LLMProvider] = None


def set_global_llm_provider(provider: Optional[LLMProvider]) -> None:
    """Set a global provider override (e.g. MockLLMProvider in automated tests)."""
    global _active_provider
    _active_provider = provider


def get_active_llm_provider() -> LLMProvider:
    """Returns the globally configured LLMProvider or instantiates default based on AI_PROVIDER."""
    global _active_provider
    if _active_provider is not None:
        return _active_provider

    provider_name = config.get_ai_provider_name()
    if provider_name == "mock":
        return MockLLMProvider()
    elif provider_name == "nvidia":
        return NvidiaLLMProvider(
            api_key=config.get_nvidia_api_key(),
            model_name=config.get_nvidia_model_name(),
            base_url=config.get_nvidia_base_url(),
        )
    else:
        return GeminiLLMProvider(
            api_key=config.get_gemini_api_key(),
            model_name=config.get_gemini_model_name(),
        )


class AIStrategyOrchestrator:
    """
    Bounded orchestrator for executing user queries with deterministic tool calling.
    """

    def __init__(
        self,
        race_id: str,
        authoritative_as_of: float,
        provider: Optional[LLMProvider] = None,
    ):
        self.race_id = race_id
        self.authoritative_as_of = float(authoritative_as_of)

        if provider is not None:
            self.provider = provider
        else:
            self.provider = get_active_llm_provider()

        self.tool_registry = ToolRegistry(self.race_id, self.authoritative_as_of)

    async def run_query(self, request: AssistantQueryRequest) -> AssistantQueryResponse:
        """
        Executes an assistant query with bounded tool execution and grounding.
        """
        # 1. Check provider availability
        if not self.provider.is_available():
            p_type = self.provider.get_provider_type()
            key_var = "NVIDIA_API_KEY" if p_type == "nvidia" else "GEMINI_API_KEY"
            provider_label = "NVIDIA" if p_type == "nvidia" else "Gemini"
            return AssistantQueryResponse(
                status="UNAVAILABLE",
                answer=(
                    f"The AI Strategy Assistant is currently unavailable because {key_var} "
                    f"is not configured in the server environment. Please set {key_var} in .env to enable AI queries."
                ),
                claims=[],
                key_numbers=[],
                evidence=[],
                confidence="UNAVAILABLE",
                limitations_notice=f"{key_var} is not configured in the backend environment (.env).",
                tool_calls_executed=[],
                as_of=self.authoritative_as_of,
                error_code="PROVIDER_UNAVAILABLE",
            )

        # 2. Check unmeasured queries in prompt
        unmeasured_guardrail = check_unmeasured_domains(request.prompt)
        if unmeasured_guardrail:
            return AssistantQueryResponse(
                status="CANNOT_TELL",
                answer=unmeasured_guardrail,
                claims=["Query involves unmeasured internal vehicle or psychological factors."],
                key_numbers=[],
                evidence=[],
                confidence="CANNOT_TELL",
                limitations_notice="Telemetry lacks internal vehicle configuration or subjective intent data.",
                tool_calls_executed=[],
                as_of=self.authoritative_as_of,
            )

        # 3. Prepare tool declarations and conversation history
        tools_spec = self.tool_registry.get_gemini_tool_declarations()

        messages: List[Dict[str, Any]] = []
        if request.conversation_history:
            for item in request.conversation_history[-AI_MAX_HISTORY_MESSAGES:]:
                messages.append({
                    "role": item.role,
                    "parts": [{"text": item.content}],
                })

        messages.append({
            "role": "user",
            "parts": [{"text": request.prompt}],
        })

        executed_tools: List[Dict[str, Any]] = []
        seen_calls: Set[str] = set()
        turns = 0
        final_answer = ""

        # 4. Bounded tool loop (maximum turns)
        while turns < AI_MAX_TOOL_TURNS:
            turns += 1

            try:
                provider_resp = await self.provider.generate_content(
                    messages=messages,
                    tools=tools_spec,
                    system_instruction=SYSTEM_INSTRUCTION,
                )
            except QuotaExhaustedError as e:
                retry_notice = f" Please retry in approximately {e.retry_delay_str}." if e.retry_delay_str else " Please try again later."
                p_type = self.provider.get_provider_type()
                p_name = "NVIDIA" if p_type == "nvidia" else ("Gemini" if p_type == "gemini" else "AI Provider")
                logger.warning(
                    "Query halted due to quota/rate limit exhaustion. Provider: %s, Model: %s, Turn: %d, Retry delay: %s",
                    p_type,
                    getattr(self.provider, "model_name", "unknown"),
                    turns,
                    e.retry_delay_str or "unspecified",
                )
                return AssistantQueryResponse(
                    status="UNAVAILABLE",
                    answer=(
                        f"The AI Strategy Assistant has reached its {p_name} API request quota / rate limit (HTTP 429).{retry_notice} "
                        "Deterministic race telemetry, lap times, tyre degradation, and pit strategies remain fully active across the dashboard."
                    ),
                    claims=[],
                    key_numbers=[],
                    evidence=[],
                    confidence="UNAVAILABLE",
                    limitations_notice=f"{p_name} API quota exhausted (HTTP 429).{retry_notice}",
                    tool_calls_executed=[rec["tool_name"] for rec in executed_tools],
                    as_of=self.authoritative_as_of,
                    error_code="quota_exhausted",
                )
            except PermissionError as e:
                clean_msg = str(e)
                logger.error(f"Authentication failure: {clean_msg}")
                p_type = self.provider.get_provider_type()
                key_var = "NVIDIA_API_KEY" if p_type == "nvidia" else "GEMINI_API_KEY"
                p_name = "NVIDIA" if p_type == "nvidia" else "Gemini"
                return AssistantQueryResponse(
                    status="ERROR",
                    answer=f"{p_name} API authentication failed. The configured {key_var} is invalid or unauthorized.",
                    claims=[],
                    key_numbers=[],
                    evidence=[],
                    confidence="UNAVAILABLE",
                    limitations_notice=f"Invalid {key_var} or authentication failure.",
                    tool_calls_executed=[rec["tool_name"] for rec in executed_tools],
                    as_of=self.authoritative_as_of,
                    error_code="invalid_api_key",
                )
            except Exception as e:
                clean_msg = str(e)
                logger.error(f"Provider error during query: {clean_msg}")
                return AssistantQueryResponse(
                    status="ERROR",
                    answer="An error occurred while communicating with the AI model provider. Please verify network connectivity and configuration.",
                    claims=[],
                    key_numbers=[],
                    evidence=[],
                    confidence="UNAVAILABLE",
                    limitations_notice=f"Provider communication failure: {clean_msg}",
                    tool_calls_executed=[rec["tool_name"] for rec in executed_tools],
                    as_of=self.authoritative_as_of,
                    error_code="provider_error",
                )

            tool_calls = provider_resp.get("tool_calls", [])

            if not tool_calls:
                # No more tool calls; model returned final answer
                final_answer = provider_resp.get("text", "")
                break

            # Execute tool calls
            tool_response_parts: List[Dict[str, Any]] = []
            for call in tool_calls:
                func_name = call.get("name")
                func_args = call.get("args", {})
                call_id = call.get("id") or f"call_{len(executed_tools)}"

                # Prevent repeated identical tool calls
                call_fingerprint = f"{func_name}:{json.dumps(func_args, sort_keys=True)}"
                if call_fingerprint in seen_calls:
                    result = {
                        "status": "ALREADY_COMPLETED",
                        "message": "This tool was already executed in a previous turn with identical arguments. Please refer to the earlier tool response in the conversation to synthesize your final answer.",
                    }
                else:
                    seen_calls.add(call_fingerprint)
                    result = self.tool_registry.execute_tool(func_name, func_args)

                executed_tools.append({
                    "tool_name": func_name,
                    "args": func_args,
                    "result": result,
                })

                tool_response_parts.append({
                    "tool_call_id": call_id,
                    "function_response": {
                        "name": func_name,
                        "response": {"result": result},
                    }
                })

            # Append the model's function call turn and the user's function response turn
            raw_model_content = provider_resp.get("raw_content")
            if raw_model_content is not None:
                messages.append(raw_model_content)
            else:
                messages.append({
                    "role": "model",
                    "parts": [{"function_call": call} for call in tool_calls],
                })
            messages.append({
                "role": "user",
                "parts": tool_response_parts,
            })

        # 5. Handle case where turn limit reached without final text
        if not final_answer:
            if turns >= AI_MAX_TOOL_TURNS:
                final_answer = (
                    "Reached the maximum tool execution limit without a synthesized answer. "
                    "Data was retrieved from deterministic tools, but no final conclusion was reached."
                )

        # 6. Validate grounding & extract evidence
        grounding_res = validate_grounding(
            prompt=request.prompt,
            answer_text=final_answer,
            tool_records=executed_tools,
            as_of=self.authoritative_as_of,
        )

        evidence_refs: List[EvidenceReference] = []
        for rec in executed_tools:
            refs = build_evidence_references(
                tool_name=rec["tool_name"],
                tool_args=rec["args"],
                tool_result=rec["result"],
                as_of=self.authoritative_as_of,
            )
            evidence_refs.extend(refs)

        return AssistantQueryResponse(
            status=grounding_res.status,
            answer=grounding_res.sanitized_answer,
            claims=grounding_res.claims,
            key_numbers=grounding_res.key_numbers,
            evidence=evidence_refs,
            confidence=grounding_res.confidence,
            limitations_notice=grounding_res.limitations_notice,
            tool_calls_executed=[rec["tool_name"] for rec in executed_tools],
            as_of=self.authoritative_as_of,
        )
