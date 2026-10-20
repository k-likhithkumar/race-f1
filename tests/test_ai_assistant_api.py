"""
Comprehensive unit and integration tests for Checkpoint 5B:
AI Strategy Assistant Backend & API.

Verifies:
1. Tool Registry (dispatching, argument validation, unknown tools, safe serialization)
2. Provider Layer (MockLLMProvider, offline operation, missing API key handling, error handling)
3. Bounded Orchestration (tool execution, max 3 turns, deduplication, error resilience)
4. Grounding and Validation (delta signs, estimates vs measurements, unmeasured guardrails)
5. Point-in-Time Safety (authoritative cutoff, future mutation test, no future leakage)
6. API Endpoints (GET status, POST query, 404/422 validation, mock provider injection)
"""

import io
import json
import math
import time
import asyncio
import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from fastapi.testclient import TestClient

from src.race.api.app import app
from src.race.api.session_manager import session_manager
from src.race.config import GEMINI_MODEL_NAME
from src.race.ai.schemas import (
    AssistantQueryRequest,
    AssistantQueryResponse,
    AssistantStatusResponse,
    EvidenceReference,
)
from src.race.ai.provider import MockLLMProvider, GeminiLLMProvider
from src.race.ai.tools import ToolRegistry, sanitize_for_json
from src.race.ai.grounding import validate_grounding, check_unmeasured_domains
from src.race.ai.evidence import build_evidence_references
from src.race.ai.orchestrator import (
    AIStrategyOrchestrator,
    set_global_llm_provider,
    get_active_llm_provider,
)

client = TestClient(app)


# ============================================================================
# 1. TOOL REGISTRY TESTS
# ============================================================================

def test_tool_registry_registration_and_dispatch():
    """Verify all 5 tools are registered and dispatch to deterministic session methods."""
    race_id = "2023_monza"
    session = session_manager.get_or_load_session(race_id)
    t_test = float(session.T_0 + 1000.0)

    registry = ToolRegistry(race_id, authoritative_as_of=t_test)
    names = registry.get_registered_tool_names()

    assert "get_race_leaderboard" in names
    assert "compare_driver_laps" in names
    assert "get_tyre_stint_degradation" in names
    assert "get_pit_strategy_and_waterfall" in names
    assert "get_battle_and_overtake_prediction" in names

    # 1. get_race_leaderboard
    res_lb = registry.execute_tool("get_race_leaderboard", {})
    assert res_lb["status"] == "SUCCESS"
    assert "leaderboard" in res_lb
    assert len(res_lb["leaderboard"]) > 0

    # 2. compare_driver_laps
    res_comp = registry.execute_tool("compare_driver_laps", {"driver_a": "VER", "driver_b": "LEC", "lap": 5})
    assert res_comp["status"] == "SUCCESS"
    assert "mean_delta_s" in res_comp

    # 3. get_tyre_stint_degradation
    res_tyre = registry.execute_tool("get_tyre_stint_degradation", {"driver": "VER", "stint": 1})
    assert "status" in res_tyre

    # 4. get_pit_strategy_and_waterfall
    res_pit = registry.execute_tool("get_pit_strategy_and_waterfall", {"driver": "ALB"})
    assert res_pit["status"] in ["SUCCESS", "PRE_RACE"]
    assert "reference_stationary_s" in res_pit

    # 5. get_battle_and_overtake_prediction
    res_battle = registry.execute_tool("get_battle_and_overtake_prediction", {"chaser": "PER", "leader": "SAI"})
    assert "status" in res_battle


def test_tool_registry_argument_validation():
    """Verify invalid driver codes, missing required args, and unknown tools are safely rejected."""
    registry = ToolRegistry("2023_monza", authoritative_as_of=2000.0)

    # Unknown tool
    err_unknown = registry.execute_tool("hack_f1_telemetry", {})
    assert err_unknown["status"] == "ERROR"
    assert "Unknown tool" in err_unknown["message"]

    # Invalid driver code (too long)
    err_driver = registry.execute_tool("compare_driver_laps", {"driver_a": "VERSTAPPEN", "driver_b": "LEC"})
    assert err_driver["status"] == "ERROR"

    # Driver comparing with self
    err_self = registry.execute_tool("compare_driver_laps", {"driver_a": "VER", "driver_b": "VER"})
    assert err_self["status"] == "ERROR"
    assert "distinct" in err_self["message"].lower()

    # Lap number out of bounds
    err_lap = registry.execute_tool("compare_driver_laps", {"driver_a": "VER", "driver_b": "LEC", "lap": 9999})
    assert err_lap["status"] == "ERROR"

    # Stint number invalid
    err_stint = registry.execute_tool("get_tyre_stint_degradation", {"driver": "VER", "stint": 0})
    assert err_stint["status"] == "ERROR"


def test_tool_registry_safe_serialization():
    """Verify that NaN, Inf, and NumPy primitives are sanitized cleanly for JSON serialization."""
    dirty_data = {
        "float_val": np.float64(12.34567),
        "int_val": np.int32(42),
        "nan_val": float("nan"),
        "inf_val": float("inf"),
        "nested": [np.bool_(True), pd.NA, None],
    }
    sanitized = sanitize_for_json(dirty_data)
    assert sanitized["float_val"] == 12.3457
    assert sanitized["int_val"] == 42
    assert sanitized["nan_val"] is None
    assert sanitized["inf_val"] is None
    assert sanitized["nested"] == [True, None, None]


# ============================================================================
# 2. PROVIDER LAYER TESTS
# ============================================================================

def test_mock_llm_provider_offline():
    """Verify MockLLMProvider functions entirely offline without API keys."""
    async def _run():
        mock = MockLLMProvider()
        assert mock.is_available() is True

        # Test prompt that triggers battle tool
        resp = await mock.generate_content(
            messages=[{"role": "user", "parts": [{"text": "Can Perez overtake Sainz?"}]}],
            tools=[],
        )
        assert len(resp["tool_calls"]) == 1
        assert resp["tool_calls"][0]["name"] == "get_battle_and_overtake_prediction"
        assert resp["tool_calls"][0]["args"]["chaser"] == "PER"
        assert resp["tool_calls"][0]["args"]["leader"] == "SAI"

        # Test custom handler override
        def custom_handler(messages, tools):
            return {"text": "Custom mock response text."}

        mock.set_custom_handler(custom_handler)
        resp2 = await mock.generate_content(messages=[], tools=[])
        assert resp2["text"] == "Custom mock response text."

    asyncio.run(_run())


def test_gemini_provider_clean_unavailability():
    """Verify GeminiLLMProvider reports unavailable cleanly when key is absent."""
    async def _run():
        provider = GeminiLLMProvider(api_key=None)
        assert provider.is_available() is False

        # Calling generate_content when unavailable must raise clean error without crashing
        with pytest.raises(RuntimeError) as exc_info:
            await provider.generate_content(messages=[], tools=[])
        assert "GEMINI_API_KEY is not configured" in str(exc_info.value)

    asyncio.run(_run())


def test_gemini_provider_contents_construction_prevents_empty_payload():
    """
    Focused test reproducing and verifying fix for 'contents are required' error.
    Ensures that orchestrator parts-based messages, tool calls, and tool responses
    are properly converted to valid types.Content structures with non-empty parts.
    """
    from google.genai import types

    # 1. Orchestrator format: {"role": "user", "parts": [{"text": "Who is leading?"}]}
    orchestrator_messages = [
        {"role": "user", "parts": [{"text": "Who is leading the race?"}]}
    ]
    contents = GeminiLLMProvider._convert_messages_to_contents(orchestrator_messages)
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 1
    assert contents[0].parts[0].text == "Who is leading the race?"

    # 2. Multi-turn format with function call and function response
    multi_turn_messages = [
        {"role": "user", "parts": [{"text": "What is Verstappen's tyre degradation?"}]},
        {
            "role": "model",
            "parts": [
                {"function_call": {"name": "get_tyre_stint_degradation", "args": {"driver": "VER", "stint": 1}}}
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "function_response": {
                        "name": "get_tyre_stint_degradation",
                        "response": {"result": {"status": "SUCCESS", "slope": 0.05}},
                    }
                }
            ],
        },
    ]
    multi_contents = GeminiLLMProvider._convert_messages_to_contents(multi_turn_messages)
    assert len(multi_contents) == 3
    assert multi_contents[0].role == "user"
    assert multi_contents[1].role == "model"
    assert multi_contents[2].role == "user"
    assert multi_contents[1].parts[0].function_call.name == "get_tyre_stint_degradation"
    assert multi_contents[2].parts[0].function_response.name == "get_tyre_stint_degradation"

    # 3. Direct types.Content preservation (e.g. raw_content from model candidate with thought_signature)
    raw_content = types.Content(
        role="model",
        parts=[types.Part.from_function_call(name="get_race_leaderboard", args={})],
    )
    preserved = GeminiLLMProvider._convert_messages_to_contents([raw_content])
    assert len(preserved) == 1
    assert preserved[0] is raw_content


def test_gemini_provider_empty_contents_validation():
    """Verify that an empty message list raises a clean ValueError prior to network dispatch."""
    provider = GeminiLLMProvider(api_key="mock_test_key")
    with pytest.raises(ValueError) as exc_info:
        provider.generate_turn(messages=[], tools=[], system_instruction="")
    assert "No valid message content" in str(exc_info.value)
# ============================================================================

def test_orchestrator_max_tool_turns_enforced():
    """Verify that orchestrator halts strictly after AI_MAX_TOOL_TURNS (3 turns)."""
    async def _run():
        call_count = 0

        def endless_tool_caller(messages, tools):
            nonlocal call_count
            call_count += 1
            return {
                "tool_calls": [
                    {
                        "name": "compare_driver_laps",
                        "args": {"driver_a": "VER", "driver_b": "LEC", "lap": call_count},
                    }
                ]
            }

        mock = MockLLMProvider()
        mock.set_custom_handler(endless_tool_caller)

        orchestrator = AIStrategyOrchestrator(
            race_id="2023_monza",
            authoritative_as_of=2000.0,
            provider=mock,
        )

        req = AssistantQueryRequest(prompt="Compare Verstappen and Leclerc infinitely.")
        resp = await orchestrator.run_query(req)

        # Should execute at most 3 turns
        assert call_count == 3
        assert len(resp.tool_calls_executed) == 3
        assert "maximum tool execution limit" in resp.answer.lower()

    asyncio.run(_run())


def test_orchestrator_duplicate_call_deduplication():
    """Verify identical repeated tool calls are deduplicated and not re-executed endlessly."""
    async def _run():
        call_step = 0

        def repeating_caller(messages, tools):
            nonlocal call_step
            call_step += 1
            if call_step == 1:
                return {
                    "tool_calls": [{"name": "compare_driver_laps", "args": {"driver_a": "VER", "driver_b": "LEC", "lap": 5}}]
                }
            elif call_step == 2:
                return {
                    "tool_calls": [{"name": "compare_driver_laps", "args": {"driver_a": "VER", "driver_b": "LEC", "lap": 5}}]
                }
            else:
                return {"text": "Finished comparison."}

        mock = MockLLMProvider()
        mock.set_custom_handler(repeating_caller)

        orchestrator = AIStrategyOrchestrator(
            race_id="2023_monza",
            authoritative_as_of=2000.0,
            provider=mock,
        )
        resp = await orchestrator.run_query(AssistantQueryRequest(prompt="Compare VER and LEC on lap 5."))
        assert resp.status in ["SUCCESS", "INSUFFICIENT_EVIDENCE"]

    asyncio.run(_run())


# ============================================================================
# 4. GROUNDING & GUARDRAILS TESTS
# ============================================================================

def test_grounding_unmeasured_query_guardrail():
    """Verify questions asking about emotions, engine modes, or damage are answered with cannot tell guardrail."""
    res_emotion = validate_grounding(
        prompt="Was Max Verstappen angry on team radio?",
        answer_text="He was furious.",
        tool_records=[],
        as_of=1000.0,
    )
    assert res_emotion.status == "CANNOT_TELL"
    assert "cannot determine" in res_emotion.sanitized_answer.lower()

    res_engine = validate_grounding(
        prompt="What engine mode did Ferrari switch to?",
        answer_text="They switched to Mode 2.",
        tool_records=[],
        as_of=1000.0,
    )
    assert res_engine.status == "CANNOT_TELL"
    assert "cannot determine" in res_engine.sanitized_answer.lower()


def test_grounding_delta_sign_preservation():
    """Verify inverted delta semantics (saying slower car was faster) are caught and flagged."""
    tool_record = {
        "tool_name": "compare_driver_laps",
        "args": {"driver_a": "VER", "driver_b": "LEC", "lap": 10},
        "result": {
            "mean_delta_s": 0.450,  # Positive: VER was slower than LEC
            "lap_time_a_s": 85.450,
            "lap_time_b_s": 85.000,
        },
    }

    # Model incorrectly claims VER was faster
    res = validate_grounding(
        prompt="Compare VER and LEC",
        answer_text="On lap 10, ver was faster than lec by half a second.",
        tool_records=[tool_record],
        as_of=1000.0,
    )
    assert res.validation_passed is False
    assert any("Inverted delta semantics" in r for r in res.reasons)
    assert "Correction" in res.limitations_notice


def test_grounding_stationary_assumption_distinction():
    """Verify reference stationary assumption (2.5s) cannot be claimed as measured stop time."""
    tool_record = {
        "tool_name": "get_pit_strategy_and_waterfall",
        "args": {"driver": "ALB"},
        "result": {
            "net_pit_loss_s": 22.4,
            "reference_stationary_s": 2.5,
        },
    }

    res = validate_grounding(
        prompt="How long was Albon's pit stop?",
        answer_text="The mechanics recorded a measured stationary time of 2.5s.",
        tool_records=[tool_record],
        as_of=1000.0,
    )
    assert res.validation_passed is False
    assert any("reference stationary time" in r for r in res.reasons)
    assert "illustrative 2.5s assumption" in res.limitations_notice


def test_grounding_overconfident_prediction():
    """Verify overtake predictions cannot be described as guaranteed or certain."""
    tool_record = {
        "tool_name": "get_battle_and_overtake_prediction",
        "args": {"chaser": "PER", "leader": "SAI"},
        "result": {
            "p_battle": 0.90,
            "p_overtake": 0.75,
        },
    }

    res = validate_grounding(
        prompt="Will Perez pass Sainz?",
        answer_text="Perez will definitely overtake Sainz on the next straight with a guaranteed overtake.",
        tool_records=[tool_record],
        as_of=1000.0,
    )
    assert res.validation_passed is False
    assert any("certainty" in r for r in res.reasons)


# ============================================================================
# 5. POINT-IN-TIME SAFETY TESTS
# ============================================================================

def test_point_in_time_future_mutation_safety():
    """
    CRITICAL REQUIREMENT:
    Verify that mutating future lap data does NOT alter the deterministic tool outputs
    or assistant response at an earlier replay cutoff timestamp.
    """
    race_id = "2023_monza"
    session = session_manager.get_or_load_session(race_id)

    # Choose an earlier cutoff (e.g. lap 10 crossing time)
    t_earlier = float(session.T_0 + 800.0)

    registry_before = ToolRegistry(race_id, authoritative_as_of=t_earlier)
    lb_before = registry_before.execute_tool("get_race_leaderboard", {})

    # Copy laps dataframe and mutate a future lap (e.g. lap 45)
    original_laps = session.laps_df.copy()
    future_mask = session.laps_df["lap_number"] > 40
    session.laps_df.loc[future_mask, "lap_time_s"] = 999.0

    try:
        registry_after = ToolRegistry(race_id, authoritative_as_of=t_earlier)
        lb_after = registry_after.execute_tool("get_race_leaderboard", {})

        # State at t_earlier must remain identical
        assert lb_before["leaderboard"] == lb_after["leaderboard"]
        assert lb_before["current_lap"] == lb_after["current_lap"]
    finally:
        # Restore original laps dataframe
        session.laps_df = original_laps


def test_point_in_time_clamp_and_future_lap_rejection():
    """Verify tool calls for laps occurring in the future relative to as_of are rejected."""
    race_id = "2023_monza"
    session = session_manager.get_or_load_session(race_id)
    t_early = float(session.T_0 + 200.0)  # Only Lap 1 or 2 completed

    registry = ToolRegistry(race_id, authoritative_as_of=t_early)
    # Request lap 40 when at t_early
    res = registry.execute_tool("compare_driver_laps", {"driver_a": "VER", "driver_b": "LEC", "lap": 40})
    assert res["status"] == "ERROR"
    assert "before or at the replay cutoff" in res["message"]


# ============================================================================
# 6. API ENDPOINT INTEGRATION TESTS
# ============================================================================

def test_api_assistant_status_endpoint():
    """Verify GET /api/races/{race_id}/assistant/status works cleanly."""
    from src.race.ai.orchestrator import get_active_llm_provider
    active_model = get_active_llm_provider().get_model_name()
    resp = client.get("/api/races/2023_monza/assistant/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "provider" in data
    assert data["model"] == active_model
    assert "tools_registered" in data
    assert len(data["tools_registered"]) == 5

    # Test unknown race returns 404
    resp_404 = client.get("/api/races/2099_spacex/assistant/status")
    assert resp_404.status_code == 404


def test_api_assistant_query_with_mock_provider():
    """Verify POST /api/races/{race_id}/assistant/query works with MockLLMProvider."""
    # Set global mock provider
    mock = MockLLMProvider()
    set_global_llm_provider(mock)

    try:
        req_payload = {
            "prompt": "Who is currently leading the race?",
            "as_of": 2000.0,
        }
        resp = client.post("/api/races/2023_monza/assistant/query", json=req_payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] in ["SUCCESS", "INSUFFICIENT_EVIDENCE"]
        assert data["as_of"] == 2000.0
        assert isinstance(data["evidence"], list)
        assert "leaderboard" in data["tool_calls_executed"][0]

        # Test unmeasured query via API returns CANNOT_TELL
        req_unmeasured = {
            "prompt": "How angry was Leclerc on the radio?",
            "as_of": 2000.0,
        }
        resp_unmeasured = client.post("/api/races/2023_monza/assistant/query", json=req_unmeasured)
        assert resp_unmeasured.status_code == 200
        data_unm = resp_unmeasured.json()
        assert data_unm["status"] == "CANNOT_TELL"
        assert "cannot determine" in data_unm["answer"].lower()

        # Test unknown race returns 404
        resp_err = client.post("/api/races/2099_spacex/assistant/query", json={"prompt": "hi"})
        assert resp_err.status_code == 404
    finally:
        set_global_llm_provider(None)


def test_api_assistant_query_unavailable_when_no_key():
    """Verify POST query returns structured UNAVAILABLE response when GEMINI_API_KEY is missing."""
    # Force provider to unavailable GeminiLLMProvider
    unavail_provider = GeminiLLMProvider(api_key="")
    set_global_llm_provider(unavail_provider)

    try:
        resp = client.post("/api/races/2023_monza/assistant/query", json={"prompt": "Who is leading?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "UNAVAILABLE"
        assert data["error_code"] == "PROVIDER_UNAVAILABLE"
        assert "GEMINI_API_KEY" in data["answer"]
    finally:
        set_global_llm_provider(None)


# ============================================================================
# 7. QUOTA EXHAUSTION & MODEL CONSISTENCY TESTS
# ============================================================================

def test_quota_exhaustion_timing_extraction():
    """Verify extract_quota_retry_info parses diverse 429 message and detail formats."""
    from src.race.ai.provider import extract_quota_retry_info

    # 1. Google API string format with sub-second precision
    msg1 = (
        "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
        "limit: 20, model: gemini-3.6-flash\nPlease retry in 16h12m11.937193772s."
    )
    secs1, str1 = extract_quota_retry_info(msg1)
    assert str1 == "16h 12m"
    assert secs1 is not None and secs1 > 50000

    # 2. Structured retryDelay in details
    class MockErrorWithDetails:
        details = {
            "error": {
                "code": 429,
                "status": "RESOURCE_EXHAUSTED",
                "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "58521s"}],
            }
        }

    secs2, str2 = extract_quota_retry_info("Resource exhausted", MockErrorWithDetails())
    assert str2 == "16h 15m"
    assert secs2 == 58521.0

    # 3. Short duration
    msg3 = "Rate limit reached. Please retry in 45s."
    secs3, str3 = extract_quota_retry_info(msg3)
    assert str3 == "45s"
    assert secs3 == 45.0


def test_quota_exhausted_error_handling_in_orchestrator():
    """Verify orchestrator returns structured UNAVAILABLE response without retrying on 429."""
    from src.race.ai.provider import QuotaExhaustedError

    class QuotaFailingProvider(MockLLMProvider):
        call_count = 0

        async def generate_content(self, messages, tools, system_instruction="", temperature=0.2):
            self.call_count += 1
            raise QuotaExhaustedError(
                "Gemini API quota exhausted (HTTP 429: RESOURCE_EXHAUSTED).",
                retry_delay_seconds=58500,
                retry_delay_str="16h 15m",
            )

    provider = QuotaFailingProvider()
    orchestrator = AIStrategyOrchestrator(
        race_id="2023_monza",
        authoritative_as_of=2000.0,
        provider=provider,
    )

    req = AssistantQueryRequest(prompt="Who is leading the race?")
    resp = asyncio.run(orchestrator.run_query(req))

    # Must fail fast in 1 turn without infinite retries
    assert provider.call_count == 1
    assert resp.status == "UNAVAILABLE"
    assert resp.error_code == "quota_exhausted"
    assert resp.confidence == "UNAVAILABLE"
    assert "HTTP 429" in resp.answer
    assert "16h 15m" in resp.answer
    assert "deterministic" in resp.answer.lower()
    assert "quota exhausted" in resp.limitations_notice.lower()


def test_quota_exhaustion_status_cooldown_and_race_usability(monkeypatch):
    """Verify that assistant status reports quota_exhausted while rest of RACE remains fully usable."""
    from src.race.ai.orchestrator import get_active_llm_provider
    provider = get_active_llm_provider()
    try:
        # Set quota cooldown on active provider
        provider.__class__._global_quota_cooldown_until = time.time() + 3600.0
        provider.__class__._global_quota_retry_str = "1h"

        # Assistant status must reflect quota_exhausted
        status_resp = client.get("/api/races/2023_monza/assistant/status")
        assert status_resp.status_code == 200
        status_data = status_resp.json()
        assert status_data["available"] is False
        assert status_data["status_reason"] == "quota_exhausted"
        assert "1h" in status_data["message"]

        # Crucial requirement: The rest of the RACE application must remain 100% usable
        races_resp = client.get("/api/races")
        assert races_resp.status_code == 200
        assert len(races_resp.json()) > 0

        track_resp = client.get("/api/races/2023_monza/track")
        assert track_resp.status_code == 200

        tyres_resp = client.get("/api/races/2023_monza/tyres?driver=VER&stint=1&fuel_correction=0.033&as_of=2000.0")
        assert tyres_resp.status_code == 200
    finally:
        provider.__class__.reset_quota_cooldown()


def test_model_consistency_across_layers(monkeypatch):
    """Verify authoritative model configuration is synchronized across config, provider, and API."""
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTest")
    from src.race.config import GEMINI_MODEL_NAME, get_gemini_model_name

    configured_model = get_gemini_model_name()
    assert GEMINI_MODEL_NAME == configured_model

    provider = GeminiLLMProvider(api_key="mock_key")
    assert provider.get_model_name() == configured_model

    status_resp = client.get("/api/races/2023_monza/assistant/status")
    assert status_resp.status_code == 200
    assert status_resp.json()["model"] == configured_model


# ============================================================================
# 8. MULTI-PROVIDER & NVIDIA NIM PROVIDER TESTS (MOCKED)
# ============================================================================

def test_provider_selection_and_automatic_detection(monkeypatch):
    """Verify provider selection obeys explicit settings and smart key-prefix detection."""
    from src.race.ai.orchestrator import get_active_llm_provider
    from src.race.ai.provider import NvidiaLLMProvider, GeminiLLMProvider, MockLLMProvider

    # 1. Explicit AI_PROVIDER=nvidia
    monkeypatch.setenv("AI_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key-123")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestKey")
    p1 = get_active_llm_provider()
    assert isinstance(p1, NvidiaLLMProvider)
    assert p1.get_provider_type() == "nvidia"

    # 2. Explicit AI_PROVIDER=gemini overrides nvapi key presence
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    p2 = get_active_llm_provider()
    assert isinstance(p2, GeminiLLMProvider)
    assert p2.get_provider_type() == "gemini"

    # 3. Explicit AI_PROVIDER=mock
    monkeypatch.setenv("AI_PROVIDER", "mock")
    p3 = get_active_llm_provider()
    assert isinstance(p3, MockLLMProvider)
    assert p3.get_provider_type() == "mock"

    # 4. Auto-detection with nvapi prefix when AI_PROVIDER is unset
    monkeypatch.delenv("AI_PROVIDER", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-auto-detected")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    p4 = get_active_llm_provider()
    assert isinstance(p4, NvidiaLLMProvider)

    # 5. Auto-detection with AIza prefix when AI_PROVIDER is unset
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyAutoDetected")
    p5 = get_active_llm_provider()
    assert isinstance(p5, GeminiLLMProvider)


def test_nvidia_provider_request_formatting():
    """Verify NVIDIA provider formats messages and tool declarations into OpenAI-compatible payload."""
    from src.race.ai.provider import NvidiaLLMProvider

    provider = NvidiaLLMProvider(api_key="nvapi-mock-key", model_name="meta/llama-3.2-11b-vision-instruct")

    # Tools formatting
    raw_tools = [
        {
            "name": "get_race_leaderboard",
            "description": "Get race order",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    formatted_tools = provider._format_tools(raw_tools)
    assert len(formatted_tools) == 1
    assert formatted_tools[0]["type"] == "function"
    assert formatted_tools[0]["function"]["name"] == "get_race_leaderboard"

    # Messages formatting across tool turns
    messages = [
        {"role": "user", "parts": [{"text": "Who is leading?"}]},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {"name": "get_race_leaderboard", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "tool_call_id": "call_123",
                    "function_response": {
                        "name": "get_race_leaderboard",
                        "response": {"result": [{"position": 1, "driver": "VER"}]},
                    },
                }
            ],
        },
    ]

    formatted_msgs = provider._format_messages(messages, system_instruction="You are F1 Assistant.")
    assert formatted_msgs[0]["role"] == "system"
    assert formatted_msgs[0]["content"] == "You are F1 Assistant."
    assert formatted_msgs[1]["role"] == "user"
    assert formatted_msgs[1]["content"] == "Who is leading?"
    assert formatted_msgs[2]["role"] == "assistant"
    assert formatted_msgs[2]["tool_calls"][0]["id"] == "call_123"
    assert formatted_msgs[3]["role"] == "tool"
    assert formatted_msgs[3]["tool_call_id"] == "call_123"
    assert "VER" in formatted_msgs[3]["content"]


def test_nvidia_provider_response_parsing_and_tool_calls(monkeypatch):
    """Verify NVIDIA provider parses chat completion responses and tool calls correctly."""
    from src.race.ai.provider import NvidiaLLMProvider
    import io
    import urllib.request

    mock_resp_json = {
        "id": "chatcmpl-test-1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc_456",
                            "type": "function",
                            "function": {
                                "name": "get_race_leaderboard",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    class MockHTTPResponse:
        def read(self):
            return json.dumps(mock_resp_json).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse())

    provider = NvidiaLLMProvider(api_key="nvapi-mock-key")
    resp = provider.generate_turn(
        messages=[{"role": "user", "content": "Who is leading?"}],
        tools=[{"name": "get_race_leaderboard", "parameters": {}}],
        system_instruction="System prompt",
    )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "get_race_leaderboard"
    assert resp.tool_calls[0].id == "call_abc_456"
    assert resp.tool_calls[0].arguments == {}


def test_nvidia_provider_multiple_tool_calls(monkeypatch):
    """Verify NVIDIA provider handles multiple parallel tool calls in one turn."""
    from src.race.ai.provider import NvidiaLLMProvider
    import urllib.request

    mock_resp_json = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_race_leaderboard", "arguments": "{}"},
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "compare_driver_laps", "arguments": "{\"driver1\": \"VER\", \"driver2\": \"LEC\"}"},
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }

    class MockHTTPResponse:
        def read(self):
            return json.dumps(mock_resp_json).encode("utf-8")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: MockHTTPResponse())

    provider = NvidiaLLMProvider(api_key="nvapi-mock-key")
    resp = provider.generate_turn(
        messages=[{"role": "user", "content": "Tell me standings and compare VER vs LEC"}],
        tools=[],
        system_instruction="",
    )

    assert len(resp.tool_calls) == 2
    assert resp.tool_calls[0].name == "get_race_leaderboard"
    assert resp.tool_calls[1].name == "compare_driver_laps"
    assert resp.tool_calls[1].arguments == {"driver1": "VER", "driver2": "LEC"}


def test_nvidia_provider_authentication_error(monkeypatch):
    """Verify NVIDIA provider raises PermissionError on HTTP 401/403 and never leaks API key."""
    from src.race.ai.provider import NvidiaLLMProvider
    import urllib.request
    import urllib.error

    secret_key = "nvapi-secret-key-to-not-leak"

    def mock_urlopen(req, timeout=None):
        fp = io.BytesIO(f"Invalid API Key: {secret_key}".encode("utf-8"))
        raise urllib.error.HTTPError(
            url="https://integrate.api.nvidia.com/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=fp,
        )

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    provider = NvidiaLLMProvider(api_key=secret_key)
    with pytest.raises(PermissionError) as exc_info:
        provider.generate_turn(messages=[{"role": "user", "content": "Hi"}], tools=[], system_instruction="")

    # API key must NEVER be leaked in exception message
    assert secret_key not in str(exc_info.value)
    assert "authentication failed" in str(exc_info.value).lower()


def test_nvidia_provider_quota_exhaustion_and_cooldown(monkeypatch):
    """Verify NVIDIA provider handles HTTP 429 by setting cooldown and raising QuotaExhaustedError."""
    from src.race.ai.provider import NvidiaLLMProvider, QuotaExhaustedError
    import urllib.request
    import urllib.error

    try:
        def mock_urlopen(req, timeout=None):
            fp = io.BytesIO(b"Rate limit exceeded. Please retry in 15m.")
            raise urllib.error.HTTPError(
                url="https://integrate.api.nvidia.com/v1/chat/completions",
                code=429,
                msg="Too Many Requests",
                hdrs={},
                fp=fp,
            )

        monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

        provider = NvidiaLLMProvider(api_key="nvapi-test")
        with pytest.raises(QuotaExhaustedError) as exc_info:
            provider.generate_turn(messages=[{"role": "user", "content": "Hi"}], tools=[], system_instruction="")

        assert "15m" in str(exc_info.value)
        # Cooldown must be active
        cooldown = provider.get_quota_cooldown_remaining()
        assert cooldown is not None
        assert cooldown[0] > 0
    finally:
        NvidiaLLMProvider.reset_quota_cooldown()


def test_nvidia_orchestrator_multi_turn_tool_execution(monkeypatch):
    """Verify orchestrator runs bounded tool loop with NvidiaLLMProvider and returns grounded response."""
    from src.race.ai.provider import NvidiaLLMProvider
    import urllib.request

    turn_count = 0

    def mock_urlopen(req, timeout=None):
        nonlocal turn_count
        turn_count += 1
        req_data = json.loads(req.data.decode("utf-8"))

        if turn_count == 1:
            # Turn 1: Model requests tool call
            resp_body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_123",
                                    "type": "function",
                                    "function": {"name": "get_race_leaderboard", "arguments": "{}"},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        else:
            # Turn 2: Model receives tool result and produces synthesized answer
            resp_body = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "Max Verstappen (VER) leads the race, followed by Charles Leclerc (LEC).",
                        },
                        "finish_reason": "stop",
                    }
                ]
            }

        class MockResp:
            def read(self):
                return json.dumps(resp_body).encode("utf-8")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        return MockResp()

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    provider = NvidiaLLMProvider(api_key="nvapi-mock-key")
    orchestrator = AIStrategyOrchestrator(
        race_id="2023_monza",
        authoritative_as_of=2000.0,
        provider=provider,
    )

    req = AssistantQueryRequest(prompt="Who is leading the race as of now?")
    response = asyncio.run(orchestrator.run_query(req))

    assert turn_count == 2
    assert response.status == "SUCCESS"
    assert "get_race_leaderboard" in response.tool_calls_executed
    assert "Verstappen" in response.answer or "VER" in response.answer
    assert len(response.evidence) > 0


def test_assistant_status_with_nvidia_provider(monkeypatch):
    """Verify status endpoint reports provider=nvidia and correct model when NVIDIA is configured."""
    monkeypatch.setenv("AI_PROVIDER", "nvidia")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("NVIDIA_MODEL", "meta/llama-3.2-11b-vision-instruct")

    status_resp = client.get("/api/races/2023_monza/assistant/status")
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["provider"] == "nvidia"
    assert data["model"] == "meta/llama-3.2-11b-vision-instruct"
    assert data["available"] is True
    assert data["status_reason"] == "available"

