"""
REST API routes for the RACE Replay Engine.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.race.api.schemas import (
    RaceSummary,
    ClockStatus,
    TrackOutline,
    ComparisonRequest,
    ComparisonResponse,
    TyreDegradationResponse,
    GapBattleResponse,
    BattleOvertakePredictionResponse,
    PitStrategyResponse,
)
from src.race import config
from src.race.config import GEMINI_MODEL_NAME
from src.race.ai.schemas import (
    AssistantStatusResponse,
    AssistantQueryRequest,
    AssistantQueryResponse,
)
from src.race.ai.orchestrator import (
    AIStrategyOrchestrator,
    get_active_llm_provider,
)
from src.race.ai.provider import MockLLMProvider, GeminiLLMProvider
from src.race.api.session_manager import session_manager
from src.race.replay.state import RaceState

router = APIRouter(prefix="/api", tags=["replay"])


@router.get("/races", response_model=List[RaceSummary])
def get_races() -> List[RaceSummary]:
    """List all available cached historical races."""
    return session_manager.list_available_races()


@router.get("/races/{race_id}/clock", response_model=ClockStatus)
def get_clock(race_id: str) -> ClockStatus:
    """Get the current clock and playback status for a race replay."""
    try:
        return session_manager.get_clock_status(race_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/races/{race_id}/state", response_model=RaceState)
def get_replay_state(
    race_id: str,
    as_of: Optional[float] = Query(None, description="Optional simulation timestamp in seconds")
) -> RaceState:
    """
    Get point-in-time public RaceState snapshot.
    If as_of is omitted, returns state at current adapter replay time.
    """
    try:
        return session_manager.get_public_state(race_id, as_of=as_of)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/races/{race_id}/track", response_model=TrackOutline)
def get_track_outline(race_id: str) -> TrackOutline:
    """Get the static track outline geometry for a race circuit."""
    try:
        return session_manager.get_track_outline(race_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Track outline for race '{race_id}' not found.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/races/{race_id}/compare", response_model=ComparisonResponse)
def compare_race_drivers(race_id: str, request: ComparisonRequest) -> ComparisonResponse:
    """
    Compare multi-driver lap and sector performance with strict point-in-time
    as_of replay chronology enforcement and fuel correction.
    """
    try:
        return session_manager.get_driver_comparison(race_id, request)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/races/{race_id}/tyres", response_model=TyreDegradationResponse)
def get_race_tyre_degradation(
    race_id: str,
    driver: str = Query(..., description="Three-letter driver code (e.g. 'VER')"),
    stint: int = Query(1, ge=1, description="Stint number (1, 2, ...)"),
    as_of: Optional[float] = Query(None, description="Optional simulation timestamp in seconds"),
    fuel_correction: float = Query(0.033, description="Fuel correction slope in seconds/lap"),
) -> TyreDegradationResponse:
    """
    Get fuel-corrected tyre degradation analysis, stint lap history,
    estimated degradation rate, and cliff detection for a driver and stint.
    """
    try:
        return session_manager.get_tyre_degradation(
            race_id=race_id,
            driver=driver,
            stint=stint,
            as_of=as_of,
            fuel_correction_s_per_lap=fuel_correction,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/races/{race_id}/battles", response_model=GapBattleResponse)
def get_race_gap_and_battles(
    race_id: str,
    driver1: str = Query(..., description="First driver code (e.g. 'VER')"),
    driver2: str = Query(..., description="Second driver code (e.g. 'LEC')"),
    as_of: Optional[float] = Query(None, description="Optional simulation timestamp in seconds"),
) -> GapBattleResponse:
    """
    Get gap history, battle intervals, completed overtakes, and active battles
    between driver1 and driver2 up to as_of time.
    """
    try:
        return session_manager.get_gap_and_battle_analysis(
            race_id=race_id,
            driver1=driver1,
            driver2=driver2,
            as_of=as_of,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/races/{race_id}/predictions", response_model=BattleOvertakePredictionResponse)
def get_race_predictions(
    race_id: str,
    driver1: str = Query(..., description="First driver code (e.g. 'VER')"),
    driver2: str = Query(..., description="Second driver code (e.g. 'LEC')"),
    as_of: Optional[float] = Query(None, description="Optional simulation timestamp in seconds"),
    horizon_laps: int = Query(5, ge=1, le=10, description="Prediction horizon in laps (default 5)"),
) -> BattleOvertakePredictionResponse:
    """
    Get point-in-time Battle & Overtake probability predictions between driver1 and driver2
    at the as_of replay timestamp, using trained regularized logistic regression pipelines.
    """
    try:
        return session_manager.get_battle_and_overtake_prediction(
            race_id=race_id,
            driver1=driver1,
            driver2=driver2,
            as_of=as_of,
            horizon_laps=horizon_laps,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/races/{race_id}/pits", response_model=PitStrategyResponse)
def get_race_pit_strategy(
    race_id: str,
    driver: str = Query(..., description="Three-letter driver code (e.g. 'VER')"),
    as_of: Optional[float] = Query(None, description="Optional simulation timestamp in seconds"),
    reference_stationary: float = Query(2.5, description="Illustrative benchmark assumption for stationary wheel change (default 2.5s); not a measured sensor value."),
) -> PitStrategyResponse:
    """
    Get point-in-time pit stop strategy and time-loss decomposition for driver
    at the as_of replay timestamp, with explicit lifecycle states, non-overlapping
    time-loss waterfall, and heuristic tactical classifications.
    """
    try:
        return session_manager.get_pit_strategy(
            race_id=race_id,
            driver=driver,
            as_of=as_of,
            reference_stationary_s=reference_stationary,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/races/{race_id}/assistant/status", response_model=AssistantStatusResponse)
def get_assistant_status(race_id: str) -> AssistantStatusResponse:
    """
    Get the availability and configuration status of the AI Strategy Assistant.
    """
    try:
        session_manager.get_or_load_session(race_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    provider = get_active_llm_provider()
    available = provider.is_available()
    provider_type = provider.get_provider_type()
    model_name = provider.get_model_name()

    quota_cooldown = provider.get_quota_cooldown_remaining()

    p_display = "NVIDIA" if provider_type == "nvidia" else ("Gemini" if provider_type == "gemini" else "Mock")
    key_var = "NVIDIA_API_KEY" if provider_type == "nvidia" else "GEMINI_API_KEY"

    if quota_cooldown is not None:
        available = False
        status_reason = "quota_exhausted"
        rem_secs, rem_str = quota_cooldown
        message = (
            f"{p_display} API request quota / rate limit is currently exhausted (HTTP 429). "
            f"Please retry in approximately {rem_str}. Deterministic race telemetry and analytics remain fully active."
        )
    elif available:
        status_reason = "available"
        message = "AI Strategy Assistant is ready to accept queries."
    else:
        has_key = bool(config.get_nvidia_api_key() if provider_type == "nvidia" else config.get_gemini_api_key())
        if not has_key:
            status_reason = "missing_api_key"
            message = (
                f"AI Strategy Assistant is currently unavailable because {key_var} is not configured "
                f"in the backend environment (.env). Set the key to enable live AI strategy queries."
            )
        else:
            status_reason = "provider_error"
            message = (
                f"AI Strategy Assistant is currently unavailable due to an LLM provider initialization error. "
                f"Verify your {key_var} and model configuration."
            )

    tool_names = [
        "get_race_leaderboard",
        "compare_driver_laps",
        "get_tyre_stint_degradation",
        "get_pit_strategy_and_waterfall",
        "get_battle_and_overtake_prediction",
    ]

    return AssistantStatusResponse(
        available=available,
        provider=provider_type,
        model=model_name,
        message=message,
        tools_registered=tool_names,
        status_reason=status_reason,
    )


@router.post("/races/{race_id}/assistant/query", response_model=AssistantQueryResponse)
async def query_assistant(race_id: str, request: AssistantQueryRequest) -> AssistantQueryResponse:
    """
    Submit a natural language race strategy question to the AI Strategy Assistant.
    Enforces server-side authoritative as_of timestamps, bounded deterministic tool calling,
    and grounding validation.
    """
    try:
        session = session_manager.get_or_load_session(race_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=404, detail=f"Race '{race_id}' not found in cache.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Authoritative as_of calculation & validation
    if request.as_of is not None:
        requested_as_of = float(request.as_of)
        if requested_as_of < session.T_0:
            authoritative_as_of = session.T_0
        elif requested_as_of > session.T_finish:
            authoritative_as_of = session.T_finish
        else:
            authoritative_as_of = requested_as_of
    else:
        clock = session_manager.get_clock_status(race_id)
        authoritative_as_of = clock.session_time_s

    orchestrator = AIStrategyOrchestrator(
        race_id=race_id,
        authoritative_as_of=authoritative_as_of,
    )
    return await orchestrator.run_query(request)


