"""
LLM Provider abstractions and implementations (Gemini + Mock).
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from src.race import config

logger = logging.getLogger(__name__)


class QuotaExhaustedError(RuntimeError):
    """Raised when the LLM provider daily/rate quota is exhausted (HTTP 429)."""

    def __init__(
        self,
        message: str,
        retry_delay_seconds: Optional[float] = None,
        retry_delay_str: Optional[str] = None,
        quota_metric: Optional[str] = None,
    ):
        super().__init__(message)
        self.retry_delay_seconds = retry_delay_seconds
        self.retry_delay_str = retry_delay_str
        self.quota_metric = quota_metric


def extract_quota_retry_info(clean_msg: str, error_obj: Optional[Any] = None) -> Tuple[Optional[float], Optional[str]]:
    """
    Extracts numerical retry delay in seconds and human-readable string from a 429 error.
    """
    retry_secs: Optional[float] = None
    retry_str: Optional[str] = None

    # 1. Check structured details if available
    if error_obj and hasattr(error_obj, "details") and isinstance(error_obj.details, dict):
        for item in error_obj.details.get("error", {}).get("details", []):
            if "retryDelay" in item:
                d = str(item["retryDelay"]).strip()
                if d.endswith("s"):
                    try:
                        retry_secs = float(d[:-1])
                    except Exception:
                        pass

    # 2. Check regex on message (e.g. 'Please retry in 16h12m11.937193772s.')
    m = re.search(r"Please retry in\s+([^\n\.]+)", clean_msg, re.IGNORECASE)
    if m:
        raw_dur = m.group(1).strip()
        hm = re.search(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", raw_dur)
        if hm:
            h, mins, s = hm.groups()
            parts = []
            if h:
                parts.append(f"{h}h")
            if mins:
                parts.append(f"{mins}m")
            if s and not h:
                parts.append(f"{int(float(s))}s")
            if parts:
                retry_str = " ".join(parts)
            if retry_secs is None:
                sec_val = 0.0
                if h:
                    sec_val += float(h) * 3600
                if mins:
                    sec_val += float(mins) * 60
                if s:
                    sec_val += float(s)
                if sec_val > 0:
                    retry_secs = sec_val
        elif not retry_str:
            retry_str = raw_dur

    # 3. Format from seconds if retry_secs is available but string is not
    if retry_secs is not None and not retry_str:
        total_seconds = int(retry_secs)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 and hours == 0:
            parts.append(f"{seconds}s")
        retry_str = " ".join(parts) if parts else f"{total_seconds}s"

    return retry_secs, retry_str


class ToolCall(BaseModel):
    """Normalized tool call request from an LLM."""
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    id: Optional[str] = None


class LLMResponse(BaseModel):
    """Normalized response from an LLM turn."""
    model_config = {"arbitrary_types_allowed": True}
    text: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    raw_content: Optional[Any] = None


class LLMProvider(ABC):
    """Abstract interface for LLM inference providers."""

    @abstractmethod
    def get_provider_type(self) -> str:
        """Return provider type identifier ('gemini', 'nvidia', or 'mock')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider is configured and available for inference."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the active model name string."""
        pass

    def get_quota_cooldown_remaining(self) -> Optional[Tuple[float, str]]:
        """Return (remaining_seconds, remaining_str) if currently in quota cooldown."""
        return None

    @abstractmethod
    def generate_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> LLMResponse:
        """Execute a single model turn given history, tool declarations, and instructions."""
        pass

    @abstractmethod
    async def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """Execute async inference turn and return dict format with text and tool_calls."""
        pass


class GeminiLLMProvider(LLMProvider):
    """
    Production LLM provider using the official google-genai SDK.
    Strictly prevents leaking API keys or secrets.
    """
    _SENTINEL = object()
    _global_quota_cooldown_until: Optional[float] = None
    _global_quota_retry_str: Optional[str] = None

    def __init__(
        self,
        api_key: Any = _SENTINEL,
        model_name: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        if api_key is self._SENTINEL:
            self.api_key = config.get_gemini_api_key()
        else:
            self.api_key = api_key or ""

        raw_model = model_name or config.get_gemini_model_name()
        # Automatically map retired model names to active Google GenAI models
        if raw_model == "gemini-2.5-flash":
            raw_model = "gemini-3.5-flash"
        self.model_name = raw_model
        self.timeout_s = timeout_s or config.AI_REQUEST_TIMEOUT_S
        self._client = None

        if self.model_name not in config.SUPPORTED_GEMINI_MODELS:
            raise ValueError(
                f"Configured model '{self.model_name}' is not in supported list: "
                f"{sorted(list(config.SUPPORTED_GEMINI_MODELS))}"
            )

        if self.api_key:
            try:
                from google import genai
                from google.genai import types
                # Fail-fast without tenacity looping on 429 quota exhaustion
                http_opts = types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
                self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
            except Exception:
                self._client = None

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
                http_opts = types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)
                )
                self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
            except Exception:
                self._client = None
        return bool(self.api_key and self._client is not None)

    def get_provider_type(self) -> str:
        return "gemini"

    def get_model_name(self) -> str:
        return self.model_name

    @classmethod
    def get_quota_cooldown_remaining(cls) -> Optional[Tuple[float, str]]:
        """Returns (remaining_seconds, remaining_str) if currently in quota cooldown."""
        if cls._global_quota_cooldown_until is not None:
            rem = cls._global_quota_cooldown_until - time.time()
            if rem > 0:
                return rem, cls._global_quota_retry_str or f"{int(rem)}s"
            cls._global_quota_cooldown_until = None
            cls._global_quota_retry_str = None
        return None

    @classmethod
    def reset_quota_cooldown(cls) -> None:
        """Reset quota cooldown state (e.g. for testing)."""
        cls._global_quota_cooldown_until = None
        cls._global_quota_retry_str = None

    @staticmethod
    def _convert_messages_to_contents(messages: List[Any]) -> List[Any]:
        """
        Converts diverse message representations (parts dictionaries, flat dicts,
        function calls, and function responses) into official google.genai types.Content objects.
        """
        from google.genai import types

        contents: List[types.Content] = []
        for msg in messages:
            if isinstance(msg, types.Content):
                contents.append(msg)
                continue
            if isinstance(msg, str):
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=msg)]))
                continue
            if not isinstance(msg, dict):
                continue

            raw_role = msg.get("role", "user")
            role = "model" if raw_role in ("assistant", "model") else "user"
            parts: List[types.Part] = []

            # 1. Unpack 'parts' array if present (standard orchestrator format)
            raw_parts = msg.get("parts", [])
            for p in raw_parts:
                if isinstance(p, types.Part):
                    parts.append(p)
                elif isinstance(p, str):
                    parts.append(types.Part.from_text(text=p))
                elif isinstance(p, dict):
                    if "text" in p and p["text"]:
                        parts.append(types.Part.from_text(text=str(p["text"])))
                    elif "function_call" in p:
                        fc = p["function_call"]
                        if hasattr(fc, "name") and hasattr(fc, "args"):
                            parts.append(types.Part.from_function_call(name=fc.name, args=fc.args or {}))
                        elif isinstance(fc, dict):
                            name = fc.get("name", "")
                            args = fc.get("args") if fc.get("args") is not None else fc.get("arguments", {})
                            parts.append(types.Part.from_function_call(name=name, args=args or {}))
                    elif "function_response" in p:
                        fr = p["function_response"]
                        if hasattr(fr, "name") and hasattr(fr, "response"):
                            resp_data = fr.response if isinstance(fr.response, dict) else {"result": fr.response}
                            parts.append(types.Part.from_function_response(name=fr.name, response=resp_data))
                        elif isinstance(fr, dict):
                            name = fr.get("name", "")
                            resp = fr.get("response", {})
                            resp_data = resp if isinstance(resp, dict) else {"result": resp}
                            parts.append(types.Part.from_function_response(name=name, response=resp_data))

            # 2. Check flat keys if parts array was empty
            if not parts:
                if "text" in msg and msg["text"]:
                    parts.append(types.Part.from_text(text=str(msg["text"])))
                if "content" in msg and msg["content"]:
                    parts.append(types.Part.from_text(text=str(msg["content"])))
                if "function_calls" in msg and msg["function_calls"]:
                    for fc in msg["function_calls"]:
                        if isinstance(fc, dict):
                            parts.append(types.Part.from_function_call(
                                name=fc.get("name", ""),
                                args=fc.get("args") or fc.get("arguments") or {},
                            ))
                        elif hasattr(fc, "name"):
                            parts.append(types.Part.from_function_call(
                                name=fc.name,
                                args=getattr(fc, "args", {}) or getattr(fc, "arguments", {}),
                            ))
                if "function_responses" in msg and msg["function_responses"]:
                    for fr in msg["function_responses"]:
                        if isinstance(fr, dict):
                            resp = fr.get("response", {})
                            resp_data = resp if isinstance(resp, dict) else {"result": resp}
                            parts.append(types.Part.from_function_response(
                                name=fr.get("name", ""),
                                response=resp_data,
                            ))

            if parts:
                contents.append(types.Content(role=role, parts=parts))

        return contents

    def generate_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> LLMResponse:
        if not self.is_available():
            raise RuntimeError("GeminiLLMProvider is unavailable (missing or invalid GEMINI_API_KEY).")

        from google.genai import types

        # Build tools declarations
        genai_tools = []
        if tools:
            func_decls = []
            for t in tools:
                func_decls.append(
                    types.FunctionDeclaration(
                        name=t["name"],
                        description=t["description"],
                        parameters=t.get("parameters", {}),
                    )
                )
            genai_tools.append(types.Tool(function_declarations=func_decls))

        # Build contents from messages with complete parts support
        contents = self._convert_messages_to_contents(messages)
        if not contents:
            raise ValueError("No valid message content could be extracted from input messages.")

        genai_config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            tools=genai_tools if genai_tools else None,
        )

        # Check active quota cooldown to avoid futile repeated requests
        cooldown = self.get_quota_cooldown_remaining()
        if cooldown is not None:
            rem_secs, rem_str = cooldown
            raise QuotaExhaustedError(
                f"Gemini API quota is currently exhausted. Please retry in approximately {rem_str}.",
                retry_delay_seconds=rem_secs,
                retry_delay_str=rem_str,
            )

        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=genai_config,
            )
        except QuotaExhaustedError:
            raise
        except Exception as e:
            clean_msg = str(e)
            if self.api_key and self.api_key in clean_msg:
                clean_msg = clean_msg.replace(self.api_key, "[REDACTED_API_KEY]")

            lower_msg = clean_msg.lower()

            # 1. Quota Exhaustion (HTTP 429 / RESOURCE_EXHAUSTED)
            is_429 = getattr(e, "code", None) == 429 or getattr(e, "status_code", None) == 429
            is_resource_exhausted = (
                "resource_exhausted" in lower_msg
                or "quota" in lower_msg
                or "429" in lower_msg
            )

            if is_429 or is_resource_exhausted:
                retry_secs, retry_str = extract_quota_retry_info(clean_msg, e)
                if retry_secs and retry_secs > 0:
                    GeminiLLMProvider._global_quota_cooldown_until = time.time() + retry_secs
                    GeminiLLMProvider._global_quota_retry_str = retry_str
                logger.warning(
                    "Gemini API quota exhausted (HTTP 429). Model: %s, Retry delay: %s",
                    self.model_name,
                    retry_str or "unspecified",
                )
                raise QuotaExhaustedError(
                    f"Gemini API quota exhausted (HTTP 429: RESOURCE_EXHAUSTED). Model: {self.model_name}.",
                    retry_delay_seconds=retry_secs,
                    retry_delay_str=retry_str,
                ) from None

            # 2. Authentication failure (401, 403, invalid key) - never confuse with quota exhaustion
            if any(err_term in lower_msg for err_term in ["api_key_invalid", "unauthenticated", "401", "403", "permission_denied"]):
                logger.error("Gemini API authentication failed. Model: %s", self.model_name)
                raise PermissionError(f"Gemini authentication failed: {clean_msg}") from None

            # 3. Standard provider error
            logger.error("Gemini API request failed. Model: %s, Error: %s", self.model_name, clean_msg)
            raise RuntimeError(f"Gemini API error ({self.model_name}): {clean_msg}") from None

        # Parse output
        tool_calls = []
        if resp.function_calls:
            for call in resp.function_calls:
                args = call.args if isinstance(call.args, dict) else {}
                tool_calls.append(ToolCall(name=call.name, arguments=args))

        response_text = resp.text if hasattr(resp, "text") and resp.text else None
        raw_content = None
        if hasattr(resp, "candidates") and resp.candidates:
            raw_content = resp.candidates[0].content

        return LLMResponse(
            text=response_text,
            tool_calls=tool_calls,
            raw_content=raw_content,
        )

    async def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("GEMINI_API_KEY is not configured in environment.")

        import asyncio
        resp = await asyncio.to_thread(
            self.generate_turn, messages, tools, system_instruction, temperature
        )
        return {
            "text": resp.text or "",
            "tool_calls": [
                {"name": tc.name, "args": tc.arguments, "id": tc.id}
                for tc in resp.tool_calls
            ],
            "raw_content": resp.raw_content,
        }


class NvidiaLLMProvider(LLMProvider):
    """
    Production LLM provider using NVIDIA NIM OpenAI-compatible API endpoint.
    Communicates with https://integrate.api.nvidia.com/v1.
    Strictly prevents leaking API keys or secrets in logs, responses, or exceptions.
    """
    _SENTINEL = object()
    _global_quota_cooldown_until: Optional[float] = None
    _global_quota_retry_str: Optional[str] = None

    def __init__(
        self,
        api_key: Any = _SENTINEL,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_s: Optional[float] = None,
    ):
        if api_key is self._SENTINEL:
            self.api_key = config.get_nvidia_api_key()
        else:
            self.api_key = api_key or ""

        self.model_name = (model_name or config.get_nvidia_model_name()).strip()
        self.base_url = (base_url or config.get_nvidia_base_url()).strip().rstrip("/")
        self.timeout_s = timeout_s or config.AI_REQUEST_TIMEOUT_S

    def get_provider_type(self) -> str:
        return "nvidia"

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_model_name(self) -> str:
        return self.model_name

    @classmethod
    def get_quota_cooldown_remaining(cls) -> Optional[Tuple[float, str]]:
        if cls._global_quota_cooldown_until is not None:
            rem = cls._global_quota_cooldown_until - time.time()
            if rem > 0:
                return rem, cls._global_quota_retry_str or f"{int(rem)}s"
            cls._global_quota_cooldown_until = None
            cls._global_quota_retry_str = None
        return None

    @classmethod
    def reset_quota_cooldown(cls) -> None:
        cls._global_quota_cooldown_until = None
        cls._global_quota_retry_str = None

    @staticmethod
    def _format_tools(tools: List[Any]) -> List[Dict[str, Any]]:
        """Formats tool definitions into OpenAI function format expected by NVIDIA."""
        formatted_tools: List[Dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                if "type" in t and "function" in t:
                    formatted_tools.append(t)
                elif "name" in t:
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {"type": "object", "properties": {}}),
                        },
                    })
        return formatted_tools

    @staticmethod
    def _format_messages(messages: List[Dict[str, Any]], system_instruction: str = "") -> List[Dict[str, Any]]:
        """
        Converts internal multi-turn messages into OpenAI-compatible chat completion messages.
        Supports text, function calls, and tool responses.
        """
        formatted: List[Dict[str, Any]] = []
        if system_instruction:
            formatted.append({"role": "system", "content": system_instruction})

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "user")

            if role == "system":
                content = msg.get("content") or msg.get("text") or ""
                formatted.append({"role": "system", "content": content})
                continue

            if role in ("assistant", "model"):
                # Check for OpenAI format raw tool calls or parts format
                tool_calls_raw = msg.get("tool_calls")
                if tool_calls_raw:
                    tcs = []
                    for idx, tc in enumerate(tool_calls_raw):
                        if isinstance(tc, dict):
                            tc_id = tc.get("id") or f"call_{idx}"
                            func = tc.get("function") or {"name": tc.get("name"), "arguments": tc.get("args") or tc.get("arguments", {})}
                            raw_args = func.get("arguments", "{}")
                            if isinstance(raw_args, dict):
                                args_str = json.dumps(raw_args)
                            else:
                                args_str = str(raw_args or "{}")
                            tcs.append({
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": func.get("name", ""),
                                    "arguments": args_str,
                                },
                            })
                    formatted.append({
                        "role": "assistant",
                        "content": msg.get("content") or None,
                        "tool_calls": tcs if tcs else None,
                    })
                    continue

                # Check for Gemini-style parts with function_call
                parts = msg.get("parts", [])
                func_calls = []
                text_parts = []
                for idx, p in enumerate(parts):
                    if isinstance(p, dict):
                        if "function_call" in p:
                            fc = p["function_call"]
                            fc_name = fc.get("name", "")
                            fc_args = fc.get("args", {})
                            func_calls.append({
                                "id": f"call_{idx}",
                                "type": "function",
                                "function": {
                                    "name": fc_name,
                                    "arguments": json.dumps(fc_args) if isinstance(fc_args, dict) else str(fc_args),
                                },
                            })
                        elif "text" in p and p["text"]:
                            text_parts.append(p["text"])

                if func_calls:
                    formatted.append({
                        "role": "assistant",
                        "content": "\n".join(text_parts) if text_parts else None,
                        "tool_calls": func_calls,
                    })
                else:
                    content = msg.get("content") or "\n".join(text_parts) or msg.get("text") or ""
                    formatted.append({"role": "assistant", "content": content})
                continue

            if role == "tool":
                formatted.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", "call_0"),
                    "name": msg.get("name", ""),
                    "content": msg.get("content", ""),
                })
                continue

            if role == "user":
                parts = msg.get("parts", [])
                # Check if parts contain function_response
                has_function_response = False
                for idx, p in enumerate(parts):
                    if isinstance(p, dict) and "function_response" in p:
                        has_function_response = True
                        fr = p["function_response"]
                        fn_name = fr.get("name", "")
                        resp_data = fr.get("response", {}).get("result", fr.get("response", {}))
                        call_id = p.get("tool_call_id") or f"call_{idx}"
                        formatted.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": fn_name,
                            "content": json.dumps(resp_data) if not isinstance(resp_data, str) else resp_data,
                        })

                if not has_function_response:
                    text_parts = []
                    if "content" in msg and msg["content"]:
                        text_parts.append(msg["content"])
                    for p in parts:
                        if isinstance(p, dict) and "text" in p and p["text"]:
                            text_parts.append(p["text"])
                        elif isinstance(p, str):
                            text_parts.append(p)
                    user_content = "\n".join(text_parts) if text_parts else msg.get("text", "")
                    formatted.append({"role": "user", "content": user_content})

        return formatted

    def generate_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str,
        temperature: float = 0.2,
    ) -> LLMResponse:
        cooldown = self.get_quota_cooldown_remaining()
        if cooldown is not None:
            rem_s, rem_str = cooldown
            raise QuotaExhaustedError(
                f"NVIDIA API rate limit or quota is exhausted (HTTP 429). Please retry in approximately {rem_str}.",
                retry_delay_seconds=rem_s,
                retry_delay_str=rem_str,
            )

        if not self.is_available():
            raise RuntimeError("NVIDIA provider is unavailable: missing NVIDIA_API_KEY.")

        openai_tools = self._format_tools(tools)
        formatted_messages = self._format_messages(messages, system_instruction)

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": float(temperature),
        }
        if openai_tools:
            payload["tools"] = openai_tools
            payload["tool_choice"] = "auto"

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(endpoint, data=data_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                resp_bytes = resp.read()
                raw_json = json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            raw_err = e.read().decode("utf-8", errors="replace")
            clean_err = raw_err
            if self.api_key:
                clean_err = clean_err.replace(self.api_key, "[REDACTED]")

            if e.code in (401, 403):
                raise PermissionError("NVIDIA API authentication failed. The configured NVIDIA_API_KEY is invalid or unauthorized.")

            if e.code == 429:
                retry_secs, retry_str = extract_quota_retry_info(clean_err)
                if retry_secs is None:
                    retry_secs = 60.0
                    retry_str = "60s"
                NvidiaLLMProvider._global_quota_cooldown_until = time.time() + retry_secs
                NvidiaLLMProvider._global_quota_retry_str = retry_str
                raise QuotaExhaustedError(
                    f"NVIDIA API rate limit or quota exceeded (HTTP 429). Please retry in {retry_str}.",
                    retry_delay_seconds=retry_secs,
                    retry_delay_str=retry_str,
                )

            if e.code == 404:
                raise RuntimeError(
                    f"NVIDIA API endpoint or model '{self.model_name}' not found (HTTP 404). "
                    "Please verify that this model is accessible for your NVIDIA account."
                )

            raise RuntimeError(f"NVIDIA API error (HTTP {e.code}): {clean_err}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"NVIDIA API connection failure: {e.reason}")
        except TimeoutError:
            raise RuntimeError(f"NVIDIA API request timed out after {self.timeout_s}s.")
        except Exception as e:
            clean_err = str(e)
            if self.api_key:
                clean_err = clean_err.replace(self.api_key, "[REDACTED]")
            raise RuntimeError(f"NVIDIA API communication failure: {clean_err}")

        choices = raw_json.get("choices", [])
        if not choices:
            return LLMResponse(text="", raw_response=raw_json)

        choice = choices[0]
        choice_msg = choice.get("message", {})
        content_text = choice_msg.get("content") or ""
        finish_reason = choice.get("finish_reason")

        tool_calls: List[ToolCall] = []
        raw_tool_calls = choice_msg.get("tool_calls", [])
        for raw_tc in raw_tool_calls:
            tc_id = raw_tc.get("id")
            func = raw_tc.get("function", {})
            fn_name = func.get("name", "")
            raw_args = func.get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args_dict = json.loads(raw_args)
                except Exception:
                    args_dict = {}
            elif isinstance(raw_args, dict):
                args_dict = raw_args
            else:
                args_dict = {}

            tool_calls.append(ToolCall(id=tc_id, name=fn_name, arguments=args_dict))

        return LLMResponse(
            text=content_text if content_text else None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            raw_response=raw_json,
            raw_content=choice_msg,
        )

    async def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        import asyncio
        resp = await asyncio.to_thread(
            self.generate_turn, messages, tools, system_instruction, temperature
        )
        return {
            "text": resp.text or "",
            "tool_calls": [
                {"name": tc.name, "args": tc.arguments, "id": tc.id}
                for tc in resp.tool_calls
            ],
            "raw_content": resp.raw_content,
        }


class MockLLMProvider(LLMProvider):
    """
    Deterministic offline mock provider for testing without external API calls or credentials.
    Supports scripted responses, custom handlers, or pattern-matching heuristic simulation.
    """

    def __init__(self, model_name: str = "mock-f1-assistant"):
        self.model_name = model_name
        self.custom_handler = None
        self.call_history: List[Dict[str, Any]] = []

    def get_provider_type(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return True

    def get_model_name(self) -> str:
        return self.model_name

    def set_custom_handler(self, handler: Optional[Any]) -> None:
        self.custom_handler = handler

    def generate_turn(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> LLMResponse:
        self.call_history.append({"messages": messages, "tools": tools})

        if self.custom_handler is not None:
            import inspect
            sig = inspect.signature(self.custom_handler)
            if len(sig.parameters) == 2:
                res = self.custom_handler(messages, tools)
            else:
                res = self.custom_handler(messages, tools, system_instruction)

            if isinstance(res, LLMResponse):
                return res
            if isinstance(res, dict):
                t_calls = []
                for tc in res.get("tool_calls", []):
                    t_calls.append(ToolCall(name=tc["name"], arguments=tc.get("args") or tc.get("arguments", {})))
                return LLMResponse(text=res.get("text"), tool_calls=t_calls)
            return LLMResponse(text=str(res))

        # Inspect the conversation so far
        last_msg = messages[-1] if messages else {}

        # Extract function responses from flat or parts format
        fr_list = []
        if "function_responses" in last_msg and last_msg["function_responses"]:
            fr_list = last_msg["function_responses"]
        elif "parts" in last_msg and last_msg["parts"]:
            for p in last_msg["parts"]:
                if isinstance(p, dict) and "function_response" in p:
                    fr_obj = p["function_response"]
                    resp_dict = fr_obj.get("response", {})
                    if "result" in resp_dict and isinstance(resp_dict["result"], dict):
                        data_val = resp_dict["result"]
                    else:
                        data_val = resp_dict
                    fr_list.append({"name": fr_obj.get("name"), "response": data_val})

        if fr_list:
            fr = fr_list[0]
            tool_name = fr.get("name")
            data = fr.get("response", {})

            if tool_name == "get_race_leaderboard":
                leaderboard = data.get("leaderboard", [])
                p1 = leaderboard[0]["driver"] if len(leaderboard) > 0 else "N/A"
                p2 = leaderboard[1]["driver"] if len(leaderboard) > 1 else "N/A"
                raw_gap = leaderboard[1].get("gap_to_leader_s") if len(leaderboard) > 1 else None
                gap_str = f"trailing the leader by {raw_gap:.3f}s" if (raw_gap is not None and isinstance(raw_gap, (int, float))) else "following closely at race start"
                return LLMResponse(
                    text=f"As of lap {data.get('current_lap', 1)}, {p1} leads the race. {p2} is running in P2, {gap_str} under track status {data.get('track_status', 'UNKNOWN')}."
                )

            elif tool_name == "compare_driver_laps":
                d1 = data.get("driver1") or data.get("driver_a")
                d2 = data.get("driver2") or data.get("driver_b")
                delta_s = data.get("mean_delta_s") if data.get("mean_delta_s") is not None else data.get("lap_delta_s", 0.0)
                fastest = d2 if delta_s > 0 else d1
                slower = d1 if delta_s > 0 else d2
                abs_delta = abs(delta_s)
                return LLMResponse(
                    text=f"Between {d1} and {d2}, {fastest} was faster by {abs_delta:.3f}s. {slower} had a lap delta of {delta_s:+.3f}s relative to the reference."
                )

            elif tool_name == "get_tyre_stint_degradation":
                driver = data.get("driver")
                stint = data.get("stint")
                compound = data.get("compound")
                wear = data.get("slope_s_per_lap") or data.get("wear_slope_s_per_lap")
                r2 = data.get("fit_r2")
                if wear is not None:
                    return LLMResponse(
                        text=f"{driver} on Stint {stint} ({compound}) showed a linear tyre degradation rate of {wear:+.3f}s/lap with an R2 fit quality of {r2:.2f} across {data.get('clean_laps_count', 0)} clean flying laps."
                    )
                else:
                    return LLMResponse(
                        text=f"Tyre degradation analysis for {driver} on Stint {stint} ({compound}) has insufficient clean flying laps to calculate a reliable slope."
                    )

            elif tool_name == "get_pit_strategy_and_waterfall":
                driver = data.get("driver")
                stops = data.get("stops", [])
                total_stops = data.get("total_completed_stops", 0)
                net_loss = data.get("net_pit_loss_s", 0.0)
                if stops or total_stops > 0:
                    return LLMResponse(
                        text=f"{driver} has completed {total_stops} official pit stop(s) with an estimated net pit loss of +{net_loss:.2f}s (reference stationary time is an illustrative 2.5s assumption)."
                    )
                else:
                    return LLMResponse(
                        text=f"No pit stops recorded for {driver} up to the requested simulation timestamp."
                    )

            elif tool_name == "get_battle_and_overtake_prediction":
                attacker = data.get("chaser") or data.get("driver1")
                defender = data.get("leader") or data.get("driver2")
                p_battle = data.get("p_battle", 0.0)
                p_overtake = data.get("p_overtake", 0.0)
                gap = data.get("gap_s", 0.0)
                return LLMResponse(
                    text=f"At a current gap of {gap:.2f}s between {attacker} and {defender}, the projected 5-lap battle probability is {p_battle * 100:.1f}%, with an on-track overtake probability of {p_overtake * 100:.1f}%."
                )

            return LLMResponse(text=f"Verified data retrieved from tool {tool_name}.")

        # If incoming user query, inspect text to decide whether unmeasured guardrail or tool call
        user_text = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                if "text" in m and m["text"]:
                    user_text = m["text"]
                    break
                for p in m.get("parts", []):
                    if isinstance(p, dict) and "text" in p and p["text"]:
                        user_text = p["text"]
                        break
                if user_text:
                    break

        lowered = user_text.lower()

        # Guardrail check for unmeasured variables
        unmeasured_patterns = [
            r"\bengine\s+mode",
            r"\bers\s+deploy",
            r"\bintent",
            r"\bmindset",
            r"\bfeeling",
            r"\bangry",
            r"\bfrustrated",
            r"\bsecret\s+strat",
            r"\bcar\s+damage\b",
        ]
        if any(re.search(p, lowered) for p in unmeasured_patterns):
            return LLMResponse(
                text="Cannot tell from the data. Official telemetry feeds do not expose engine modes, driver intent, internal damage, or unmeasured psychological factors."
            )

        # Detect driver codes and names from text
        driver_name_map = {
            "verstappen": "VER", "max": "VER", "ver": "VER",
            "perez": "PER", "checo": "PER", "per": "PER",
            "leclerc": "LEC", "charles": "LEC", "lec": "LEC",
            "sainz": "SAI", "carlos": "SAI", "sai": "SAI",
            "hamilton": "HAM", "lewis": "HAM", "ham": "HAM",
            "russell": "RUS", "george": "RUS", "rus": "RUS",
            "norris": "NOR", "lando": "NOR", "nor": "NOR",
            "piastri": "PIA", "oscar": "PIA", "pia": "PIA",
            "alonso": "ALO", "fernando": "ALO", "alo": "ALO",
            "stroll": "STR", "lance": "STR", "str": "STR",
            "gasly": "GAS", "pierre": "GAS", "gas": "GAS",
            "ocon": "OCO", "esteban": "OCO", "oco": "OCO",
            "albon": "ALB", "alex": "ALB", "alb": "ALB",
            "sargeant": "SAR", "logan": "SAR", "sar": "SAR",
            "tsunoda": "TSU", "yuki": "TSU", "tsu": "TSU",
            "ricciardo": "RIC", "daniel": "RIC", "ric": "RIC",
            "bottas": "BOT", "valtteri": "BOT", "bot": "BOT",
            "zhou": "ZHO", "guanyu": "ZHO", "zho": "ZHO",
            "magnussen": "MAG", "kevin": "MAG", "mag": "MAG",
            "hulkenberg": "HUL", "nico": "HUL", "hul": "HUL",
            "lawson": "LAW", "liam": "LAW", "law": "LAW",
        }
        found_drivers = []
        for word in re.findall(r"[A-Za-z]+", user_text):
            w_lower = word.lower()
            if w_lower in driver_name_map:
                code = driver_name_map[w_lower]
                if code not in found_drivers:
                    found_drivers.append(code)
            elif len(word) == 3 and word.isupper():
                if word not in found_drivers:
                    found_drivers.append(word)

        d1 = found_drivers[0] if len(found_drivers) > 0 else "VER"
        d2 = found_drivers[1] if len(found_drivers) > 1 else ("LEC" if d1 != "LEC" else "SAI")

        # Match query type to tool call
        if any(w in lowered for w in ["p2", "leaderboard", "position", "leading", "who is", "order", "standings"]):
            return LLMResponse(
                tool_calls=[ToolCall(name="get_race_leaderboard", arguments={})]
            )

        if any(w in lowered for w in ["slower", "faster", "compare", "gap between", "delta", "losing time", "lost time"]):
            return LLMResponse(
                tool_calls=[ToolCall(name="compare_driver_laps", arguments={"driver1": d1, "driver2": d2, "driver_a": d1, "driver_b": d2, "mode": "single"})]
            )

        if any(w in lowered for w in ["tyre", "tire", "degradation", "wear", "stint"]):
            return LLMResponse(
                tool_calls=[ToolCall(name="get_tyre_stint_degradation", arguments={"driver": d1, "stint": 1})]
            )

        if any(w in lowered for w in ["pit", "stop", "waterfall", "undercut", "overcut"]):
            return LLMResponse(
                tool_calls=[ToolCall(name="get_pit_strategy_and_waterfall", arguments={"driver": d1})]
            )

        if any(w in lowered for w in ["overtake", "pass", "battle", "chance", "probability", "odds"]):
            return LLMResponse(
                tool_calls=[ToolCall(name="get_battle_and_overtake_prediction", arguments={"chaser": d1, "leader": d2, "driver1": d1, "driver2": d2})]
            )

        # Default fallback: leaderboard tool
        return LLMResponse(
            tool_calls=[ToolCall(name="get_race_leaderboard", arguments={})]
        )

    async def generate_content(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Any],
        system_instruction: str = "",
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        resp = self.generate_turn(messages, tools, system_instruction, temperature)
        return {
            "text": resp.text or "",
            "tool_calls": [
                {"name": tc.name, "args": tc.arguments}
                for tc in resp.tool_calls
            ],
            "raw_content": getattr(resp, "raw_content", None),
        }
