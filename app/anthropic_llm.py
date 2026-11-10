"""`LLMClient` adapter for the Anthropic Messages API (Claude).

The only module that imports the provider SDK. Everything it must guarantee
is listed in `app/llm.py`.
"""

from collections.abc import Sequence
from typing import Any

import anthropic

from app.llm import (
    AssistantMessage,
    LLMError,
    LLMRateLimited,
    LLMResponse,
    LLMUnavailable,
    Message,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
    UserMessage,
)

REFUSAL_FALLBACK_BETA = "server-side-fallback-2026-07-01"
SDK_RETRIES = 1
_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "final",
    "tool_use": "tool_calls",
    "refusal": "refusal",
    "max_tokens": "truncated",
    "model_context_window_exceeded": "truncated",
}


def _to_param(message: Message) -> dict[str, Any]:
    if isinstance(message, UserMessage):
        return {"role": "user", "content": message.text}
    if isinstance(message, AssistantMessage):
        # The provider's own blocks go back untouched: thinking blocks are only
        # valid when replayed exactly as they came.
        return {"role": "assistant", "content": message.response.raw}
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": result.call_id,
                "content": result.content,
                "is_error": result.is_error,
            }
            for result in message.results
        ],
    }


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        effort: str | None = "low",
        max_tokens: int = 4096,
        server_fallback: bool = True,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.server_fallback = server_fallback
        self._client = client or anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any],
        timeout: float,
    ) -> LLMResponse:
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": response_schema}
        }
        if self.effort:
            output_config["effort"] = self.effort
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,  # thinking counts toward it too
            # Stable text, so it is cacheable. Below the model's minimum cacheable
            # length the marker is simply ignored.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [_to_param(message) for message in messages],
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.parameters,
                    "strict": True,
                }
                for tool in tools
            ],
            "output_config": output_config,
        }
        # One retry rides out a transient overload. Each attempt gets half the
        # time left, so both together stay inside the question's deadline.
        client = self._client.with_options(
            timeout=timeout / (SDK_RETRIES + 1), max_retries=SDK_RETRIES
        )
        try:
            if self.server_fallback:
                # A safety decline is re-run on the model Anthropic recommends
                # for that category, inside the same call.
                response = client.beta.messages.create(
                    betas=[REFUSAL_FALLBACK_BETA], fallbacks="default", **request
                )
            else:
                response = client.messages.create(**request)
        except anthropic.RateLimitError as error:
            retry_after = error.response.headers.get("retry-after", "")
            raise LLMRateLimited(int(retry_after) if retry_after.isdigit() else None) from error
        except anthropic.APIStatusError as error:
            if error.status_code >= 500:
                raise LLMUnavailable(f"provider {error.status_code}") from error
            raise LLMError(f"provider {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:  # includes timeouts
            raise LLMUnavailable(type(error).__name__) from error
        return _to_response(response)


def _to_response(response: Any) -> LLMResponse:
    stop: StopReason = _STOP_REASONS.get(response.stop_reason, "other")
    text = next((block.text for block in response.content if block.type == "text"), "")
    tool_calls = tuple(
        ToolCall(id=block.id, name=block.name, arguments=block.input)
        for block in response.content
        if block.type == "tool_use"
    )
    if (stop == "tool_calls" and not tool_calls) or (stop == "final" and not text):
        stop = "other"
    usage = response.usage
    return LLMResponse(
        stop=stop,
        text=text,
        tool_calls=tool_calls,
        usage=Usage(
            input_tokens=usage.input_tokens
            + (usage.cache_creation_input_tokens or 0)
            + (usage.cache_read_input_tokens or 0),
            output_tokens=usage.output_tokens,
        ),
        raw=response.content,
    )
