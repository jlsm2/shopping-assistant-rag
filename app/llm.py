"""Provider-neutral LLM interface.

The assistant only talks to `LLMClient`. The concrete provider (which API, which
model) is not chosen yet; when it is, write one adapter that implements
`LLMClient` and return it from `app.api.build_llm()`.

What an adapter must guarantee, because the security model depends on it:
- `system` goes in the provider's system/developer channel, never merged into a
  user turn.
- `response_schema` is enforced with the provider's structured-output feature,
  so `LLMResponse.text` is JSON matching the schema when `stop == "final"`.
- Tool arguments are schema-validated by the provider when it supports it
  (every `ToolSpec.parameters` already sets `additionalProperties: false`).
- A safety refusal maps to `stop="refusal"` and an output cut by the token
  limit maps to `stop="truncated"`. Neither may be reported as "final".
- `AssistantMessage.response.raw` (the provider-native assistant turn) is sent
  back unchanged on the next request of the same tool loop.
- Provider errors are translated to the `LLMError` family below; their
  messages are never shown to the end user.
- `timeout` (seconds left for this question) is a hard limit for the whole
  call, SDK retries included; going over it raises `LLMUnavailable`.
- Output tokens are capped per call, and `LLMResponse.usage` reports what the
  provider billed, because the daily spend ceiling is counted from it.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

StopReason = Literal["final", "tool_calls", "refusal", "truncated"]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Any  # untrusted until validated by the assistant


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


@dataclass(frozen=True)
class LLMResponse:
    stop: StopReason
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = Usage()
    raw: Any = None


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class AssistantMessage:
    response: LLMResponse


@dataclass(frozen=True)
class ToolMessage:
    """All tool results of one model turn, delivered together."""

    results: tuple[ToolResult, ...]


Message = UserMessage | AssistantMessage | ToolMessage


class LLMError(Exception):
    """Any provider failure. The message is for logs only."""


class LLMRateLimited(LLMError):
    def __init__(self, retry_after: int | None = None) -> None:
        super().__init__("provider rate limit")
        self.retry_after = retry_after


class LLMUnavailable(LLMError):
    """Connection failure, timeout or provider 5xx."""


class LLMClient(Protocol):
    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any],
        timeout: float,
    ) -> LLMResponse: ...
