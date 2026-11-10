"""`LLMClient` adapter for any API that speaks the OpenAI Chat Completions format.

That covers OpenAI itself and the many services and local servers that copy
its format; `LLM_BASE_URL` says which one. The model must support tool calling
and JSON-schema output. Everything an adapter must guarantee is in `app/llm.py`.
"""

import json
from collections.abc import Sequence
from typing import Any

import openai

from app.llm import (
    SDK_RETRIES,
    AssistantMessage,
    LLMResponse,
    Message,
    StopReason,
    ToolCall,
    ToolSpec,
    Usage,
    UserMessage,
    neutral_error,
)

ANSWER_SCHEMA_NAME = "answer"


def _to_params(message: Message) -> list[dict[str, Any]]:
    if isinstance(message, UserMessage):
        return [{"role": "user", "content": message.text}]
    if isinstance(message, AssistantMessage):
        return [message.response.raw]  # the provider's own turn, sent back as it came
    # This format has no error flag on a tool result, so it goes in the text.
    return [
        {
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": f"ERRO: {result.content}" if result.is_error else result.content,
        }
        for result in message.results
    ]


def _parse_arguments(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return None  # the assistant rejects it as invalid tool input


class OpenAILLM:
    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        effort: str | None = None,
        max_tokens: int = 4096,
    ) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any],
        timeout: float,
    ) -> LLMResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                *(param for message in messages for param in _to_params(message)),
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": True,
                    },
                }
                for tool in tools
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": ANSWER_SCHEMA_NAME,
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "max_completion_tokens": self.max_tokens,
        }
        if self.effort:
            request["reasoning_effort"] = self.effort
        client = self._client.with_options(
            timeout=timeout / (SDK_RETRIES + 1), max_retries=SDK_RETRIES
        )
        try:
            completion = client.chat.completions.create(**request)
        except (openai.APIStatusError, openai.APIConnectionError) as error:
            raise neutral_error(error, openai) from error
        return _to_response(completion)


def _to_response(completion: Any) -> LLMResponse:
    usage = Usage(
        input_tokens=getattr(completion.usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(completion.usage, "completion_tokens", 0) or 0,
    )
    if not completion.choices:
        return LLMResponse(stop="other", usage=usage)
    choice = completion.choices[0]
    message = choice.message
    function_calls = [call for call in message.tool_calls or [] if call.type == "function"]
    text = message.content or ""

    stop: StopReason
    if choice.finish_reason == "length":
        stop = "truncated"
    elif choice.finish_reason == "content_filter" or getattr(message, "refusal", None):
        stop = "refusal"
    elif function_calls:
        stop = "tool_calls"
    elif choice.finish_reason == "stop" and text:
        stop = "final"
    else:
        stop = "other"

    raw: dict[str, Any] = {"role": "assistant", "content": message.content}
    if function_calls:
        # model_dump keeps provider-specific extras that must be echoed back.
        raw["tool_calls"] = [call.model_dump(exclude_none=True) for call in function_calls]
    return LLMResponse(
        stop=stop,
        text=text,
        tool_calls=tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
            )
            for call in function_calls
        ),
        usage=usage,
        raw=raw,
    )
