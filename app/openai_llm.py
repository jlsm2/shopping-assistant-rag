"""`LLMClient` adapter for any API that speaks the OpenAI Chat Completions format.

That covers OpenAI itself and the many services and local servers that copy
its format; `LLM_BASE_URL` says which one. The model must support tool calling
and JSON-schema output. Everything an adapter must guarantee is in `app/llm.py`.
"""

import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import openai

from app.llm import (
    SDK_RETRIES,
    AssistantMessage,
    LLMError,
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
        schema_with_tools: bool = True,
    ) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        # False for APIs that reject tools and a json_schema in one request (Groq).
        self.schema_with_tools = schema_with_tools

    def complete(
        self,
        *,
        system: str,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        response_schema: dict[str, Any],
        timeout: float,
    ) -> LLMResponse:
        base: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                *(param for message in messages for param in _to_params(message)),
            ],
            "max_completion_tokens": self.max_tokens,
        }
        if self.effort:
            base["reasoning_effort"] = self.effort
        with_tools = {
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
            ]
        }
        with_schema = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": ANSWER_SCHEMA_NAME,
                    "strict": True,
                    "schema": response_schema,
                },
            }
        }
        if self.schema_with_tools:
            return self._send({**base, **with_tools, **with_schema}, timeout)

        # Ask with tools first; once the model stops calling them, impose the
        # schema in a call without tools. The deadline covers both calls.
        try:
            first = self._send({**base, **with_tools}, timeout / 2)
        except LLMError as error:
            if type(error) is not LLMError:
                raise  # rate limit or outage: a second call would not help
            # gpt-oss on Groq may write the answer as a call to an undeclared
            # tool, which the API rejects; the call without tools cannot do that.
            return self._send({**base, **with_schema}, timeout / 2)
        if first.stop != "final" or _has_schema_fields(first.text, response_schema):
            return first
        second = self._send({**base, **with_schema}, timeout / 2)
        return replace(second, usage=first.usage + second.usage)

    def _send(self, request: dict[str, Any], timeout: float) -> LLMResponse:
        client = self._client.with_options(
            timeout=timeout / (SDK_RETRIES + 1), max_retries=SDK_RETRIES
        )
        try:
            completion = client.chat.completions.create(**request)
        except (openai.APIStatusError, openai.APIConnectionError) as error:
            raise neutral_error(error, openai) from error
        return _to_response(completion)


def _has_schema_fields(text: str, schema: dict[str, Any]) -> bool:
    """Saves the second call when the model already wrote the answer object.

    Only the field names are checked here; the assistant still validates the
    answer strictly and falls back when it does not fit.
    """
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return isinstance(value, dict) and set(value) == set(schema.get("required", ()))


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
