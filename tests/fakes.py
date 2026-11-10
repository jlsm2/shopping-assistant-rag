"""Scripted LLM client: no network, deterministic answers."""

import json
from typing import Any

from app.llm import LLMResponse, ToolCall, Usage


class FakeLLM:
    """Returns the scripted steps in order and records every call it receives."""

    def __init__(self, *script: LLMResponse | Exception) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system, messages, tools, response_schema) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "tools": tools,
                "response_schema": response_schema,
            }
        )
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return step


def final(
    answer: str = "",
    citations: tuple[str, ...] = (),
    products: tuple[str, ...] = (),
    outcome: str = "answered",
) -> LLMResponse:
    text = json.dumps(
        {
            "answer": answer,
            "citations": list(citations),
            "products": list(products),
            "outcome": outcome,
        }
    )
    return LLMResponse(stop="final", text=text, usage=Usage(100, 20))


def tool_calls(*calls: tuple[str, Any]) -> LLMResponse:
    return LLMResponse(
        stop="tool_calls",
        tool_calls=tuple(
            ToolCall(id=f"call_{position}", name=name, arguments=arguments)
            for position, (name, arguments) in enumerate(calls)
        ),
        usage=Usage(100, 10),
    )
