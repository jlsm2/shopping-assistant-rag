"""The Anthropic adapter, tested offline against a fake SDK client."""

import json
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.anthropic_llm import AnthropicLLM
from app.assistant import ANSWER_SCHEMA, TOOLS, Assistant
from app.llm import (
    AssistantMessage,
    LLMError,
    LLMRateLimited,
    LLMUnavailable,
    ToolMessage,
    ToolResult,
    UserMessage,
)

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
TEXT = SimpleNamespace(type="text", text='{"answer": "ok"}')
THINKING = SimpleNamespace(type="thinking", thinking="")
TOOL_USE = SimpleNamespace(
    type="tool_use", id="toolu_1", name="search_products", input={"query": "nude", "max_price": 70}
)


def response(stop_reason="end_turn", content=(TEXT,), **usage) -> SimpleNamespace:
    counts = {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
        **usage,
    }
    return SimpleNamespace(
        stop_reason=stop_reason, content=list(content), usage=SimpleNamespace(**counts)
    )


class FakeAnthropic:
    """Records how the adapter calls the SDK and replays one scripted result."""

    def __init__(self, result) -> None:
        self.result = result
        self.options: dict = {}
        self.endpoint = ""
        self.request: dict = {}
        self.messages = SimpleNamespace(create=lambda **kw: self._create("messages", kw))
        self.beta = SimpleNamespace(
            messages=SimpleNamespace(create=lambda **kw: self._create("beta.messages", kw))
        )

    def with_options(self, **options):
        self.options = options
        return self

    def _create(self, endpoint: str, request: dict):
        self.endpoint, self.request = endpoint, request
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


QUESTION = (UserMessage("pergunta"),)


def complete(result=None, messages=QUESTION, timeout=30.0, **adapter):
    client = FakeAnthropic(result or response())
    llm = AnthropicLLM(client=client, **{"model": "claude-opus-5-5", **adapter})
    outcome = None
    error = None
    try:
        outcome = llm.complete(
            system="REGRAS DO SISTEMA",
            messages=list(messages),
            tools=TOOLS,
            response_schema=ANSWER_SCHEMA,
            timeout=timeout,
        )
    except LLMError as raised:
        error = raised
    return client, outcome, error


# --- request ---------------------------------------------------------------------


def test_system_prompt_goes_in_the_system_channel_and_the_user_text_only_in_messages():
    client, _, _ = complete()

    assert client.request["system"] == [
        {"type": "text", "text": "REGRAS DO SISTEMA", "cache_control": {"type": "ephemeral"}}
    ]
    assert client.request["messages"] == [{"role": "user", "content": "pergunta"}]


def test_model_effort_and_token_cap_come_from_configuration():
    client, _, _ = complete(model="claude-haiku-5-5", effort="medium", max_tokens=1500)

    assert client.request["model"] == "claude-haiku-5-5"
    assert client.request["max_tokens"] == 1500
    assert client.request["output_config"]["effort"] == "medium"


def test_effort_is_omitted_when_not_configured():
    client, _, _ = complete(effort=None)

    assert "effort" not in client.request["output_config"]


def test_answer_schema_is_enforced_with_structured_output():
    client, _, _ = complete()

    assert client.request["output_config"]["format"] == {
        "type": "json_schema",
        "schema": ANSWER_SCHEMA,
    }


def test_tools_are_strict_and_tool_choice_is_left_on_auto():
    client, _, _ = complete()

    names = [tool["name"] for tool in client.request["tools"]]
    assert names == ["search_products", "get_product"]
    assert all(tool["strict"] is True for tool in client.request["tools"])
    assert all(
        tool["input_schema"]["additionalProperties"] is False for tool in client.request["tools"]
    )
    # These return a 400 on current models, so they must never be sent.
    assert not {"tool_choice", "temperature", "top_p", "top_k", "thinking"} & set(client.request)


def test_server_side_refusal_fallback_is_on_by_default_and_can_be_turned_off():
    with_fallback, _, _ = complete()
    without_fallback, _, _ = complete(refusal_fallback=False)

    assert with_fallback.endpoint == "beta.messages"
    assert with_fallback.request["betas"] == ["server-side-fallback-2026-07-01"]
    assert with_fallback.request["fallbacks"] == "default"
    assert without_fallback.endpoint == "messages"
    assert not {"betas", "fallbacks"} & set(without_fallback.request)


def test_deadline_covers_the_single_retry():
    client, _, _ = complete(timeout=20.0)

    assert client.options == {"timeout": 10.0, "max_retries": 1}


def test_tool_loop_messages_are_translated_and_the_assistant_turn_goes_back_unchanged():
    native_turn = [THINKING, TOOL_USE]
    _, first, _ = complete(response("tool_use", native_turn))
    results = (
        ToolResult("toolu_1", "<tool_data>[]</tool_data>"),
        ToolResult("toolu_2", "Parâmetros inválidos.", is_error=True),
    )

    client, _, _ = complete(
        messages=(UserMessage("pergunta"), AssistantMessage(first), ToolMessage(results))
    )

    _, assistant_turn, tool_turn = client.request["messages"]
    assert assistant_turn == {"role": "assistant", "content": native_turn}
    assert assistant_turn["content"] is first.raw  # thinking blocks must not be rebuilt
    assert tool_turn == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "<tool_data>[]</tool_data>",
                "is_error": False,
            },
            {
                "type": "tool_result",
                "tool_use_id": "toolu_2",
                "content": "Parâmetros inválidos.",
                "is_error": True,
            },
        ],
    }


# --- response --------------------------------------------------------------------


def test_final_answer_is_the_text_block_after_any_thinking():
    _, outcome, _ = complete(response("end_turn", (THINKING, TEXT)))

    assert outcome.stop == "final"
    assert outcome.text == '{"answer": "ok"}'
    assert outcome.tool_calls == ()


def test_tool_use_turn_becomes_tool_calls():
    _, outcome, _ = complete(response("tool_use", (THINKING, TOOL_USE)))

    assert outcome.stop == "tool_calls"
    (call,) = outcome.tool_calls
    assert (call.id, call.name) == ("toolu_1", "search_products")
    assert call.arguments == {"query": "nude", "max_price": 70}


@pytest.mark.parametrize(
    ("stop_reason", "content", "expected"),
    [
        ("refusal", (), "refusal"),
        ("max_tokens", (TEXT,), "truncated"),
        ("model_context_window_exceeded", (), "truncated"),
        ("pause_turn", (), "other"),
        ("stop_sequence", (TEXT,), "other"),
        ("tool_use", (TEXT,), "other"),  # said tool_use but sent no tool call
        ("end_turn", (THINKING,), "other"),  # finished without any text
    ],
)
def test_anything_but_a_clean_finish_is_never_reported_as_final(stop_reason, content, expected):
    _, outcome, _ = complete(response(stop_reason, content))

    assert outcome.stop == expected


def test_usage_counts_every_billed_input_token_including_cache():
    usage = {"input_tokens": 100, "output_tokens": 40}
    cache = {"cache_creation_input_tokens": 900, "cache_read_input_tokens": 50}

    _, outcome, _ = complete(response(**usage, **cache))

    assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (1050, 40)


# --- errors ----------------------------------------------------------------------


def status_error(error_class, status: int, **headers) -> anthropic.APIStatusError:
    reply = httpx2.Response(status, headers=headers, request=REQUEST)
    return error_class("provider says: secret detail", response=reply, body=None)


def test_rate_limit_carries_the_retry_after_header():
    _, _, error = complete(status_error(anthropic.RateLimitError, 429, **{"retry-after": "7"}))

    assert isinstance(error, LLMRateLimited)
    assert error.retry_after == 7


def test_rate_limit_without_a_usable_header_has_no_retry_after():
    _, _, error = complete(status_error(anthropic.RateLimitError, 429, **{"retry-after": "soon"}))

    assert isinstance(error, LLMRateLimited) and error.retry_after is None


@pytest.mark.parametrize(
    "provider_error",
    [
        status_error(anthropic.InternalServerError, 500),
        status_error(anthropic.APIStatusError, 529),
        anthropic.APITimeoutError(request=REQUEST),
        anthropic.APIConnectionError(request=REQUEST),
    ],
)
def test_outage_timeout_and_network_failure_are_unavailable(provider_error):
    _, _, error = complete(provider_error)

    assert isinstance(error, LLMUnavailable)


@pytest.mark.parametrize(
    ("error_class", "status"),
    [
        (anthropic.BadRequestError, 400),
        (anthropic.AuthenticationError, 401),
        (anthropic.NotFoundError, 404),
    ],
)
def test_request_and_credential_problems_are_plain_errors(error_class, status):
    _, _, error = complete(status_error(error_class, status))

    assert type(error) is LLMError
    assert str(status) in str(error)


# --- through the real SDK --------------------------------------------------------


def wire_message(stop_reason: str, content: list[dict]) -> dict:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 900, "output_tokens": 30},
    }


def test_full_tool_loop_through_the_real_sdk_with_a_fake_network(index, catalog, store):
    """Nothing is mocked above HTTP: the real SDK builds and parses every request,
    so a wrong parameter name or a block that cannot be replayed fails here."""
    answer = {
        "answer": "O Set Nude Curto custa R$ 59,90.",
        "citations": [],
        "products": ["nude-curto"],
        "outcome": "answered",
    }
    replies = [
        wire_message(
            "tool_use",
            [
                {"type": "thinking", "thinking": "", "signature": "sig-abc"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "search_products",
                    "input": {"query": "nude", "max_price": 70},
                },
            ],
        ),
        wire_message("end_turn", [{"type": "text", "text": json.dumps(answer)}]),
    ]
    sent: list[httpx2.Request] = []

    def network(request: httpx2.Request) -> httpx2.Response:
        sent.append(request)
        return httpx2.Response(200, json=replies[len(sent) - 1])

    client = anthropic.Anthropic(
        api_key="sk-ant-test-not-a-real-key",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(network)),
    )
    assistant = Assistant(
        llm=AnthropicLLM(client, model="claude-opus-5-5", effort="low"),
        index=index,
        catalog=catalog,
        store=store,
    )

    reply = assistant.answer("Tem algum set nude até R$ 70?")

    assert reply.fallback is False, reply.reason
    assert [product["id"] for product in reply.products] == ["nude-curto"]
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (1800, 60)

    first, second = (json.loads(request.content) for request in sent)
    assert "server-side-fallback-2026-07-01" in sent[0].headers["anthropic-beta"]
    assert first["fallbacks"] == "default"
    assert first["model"] == "claude-opus-5-5"
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"] == {
        "effort": "low",
        "format": {"type": "json_schema", "schema": ANSWER_SCHEMA},
    }
    assert [tool["strict"] for tool in first["tools"]] == [True, True]
    assert "<customer_message>" in first["messages"][0]["content"]

    _, assistant_turn, tool_turn = second["messages"]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["content"][0] == {
        "type": "thinking",
        "thinking": "",
        "signature": "sig-abc",
    }
    assert assistant_turn["content"][1]["id"] == "toolu_1"
    (tool_result,) = tool_turn["content"]
    assert tool_result["tool_use_id"] == "toolu_1" and tool_result["is_error"] is False
    assert "nude-curto" in tool_result["content"]
