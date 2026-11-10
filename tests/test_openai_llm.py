"""The OpenAI-compatible adapter, tested offline against a fake SDK client and
then through the real SDK over a fake network."""

import json
from types import SimpleNamespace

import httpx
import openai
import pytest

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
from app.openai_llm import OpenAILLM

REQUEST = httpx.Request("POST", "https://api.example/v1/chat/completions")
QUESTION = (UserMessage("pergunta"),)
ARGUMENTS = '{"query": "nude", "max_price": 70}'


def tool_call(arguments: str = ARGUMENTS, kind: str = "function") -> SimpleNamespace:
    payload = {
        "id": "call_1",
        "type": kind,
        "function": {"name": "search_products", "arguments": arguments},
    }
    return SimpleNamespace(
        id="call_1",
        type=kind,
        function=SimpleNamespace(name="search_products", arguments=arguments),
        model_dump=lambda exclude_none: payload,
    )


def completion(finish_reason="stop", content='{"answer": "ok"}', tool_calls=None, refusal=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls, refusal=refusal)
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
    )


class FakeOpenAI:
    """Records how the adapter calls the SDK and replays scripted results in order
    (the last one repeats)."""

    def __init__(self, result) -> None:
        self.results = result if isinstance(result, list) else [result]
        self.options: dict = {}
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    @property
    def request(self) -> dict:
        return self.requests[-1]

    def with_options(self, **options):
        self.options = options
        return self

    def _create(self, **request):
        self.requests.append(request)
        result = self.results[min(len(self.requests), len(self.results)) - 1]
        if isinstance(result, Exception):
            raise result
        return result


def complete(result=None, messages=QUESTION, timeout=30.0, **adapter):
    client = FakeOpenAI(result or completion())
    llm = OpenAILLM(client, **{"model": "some-model", **adapter})
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


def test_system_prompt_is_a_system_message_and_the_user_text_a_user_message():
    client, _, _ = complete()

    assert client.request["messages"] == [
        {"role": "system", "content": "REGRAS DO SISTEMA"},
        {"role": "user", "content": "pergunta"},
    ]


def test_model_effort_and_token_cap_come_from_configuration():
    client, _, _ = complete(model="outro-modelo", effort="high", max_tokens=1500)

    assert client.request["model"] == "outro-modelo"
    assert client.request["max_completion_tokens"] == 1500
    assert client.request["reasoning_effort"] == "high"


def test_effort_is_not_sent_unless_configured():
    client, _, _ = complete()

    assert "reasoning_effort" not in client.request


def test_answer_schema_and_tools_are_strict():
    client, _, _ = complete()

    assert client.request["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": ANSWER_SCHEMA},
    }
    functions = [tool["function"] for tool in client.request["tools"]]
    assert [function["name"] for function in functions] == ["search_products", "get_product"]
    assert all(tool["type"] == "function" for tool in client.request["tools"])
    assert all(function["strict"] is True for function in functions)
    assert not {"tool_choice", "temperature", "top_p"} & set(client.request)


def test_deadline_covers_the_single_retry():
    client, _, _ = complete(timeout=20.0)

    assert client.options == {"timeout": 10.0, "max_retries": 1}


def test_tool_loop_messages_are_translated_and_the_assistant_turn_goes_back_as_it_came():
    _, first, _ = complete(completion("tool_calls", content=None, tool_calls=[tool_call()]))
    results = (
        ToolResult("call_1", "<tool_data>[]</tool_data>"),
        ToolResult("call_2", "Parâmetros inválidos.", is_error=True),
    )

    client, _, _ = complete(
        messages=(UserMessage("pergunta"), AssistantMessage(first), ToolMessage(results))
    )

    assert client.request["messages"][2:] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search_products", "arguments": ARGUMENTS},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "<tool_data>[]</tool_data>"},
        {"role": "tool", "tool_call_id": "call_2", "content": "ERRO: Parâmetros inválidos."},
    ]


# --- APIs that refuse tools and a schema in one request (Groq) -------------------

IN_FORMAT = json.dumps({"answer": "ok", "citations": [], "products": [], "outcome": "smalltalk"})


def test_split_mode_asks_with_tools_only_and_returns_a_tool_call_as_is():
    client, outcome, _ = complete(
        completion("tool_calls", content=None, tool_calls=[tool_call()]),
        schema_with_tools=False,
    )

    (request,) = client.requests
    assert "response_format" not in request
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "search_products",
        "get_product",
    ]
    assert outcome.stop == "tool_calls"


def test_split_mode_enforces_the_schema_in_a_second_call_without_tools():
    client, outcome, _ = complete(
        [completion(content="Resposta solta, sem JSON."), completion(content=IN_FORMAT)],
        schema_with_tools=False,
    )

    first, second = client.requests
    assert "tools" not in second
    assert second["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA
    assert second["messages"] == first["messages"]
    assert (outcome.stop, outcome.text) == ("final", IN_FORMAT)
    assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (200, 80)


def test_split_mode_skips_the_second_call_when_the_answer_already_has_the_fields():
    client, outcome, _ = complete(completion(content=IN_FORMAT), schema_with_tools=False)

    assert len(client.requests) == 1
    assert outcome.text == IN_FORMAT


@pytest.mark.parametrize("text", ['{"answer": "ok"}', "[]", "não é json"])
def test_split_mode_does_not_trust_a_partial_or_non_object_answer(text):
    client, _, _ = complete(
        [completion(content=text), completion(content=IN_FORMAT)], schema_with_tools=False
    )

    assert len(client.requests) == 2


def test_split_mode_recovers_when_the_tools_call_is_rejected():
    # gpt-oss on Groq may write the answer as a call to an undeclared tool: 400.
    rejected = status_error(openai.BadRequestError, 400)

    client, outcome, _ = complete(
        [rejected, completion(content=IN_FORMAT)], schema_with_tools=False
    )

    assert len(client.requests) == 2
    assert "tools" not in client.request
    assert (outcome.stop, outcome.text) == ("final", IN_FORMAT)


def test_split_mode_does_not_retry_a_rate_limit():
    limited = status_error(openai.RateLimitError, 429)

    client, _, error = complete([limited, completion(content=IN_FORMAT)], schema_with_tools=False)

    assert len(client.requests) == 1
    assert isinstance(error, LLMRateLimited)


def test_split_mode_does_not_retry_a_truncated_or_refused_turn():
    client, outcome, _ = complete(completion("length"), schema_with_tools=False)

    assert len(client.requests) == 1
    assert outcome.stop == "truncated"


# --- response --------------------------------------------------------------------


def test_final_answer_is_the_message_content():
    _, outcome, _ = complete()

    assert (outcome.stop, outcome.text, outcome.tool_calls) == ("final", '{"answer": "ok"}', ())
    assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (100, 40)


def test_tool_call_turn_becomes_tool_calls_with_parsed_arguments():
    _, outcome, _ = complete(completion("tool_calls", content=None, tool_calls=[tool_call()]))

    assert outcome.stop == "tool_calls"
    (call,) = outcome.tool_calls
    assert (call.id, call.name) == ("call_1", "search_products")
    assert call.arguments == {"query": "nude", "max_price": 70}


def test_tool_arguments_that_are_not_json_reach_the_assistant_as_invalid():
    broken = completion("tool_calls", content=None, tool_calls=[tool_call('{"query": "nu')])

    _, outcome, _ = complete(broken)

    assert outcome.tool_calls[0].arguments is None


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (completion("length"), "truncated"),
        (completion("length", content=None, tool_calls=[tool_call()]), "truncated"),
        (completion("content_filter", content=None), "refusal"),
        (completion("stop", content=None, refusal="I can't help with that."), "refusal"),
        (completion("stop", content=None), "other"),
        (completion("stop", content=""), "other"),
        (completion("function_call"), "other"),
        (completion("tool_calls", content=None, tool_calls=[tool_call(kind="custom")]), "other"),
        (SimpleNamespace(choices=[], usage=None), "other"),
    ],
)
def test_anything_but_a_clean_finish_is_never_reported_as_final(result, expected):
    _, outcome, _ = complete(result)

    assert outcome.stop == expected


def test_missing_usage_counts_as_zero_tokens():
    result = completion()
    result.usage = None

    _, outcome, _ = complete(result)

    assert (outcome.usage.input_tokens, outcome.usage.output_tokens) == (0, 0)


# --- errors ----------------------------------------------------------------------


def status_error(error_class, status: int, **headers) -> openai.APIStatusError:
    reply = httpx.Response(status, headers=headers, request=REQUEST)
    return error_class("provider says: secret detail", response=reply, body=None)


def test_rate_limit_carries_the_retry_after_header():
    _, _, error = complete(status_error(openai.RateLimitError, 429, **{"retry-after": "7"}))

    assert isinstance(error, LLMRateLimited)
    assert error.retry_after == 7


@pytest.mark.parametrize(
    "provider_error",
    [
        status_error(openai.InternalServerError, 503),
        openai.APITimeoutError(request=REQUEST),
        openai.APIConnectionError(request=REQUEST),
    ],
)
def test_outage_timeout_and_network_failure_are_unavailable(provider_error):
    _, _, error = complete(provider_error)

    assert isinstance(error, LLMUnavailable)


@pytest.mark.parametrize(
    ("error_class", "status"),
    [(openai.BadRequestError, 400), (openai.AuthenticationError, 401)],
)
def test_request_and_credential_problems_are_plain_errors(error_class, status):
    _, _, error = complete(status_error(error_class, status))

    assert type(error) is LLMError
    assert str(status) in str(error)


# --- through the real SDK --------------------------------------------------------


def wire_completion(finish_reason: str, message: dict) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "some-model",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
        "usage": {"prompt_tokens": 900, "completion_tokens": 30, "total_tokens": 930},
    }


def test_full_tool_loop_through_the_real_sdk_with_a_fake_network(index, catalog, store):
    """Nothing is mocked above HTTP: the real SDK builds and parses every request."""
    answer = {
        "answer": "O Set Nude Curto custa R$ 59,90.",
        "citations": [],
        "products": ["nude-curto"],
        "outcome": "answered",
    }
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "search_products", "arguments": ARGUMENTS},
        "extra_content": {"provider": {"signature": "sig-abc"}},  # must be echoed back
    }
    replies = [
        wire_completion("tool_calls", {"role": "assistant", "content": None, "tool_calls": [call]}),
        wire_completion("stop", {"role": "assistant", "content": json.dumps(answer)}),
    ]
    sent: list[httpx.Request] = []

    def network(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=replies[len(sent) - 1])

    client = openai.OpenAI(
        api_key="local",
        base_url="http://localhost:1234/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(network)),
    )
    assistant = Assistant(
        llm=OpenAILLM(client, model="some-model"), index=index, catalog=catalog, store=store
    )

    reply = assistant.answer("Tem algum set nude até R$ 70?")

    assert reply.fallback is False, reply.reason
    assert [product["id"] for product in reply.products] == ["nude-curto"]
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (1800, 60)

    assert str(sent[0].url) == "http://localhost:1234/v1/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer local"
    first, second = (json.loads(request.content) for request in sent)
    assert first["model"] == "some-model"
    assert first["messages"][0]["role"] == "system"
    assert "<customer_message>" in first["messages"][1]["content"]
    assert first["response_format"]["json_schema"]["schema"] == ANSWER_SCHEMA
    assert [tool["function"]["strict"] for tool in first["tools"]] == [True, True]

    assistant_turn, tool_turn = second["messages"][2:]
    assert assistant_turn["tool_calls"] == [call]
    assert tool_turn["role"] == "tool" and tool_turn["tool_call_id"] == "call_1"
    assert "nude-curto" in tool_turn["content"]
