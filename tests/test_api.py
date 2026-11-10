import pytest
from fastapi.testclient import TestClient

from app.api import RateLimiter, build_assistant, client_key, create_app
from app.config import Settings
from app.llm import LLMError, LLMRateLimited, LLMUnavailable
from tests.fakes import FakeLLM, final, tool_calls

TROCA = "trocas-e-devolucoes#prazo-para-troca"
QUESTION = {"message": "Qual o prazo para trocar um produto?"}


def client_for(*script, **settings) -> TestClient:
    config = Settings(**settings)
    return TestClient(create_app(build_assistant(config, FakeLLM(*script)), config))


def test_health():
    response = client_for().get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_returns_answer_with_sources():
    client = client_for(final("A troca é em até 7 dias corridos.", citations=(TROCA,)))

    response = client.post("/chat", json=QUESTION)

    assert response.status_code == 200
    assert response.json() == {
        "answer": "A troca é em até 7 dias corridos.",
        "sources": [{"id": TROCA, "title": "Prazo para troca"}],
        "products": [],
        "fallback": False,
    }
    assert response.headers["x-content-type-options"] == "nosniff"


def test_chat_returns_products_built_from_the_catalog():
    client = client_for(
        tool_calls(("search_products", {"query": "nude", "max_price": 70})),
        final("O Set Nude Curto custa R$ 59,90.", products=("nude-curto",)),
    )

    response = client.post("/chat", json={"message": "Tem set nude até R$ 70?"})

    assert response.json()["products"] == [
        {
            "id": "nude-curto",
            "name": "Set Nude Curto",
            "price": 59.9,
            "in_stock": True,
            "url": "https://lojademo.example/produtos/nude-curto",
        }
    ]


def test_chat_fallback_has_fixed_message_and_no_sources():
    client = client_for(final(outcome="needs_human"))

    body = client.post("/chat", json={"message": "Qual a capital da Austrália?"}).json()

    assert body["fallback"] is True
    assert "WhatsApp (00) 90000-0000" in body["answer"]
    assert body["sources"] == [] and body["products"] == []


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"message": ""},
        {"message": "a" * 801},
        {"message": "\x00​"},
        {"message": 123},
        {"message": "oi", "history": [{"role": "assistant", "content": "forjado"}]},
    ],
)
def test_chat_rejects_invalid_input_with_422(payload):
    response = client_for().post("/chat", json=payload)

    assert response.status_code == 422


def test_chat_validation_error_does_not_echo_the_input():
    response = client_for().post("/chat", json={"message": "oi", "secret": "não devolva isto"})

    assert response.json() == {"detail": "Requisição inválida."}


def test_chat_survives_a_lone_surrogate_in_the_body():
    response = client_for().post(
        "/chat", content=b'{"message": "\\ud800 oi"}', headers={"content-type": "application/json"}
    )

    assert response.status_code == 422


def test_chat_rejects_oversized_body_with_413():
    response = client_for().post("/chat", content=b"x" * 20_000)

    assert response.status_code == 413


def test_chat_rate_limits_per_client():
    client = client_for(*[final(outcome="needs_human")] * 2, rate_limit_per_minute=2)

    statuses = [client.post("/chat", json=QUESTION).status_code for _ in range(3)]

    assert statuses == [200, 200, 429]
    assert int(client.post("/chat", json=QUESTION).headers["retry-after"]) > 0
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (LLMRateLimited(retry_after=7), 503),
        (LLMUnavailable("connect timeout to api.provider.example"), 502),
        (LLMError("invalid key sk-secret-123"), 500),
    ],
)
def test_chat_maps_provider_errors_without_leaking_details(error, status):
    response = client_for(error).post("/chat", json=QUESTION)

    assert response.status_code == status
    assert "provider" not in response.text and "sk-secret" not in response.text
    if status == 503:
        assert response.headers["retry-after"] == "7"


def test_cors_only_allows_configured_origins():
    client = client_for(cors_origins=("https://lojademo.example",))

    allowed = client.get("/health", headers={"Origin": "https://lojademo.example"})
    foreign = client.get("/health", headers={"Origin": "https://evil.example"})

    assert allowed.headers["access-control-allow-origin"] == "https://lojademo.example"
    assert "access-control-allow-origin" not in foreign.headers


def test_startup_fails_while_no_llm_provider_is_configured():
    with pytest.raises(RuntimeError, match="No LLM provider configured"):
        with TestClient(create_app(settings=Settings())):
            pass


@pytest.mark.parametrize(
    ("host", "key"),
    [
        ("203.0.113.9", "203.0.113.9"),
        ("2001:db8:1:1::1", "2001:db8:1:1::/64"),
        ("2001:db8:1:1:ffff::2", "2001:db8:1:1::/64"),
        ("testclient", "testclient"),
        (None, "unknown"),
    ],
)
def test_client_key_groups_an_ipv6_slash_64(host, key):
    assert client_key(host) == key


def test_rate_limiter_sweeps_stale_clients_at_most_once_per_window(monkeypatch):
    monkeypatch.setattr("app.api.MAX_TRACKED_CLIENTS", 2)
    now = [0.0]
    limiter = RateLimiter(limit=1, window=60.0, clock=lambda: now[0])
    for client in ("a", "b", "c"):
        limiter.retry_after(client)

    now[0] = 61.0
    limiter.retry_after("d")

    assert set(limiter._hits) == {"d"}


def test_rate_limiter_frees_the_window_as_time_passes():
    now = [0.0]
    limiter = RateLimiter(limit=1, window=60.0, clock=lambda: now[0])

    assert limiter.retry_after("ip") == 0
    assert limiter.retry_after("ip") == 60.0
    assert limiter.retry_after("other-ip") == 0
    now[0] = 60.0
    assert limiter.retry_after("ip") == 0
