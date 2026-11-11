from pathlib import Path

import pytest

from app.config import DEFAULT_DATA_DIR, Settings, Store, load_env_file


def test_settings_defaults_when_environment_is_empty(monkeypatch):
    for name in ("DATA_DIR", "RETRIEVAL_TOP_K", "CORS_ORIGINS", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(name, raising=False)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_EFFORT", "LLM_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    for name in ("DAILY_TOKEN_BUDGET", "LLM_TIMEOUT_SECONDS", "LLM_REFUSAL_FALLBACK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LLM_SCHEMA_WITH_TOOLS", raising=False)
    for name in ("SERVICE_TOKENS", "RETRIEVAL_MIN_SCORE", "MAX_TOOL_ITERATIONS"):
        monkeypatch.delenv(name, raising=False)

    assert Settings.from_env() == Settings()


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/srv/store")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "6")
    monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "2.5")
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "2")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")
    monkeypatch.setenv("SERVICE_TOKENS", f"{'a' * 32}, {'b' * 40}")
    monkeypatch.setenv("DAILY_TOKEN_BUDGET", "500000")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("LLM_PROVIDER", " OpenAI ")
    monkeypatch.setenv("LLM_MODEL", "some-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("LLM_EFFORT", "medium")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1500")
    monkeypatch.setenv("LLM_REFUSAL_FALLBACK", "false")
    monkeypatch.setenv("LLM_SCHEMA_WITH_TOOLS", "false")
    monkeypatch.setenv("LLM_API_KEY", "never-stored-in-settings")

    assert "never-stored" not in repr(Settings.from_env())
    assert Settings.from_env() == Settings(
        llm_provider="openai",
        llm_model="some-model",
        llm_base_url="http://localhost:1234/v1",
        llm_effort="medium",
        llm_max_tokens=1500,
        llm_refusal_fallback=False,
        llm_schema_with_tools=False,
        daily_token_budget=500_000,
        llm_timeout_seconds=12.5,
        service_tokens=("a" * 32, "b" * 40),
        data_dir=Path("/srv/store"),
        retrieval_top_k=6,
        retrieval_min_score=2.5,
        max_tool_iterations=2,
        cors_origins=("https://a.example", "https://b.example"),
        rate_limit_per_minute=5,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"cors_origins": ("*",)},
        {"service_tokens": ("short-secret",)},
        {"daily_token_budget": 0},
        {"llm_provider": "skynet"},
        {"llm_max_tokens": 0},
        {"llm_timeout_seconds": 0},
        {"rate_limit_per_minute": 0},
        {"max_tool_iterations": 0},
        {"retrieval_top_k": 0},
    ],
)
def test_settings_reject_unsafe_values(invalid):
    with pytest.raises(ValueError):
        Settings(**invalid)


def test_load_env_file_reads_key_value_lines(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "﻿# comentário\n"  # Notepad on Windows writes a BOM
        "\n"
        "SERVICE_TOKENS=abc123\n"
        '  QUOTED = "com aspas"  \n'
        "SINGLE='x'\n"
        "WITH_EQUALS=a=b=c\n"
        "EMPTY=\n"
        "linha sem igual\n"
        "=sem chave\n",
        encoding="utf-8",
    )
    environ: dict[str, str] = {}

    load_env_file(env_file, environ)

    assert environ == {
        "SERVICE_TOKENS": "abc123",
        "QUOTED": "com aspas",
        "SINGLE": "x",
        "WITH_EQUALS": "a=b=c",
        "EMPTY": "",
    }


def test_load_env_file_never_overrides_a_real_environment_variable(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("RATE_LIMIT_PER_MINUTE=5\nNEW=1\n", encoding="utf-8")
    environ = {"RATE_LIMIT_PER_MINUTE": "99"}

    load_env_file(env_file, environ)

    assert environ == {"RATE_LIMIT_PER_MINUTE": "99", "NEW": "1"}


def test_load_env_file_ignores_a_missing_file(tmp_path):
    environ: dict[str, str] = {}

    load_env_file(tmp_path / "nao-existe.env", environ)

    assert environ == {}


def test_settings_from_env_reads_the_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "loja.env"
    env_file.write_text("RETRIEVAL_TOP_K=9\n", encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.setenv("RETRIEVAL_TOP_K", "placeholder")  # so the value is removed after the test
    monkeypatch.delenv("RETRIEVAL_TOP_K")

    assert Settings.from_env().retrieval_top_k == 9


def test_store_fills_contact_and_name_into_the_fixed_messages():
    store = Store.load(DEFAULT_DATA_DIR / "store.json")

    assert "(00) 90000-0000" in store.fallback_message
    assert "Loja Demo" in store.greeting_message
    assert "unhas" in store.off_topic_message
    assert store.allowed_domains == ("lojademo.example",)
