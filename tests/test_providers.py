"""Provider selection: everything about the LLM comes from configuration."""

import pytest

from app.anthropic_llm import AnthropicLLM
from app.config import Settings
from app.openai_llm import OpenAILLM
from app.providers import build_llm

KEY_VARIABLES = ("LLM_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY")


@pytest.fixture(autouse=True)
def no_keys_in_the_environment(monkeypatch):
    for name in KEY_VARIABLES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    "settings",
    [
        Settings(),
        Settings(llm_provider="anthropic"),
        Settings(llm_model="some-model"),
    ],
)
def test_there_is_no_default_provider_or_model(settings, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "any-key")

    with pytest.raises(RuntimeError, match="LLM_PROVIDER .* and LLM_MODEL must be set"):
        build_llm(settings)


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_missing_api_key_stops_at_startup(provider):
    with pytest.raises(RuntimeError, match="LLM_API_KEY is not set"):
        build_llm(Settings(llm_provider=provider, llm_model="some-model"))


def test_anthropic_provider_is_built_from_the_settings(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "key-from-env")
    settings = Settings(
        llm_provider="anthropic",
        llm_model="claude-haiku-5-5",
        llm_effort="medium",
        llm_max_tokens=900,
        llm_refusal_fallback=False,
    )

    llm = build_llm(settings)

    assert isinstance(llm, AnthropicLLM)
    assert (llm.model, llm.effort, llm.max_tokens) == ("claude-haiku-5-5", "medium", 900)
    assert llm.refusal_fallback is False
    assert llm._client.api_key == "key-from-env"


def test_openai_compatible_provider_takes_any_base_url_and_model(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "local")
    settings = Settings(
        llm_provider="openai", llm_model="qualquer-modelo", llm_base_url="http://localhost:1234/v1"
    )

    llm = build_llm(settings)

    assert isinstance(llm, OpenAILLM)
    assert (llm.model, llm.effort, llm.max_tokens) == ("qualquer-modelo", None, 4096)
    assert llm._client.api_key == "local"
    assert str(llm._client.base_url) == "http://localhost:1234/v1/"


@pytest.mark.parametrize(
    ("provider", "native_variable", "adapter"),
    [
        ("anthropic", "ANTHROPIC_API_KEY", AnthropicLLM),
        ("openai", "OPENAI_API_KEY", OpenAILLM),
    ],
)
def test_the_sdk_own_key_variable_also_works(provider, native_variable, adapter, monkeypatch):
    monkeypatch.setenv(native_variable, "native-key")

    llm = build_llm(Settings(llm_provider=provider, llm_model="some-model"))

    assert isinstance(llm, adapter)
    assert llm._client.api_key == "native-key"
