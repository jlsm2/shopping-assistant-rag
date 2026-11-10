from pathlib import Path

import pytest

from app.config import DEFAULT_DATA_DIR, Settings, Store


def test_settings_defaults_when_environment_is_empty(monkeypatch):
    for name in ("DATA_DIR", "RETRIEVAL_TOP_K", "CORS_ORIGINS", "RATE_LIMIT_PER_MINUTE"):
        monkeypatch.delenv(name, raising=False)

    assert Settings.from_env() == Settings()


def test_settings_read_from_environment(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/srv/store")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "6")
    monkeypatch.setenv("RETRIEVAL_MIN_SCORE", "2.5")
    monkeypatch.setenv("MAX_TOOL_ITERATIONS", "2")
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example, https://b.example")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "5")

    assert Settings.from_env() == Settings(
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
        {"rate_limit_per_minute": 0},
        {"max_tool_iterations": 0},
        {"retrieval_top_k": 0},
    ],
)
def test_settings_reject_unsafe_values(invalid):
    with pytest.raises(ValueError):
        Settings(**invalid)


def test_store_fills_contact_and_name_into_the_fixed_messages():
    store = Store.load(DEFAULT_DATA_DIR / "store.json")

    assert "(00) 90000-0000" in store.fallback_message
    assert "Loja Demo" in store.greeting_message
    assert store.allowed_domains == ("lojademo.example",)
