"""Runtime settings (environment) and store configuration (data/store.json)."""

import json
import os
from collections.abc import MutableMapping
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MIN_SERVICE_TOKEN_CHARS = 32
LLM_PROVIDERS = ("anthropic", "openai")  # "openai" is any OpenAI-compatible API


def load_env_file(path: Path, environ: MutableMapping[str, str] = os.environ) -> None:
    """Copy the KEY=VALUE lines of a .env file into the environment.

    Real environment variables win, so a deployment can override the file.
    # shortcut: no `export` prefix, inline comments or multi-line values;
    # switch to python-dotenv if the file ever needs them
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():  # -sig drops a Windows BOM
        key, separator, value = line.strip().partition("=")
        key = key.strip()
        if separator and key and not key.startswith("#"):
            environ.setdefault(key, value.strip().strip("\"'"))


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    retrieval_top_k: int = 4
    retrieval_min_score: float = 1.0
    max_tool_iterations: int = 4
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    rate_limit_per_minute: int = 20
    service_tokens: tuple[str, ...] = ()  # more than one only while rotating
    daily_token_budget: int = 2_000_000  # LLM tokens (input + output) per UTC day
    llm_timeout_seconds: float = 30.0  # for one question, tool rounds included
    # The LLM is chosen entirely by configuration: there is no default provider
    # or model. The API key is read from the environment, never stored here.
    llm_provider: str = ""  # one of LLM_PROVIDERS
    llm_model: str = ""
    llm_base_url: str = ""  # empty: the provider's own address
    llm_effort: str = ""  # reasoning effort; empty: not sent
    llm_max_tokens: int = 4096  # output cap per call, reasoning included
    llm_refusal_fallback: bool = True  # Anthropic only: re-run a safety decline
    llm_schema_with_tools: bool = True  # "openai" only: false for APIs like Groq

    def __post_init__(self) -> None:
        if self.llm_provider not in ("", *LLM_PROVIDERS):
            raise ValueError(f"LLM_PROVIDER must be one of {LLM_PROVIDERS}")
        if self.llm_max_tokens < 1:
            raise ValueError("LLM_MAX_TOKENS must be >= 1")
        if self.daily_token_budget < 1 or self.llm_timeout_seconds <= 0:
            raise ValueError("DAILY_TOKEN_BUDGET and LLM_TIMEOUT_SECONDS must be positive")
        if any(len(token) < MIN_SERVICE_TOKEN_CHARS for token in self.service_tokens):
            raise ValueError(
                f"each SERVICE_TOKENS entry needs at least {MIN_SERVICE_TOKEN_CHARS} characters"
            )
        if self.retrieval_top_k < 1 or self.max_tool_iterations < 1:
            raise ValueError("RETRIEVAL_TOP_K and MAX_TOOL_ITERATIONS must be >= 1")
        if self.rate_limit_per_minute < 1:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be >= 1")
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS must list explicit origins, not '*'")

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file(Path(os.environ.get("ENV_FILE", ".env")))
        env = os.environ
        default = cls()
        origins = env.get("CORS_ORIGINS")
        return cls(
            data_dir=Path(env.get("DATA_DIR") or default.data_dir),
            retrieval_top_k=int(env.get("RETRIEVAL_TOP_K") or default.retrieval_top_k),
            retrieval_min_score=float(
                env.get("RETRIEVAL_MIN_SCORE") or default.retrieval_min_score
            ),
            max_tool_iterations=int(env.get("MAX_TOOL_ITERATIONS") or default.max_tool_iterations),
            cors_origins=_csv(origins) if origins is not None else default.cors_origins,
            rate_limit_per_minute=int(
                env.get("RATE_LIMIT_PER_MINUTE") or default.rate_limit_per_minute
            ),
            service_tokens=_csv(env.get("SERVICE_TOKENS", "")),
            daily_token_budget=int(env.get("DAILY_TOKEN_BUDGET") or default.daily_token_budget),
            llm_timeout_seconds=float(
                env.get("LLM_TIMEOUT_SECONDS") or default.llm_timeout_seconds
            ),
            llm_provider=env.get("LLM_PROVIDER", "").strip().lower(),
            llm_model=env.get("LLM_MODEL", "").strip(),
            llm_base_url=env.get("LLM_BASE_URL", "").strip(),
            llm_effort=env.get("LLM_EFFORT", "").strip(),
            llm_max_tokens=int(env.get("LLM_MAX_TOKENS") or default.llm_max_tokens),
            llm_refusal_fallback=env.get("LLM_REFUSAL_FALLBACK", "true").strip().lower()
            in ("1", "true", "yes"),
            llm_schema_with_tools=env.get("LLM_SCHEMA_WITH_TOOLS", "true").strip().lower()
            in ("1", "true", "yes"),
        )


@dataclass(frozen=True)
class Store:
    name: str
    allowed_domains: tuple[str, ...]
    human_contact: str
    fallback_message: str
    greeting_message: str
    off_topic_message: str
    closing_message: str  # a thanks or goodbye in the middle of a conversation

    @classmethod
    def load(cls, path: Path) -> "Store":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            allowed_domains=tuple(d.lower() for d in raw["allowed_domains"]),
            human_contact=raw["human_contact"],
            fallback_message=raw["fallback_message"].format(contact=raw["human_contact"]),
            greeting_message=raw["greeting_message"].format(name=raw["name"]),
            off_topic_message=raw["off_topic_message"].format(name=raw["name"]),
            closing_message=raw["closing_message"].format(name=raw["name"]),
        )
