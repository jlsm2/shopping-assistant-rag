"""Runtime settings (environment) and store configuration (data/store.json)."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Settings:
    data_dir: Path = DEFAULT_DATA_DIR
    retrieval_top_k: int = 4
    retrieval_min_score: float = 1.0
    max_tool_iterations: int = 4
    cors_origins: tuple[str, ...] = ("http://localhost:3000",)
    rate_limit_per_minute: int = 20

    def __post_init__(self) -> None:
        if self.retrieval_top_k < 1 or self.max_tool_iterations < 1:
            raise ValueError("RETRIEVAL_TOP_K and MAX_TOOL_ITERATIONS must be >= 1")
        if self.rate_limit_per_minute < 1:
            raise ValueError("RATE_LIMIT_PER_MINUTE must be >= 1")
        if "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS must list explicit origins, not '*'")

    @classmethod
    def from_env(cls) -> "Settings":
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
            cors_origins=(
                tuple(o.strip() for o in origins.split(",") if o.strip())
                if origins is not None
                else default.cors_origins
            ),
            rate_limit_per_minute=int(
                env.get("RATE_LIMIT_PER_MINUTE") or default.rate_limit_per_minute
            ),
        )


@dataclass(frozen=True)
class Store:
    name: str
    allowed_domains: tuple[str, ...]
    human_contact: str
    fallback_message: str
    greeting_message: str

    @classmethod
    def load(cls, path: Path) -> "Store":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            allowed_domains=tuple(d.lower() for d in raw["allowed_domains"]),
            human_contact=raw["human_contact"],
            fallback_message=raw["fallback_message"].format(contact=raw["human_contact"]),
            greeting_message=raw["greeting_message"].format(name=raw["name"]),
        )
