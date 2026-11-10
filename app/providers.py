"""Picks the LLM adapter from configuration. No provider or model is built in."""

import os

import anthropic
import openai

from app.anthropic_llm import AnthropicLLM
from app.config import LLM_PROVIDERS, Settings
from app.llm import LLMClient
from app.openai_llm import OpenAILLM

# Each SDK also reads its own variable; LLM_API_KEY works for every provider.
_NATIVE_KEY_VARIABLES = {
    "anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "openai": ("OPENAI_API_KEY",),
}


def build_llm(settings: Settings) -> LLMClient:
    """The single place where the LLM provider is chosen.

    Raises RuntimeError with what is missing, so a bad configuration stops the
    server at startup instead of failing on the first customer.
    """
    provider = settings.llm_provider
    if not provider or not settings.llm_model:
        raise RuntimeError(
            f"LLM_PROVIDER (one of {LLM_PROVIDERS}) and LLM_MODEL must be set in .env."
        )
    api_key = os.environ.get("LLM_API_KEY") or None
    if not api_key and not any(os.environ.get(name) for name in _NATIVE_KEY_VARIABLES[provider]):
        raise RuntimeError(
            "LLM_API_KEY is not set in .env. A local server that needs no key "
            "still requires a placeholder, for example LLM_API_KEY=local."
        )
    base_url = settings.llm_base_url or None
    effort = settings.llm_effort or None
    if provider == "anthropic":
        return AnthropicLLM(
            anthropic.Anthropic(api_key=api_key, base_url=base_url),
            model=settings.llm_model,
            effort=effort,
            max_tokens=settings.llm_max_tokens,
            refusal_fallback=settings.llm_refusal_fallback,
        )
    return OpenAILLM(
        openai.OpenAI(api_key=api_key, base_url=base_url),
        model=settings.llm_model,
        effort=effort,
        max_tokens=settings.llm_max_tokens,
        schema_with_tools=settings.llm_schema_with_tools,
    )
