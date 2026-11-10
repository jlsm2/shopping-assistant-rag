"""HTTP API: POST /chat, GET /health and GET /metrics."""

import hmac
import ipaddress
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from app.anthropic_llm import AnthropicLLM
from app.assistant import Assistant
from app.budget import TokenBudget
from app.catalog import JsonCatalog
from app.config import Settings, Store
from app.guard import MAX_INPUT_CHARS, InvalidInput
from app.llm import LLMClient, LLMError, LLMRateLimited, LLMUnavailable, Usage
from app.retrieval import Index, load_chunks
from app.telemetry import Metrics, accept_request_id, request_id

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 16_384
RATE_WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 10_000
DEFAULT_RETRY_AFTER_SECONDS = 30
END_USER_HEADER = "x-end-user-id"
MAX_END_USER_CHARS = 128
KNOWN_PATHS = frozenset({"/chat", "/health", "/metrics"})  # bounds the metric labels
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)


class Source(BaseModel):
    id: str
    title: str


class Product(BaseModel):
    id: str
    name: str
    price: float
    in_stock: bool
    url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    products: list[Product]
    fallback: bool


class RateLimiter:
    """Sliding window per client key.

    # shortcut: in-memory and per-process; use Redis once there is more than one worker
    """

    def __init__(
        self,
        limit: int,
        window: float = RATE_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._limit = limit
        self._window = window
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._last_prune = clock()
        self._lock = threading.Lock()

    def retry_after(self, key: str) -> float:
        """Record a hit. Returns 0 when allowed, else the seconds to wait."""
        now = self._clock()
        with self._lock:
            # At most one sweep per window, so a flood of keys cannot make every
            # request pay for a full rebuild.
            if len(self._hits) > MAX_TRACKED_CLIENTS and now - self._last_prune >= self._window:
                self._last_prune = now
                self._hits = {
                    k: hits for k, hits in self._hits.items() if now - hits[-1] < self._window
                }
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= self._window:
                hits.popleft()
            if len(hits) >= self._limit:
                return self._window - (now - hits[0])
            hits.append(now)
            return 0.0


def client_key(host: str | None) -> str:
    """Rate-limit key. An IPv6 client owns a whole /64, so the key is the /64."""
    try:
        address = ipaddress.ip_address(host or "")
    except ValueError:
        return host or "unknown"
    if address.version == 6:
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


def build_llm(settings: Settings) -> LLMClient:
    """The single place where the LLM provider is chosen."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Create a key at console.anthropic.com "
            "and add it to .env."
        )
    return AnthropicLLM(
        model=settings.assistant_model,
        effort=settings.assistant_effort or None,
        max_tokens=settings.assistant_max_tokens,
        server_fallback=settings.assistant_server_fallback,
    )


def build_assistant(settings: Settings, llm: LLMClient) -> Assistant:
    return Assistant(
        llm=llm,
        index=Index(
            load_chunks(settings.data_dir / "knowledge"),
            top_k=settings.retrieval_top_k,
            min_score=settings.retrieval_min_score,
        ),
        catalog=JsonCatalog(settings.data_dir / "catalog.json"),
        store=Store.load(settings.data_dir / "store.json"),
        max_tool_iterations=settings.max_tool_iterations,
        timeout_seconds=settings.llm_timeout_seconds,
        budget=TokenBudget(settings.daily_token_budget),
    )


def _map_error(error: Exception) -> tuple[str, HTTPException]:
    """Metric label and HTTP error for a failed question. No provider detail leaks."""
    if isinstance(error, InvalidInput):
        return "invalid_input", HTTPException(422, str(error))
    if isinstance(error, LLMRateLimited):
        return "llm_rate_limited", HTTPException(
            503,
            "O assistente está sobrecarregado. Tente de novo em instantes.",
            headers={"Retry-After": str(error.retry_after or DEFAULT_RETRY_AFTER_SECONDS)},
        )
    if isinstance(error, LLMUnavailable):
        return "llm_unavailable", HTTPException(502, "O assistente está indisponível no momento.")
    return "llm_error", HTTPException(500, "Erro interno.")


def create_app(assistant: Assistant | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    limiter = RateLimiter(settings.rate_limit_per_minute)
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Fail at startup, not on the first customer, when something is missing.
        if not settings.service_tokens:
            raise RuntimeError(
                "SERVICE_TOKENS is not set. Generate one with "
                '`python -c "import secrets; print(secrets.token_urlsafe(32))"`.'
            )
        app.state.assistant = assistant or build_assistant(settings, build_llm(settings))
        yield

    def require_service_token(request: Request) -> None:
        """Only the store backend may call /chat. Fails closed without tokens."""
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        if scheme.lower() != "bearer" or not any(
            hmac.compare_digest(token.encode(), valid.encode()) for valid in settings.service_tokens
        ):
            raise HTTPException(401, "Não autorizado.", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="shopping-assistant-rag", lifespan=lifespan)
    app.state.assistant = assistant

    @app.middleware("http")
    async def limit_body_and_set_headers(request: Request, call_next: Callable) -> Response:
        request.state.request_id = accept_request_id(request.headers.get("x-request-id"))
        # shortcut: trusts Content-Length; enforce the body limit at the reverse proxy too
        declared = request.headers.get("content-length", "0")
        if not (declared.isascii() and declared.isdigit()) or int(declared) > MAX_BODY_BYTES:
            response: Response = JSONResponse(
                {"detail": "Requisição grande demais."}, status_code=413
            )
        else:
            response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Request-Id"] = request.state.request_id
        path = request.url.path if request.url.path in KNOWN_PATHS else "other"
        metrics.inc("http_responses_total", path=path, status=str(response.status_code))
        return response

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError) -> JSONResponse:
        # The default handler echoes the rejected input back; this one does not.
        return JSONResponse({"detail": "Requisição inválida."}, status_code=422)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics", dependencies=[Depends(require_service_token)])
    def prometheus_metrics(request: Request) -> PlainTextResponse:
        budget = request.app.state.assistant.budget
        gauges = (
            {"llm_daily_token_budget": budget.daily_limit, "llm_daily_tokens_used": budget.used()}
            if budget
            else {}
        )
        return PlainTextResponse(metrics.render(gauges), media_type=PROMETHEUS_CONTENT_TYPE)

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_service_token)])
    def chat(body: ChatRequest, request: Request) -> ChatResponse:
        # Every request comes from the store backend, so its IP says nothing
        # about the customer. The backend names the customer (session id or IP)
        # in a header, which is trusted because the request is authenticated.
        end_user = request.headers.get(END_USER_HEADER, "")[:MAX_END_USER_CHARS]
        connection = request.client.host if request.client else None
        wait = limiter.retry_after(client_key(end_user or connection))
        if wait:
            raise HTTPException(
                429,
                "Muitas mensagens em pouco tempo. Tente de novo em instantes.",
                headers={"Retry-After": str(math.ceil(wait))},
            )
        request_id.set(request.state.request_id)
        started = time.monotonic()

        def record(outcome: str, reason: str | None, usage: Usage) -> None:
            """One log line and the metrics for this question. Never the message."""
            seconds = time.monotonic() - started
            metrics.observe(seconds)
            metrics.inc("chat_outcomes_total", outcome=outcome)
            metrics.inc("llm_tokens_total", usage.input_tokens, direction="input")
            metrics.inc("llm_tokens_total", usage.output_tokens, direction="output")
            logger.info(
                "chat request_id=%s outcome=%s reason=%s duration_ms=%d tokens_in=%d tokens_out=%d",
                request.state.request_id,
                outcome,
                reason,
                seconds * 1000,
                usage.input_tokens,
                usage.output_tokens,
            )

        try:
            reply = request.app.state.assistant.answer(body.message)
        except (InvalidInput, LLMError) as error:
            kind, http_error = _map_error(error)
            metrics.inc("chat_errors_total", kind=kind)
            record("error", kind, Usage())
            if kind == "llm_error":
                # The provider message stays in the server log, never in the response.
                logger.exception("llm_error request_id=%s", request.state.request_id)
            raise http_error from error
        if reply.fallback:
            metrics.inc("chat_fallbacks_total", reason=reply.reason or "unknown")
        record("fallback" if reply.fallback else "answered", reply.reason, reply.usage)
        return ChatResponse(
            answer=reply.answer,
            sources=[Source(id=chunk.id, title=chunk.title) for chunk in reply.sources],
            products=[Product(**product) for product in reply.products],
            fallback=reply.fallback,
        )

    return app


app = create_app()
