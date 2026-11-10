"""HTTP API: POST /chat and GET /health."""

import hmac
import ipaddress
import logging
import math
import threading
import time
from collections import deque
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app.assistant import Assistant
from app.catalog import JsonCatalog
from app.config import Settings, Store
from app.guard import MAX_INPUT_CHARS, InvalidInput
from app.llm import LLMClient, LLMError, LLMRateLimited, LLMUnavailable
from app.retrieval import Index, load_chunks

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 16_384
RATE_WINDOW_SECONDS = 60.0
MAX_TRACKED_CLIENTS = 10_000
DEFAULT_RETRY_AFTER_SECONDS = 30
END_USER_HEADER = "x-end-user-id"
MAX_END_USER_CHARS = 128


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


def build_llm() -> LLMClient:
    """The single place where the LLM provider gets plugged in."""
    raise RuntimeError(
        "No LLM provider configured yet. Implement app.llm.LLMClient for the chosen "
        "API and return it from app.api.build_llm()."
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
    )


def create_app(assistant: Assistant | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    limiter = RateLimiter(settings.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Fail at startup, not on the first customer, when something is missing.
        if not settings.service_tokens:
            raise RuntimeError(
                "SERVICE_TOKENS is not set. Generate one with "
                '`python -c "import secrets; print(secrets.token_urlsafe(32))"`.'
            )
        app.state.assistant = assistant or build_assistant(settings, build_llm())
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
        # shortcut: trusts Content-Length; enforce the body limit at the reverse proxy too
        declared = request.headers.get("content-length", "0")
        if not (declared.isascii() and declared.isdigit()) or int(declared) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "Requisição grande demais."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
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
        try:
            reply = request.app.state.assistant.answer(body.message)
        except InvalidInput as error:
            raise HTTPException(422, str(error)) from error
        except LLMRateLimited as error:
            logger.warning("llm_error kind=rate_limited")
            retry_after = error.retry_after or DEFAULT_RETRY_AFTER_SECONDS
            raise HTTPException(
                503,
                "O assistente está sobrecarregado. Tente de novo em instantes.",
                headers={"Retry-After": str(retry_after)},
            ) from error
        except LLMUnavailable as error:
            logger.warning("llm_error kind=unavailable")
            raise HTTPException(502, "O assistente está indisponível no momento.") from error
        except LLMError as error:
            # The provider message stays in the server log, never in the response.
            logger.exception("llm_error kind=other")
            raise HTTPException(500, "Erro interno.") from error
        return ChatResponse(
            answer=reply.answer,
            sources=[Source(id=chunk.id, title=chunk.title) for chunk in reply.sources],
            products=[Product(**product) for product in reply.products],
            fallback=reply.fallback,
        )

    return app


app = create_app()
