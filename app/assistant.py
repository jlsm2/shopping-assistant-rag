"""Orchestration: retrieve -> tool loop -> validate.

Single-turn on purpose: history sent by the client could forge assistant turns.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app import guard
from app.catalog import Catalog, public_view
from app.config import Store
from app.llm import (
    AssistantMessage,
    LLMClient,
    Message,
    ToolCall,
    ToolMessage,
    ToolResult,
    ToolSpec,
    Usage,
    UserMessage,
)
from app.prompts import CANARY, build_system_prompt
from app.retrieval import Chunk, Index

logger = logging.getLogger(__name__)

MAX_PRODUCT_RESULTS = 5
MAX_TOOL_CALLS_PER_TURN = 5

ANSWER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "products": {"type": "array", "items": {"type": "string"}},
        "outcome": {"type": "string", "enum": ["answered", "smalltalk", "needs_human"]},
    },
    "required": ["answer", "citations", "products", "outcome"],
    "additionalProperties": False,
}


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str]
    products: list[str]
    outcome: Literal["answered", "smalltalk", "needs_human"]


class SearchProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=200)
    max_price: float | None = Field(ge=0)


class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(min_length=1, max_length=100)


# The allowlist. Both tools only read; nothing the model emits can change state.
TOOL_INPUTS: dict[str, type[BaseModel]] = {
    "search_products": SearchProductsInput,
    "get_product": GetProductInput,
}

TOOLS = (
    ToolSpec(
        name="search_products",
        description=(
            "Busca produtos no catálogo da loja por palavras-chave (cor, formato, tipo). "
            "Retorna até 5 produtos com id, nome, preço, disponibilidade e descrição. "
            "Use max_price quando a cliente informar um teto de preço, senão envie null."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_price": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            },
            "required": ["query", "max_price"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="get_product",
        description=(
            "Consulta um produto pelo id, para confirmar preço e disponibilidade atuais. "
            'Retorna o produto ou {"error": "not_found"}.'
        ),
        parameters={
            "type": "object",
            "properties": {"product_id": {"type": "string"}},
            "required": ["product_id"],
            "additionalProperties": False,
        },
    ),
)


@dataclass(frozen=True)
class AssistantReply:
    answer: str
    sources: tuple[Chunk, ...] = ()
    products: tuple[dict[str, Any], ...] = ()
    fallback: bool = False
    reason: str | None = None  # why it fell back; feeds logs and eval metrics
    usage: Usage = field(default_factory=Usage)


class Assistant:
    def __init__(
        self,
        llm: LLMClient,
        index: Index,
        catalog: Catalog,
        store: Store,
        max_tool_iterations: int = 4,
    ) -> None:
        self._llm = llm
        self._index = index
        self._catalog = catalog
        self._store = store
        self._max_tool_iterations = max_tool_iterations
        self._system_prompt = build_system_prompt(store.name)

    def answer(self, question: str) -> AssistantReply:
        """Answer one customer message. Raises `guard.InvalidInput` and `LLMError`."""
        question = guard.clean_input(question)
        chunks = self._index.search(question)
        evidence = guard.Evidence(
            chunks={chunk.id: chunk for chunk in chunks},
            store_facts=(self._store.human_contact,),
        )
        messages: list[Message] = [UserMessage(_render_user_turn(chunks, question))]
        usage = Usage()

        for iteration in range(self._max_tool_iterations + 1):
            response = self._llm.complete(
                system=self._system_prompt,
                messages=messages,
                tools=TOOLS,
                response_schema=ANSWER_SCHEMA,
            )
            usage += response.usage
            if response.stop != "tool_calls":
                break
            if iteration == self._max_tool_iterations:
                return self._fallback("tool_iteration_limit", usage)
            if len(response.tool_calls) > MAX_TOOL_CALLS_PER_TURN:
                return self._fallback("too_many_tool_calls", usage)
            results = tuple(self._run_tool(call, evidence) for call in response.tool_calls)
            messages += [AssistantMessage(response), ToolMessage(results)]

        if response.stop != "final":
            return self._fallback(f"stop_{response.stop}", usage)
        try:
            parsed = ModelAnswer.model_validate_json(response.text)
        except ValidationError:
            return self._fallback("invalid_json", usage)

        if parsed.outcome == "needs_human":
            return self._fallback("needs_human", usage)
        if parsed.outcome == "smalltalk":
            # Canned text: a greeting never carries model-written claims.
            return AssistantReply(answer=self._store.greeting_message, usage=usage)
        if len(parsed.answer) > guard.MAX_ANSWER_CHARS:
            return self._fallback("answer_too_long", usage)
        try:
            answer = guard.clean_output(parsed.answer)
            reason = guard.review_answer(
                answer,
                parsed.citations,
                parsed.products,
                evidence,
                self._store.allowed_domains,
                self._system_prompt,
                CANARY,
            )
        except Exception:
            # A guard that crashes must fail closed, not become a 500 or a pass.
            logger.exception("guard_failed")
            reason = "guard_error"
        if reason:
            return self._fallback(reason, usage)

        products = tuple(evidence.products[i] for i in dict.fromkeys(parsed.products))
        sold_out = [product["name"] for product in products if not product["in_stock"]]
        if sold_out and not guard.mentions_unavailability(answer):
            # Server-written, so stock is right even if the model glossed over it.
            answer += f" Indisponível no momento: {', '.join(sold_out)}."
        return AssistantReply(
            answer=answer,
            sources=tuple(evidence.chunks[i] for i in dict.fromkeys(parsed.citations)),
            products=products,
            usage=usage,
        )

    def _run_tool(self, call: ToolCall, evidence: guard.Evidence) -> ToolResult:
        input_model = TOOL_INPUTS.get(call.name)
        if input_model is None:
            logger.warning("tool_rejected reason=not_allowed name=%.40r", call.name)
            return ToolResult(call.id, "Ferramenta inexistente.", is_error=True)
        try:
            arguments = input_model.model_validate(call.arguments)
        except ValidationError:
            logger.warning("tool_rejected reason=invalid_input name=%s", call.name)
            return ToolResult(call.id, "Parâmetros inválidos.", is_error=True)
        try:
            if isinstance(arguments, SearchProductsInput):
                found = self._catalog.search(arguments.query, arguments.max_price)
                views = [public_view(product) for product in found[:MAX_PRODUCT_RESULTS]]
                payload: Any = views
            else:
                product = self._catalog.get(arguments.product_id)
                views = [public_view(product)] if product else []
                payload = views[0] if views else {"error": "not_found"}
        except Exception:
            # A future HttpCatalog can fail; the model gets an error, not a crash.
            logger.exception("tool_failed name=%s", call.name)
            return ToolResult(call.id, "Catálogo indisponível no momento.", is_error=True)
        evidence.products.update({view["id"]: view for view in views})
        return ToolResult(call.id, guard.wrap_tool_data(payload))

    def _fallback(self, reason: str, usage: Usage) -> AssistantReply:
        # The customer message is deliberately not logged.
        logger.info("fallback reason=%s", reason)
        return AssistantReply(
            answer=self._store.fallback_message, fallback=True, reason=reason, usage=usage
        )


def _render_user_turn(chunks: list[Chunk], question: str) -> str:
    documents = "\n".join(guard.wrap_document(chunk) for chunk in chunks)
    return (
        f"<documents>\n{documents or '(nenhum documento encontrado)'}\n</documents>\n\n"
        f"{guard.wrap_customer_message(question)}"
    )
