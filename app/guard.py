"""Deterministic checks around the model: input cleaning, data envelopes for
untrusted content, output sanitizing and grounding validation.

Nothing here asks a model for an opinion. Each check is plain code, so it
behaves the same no matter what the model was talked into.
"""

import html
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from app.retrieval import Chunk, normalize

MAX_INPUT_CHARS = 800
MIN_LEAK_SENTENCE_CHARS = 40
REMOVED_LINK = "[link removido]"


class InvalidInput(ValueError):
    """The customer message cannot be processed. The message is safe to show."""


# --- input -------------------------------------------------------------------


def clean_input(text: str) -> str:
    """Drop control and invisible characters, collapse whitespace, check size.

    Unicode category C covers control codes and also format characters
    (zero-width, bidi overrides, tag characters) used to hide instructions.
    """
    text = unicodedata.normalize("NFC", text)
    visible = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch)[0] != "C"
    )
    cleaned = " ".join(visible.split())
    if not cleaned:
        raise InvalidInput("A mensagem está vazia.")
    if len(cleaned) > MAX_INPUT_CHARS:
        raise InvalidInput(f"A mensagem passa de {MAX_INPUT_CHARS} caracteres.")
    return cleaned


# --- envelopes ---------------------------------------------------------------


def escape(text: str) -> str:
    """Neutralize markup so untrusted text cannot close or forge an envelope tag."""
    return html.escape(text, quote=False)


def wrap_document(chunk: Chunk) -> str:
    return (
        f'<document id="{html.escape(chunk.id)}" title="{html.escape(chunk.title)}">\n'
        f"{escape(chunk.text)}\n</document>"
    )


def wrap_tool_data(payload: Any) -> str:
    return f"<tool_data>\n{escape(json.dumps(payload, ensure_ascii=False))}\n</tool_data>"


def wrap_customer_message(text: str) -> str:
    return f"<customer_message>\n{escape(text)}\n</customer_message>"


# --- output ------------------------------------------------------------------

# Case-insensitive on purpose: a chat widget that auto-links text would link
# "EVIL.COM" too. The cost is that a typo like "fim.Depois" reads as a domain.
_LINK = re.compile(
    r"(?:\b[a-z][a-z0-9+.-]*://|\bwww\.)[^\s<>\"')\]]+"  # scheme://... or www....
    r"|\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b(?:/[^\s<>\"')\]]*)?",  # bare domain.tld/path
    re.IGNORECASE,
)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)


def _host(link: str) -> str:
    authority = re.split(r"[/?#]", _SCHEME.sub("", link), maxsplit=1)[0]
    host = authority.rsplit("@", 1)[-1].split(":", 1)[0]
    return host.lower().strip(".")


def _is_allowed(host: str, allowed_domains: Iterable[str]) -> bool:
    return any(host == domain or host.endswith("." + domain) for domain in allowed_domains)


def strip_external_links(text: str, allowed_domains: Iterable[str]) -> str:
    """Replace every link whose host is not the store's (or a subdomain of it)."""
    allowed = tuple(allowed_domains)
    return _LINK.sub(
        lambda match: match.group() if _is_allowed(_host(match.group()), allowed) else REMOVED_LINK,
        text,
    )


def _squash(text: str) -> str:
    return " ".join(normalize(text).split())


def leaks_prompt(text: str, system_prompt: str, canary: str) -> bool:
    """True when the output carries the canary or a full sentence of the system prompt."""
    haystack = _squash(text)
    if _squash(canary) in haystack:
        return True
    sentences = re.split(r"(?<=[.!?:])\s+|\n+", system_prompt)
    return any(
        len(sentence) >= MIN_LEAK_SENTENCE_CHARS and _squash(sentence) in haystack
        for sentence in sentences
    )


# --- grounding ---------------------------------------------------------------

_NUMBER = r"(?:\d{1,3}(?:\.\d{3})+(?:,\d{1,2})?|\d+(?:[.,]\d{1,2})?)"
_MONEY = re.compile(rf"R\$\s*({_NUMBER})|(?<![\d.,])({_NUMBER})\s*reais\b", re.IGNORECASE)
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
_CENTS = Decimal("0.01")

# Runs on normalized text (lowercase, no accents). Captures "7 dias",
# "3 a 5 dias", "10%", "3x", "24h".
_QUANTITY = re.compile(
    r"(\d+)(?:\s*(?:a|e|ou|-|–)\s*(\d+))?\s*"
    r"(%|x\b|vezes\b|h\b|horas?\b|dias?\b|semanas?\b|mes(?:es)?\b|minutos?\b|segundos?\b|anos?\b)"
)
_UNITS = {"vezes": "x", "h": "hora", "meses": "mes", "mes": "mes"}

# Phone numbers, Pix keys, CPF/CNPJ, bank accounts: 8+ digits, separators allowed.
_LONG_NUMBER = re.compile(r"\d[\d\s().\-/]{6,}\d")
MIN_LONG_NUMBER_DIGITS = 8


def parse_brl(raw: str) -> Decimal:
    """'59,90', '59.90', '1.299,00' and '1.299' as Decimal with two places."""
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    elif _THOUSANDS.fullmatch(raw):
        raw = raw.replace(".", "")
    return Decimal(raw).quantize(_CENTS)


def extract_prices(text: str) -> set[Decimal]:
    return {parse_brl(a or b) for a, b in _MONEY.findall(text)}


def extract_quantities(text: str) -> set[tuple[int, str]]:
    """(number, unit) pairs for durations, percentages and installments."""
    found = set()
    for first, second, unit in _QUANTITY.findall(normalize(text)):
        unit = _UNITS.get(unit, unit if unit in "%x" else unit.rstrip("s"))
        found.update((int(number), unit) for number in (first, second) if number)
    return found


def extract_long_numbers(text: str) -> set[str]:
    """Digit strings that could be a contact or payment identifier."""
    digits = (re.sub(r"\D", "", match) for match in _LONG_NUMBER.findall(text))
    return {number for number in digits if len(number) >= MIN_LONG_NUMBER_DIGITS}


@dataclass
class Evidence:
    """Everything the model was shown for one question. Nothing else may be claimed."""

    chunks: dict[str, Chunk] = field(default_factory=dict)
    products: dict[str, dict[str, Any]] = field(default_factory=dict)  # public views
    tool_texts: list[str] = field(default_factory=list)
    store_facts: tuple[str, ...] = ()  # always true, e.g. the human contact

    def texts(self) -> list[str]:
        chunk_texts = [f"{chunk.title}\n{chunk.text}" for chunk in self.chunks.values()]
        return [*chunk_texts, *self.tool_texts, *self.store_facts]

    def prices(self) -> set[Decimal]:
        from_products = {
            Decimal(str(view["price"])).quantize(_CENTS) for view in self.products.values()
        }
        return from_products.union(*(extract_prices(text) for text in self.texts()))

    def quantities(self) -> set[tuple[int, str]]:
        return set().union(*(extract_quantities(text) for text in self.texts()))

    def long_numbers(self) -> set[str]:
        return set().union(*(extract_long_numbers(text) for text in self.texts()))


def validate_grounding(
    answer: str, citations: list[str], products: list[str], evidence: Evidence
) -> str | None:
    """Return the reason the answer is not grounded, or None when it is."""
    if any(citation not in evidence.chunks for citation in citations):
        return "unknown_citation"
    if any(product not in evidence.products for product in products):
        return "unknown_product"
    if not citations and not products:
        return "no_evidence_cited"
    if extract_prices(answer) - evidence.prices():
        return "ungrounded_price"
    if extract_quantities(answer) - evidence.quantities():
        return "ungrounded_quantity"
    if extract_long_numbers(answer) - evidence.long_numbers():
        # A made-up phone or Pix key is a fraud vector, not just a wrong answer.
        return "ungrounded_contact"
    return None
