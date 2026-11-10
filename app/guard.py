"""Deterministic checks around the model: input cleaning, data envelopes for
untrusted content, and review of the model's answer.

Nothing here asks a model for an opinion. Each check is plain code, so it
behaves the same no matter what the model was talked into. The answer review
assumes the model is compromised and can emit any text it likes.
"""

import html
import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from app.retrieval import Chunk, normalize

MAX_INPUT_CHARS = 800
MAX_ANSWER_CHARS = 1200
LEAK_WINDOW_WORDS = 8
MIN_LONG_NUMBER_DIGITS = 8
MIN_CODE_CHARS = 4


class InvalidInput(ValueError):
    """The customer message cannot be processed. The message is safe to show."""


# --- cleaning ----------------------------------------------------------------


def _visible(text: str, form: str) -> str:
    """Normalize, drop invisible characters and collapse whitespace.

    Unicode category C covers control codes and also format characters
    (zero-width, bidi overrides, tag characters) used to hide text.
    """
    text = unicodedata.normalize(form, text)
    kept = "".join(
        " " if ch.isspace() else ch
        for ch in text
        if ch.isspace() or unicodedata.category(ch)[0] != "C"
    )
    return " ".join(kept.split())


def clean_input(text: str) -> str:
    cleaned = _visible(text, "NFC")
    if not cleaned:
        raise InvalidInput("A mensagem está vazia.")
    if len(cleaned) > MAX_INPUT_CHARS:
        raise InvalidInput(f"A mensagem passa de {MAX_INPUT_CHARS} caracteres.")
    return cleaned


def clean_output(text: str) -> str:
    """Same cleaning for the model's answer, before any check reads it.

    NFKC also folds look-alikes such as fullwidth digits and "＄" into ASCII,
    so "R＄ 5" cannot dodge the number checks.
    """
    return _visible(text, "NFKC")


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


# --- links -------------------------------------------------------------------

_URL_BODY = r"[^\s<>\"')\]]+"
_LABEL = r"[^\W_](?:[^\W_]|-)*"  # Unicode-aware, so "promoção.com" is a domain too
# Case-insensitive on purpose: a chat widget that auto-links text links "EVIL.COM".
# The cost is that a typo like "fim.Depois" reads as a domain.
_LINK = re.compile(
    rf"(?:\b[a-z][a-z0-9+.-]*:)?//{_URL_BODY}"  # scheme://host or //host
    rf"|\bwww\.{_URL_BODY}"
    rf"|(?:{_LABEL}\.)+[^\W\d_]{{2,}}(?:/{_URL_BODY})?"  # bare domain.tld/path
    r"|\b\d{1,3}(?:\.\d{1,3}){3}\b",  # IPv4
    re.IGNORECASE,
)
_SCHEME = re.compile(r"^(?:[a-z][a-z0-9+.-]*:)?//", re.IGNORECASE)


def _host(link: str) -> str:
    # Browsers read "\" as "/" in http(s) URLs, so it ends the authority too.
    authority = re.split(r"[/?#\\]", _SCHEME.sub("", link), maxsplit=1)[0]
    host = authority.rsplit("@", 1)[-1].split(":", 1)[0]
    return host.lower().strip(".")


def find_external_links(text: str, allowed_domains: Iterable[str]) -> list[str]:
    """Links whose host is not the store's domain or a subdomain of it."""
    allowed = tuple(allowed_domains)
    return [
        link
        for link in (match.group() for match in _LINK.finditer(text))
        if not any(_host(link) == d or _host(link).endswith("." + d) for d in allowed)
    ]


# --- prompt leak -------------------------------------------------------------


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize(text))


def leaks_prompt(text: str, system_prompt: str, canary: str) -> bool:
    """True when the text carries the canary or a run of words copied from the prompt.

    Punctuation, case and accents are ignored, so re-punctuating the prompt
    does not hide it.
    """
    words = _words(text)
    if "".join(_words(canary)) in "".join(words):
        return True
    prompt_words = _words(system_prompt)
    size = LEAK_WINDOW_WORDS
    windows = {tuple(prompt_words[i : i + size]) for i in range(len(prompt_words) - size + 1)}
    return any(tuple(words[i : i + size]) in windows for i in range(len(words) - size + 1))


# --- grounding ---------------------------------------------------------------

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_THOUSANDS = re.compile(r"\d{1,3}(?:\.\d{3})+")
_CODE = re.compile(r"[a-z0-9]+")
# Phone numbers, Pix keys, CPF/CNPJ, bank accounts: digits joined by separators.
_DIGIT_RUN = re.compile(r"\d(?:[\W_]{0,3}\d)+")
_SENSITIVE = re.compile(r"\b(?:senha|cvv|cpf|codigo de seguranca|(?:numero|dados) do cartao)\b")
_UNAVAILABLE = re.compile(r"indisponi|esgotad|sem estoque|fora de estoque|em falta")


def canonical_number(raw: str) -> str:
    """One spelling per value: '59,90', '59.90' and '59.9' -> '59.9'; '1.299,00' -> '1299'."""
    if "," in raw:
        candidate = raw.replace(".", "").replace(",", ".")
    elif _THOUSANDS.fullmatch(raw):
        candidate = raw.replace(".", "")
    else:
        candidate = raw
    integer, _, fraction = candidate.partition(".")
    if not integer.isdigit() or (fraction and not fraction.isdigit()):  # e.g. "1.2.3"
        return re.sub(r"\D", "", raw)
    integer, fraction = integer.lstrip("0") or "0", fraction.rstrip("0")
    return f"{integer}.{fraction}" if fraction else integer


def extract_numbers(text: str) -> set[str]:
    """Every number in the text, whatever unit or currency sign is next to it."""
    return {canonical_number(raw) for raw in _NUMBER.findall(text)}


def extract_codes(text: str) -> set[str]:
    """Tokens mixing letters and digits, such as coupon codes ("BEMVINDA10")."""
    return {
        token
        for token in _CODE.findall(normalize(text))
        if len(token) >= MIN_CODE_CHARS and not token.isdigit() and not token.isalpha()
    }


def _long_numbers(text: str) -> set[str]:
    digits = (re.sub(r"\D", "", run) for run in _DIGIT_RUN.findall(text))
    return {number for number in digits if len(number) >= MIN_LONG_NUMBER_DIGITS}


def has_unknown_long_number(text: str, evidence_texts: Iterable[str]) -> bool:
    """A phone/Pix/document-like number that the evidence does not contain."""
    known = set().union(*(_long_numbers(evidence) for evidence in evidence_texts))
    for number in known:  # blank out the known ones, whatever separators were used
        text = re.sub(r"[\W_]{0,3}".join(number), " ", text)
    return any(not any(number in other for other in known) for number in _long_numbers(text))


def mentions_unavailability(text: str) -> bool:
    return bool(_UNAVAILABLE.search(normalize(text)))


@dataclass
class Evidence:
    """What the model was shown for one question. Nothing else may be claimed."""

    chunks: dict[str, Chunk] = field(default_factory=dict)
    products: dict[str, dict[str, Any]] = field(default_factory=dict)  # public views
    store_facts: tuple[str, ...] = ()  # always true, e.g. the human contact

    def cited_texts(self, citations: Iterable[str], products: Iterable[str]) -> list[str]:
        """Only what the answer cites counts, so a number cannot be borrowed
        from a neighbouring document or from another product."""
        documents = [f"{self.chunks[i].title}\n{self.chunks[i].text}" for i in citations]
        views = [json.dumps(self.products[i], ensure_ascii=False) for i in products]
        return [*documents, *views, *self.store_facts]


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
    texts = evidence.cited_texts(citations, products)
    if extract_numbers(answer) - set().union(*map(extract_numbers, texts)):
        return "ungrounded_number"
    if extract_codes(answer) - set().union(*map(extract_codes, texts)):
        return "ungrounded_code"
    if has_unknown_long_number(answer, texts):
        # A made-up phone or Pix key is a fraud vector, not just a wrong answer.
        return "ungrounded_contact"
    sensitive = set(_SENSITIVE.findall(normalize(answer)))
    if sensitive - set(_SENSITIVE.findall(normalize(" ".join(texts)))):
        return "sensitive_request"
    return None


def review_answer(
    answer: str,
    citations: list[str],
    products: list[str],
    evidence: Evidence,
    allowed_domains: Iterable[str],
    system_prompt: str,
    canary: str,
) -> str | None:
    """Every check on an already cleaned answer. Returns the fallback reason, or None."""
    if not answer:
        return "empty_answer"
    if "<" in answer or ">" in answer:
        return "markup_in_answer"
    if leaks_prompt(answer, system_prompt, canary):
        return "prompt_leak"
    if find_external_links(answer, allowed_domains):
        return "external_link"
    return validate_grounding(answer, citations, products, evidence)
