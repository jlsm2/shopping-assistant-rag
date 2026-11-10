"""Knowledge base chunking and BM25 search.

# shortcut: BM25 em memória; migrar para pgvector + busca híbrida quando o corpus crescer
The v2 swap keeps the `Index.search(query, k) -> list[Chunk]` interface.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

K1 = 1.5
B = 0.75
TITLE_WEIGHT = 2
MIN_STEM_LENGTH = 3

STOPWORDS = frozenset(
    """
    a o as os um uma uns umas de do da dos das em no na nos nas por pelo pela pelos pelas
    para pra com sem sobre entre ate e ou mas que se como quando onde qual quais quanto
    quanta quantos quantas eu voce voces ele ela eles elas me te lhe meu minha meus minhas
    seu sua seus suas esse essa isso este esta isto aquele aquela aquilo ser sao foi era
    estar estao ter tem tenho ha ja nao sim mais menos muito tambem so ao aos la ai aqui
    posso pode podem quero queria gostaria fazer faco faz vou vai oi ola bom boa favor
    obrigada obrigado
    """.split()
)

# Longest first, so "amentos" wins over "os".
_SUFFIXES = sorted(
    """
    amentos imentos amento imento acoes icoes ucoes acao icao ucao mente aveis iveis avel
    ivel amos emos imos ando endo indo adas idas ados idos ada ida ado ido oes aes ais eis
    ar er ir am em
    as os es a o e s
    """.split(),
    key=len,
    reverse=True,
)

_TOKEN = re.compile(r"[a-z0-9]+")
_SLUG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase and strip accents, so "Devolução" and "devolucao" match."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).lower()


def stem(token: str) -> str:
    """Light Portuguese suffix stripping: troca/trocar/trocas -> troc."""
    if any(ch.isdigit() for ch in token):
        return token
    if token.endswith("ns"):  # cupons -> cupom
        token = token[:-2] + "m"
    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= MIN_STEM_LENGTH:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [
        stem(token)
        for token in _TOKEN.findall(normalize(text))
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def slugify(title: str) -> str:
    return _SLUG_SEPARATOR.sub("-", normalize(title)).strip("-")


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str


def chunk_markdown(source: str, markdown: str) -> list[Chunk]:
    """One chunk per `## ` section. Text before the first section is ignored."""
    chunks: list[Chunk] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if title is not None:
            chunks.append(Chunk(f"{source}#{slugify(title)}", title, "\n".join(body).strip()))

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            title, body = line[3:].strip(), []
        elif title is not None:
            body.append(line)
    flush()
    return chunks


def load_chunks(directory: Path) -> list[Chunk]:
    chunks = [
        chunk
        for path in sorted(directory.glob("*.md"))
        for chunk in chunk_markdown(path.stem, path.read_text(encoding="utf-8"))
    ]
    ids = [chunk.id for chunk in chunks]
    duplicated = {i for i in ids if ids.count(i) > 1}
    if duplicated:
        # Ids are what the model cites; two chunks with one id would be ambiguous.
        raise ValueError(f"duplicate chunk ids: {sorted(duplicated)}")
    return chunks


class Index:
    def __init__(self, chunks: list[Chunk], top_k: int = 4, min_score: float = 1.0) -> None:
        self._chunks = chunks
        self._top_k = top_k
        self._min_score = min_score
        self._term_counts = [
            Counter(tokenize(c.title) * TITLE_WEIGHT + tokenize(c.text)) for c in chunks
        ]
        self._lengths = [sum(counts.values()) for counts in self._term_counts]
        self._average_length = sum(self._lengths) / len(chunks) if chunks else 0.0
        document_frequency = Counter(term for counts in self._term_counts for term in counts)
        self._idf = {
            term: math.log(1 + (len(chunks) - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def get(self, chunk_id: str) -> Chunk | None:
        return next((c for c in self._chunks if c.id == chunk_id), None)

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        """Best chunks first. Empty when nothing reaches the minimum score."""
        terms = set(tokenize(query))
        scored = []
        for position, counts in enumerate(self._term_counts):
            score = sum(self._bm25(term, counts, self._lengths[position]) for term in terms)
            if score >= self._min_score:
                scored.append((score, position))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [self._chunks[position] for _, position in scored[: k or self._top_k]]

    def _bm25(self, term: str, counts: Counter, length: int) -> float:
        frequency = counts.get(term, 0)
        if not frequency:
            return 0.0
        norm = K1 * (1 - B + B * length / self._average_length)
        return self._idf[term] * frequency * (K1 + 1) / (frequency + norm)
