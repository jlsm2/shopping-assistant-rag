"""Conversation memory, kept on the server so the customer can ask follow-ups.

The client holds only an unguessable id and never sends the history back, so it
cannot forge what the assistant said. A conversation belongs to the customer
that started it (the same key the rate limit uses).

# shortcut: in memory and per process, like the rate limit; use Redis with more than one worker
"""

import secrets
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

CONVERSATION_TTL_SECONDS = 30 * 60
MAX_TURNS = 4
MAX_CONVERSATIONS = 10_000
ID_BYTES = 24  # 32 url-safe characters


@dataclass(frozen=True)
class Turn:
    question: str  # already cleaned, personal data masked
    answer: str  # what the customer received, never a rejected draft
    source_ids: tuple[str, ...] = ()
    product_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Conversation:
    owner: str
    turns: tuple[Turn, ...]
    last_seen: float


class ConversationStore:
    def __init__(
        self,
        ttl_seconds: float = CONVERSATION_TTL_SECONDS,
        max_turns: int = MAX_TURNS,
        max_conversations: int = MAX_CONVERSATIONS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_turns = max_turns
        self._max_conversations = max_conversations
        self._clock = clock
        self._items: OrderedDict[str, _Conversation] = OrderedDict()  # least recent first
        self._lock = threading.Lock()

    def open(self, conversation_id: str | None, owner: str) -> tuple[str, tuple[Turn, ...]]:
        """The id to use and its history. Unknown, expired or someone else's
        conversations start over with a new id."""
        with self._lock:
            item = self._live(conversation_id, owner) if conversation_id else None
        if conversation_id is None or item is None:
            return secrets.token_urlsafe(ID_BYTES), ()
        return conversation_id, item.turns

    def append(self, conversation_id: str, owner: str, turn: Turn) -> None:
        now = self._clock()
        with self._lock:
            existing = self._items.get(conversation_id)
            if existing and existing.owner != owner:
                return  # never write into another customer's conversation
            item = self._live(conversation_id, owner)
            turns = (*(item.turns if item else ()), turn)[-self._max_turns :]
            self._items.pop(conversation_id, None)
            self._items[conversation_id] = _Conversation(owner, turns, now)
            self._evict(now)

    def _live(self, conversation_id: str, owner: str) -> _Conversation | None:
        item = self._items.get(conversation_id)
        if item and item.owner == owner and self._clock() - item.last_seen < self._ttl:
            return item
        return None

    def _evict(self, now: float) -> None:
        while self._items:
            oldest = next(iter(self._items.values()))
            if len(self._items) <= self._max_conversations and now - oldest.last_seen < self._ttl:
                return
            self._items.popitem(last=False)
