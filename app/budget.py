"""Daily spend ceiling, counted in LLM tokens.

A public chat endpoint is a way to spend someone else's money. The per-client
rate limit slows one abuser down; this caps the total for the day. Tokens are
the unit because they are the same for every provider.
"""

import threading
import time
from collections.abc import Callable

from app.llm import Usage

SECONDS_PER_DAY = 86_400


class TokenBudget:
    """Tokens (input + output) allowed per UTC day.

    # shortcut: in-memory and per-process; share it in Redis with more than one worker
    """

    def __init__(self, daily_limit: int, clock: Callable[[], float] = time.time) -> None:
        self.daily_limit = daily_limit
        self._clock = clock
        self._day = self._today()
        self._used = 0
        self._lock = threading.Lock()

    def _today(self) -> int:
        return int(self._clock() // SECONDS_PER_DAY)

    def used(self) -> int:
        with self._lock:
            if self._today() != self._day:
                self._day, self._used = self._today(), 0
            return self._used

    def exhausted(self) -> bool:
        return self.used() >= self.daily_limit

    def spend(self, usage: Usage) -> None:
        self.used()  # roll the day over first
        with self._lock:
            self._used += usage.input_tokens + usage.output_tokens
