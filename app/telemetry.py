"""Request correlation and in-process metrics in Prometheus text format."""

import re
import threading
import uuid
from collections import Counter
from contextvars import ContextVar

# Set by the API for each request; read by any log line deeper in the stack.
request_id: ContextVar[str] = ContextVar("request_id", default="-")

_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")
LATENCY_BUCKETS = (0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


def accept_request_id(candidate: str | None) -> str:
    """Reuse the caller's id when it is safe to put in a log line, else make one."""
    if candidate and _SAFE_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    escaped = (
        (key, value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))
        for key, value in labels
    )
    return "{" + ",".join(f'{key}="{value}"' for key, value in escaped) + "}"


class Metrics:
    """Counters plus one latency histogram.

    # shortcut: in-memory and per-process; with several workers, scrape each one
    # or switch to prometheus_client in multiprocess mode
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._buckets = [0] * len(LATENCY_BUCKETS)
        self._duration_sum = 0.0
        self._duration_count = 0

    def inc(self, name: str, amount: int = 1, **labels: str) -> None:
        with self._lock:
            self._counters[(name, tuple(sorted(labels.items())))] += amount

    def observe(self, seconds: float) -> None:
        with self._lock:
            self._duration_sum += seconds
            self._duration_count += 1
            for position, limit in enumerate(LATENCY_BUCKETS):
                if seconds <= limit:  # buckets are cumulative
                    self._buckets[position] += 1

    def render(self, gauges: dict[str, float]) -> str:
        with self._lock:
            counters = sorted(self._counters.items())
            buckets = list(self._buckets)
            total, count = self._duration_sum, self._duration_count
        lines = []
        for name in dict.fromkeys(name for (name, _), _ in counters):
            lines.append(f"# TYPE {name} counter")
            lines += [
                f"{name}{_labels(labels)} {value}" for (n, labels), value in counters if n == name
            ]
        lines.append("# TYPE chat_duration_seconds histogram")
        lines += [
            f'chat_duration_seconds_bucket{{le="{limit}"}} {hits}'
            for limit, hits in zip(LATENCY_BUCKETS, buckets, strict=True)
        ]
        lines.append(f'chat_duration_seconds_bucket{{le="+Inf"}} {count}')
        lines.append(f"chat_duration_seconds_sum {round(total, 3)}")
        lines.append(f"chat_duration_seconds_count {count}")
        for name, value in gauges.items():
            lines += [f"# TYPE {name} gauge", f"{name} {value}"]
        return "\n".join(lines) + "\n"
