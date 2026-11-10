import pytest

from app.budget import SECONDS_PER_DAY, TokenBudget
from app.llm import Usage
from app.telemetry import Metrics, accept_request_id

# --- request id ----------------------------------------------------------------


@pytest.mark.parametrize("candidate", ["req-123", "a.b_c-D", "f" * 64])
def test_accept_request_id_keeps_a_safe_caller_id(candidate):
    assert accept_request_id(candidate) == candidate


@pytest.mark.parametrize(
    "candidate", [None, "", "f" * 65, "id with space", "id\ninjected=1", "<script>", "é"]
)
def test_accept_request_id_replaces_anything_unsafe_to_log(candidate):
    generated = accept_request_id(candidate)

    assert generated != candidate
    assert len(generated) == 32 and generated.isalnum()


def test_accept_request_id_generates_a_new_id_each_time():
    assert accept_request_id(None) != accept_request_id(None)


# --- metrics -------------------------------------------------------------------


def test_metrics_render_counters_with_labels_in_prometheus_format():
    metrics = Metrics()
    metrics.inc("chat_fallbacks_total", reason="needs_human")
    metrics.inc("chat_fallbacks_total", reason="needs_human")
    metrics.inc("chat_fallbacks_total", reason="ungrounded_number")
    metrics.inc("llm_tokens_total", 120, direction="input")

    text = metrics.render({})

    assert "# TYPE chat_fallbacks_total counter" in text
    assert 'chat_fallbacks_total{reason="needs_human"} 2' in text
    assert 'chat_fallbacks_total{reason="ungrounded_number"} 1' in text
    assert 'llm_tokens_total{direction="input"} 120' in text
    assert text.endswith("\n")


def test_metrics_render_a_cumulative_latency_histogram():
    metrics = Metrics()
    for seconds in (0.2, 0.7, 3.0, 99.0):
        metrics.observe(seconds)

    text = metrics.render({})

    assert "# TYPE chat_duration_seconds histogram" in text
    assert 'chat_duration_seconds_bucket{le="0.5"} 1' in text
    assert 'chat_duration_seconds_bucket{le="1.0"} 2' in text
    assert 'chat_duration_seconds_bucket{le="5.0"} 3' in text
    assert 'chat_duration_seconds_bucket{le="30.0"} 3' in text
    assert 'chat_duration_seconds_bucket{le="+Inf"} 4' in text
    assert "chat_duration_seconds_count 4" in text
    assert "chat_duration_seconds_sum 102.9" in text


def test_metrics_render_gauges_and_escape_label_values():
    metrics = Metrics()
    metrics.inc("chat_errors_total", kind='odd"kind\\\n')

    text = metrics.render({"llm_daily_tokens_used": 42})

    assert "# TYPE llm_daily_tokens_used gauge\nllm_daily_tokens_used 42" in text
    assert 'chat_errors_total{kind="odd\\"kind\\\\\\n"} 1' in text


# --- daily token budget ----------------------------------------------------------


def test_budget_is_exhausted_once_the_daily_limit_is_spent():
    budget = TokenBudget(daily_limit=1000, clock=lambda: 0.0)

    budget.spend(Usage(600, 300))
    assert budget.used() == 900 and not budget.exhausted()

    budget.spend(Usage(90, 10))
    assert budget.exhausted()


def test_budget_starts_again_on_the_next_utc_day():
    now = [SECONDS_PER_DAY - 1.0]
    budget = TokenBudget(daily_limit=100, clock=lambda: now[0])
    budget.spend(Usage(100, 0))
    assert budget.exhausted()

    now[0] = SECONDS_PER_DAY + 1.0

    assert budget.used() == 0 and not budget.exhausted()
