"""Per-inference metrics, published-rate pricing, and isolated stats storage."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

from core.metrics import (
    TIER_LARGE,
    TIER_MID,
    TIER_SMALL,
    USAGE_ESTIMATED,
    USAGE_REPORTED,
    USAGE_UNAVAILABLE,
    InferenceCall,
    TokenUsage,
    aggregate_calls,
    build_call,
    estimate_tokens,
    summarize_usage,
)
from core.metrics_store import MetricsStore
from core.pricing import DEFAULT_PRICES, ModelPrice, PricingTable

PRICING = PricingTable({
    "small-x": ModelPrice(Decimal("0.15"), Decimal("0.50")),
    "mid-x": ModelPrice(Decimal("1.40"), Decimal("4.40")),
    "large-x": ModelPrice(Decimal("2.00"), Decimal("6.00")),
})


def call(
    tier: str,
    model: str,
    latency_ms: float,
    prompt: int | None,
    completion: int | None,
    *,
    route: str = "test-route",
) -> InferenceCall:
    return build_call(
        tier=tier,
        route=route,
        requested_model=model,
        response_model=model,
        latency_ms=latency_ms,
        usage=TokenUsage(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=(
                None if prompt is None or completion is None else prompt + completion
            ),
            source=USAGE_REPORTED if prompt is not None else USAGE_UNAVAILABLE,
        ),
        pricing=PRICING,
    )


def test_aggregate_counts_latency_spend_and_savings() -> None:
    calls = [
        call(TIER_SMALL, "small-x", 10.0, 1000, 200,
             route="client:small"),
        call(TIER_LARGE, "large-x", 30.0, 500, 100,
             route="client:direct-large"),
    ]

    report = aggregate_calls(calls, PRICING, "large-x")

    assert report.total_calls == 2
    assert report.calls_by_tier == {TIER_SMALL: 1, TIER_LARGE: 1}
    assert report.calls_by_route == {"client:small": 1, "client:direct-large": 1}
    assert report.calls_by_model == {"small-x": 1, "large-x": 1}
    assert report.average_latency_ms == 20.0
    # 1000 * 0.15/1e6 + 200 * 0.50/1e6 = 0.00025; 500 * 2/1e6 + 100 * 6/1e6 = 0.0016
    assert report.estimated_spend_usd == Decimal("0.00185")
    # The same usage priced at the large-model rates: 0.0032 + 0.0016
    assert report.large_model_baseline_usd == Decimal("0.0048")
    assert report.estimated_savings_usd == Decimal("0.00295")
    assert report.savings_calls == 2
    assert report.calls_without_usage == 0
    assert report.cost_label == "estimated"
    assert report.savings_available is True


def test_unavailable_usage_is_excluded_from_cost_and_savings() -> None:
    calls = [
        call(TIER_MID, "mid-x", 5.0, 400, 40),
        call(TIER_MID, "mid-x", 7.0, None, None),
    ]

    report = aggregate_calls(calls, PRICING, "large-x")

    assert report.total_calls == 2
    assert report.calls_without_usage == 1
    assert report.priced_calls == 1
    assert report.savings_calls == 1
    # 400 * 1.4/1e6 + 40 * 4.4/1e6 = 0.00056 + 0.000176
    assert report.estimated_spend_usd == Decimal("0.000736")
    assert report.average_latency_ms == 6.0
    assert "excluded" in " ".join(report.notes)


def test_savings_are_unavailable_when_the_large_model_is_unpriced() -> None:
    report = aggregate_calls(
        [call(TIER_SMALL, "small-x", 4.0, 100, 20)], PRICING, "unknown-model"
    )

    assert report.estimated_spend_usd == Decimal("0.000025")
    assert report.large_model_baseline_usd is None
    assert report.estimated_savings_usd is None
    assert report.savings_available is False
    assert report.savings_calls == 0


def test_default_prices_match_documented_digitalocean_rates() -> None:
    assert DEFAULT_PRICES["glm-5.3-flash"] == ModelPrice(
        Decimal("0.15"), Decimal("0.50")
    )
    assert DEFAULT_PRICES["glm-5.3"] == ModelPrice(Decimal("1.40"), Decimal("4.40"))
    assert DEFAULT_PRICES["qwen3.8-max"] == ModelPrice(Decimal("2.00"), Decimal("6.00"))
    assert PricingTable().estimate_cost("glm-5.3", 1_000_000, 0) == Decimal("1.4")
    assert PricingTable().estimate_cost("glm-5.3", 0, 1_000_000) == Decimal("4.4")


def test_estimated_usage_is_labelled_and_never_reported_as_reported() -> None:
    usage = TokenUsage.estimate("four four four four", "eight eight")

    assert usage.source == USAGE_ESTIMATED
    assert usage.prompt_tokens == estimate_tokens("four four four four")
    assert usage.total_tokens == (usage.prompt_tokens or 0) + (
        usage.completion_tokens or 0
    )
    assert TokenUsage.from_completion(None).source == USAGE_UNAVAILABLE


def test_summarize_usage_reports_mixed_and_partial_sources() -> None:
    mixed = summarize_usage([
        call(TIER_SMALL, "small-x", 1.0, 10, 2),
        build_call(
            tier=TIER_SMALL, route="mock", requested_model="small-x",
            response_model=None, latency_ms=1.0,
            usage=TokenUsage.estimate("hi there", "ok"), pricing=PRICING,
        ),
    ])
    assert mixed.source == "mixed"
    # Reported and estimated token counts are summed, never silently upgraded.
    assert mixed.prompt_tokens == 10 + estimate_tokens("hi there")

    partial = summarize_usage([
        call(TIER_SMALL, "small-x", 1.0, 10, 2),
        call(TIER_SMALL, "small-x", 1.0, None, None),
    ])
    assert partial.source == "partial"

    assert summarize_usage([]).source == "unavailable"


def test_metrics_store_isolates_and_persists_calls_across_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stats.sqlite3"
    first = MetricsStore(path)
    first.record(call(TIER_MID, "mid-x", 12.5, 300, 30))

    second = MetricsStore(path)
    stored = second.calls()

    assert len(stored) == 1
    assert stored[0].tier == TIER_MID
    assert stored[0].model == "mid-x"
    assert stored[0].latency_ms == 12.5
    assert stored[0].prompt_tokens == 300
    assert stored[0].estimated_cost_usd == Decimal("0.000552")
    assert second.stats(PRICING, "large-x").total_calls == 1

    second.clear()
    assert MetricsStore(path).calls() == []
