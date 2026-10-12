"""Per-inference-call metrics, honest usage handling, and aggregate stats.

Every classifier, agent, and tool follow-up completion is one inference call.
Usage and cost stay `None`/`unavailable` whenever the provider omits them; the
only estimated token counts come from the label-free MOCK path, and they are
labelled `estimated` so an offline rehearsal is never reported as a real bill.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from core.pricing import (
    COST_QUANTUM,
    PRICE_SOURCE,
    PRICE_SOURCE_DATE,
    PRICE_SOURCE_NOTE,
    PricingTable,
)

USAGE_REPORTED = "reported"
USAGE_ESTIMATED = "estimated"
USAGE_MIXED = "mixed"
USAGE_PARTIAL = "partial"
USAGE_UNAVAILABLE = "unavailable"

TIER_SMALL = "small"
TIER_MID = "mid"
TIER_LARGE = "large"
TIER_ROUTER = "router"

COST_LABEL = "estimated"
CHARS_PER_TOKEN = 4
LATENCY_DECIMALS = 3


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def estimate_tokens(text: str) -> int:
    """Estimate tokens from text length for the offline MOCK path only.

    The estimate is a deterministic four-characters-per-token approximation so
    local rehearsals have reproducible numbers; it is always labelled
    `estimated` and never presented as provider-reported usage.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, math.ceil(len(stripped) / CHARS_PER_TOKEN))


def message_text(messages: Sequence[Any]) -> str:
    """Flatten chat messages into the text used for a mock token estimate."""
    parts = []
    for message in messages:
        if isinstance(message, dict):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)
        if isinstance(content, str):
            parts.append(content)
    return "\n".join(parts)


def completion_text(completion: Any) -> str:
    """Read the assistant text from a completion without fabricating content."""
    try:
        content = completion.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        return ""
    return content if isinstance(content, str) else ""


@dataclass(frozen=True)
class TokenUsage:
    """Token counts for one inference call plus how they were obtained."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    source: str = USAGE_UNAVAILABLE

    @property
    def available(self) -> bool:
        return self.prompt_tokens is not None and self.completion_tokens is not None

    @classmethod
    def from_completion(cls, completion: Any) -> TokenUsage:
        """Keep provider-reported usage; stay unavailable when the provider omits it."""
        raw = getattr(completion, "usage", None)
        if raw is None:
            return cls()
        prompt = _non_negative_int(getattr(raw, "prompt_tokens", None))
        completion_tokens = _non_negative_int(getattr(raw, "completion_tokens", None))
        total = _non_negative_int(getattr(raw, "total_tokens", None))
        if prompt is None or completion_tokens is None:
            return cls()
        return cls(
            prompt,
            completion_tokens,
            total if total is not None else prompt + completion_tokens,
            USAGE_REPORTED,
        )

    @classmethod
    def estimate(cls, prompt_text: str, completion_content: str) -> TokenUsage:
        """Deterministic offline estimate used only for MOCK completions."""
        prompt = estimate_tokens(prompt_text)
        completion = estimate_tokens(completion_content)
        return cls(prompt, completion, prompt + completion, USAGE_ESTIMATED)


@dataclass(frozen=True)
class InferenceCall:
    """One recorded inference invocation."""

    tier: str
    route: str
    model: str | None
    response_model: str | None
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    usage_source: str = USAGE_UNAVAILABLE
    estimated_cost_usd: Decimal | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "route": self.route,
            "model": self.model,
            "response_model": self.response_model,
            "latency_ms": round(self.latency_ms, LATENCY_DECIMALS),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage_source": self.usage_source,
            "estimated_cost_usd": as_usd(self.estimated_cost_usd),
            "cost_label": COST_LABEL,
        }


def as_usd(value: Decimal | None) -> float | None:
    """Render an estimated USD amount as a JSON-friendly float."""
    return None if value is None else float(value.quantize(COST_QUANTUM))


def build_call(
    *,
    tier: str,
    route: str,
    requested_model: str | None,
    response_model: str | None,
    latency_ms: float,
    usage: TokenUsage,
    pricing: PricingTable,
) -> InferenceCall:
    """Assemble one call record, pricing the actual serving model when known."""
    model = response_model or requested_model
    cost = (
        pricing.estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        if usage.available
        else None
    )
    return InferenceCall(
        tier=tier,
        route=route,
        model=model,
        response_model=response_model,
        latency_ms=latency_ms,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        usage_source=usage.source,
        estimated_cost_usd=cost,
    )


@dataclass(frozen=True)
class UsageSummary:
    """Token totals for one chat turn, with an honest source label."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "source": self.source,
        }


def summarize_usage(calls: Iterable[InferenceCall]) -> UsageSummary:
    """Sum the available token counts and label how complete they are."""
    records = list(calls)
    with_usage = [call for call in records if call.prompt_tokens is not None]
    if not records or not with_usage:
        return UsageSummary(None, None, None, USAGE_UNAVAILABLE)
    prompt = sum(call.prompt_tokens or 0 for call in with_usage)
    completion = sum(call.completion_tokens or 0 for call in with_usage)
    total = sum(call.total_tokens or 0 for call in with_usage)
    if len(with_usage) != len(records):
        source = USAGE_PARTIAL
    else:
        sources = {call.usage_source for call in records}
        source = sources.pop() if len(sources) == 1 else USAGE_MIXED
    return UsageSummary(prompt, completion, total, source)


@dataclass(frozen=True)
class StatsReport:
    """Aggregate per-inference-call statistics with labelled estimates."""

    total_calls: int
    calls_by_tier: dict[str, int] = field(default_factory=dict)
    calls_by_model: dict[str, int] = field(default_factory=dict)
    calls_by_route: dict[str, int] = field(default_factory=dict)
    calls_by_usage_source: dict[str, int] = field(default_factory=dict)
    calls_without_usage: int = 0
    calls_with_unpriced_model: int = 0
    priced_calls: int = 0
    average_latency_ms: float | None = None
    average_latency_ms_by_tier: dict[str, float] = field(default_factory=dict)
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None
    total_tokens: int | None = None
    estimated_spend_usd: Decimal = Decimal(0)
    large_model: str | None = None
    large_model_baseline_usd: Decimal | None = None
    estimated_savings_usd: Decimal | None = None
    savings_calls: int = 0
    savings_available: bool = False
    cost_label: str = COST_LABEL
    price_source: str = PRICE_SOURCE
    price_source_date: str = PRICE_SOURCE_DATE
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "calls_by_tier": dict(self.calls_by_tier),
            "calls_by_model": dict(self.calls_by_model),
            "calls_by_route": dict(self.calls_by_route),
            "calls_by_usage_source": dict(self.calls_by_usage_source),
            "calls_without_usage": self.calls_without_usage,
            "calls_with_unpriced_model": self.calls_with_unpriced_model,
            "priced_calls": self.priced_calls,
            "average_latency_ms": self.average_latency_ms,
            "average_latency_ms_by_tier": dict(self.average_latency_ms_by_tier),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_spend_usd": as_usd(self.estimated_spend_usd),
            "large_model": self.large_model,
            "large_model_baseline_usd": as_usd(self.large_model_baseline_usd),
            "estimated_savings_usd": as_usd(self.estimated_savings_usd),
            "savings_calls": self.savings_calls,
            "savings_available": self.savings_available,
            "currency": "USD",
            "cost_label": self.cost_label,
            "price_source": self.price_source,
            "price_source_date": self.price_source_date,
            "price_source_note": PRICE_SOURCE_NOTE,
            "notes": list(self.notes),
        }


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    counted: dict[str, int] = {}
    for value in values:
        key = value if value is not None else "unknown"
        counted[key] = counted.get(key, 0) + 1
    return counted


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), LATENCY_DECIMALS)


def aggregate_calls(
    calls: Iterable[InferenceCall],
    pricing: PricingTable | None = None,
    large_model: str | None = None,
) -> StatsReport:
    """Aggregate recorded calls into the reported stats.

    Savings compare the same token usage priced at the configured large-model
    rates against the recorded estimated spend. Only calls with both known
    usage and a known rate for both models participate, so an incomplete or
    unpriced call never inflates the reported saving.
    """
    table = pricing or PricingTable()
    records = list(calls)
    latencies = [call.latency_ms for call in records]
    usage_summary = summarize_usage(records)

    spend = Decimal(0)
    baseline = Decimal(0)
    savings_spend = Decimal(0)
    without_usage = 0
    unpriced = 0
    priced = 0
    savings_calls = 0
    large_price_known = bool(large_model) and table.price_for(large_model) is not None

    for call in records:
        if call.estimated_cost_usd is None:
            if not _has_usage(call):
                without_usage += 1
            else:
                unpriced += 1
            continue
        priced += 1
        spend += call.estimated_cost_usd
        baseline_cost = (
            table.estimate_cost(
                large_model, call.prompt_tokens, call.completion_tokens
            )
            if large_price_known
            else None
        )
        if baseline_cost is not None:
            baseline += baseline_cost
            savings_spend += call.estimated_cost_usd
            savings_calls += 1

    by_tier: dict[str, list[float]] = {}
    for call in records:
        by_tier.setdefault(call.tier, []).append(call.latency_ms)
    per_tier_average = {
        tier: round(sum(values) / len(values), LATENCY_DECIMALS)
        for tier, values in by_tier.items()
    }

    notes: list[str] = []
    if without_usage:
        notes.append(
            f"{without_usage} call(s) had no provider-reported usage and are "
            "excluded from cost and savings estimates."
        )
    if unpriced:
        notes.append(
            f"{unpriced} call(s) used a model with no configured published rate; "
            "their cost is not estimated."
        )
    if records and not large_price_known:
        notes.append(
            "The configured large model has no published rate entry, so "
            "large-model savings cannot be estimated."
        )
    if not records:
        notes.append("No inference calls recorded yet.")

    return StatsReport(
        total_calls=len(records),
        calls_by_tier=_counts(call.tier for call in records),
        calls_by_model=_counts(call.model for call in records),
        calls_by_route=_counts(call.route for call in records),
        calls_by_usage_source=_counts(call.usage_source for call in records),
        calls_without_usage=without_usage,
        calls_with_unpriced_model=unpriced,
        priced_calls=priced,
        average_latency_ms=_average(latencies),
        average_latency_ms_by_tier=per_tier_average,
        total_prompt_tokens=usage_summary.prompt_tokens,
        total_completion_tokens=usage_summary.completion_tokens,
        total_tokens=usage_summary.total_tokens,
        estimated_spend_usd=spend,
        large_model=large_model,
        large_model_baseline_usd=baseline if large_price_known else None,
        estimated_savings_usd=(
            baseline - savings_spend if large_price_known else None
        ),
        savings_calls=savings_calls,
        savings_available=large_price_known,
        notes=notes,
    )


def _has_usage(call: InferenceCall) -> bool:
    return call.prompt_tokens is not None and call.completion_tokens is not None
