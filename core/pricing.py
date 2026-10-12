"""Published model rates used for estimated inference costs.

The table below holds the DigitalOcean Serverless Inference rates for the
configured tier model IDs recorded in `MODEL_SELECTION.md`. Values are
estimates for local accounting, not a bill, and any model without an entry is
reported as unpriced instead of being guessed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

TOKENS_PER_MILLION = Decimal(1_000_000)
COST_QUANTUM = Decimal("0.00000001")
PRICE_SOURCE = "https://docs.digitalocean.com/products/inference/details/pricing/"
PRICE_SOURCE_DATE = "2026-10-12"
PRICE_SOURCE_NOTE = "published DigitalOcean inference rates (page updated 2026-10-09)"


@dataclass(frozen=True)
class ModelPrice:
    """USD per one million input and output tokens."""

    input_per_million: Decimal
    output_per_million: Decimal


DEFAULT_PRICES: Mapping[str, ModelPrice] = {
    # MODEL_SMALL: lowest-priced preferred candidate for classification.
    "glm-5.3-flash": ModelPrice(Decimal("0.15"), Decimal("0.50")),
    # MODEL_MID: general-purpose candidate, priced below the large tier.
    "glm-5.3": ModelPrice(Decimal("1.40"), Decimal("4.40")),
    # MODEL_LARGE: high-capability candidate for direct urgent replies.
    "qwen3.8-max": ModelPrice(Decimal("2.00"), Decimal("6.00")),
}


def quantize_cost(value: Decimal) -> Decimal:
    """Round a USD amount to eight decimal places with deterministic rounding."""
    return value.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class PricingTable:
    """Model-ID to published-rate lookups with honest handling of unknowns."""

    prices: Mapping[str, ModelPrice] = field(
        default_factory=lambda: dict(DEFAULT_PRICES)
    )

    def price_for(self, model: str | None) -> ModelPrice | None:
        if not model:
            return None
        return self.prices.get(model)

    def estimate_cost(
        self,
        model: str | None,
        prompt_tokens: int | None,
        completion_tokens: int | None,
    ) -> Decimal | None:
        """Estimate USD for one call, or None when usage or rate is unavailable."""
        price = self.price_for(model)
        if price is None or prompt_tokens is None or completion_tokens is None:
            return None
        cost = (
            Decimal(prompt_tokens) * price.input_per_million
            + Decimal(completion_tokens) * price.output_per_million
        ) / TOKENS_PER_MILLION
        return quantize_cost(cost)
