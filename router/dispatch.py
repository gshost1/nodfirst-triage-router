"""Route decisions for the mock, client, and DO routing modes."""

from __future__ import annotations

from dataclasses import dataclass

from core.llm import MOCK_MODEL
from router.classify import ClassificationResult, MessageClass
from router.config import RouterConfig, RouterMode

URGENT_ESCALATION_NOTE = (
    "This sounds potentially urgent. If this could be an emergency, please "
    "call 911 now. Otherwise, call Mom's doctor right away — don't wait for "
    "a reply from me, and please don't rely on me to judge how serious it is."
)


@dataclass(frozen=True)
class RouteDecision:
    """The dispatch target chosen for one message.

    `model` is the model ID to request; `route` is the dispatch path label;
    `override_reason` records a keyword-safety-net override; and
    `escalation_note` is the required non-diagnostic 911/doctor guidance the
    app must surface with urgent replies.
    """

    message_class: MessageClass
    mode: RouterMode
    route: str
    model: str
    reason: str
    override_reason: str | None = None
    escalation_note: str | None = None

    @property
    def is_urgent(self) -> bool:
        return self.message_class is MessageClass.URGENT


def dispatch_route(
    config: RouterConfig,
    classification: ClassificationResult,
    *,
    fallback_model: str = MOCK_MODEL,
) -> RouteDecision:
    """Choose the dispatch target for a classification under the active mode.

    Urgent messages bypass every router in every mode and go directly to the
    configured large model. `fallback_model` is only used in mock mode when
    the tier environment variables are not configured.
    """
    override_reason = (
        f"urgent keyword safety net: {classification.safety_override}"
        if classification.safety_override is not None
        else None
    )
    if config.mode is RouterMode.MOCK:
        return _mock_decision(config, classification, fallback_model, override_reason)
    if config.mode is RouterMode.CLIENT:
        return _client_decision(config, classification, override_reason)
    return _do_decision(config, classification, override_reason)


def _mock_decision(
    config: RouterConfig,
    classification: ClassificationResult,
    fallback_model: str,
    override_reason: str | None,
) -> RouteDecision:
    """Canned deterministic routing metadata without external inference."""
    if classification.is_urgent:
        return RouteDecision(
            message_class=MessageClass.URGENT,
            mode=RouterMode.MOCK,
            route="mock:direct-large",
            model=config.large_model or fallback_model,
            reason=(
                "ROUTER_MODE=mock: urgent messages bypass routing and target "
                "the configured large tier (canned metadata)."
            ),
            override_reason=override_reason,
            escalation_note=URGENT_ESCALATION_NOTE,
        )
    tier_model = (
        config.small_model
        if classification.message_class is MessageClass.ROUTINE
        else config.mid_model
    )
    return RouteDecision(
        message_class=classification.message_class,
        mode=RouterMode.MOCK,
        route="mock",
        model=tier_model or fallback_model,
        reason=(
            f"ROUTER_MODE=mock: canned route metadata for class "
            f"'{classification.message_class.value}'."
        ),
    )


def _client_decision(
    config: RouterConfig,
    classification: ClassificationResult,
    override_reason: str | None,
) -> RouteDecision:
    """Client-side routing: each class maps to a configured tier model ID."""
    if classification.is_urgent:
        return RouteDecision(
            message_class=MessageClass.URGENT,
            mode=RouterMode.CLIENT,
            route="client:direct-large",
            model=config.large_model,
            reason=(
                "ROUTER_MODE=client: urgent messages bypass routing and "
                "dispatch to the configured large model."
            ),
            override_reason=override_reason,
            escalation_note=URGENT_ESCALATION_NOTE,
        )
    if classification.message_class is MessageClass.ROUTINE:
        route = "client:small"
        model = config.small_model
        reason = (
            "ROUTER_MODE=client: routine messages dispatch to the "
            "configured small tier."
        )
    else:
        route = "client:mid"
        model = config.mid_model
        reason = (
            "ROUTER_MODE=client: logistics messages dispatch to the "
            "configured mid tier."
        )
    return RouteDecision(
        message_class=classification.message_class,
        mode=RouterMode.CLIENT,
        route=route,
        model=model,
        reason=reason,
        override_reason=override_reason,
    )


def _do_decision(
    config: RouterConfig,
    classification: ClassificationResult,
    override_reason: str | None,
) -> RouteDecision:
    """DO routing: routine/logistics go through the inference router."""
    if classification.is_urgent:
        return RouteDecision(
            message_class=MessageClass.URGENT,
            mode=RouterMode.DO,
            route="do:direct-large",
            model=config.large_model,
            reason=(
                "ROUTER_MODE=do: urgent messages bypass the inference "
                "router and dispatch directly to the configured large model."
            ),
            override_reason=override_reason,
            escalation_note=URGENT_ESCALATION_NOTE,
        )
    return RouteDecision(
        message_class=classification.message_class,
        mode=RouterMode.DO,
        route="do:inference-router",
        model=config.do_router_model,
        reason=(
            f"ROUTER_MODE=do: class '{classification.message_class.value}' "
            f"dispatches through the DigitalOcean inference router "
            f"({config.do_router_model}); the router chooses the serving "
            "model itself."
        ),
        override_reason=override_reason,
    )
