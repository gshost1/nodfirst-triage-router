"""Caregiving message classification and routing.

Composition for the app layer:

1. `TriageClassifier.classify(message)` -> `ClassificationResult` (the urgent
   keyword safety net always overrides a lower-risk class).
2. `dispatch_route(config, classification)` -> `RouteDecision` (urgent always
   dispatches directly to the configured large model; the DO router is only
   used for routine/logistics in DO mode).
3. `RouteExecutor.execute(decision, messages)` -> `CompletionOutcome` (carries
   the provider-reported serving model and router selected route when
   available).

Urgent replies must be surfaced with `RouteDecision.escalation_note`: the
non-diagnostic 911/doctor guidance.
"""

from router.classify import (
    URGENT_KEYWORD_CATEGORIES,
    ClassificationResult,
    MessageClass,
    TriageClassifier,
    keyword_classify,
    model_classify,
    urgent_keyword_category,
)
from router.config import (
    DEFAULT_DO_ROUTER_MODEL,
    RouterConfig,
    RouterConfigError,
    RouterMode,
)
from router.dispatch import (
    URGENT_ESCALATION_NOTE,
    RouteDecision,
    dispatch_route,
)
from router.do_adapter import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    ROUTER_SELECTED_ROUTE_HEADER,
    DORouterAdapter,
    RouterDispatch,
)
from router.execute import CompletionOutcome, RouteExecutor

__all__ = [
    "DEFAULT_DO_ROUTER_MODEL",
    "DEFAULT_MAX_COMPLETION_TOKENS",
    "ROUTER_SELECTED_ROUTE_HEADER",
    "URGENT_ESCALATION_NOTE",
    "URGENT_KEYWORD_CATEGORIES",
    "ClassificationResult",
    "CompletionOutcome",
    "DORouterAdapter",
    "MessageClass",
    "RouteDecision",
    "RouteExecutor",
    "RouterConfig",
    "RouterConfigError",
    "RouterDispatch",
    "RouterMode",
    "TriageClassifier",
    "dispatch_route",
    "keyword_classify",
    "model_classify",
    "urgent_keyword_category",
]
