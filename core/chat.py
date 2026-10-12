"""One chat turn: classify, route, run the caregiver agent, record metrics.

This service is the single place the API surface calls. It keeps routing,
agent tool execution, and per-inference accounting together so `POST /chat`
and `GET /stats` cannot drift apart.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from core.agent import AgentResult, CaregiverAgent
from core.llm import ProviderError
from core.metrics import (
    COST_LABEL,
    TIER_LARGE,
    TIER_MID,
    TIER_ROUTER,
    TIER_SMALL,
    InferenceCall,
    StatsReport,
    TokenUsage,
    UsageSummary,
    aggregate_calls,
    as_usd,
    build_call,
    completion_text,
    message_text,
    summarize_usage,
)
from core.metrics_store import MetricsStore
from core.mock_agent import mock_agent_completion
from core.notify import BotmailNotifier, NullNotifier
from core.pricing import PricingTable
from core.speech import DOSpeech, NullSpeech, SpeechClip
from core.store import CareStore
from core.tools import ToolAction
from core.transcribe import NullTranscriber, Transcript, WhisperCLI
from router.classify import (
    MessageClass,
    TriageClassifier,
    keyword_classify,
    model_classification_call,
)
from router.config import RouterConfig, RouterMode
from router.dispatch import RouteDecision, dispatch_route
from router.do_adapter import DORouterAdapter
from router.execute import CompletionOutcome, RouteExecutor

NOT_MEDICAL_ADVICE = (
    "Not medical advice. If this could be an emergency, call 911 now; "
    "otherwise contact Mom's doctor."
)
AGENT_MAX_COMPLETION_TOKENS = 512
HISTORY_TURNS = 8  # prior thread messages (user + assistant) sent to the model
DEFAULT_SENDER = "You"


def tier_for_decision(decision: RouteDecision) -> str:
    """Label the tier a decision targets, for per-inference stats."""
    if decision.is_urgent:
        return TIER_LARGE
    if decision.mode is RouterMode.DO:
        return TIER_ROUTER
    if decision.message_class is MessageClass.ROUTINE:
        return TIER_SMALL
    return TIER_MID


class _TurnRecorder:
    """Collect one turn's inference calls and persist each immediately."""

    def __init__(
        self, metrics: MetricsStore, pricing: PricingTable, clock: Callable[[], float]
    ) -> None:
        self.metrics = metrics
        self.pricing = pricing
        self.clock = clock
        self.calls: list[InferenceCall] = []
        self.selected_route: str | None = None

    def add(self, call: InferenceCall) -> InferenceCall:
        self.calls.append(call)
        self.metrics.record(call)
        return call


@dataclass(frozen=True)
class ChatTurn:
    """The result of one `POST /chat` turn, ready for JSON serialization."""

    reply: str
    decision: RouteDecision
    tool_actions: list[ToolAction]
    calls: list[InferenceCall]
    usage: UsageSummary
    latency_ms: float
    estimated_cost_usd: Decimal | None
    model: str | None
    response_model: str | None
    selected_route: str | None
    sender: str = DEFAULT_SENDER
    alert: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "reply": self.reply,
            "sender": self.sender,
            "alert": self.alert,
            "message_class": self.decision.message_class.value,
            "urgent": self.decision.is_urgent,
            "mode": self.decision.mode.value,
            "route": self.decision.route,
            "model": self.model,
            "response_model": self.response_model,
            "selected_route": self.selected_route,
            "reason": self.decision.reason,
            "override_reason": self.decision.override_reason,
            "escalation_note": self.decision.escalation_note,
            "latency_ms": round(self.latency_ms, 3),
            "usage": self.usage.as_dict(),
            "estimated_cost_usd": as_usd(self.estimated_cost_usd),
            "cost_label": COST_LABEL,
            "calls": [call.as_dict() for call in self.calls],
            "tool_actions": [asdict(action) for action in self.tool_actions],
            "disclaimer": NOT_MEDICAL_ADVICE,
        }


logger = logging.getLogger(__name__)

OVERLOADED_REPLY = (
    "The assistant could not reach its model just now (the provider is "
    "overloaded). Nothing was recorded; please send that again in a moment."
)


def overloaded_reply(escalation_note: str | None) -> str:
    if escalation_note:
        return f"{OVERLOADED_REPLY}\n\n{escalation_note}"
    return OVERLOADED_REPLY


class ChatService:
    """Coordinate classification, dispatch, the agent, and metrics.

    MOCK mode runs entirely offline through the agent-layer double, so the
    demo and tests never touch a provider. Every other mode dispatches through
    `RouteExecutor`, which keeps DO-router traffic inside the adapter.
    """

    def __init__(
        self,
        *,
        client: Any,
        config: RouterConfig,
        store: CareStore,
        metrics: MetricsStore,
        pricing: PricingTable | None = None,
        clock: Callable[[], float] = time.perf_counter,
        mock_completion: Callable[..., Any] | None = None,
        notifier: BotmailNotifier | NullNotifier | None = None,
        speech: DOSpeech | NullSpeech | None = None,
        transcriber: WhisperCLI | NullTranscriber | None = None,
    ) -> None:
        self.notifier = notifier or NullNotifier()
        self.speech = speech or NullSpeech()
        self.transcriber = transcriber or NullTranscriber("speech-to-text off")
        self.client = client
        self.config = config
        self.store = store
        self.metrics = metrics
        self.pricing = pricing or PricingTable()
        self.clock = clock
        self.mock_completion = mock_completion or mock_agent_completion
        self.executor = RouteExecutor(
            client, DORouterAdapter.from_client(client, config.do_router_model)
        )

    @property
    def is_mock(self) -> bool:
        return self.config.mode is RouterMode.MOCK

    def handle(
        self, message: str, *, sender: str | None = None, voice: bool = False
    ) -> ChatTurn:
        """Run one turn in the shared sibling thread.

        The last `HISTORY_TURNS` stored messages are sent as context, the new
        user message and the reply are persisted, and an urgent decision also
        records an alert for the sibling group.
        """
        sender = (sender or "").strip() or DEFAULT_SENDER
        history = self.store.recent_messages(HISTORY_TURNS)
        recorder = _TurnRecorder(self.metrics, self.pricing, self.clock)
        classifier = TriageClassifier(
            self.client,
            self.config,
            classify_with=(
                None
                if self.is_mock
                else lambda text: self._classify_with_metrics(recorder, text)
            ),
        )
        classification = classifier.classify(message)
        decision = dispatch_route(self.config, classification)
        agent = CaregiverAgent(self.store, self._completion(recorder, decision))
        try:
            result = agent.run(
                message, urgent=decision.is_urgent, sender=sender, history=history,
                voice=voice,
            )
        except ProviderError as error:
            logger.warning("agent completion failed (%s); honest fallback", error)
            result = AgentResult(
                overloaded_reply(decision.escalation_note), [], []
            )

        message_class = decision.message_class.value
        self.store.add_message(sender, "user", message, message_class)
        self.store.add_message("assistant", "assistant", result.reply, message_class)
        alert = None
        if decision.is_urgent:
            alert = self.store.add_alert(
                sender, message, decision.escalation_note or ""
            )
            # Best effort: a failed or disabled email is recorded, never raised.
            sent = self.notifier.send_alert(alert)
            self.store.record_alert_email(
                alert["id"], sent.status, sent.detail, sent.recipients
            )
            alert.update(
                email_status=sent.status, email_detail=sent.detail,
                email_recipients=list(sent.recipients),
            )

        calls = recorder.calls
        last = calls[-1] if calls else None
        costs = [call.estimated_cost_usd for call in calls
                 if call.estimated_cost_usd is not None]
        return ChatTurn(
            reply=result.reply,
            decision=decision,
            tool_actions=list(result.tool_actions),
            calls=list(calls),
            usage=summarize_usage(calls),
            latency_ms=sum(call.latency_ms for call in calls),
            estimated_cost_usd=sum(costs, Decimal(0)) if costs else None,
            model=last.model if last else decision.model,
            response_model=last.response_model if last else None,
            selected_route=recorder.selected_route,
            sender=sender,
            alert=alert,
        )

    def thread(self, limit: int = 50) -> dict[str, Any]:
        """The shared thread plus the sibling roster, for page load."""
        return {
            "siblings": self.store.siblings(),
            "messages": self.store.recent_messages(limit),
            "mode": self.config.mode.value,
            "tiers": {
                "routine": self.config.small_model or "mock",
                "logistics": self.config.mid_model or "mock",
                "urgent": self.config.large_model or "mock",
                "router": self.config.do_router_model,
            },
        }

    def alerts(self) -> list[dict[str, Any]]:
        return self.store.alerts()

    def digest(self) -> dict[str, Any]:
        """The store-grounded daily digest, with no model call."""
        return {**self.store.daily_digest(), "speech": self.speech.status()}

    def speak_digest(self) -> SpeechClip:
        """Read the digest aloud through text-to-speech (raises SpeechError)."""
        return self.speech.synthesize(self.store.daily_digest()["summary"])

    def transcribe(self, audio: bytes, media_type: str) -> Transcript:
        return self.transcriber.transcribe(audio, media_type)

    def speak(self, text: str) -> SpeechClip:
        """Read any reply aloud (raises SpeechError)."""
        return self.speech.synthesize(text)

    def email_digest(self) -> dict[str, Any]:
        """Email the digest to the alert recipients through the notifier."""
        digest = self.store.daily_digest()
        sent = self.notifier.send_digest(digest)
        return {"digest": digest, "email": sent.as_dict()}

    def alert_delivery(self) -> dict[str, Any]:
        """How urgent alerts leave the app, for the UI panel."""
        if isinstance(self.notifier, BotmailNotifier):
            return {"enabled": True, "provider": "botmail",
                    "recipients": list(self.notifier.recipients)}
        return {"enabled": False, "provider": None,
                "recipients": [], "reason": self.notifier.reason}

    def acknowledge_alert(self, alert_id: int) -> dict[str, Any] | None:
        return self.store.acknowledge_alert(alert_id)

    def stats(self) -> StatsReport:
        return aggregate_calls(
            self.metrics.calls(), self.pricing, self.config.large_model or None
        )

    # -- inference plumbing -------------------------------------------------

    def _classify_with_metrics(
        self, recorder: _TurnRecorder, message: str
    ) -> MessageClass:
        start = self.clock()
        try:
            call = model_classification_call(self.client, self.config, message)
        except ProviderError as error:
            # Live 2026-10-12: a 529 "Platform overloaded" here turned the whole
            # turn into a 500. The keyword classifier keeps the turn alive and
            # the urgent safety net still applies on top of it.
            logger.warning("classifier unavailable (%s); using keywords", error)
            return keyword_classify(message)
        latency_ms = (self.clock() - start) * 1000.0
        recorder.add(
            build_call(
                tier=TIER_SMALL,
                route="classifier",
                requested_model=self.config.small_model,
                response_model=_reported_model(call.completion),
                latency_ms=latency_ms,
                usage=TokenUsage.from_completion(call.completion),
                pricing=self.pricing,
            )
        )
        return call.message_class

    def _completion(
        self, recorder: _TurnRecorder, decision: RouteDecision
    ) -> Callable[..., Any]:
        tier = tier_for_decision(decision)

        if self.is_mock:
            def complete(
                *,
                messages: Sequence[Any],
                max_completion_tokens: int = AGENT_MAX_COMPLETION_TOKENS,
                **_kwargs: Any,
            ) -> CompletionOutcome:
                start = self.clock()
                completion = self.mock_completion(
                    messages=messages, model=decision.model
                )
                latency_ms = (self.clock() - start) * 1000.0
                recorder.add(
                    build_call(
                        tier=tier,
                        route=decision.route,
                        requested_model=decision.model,
                        response_model=None,
                        latency_ms=latency_ms,
                        usage=TokenUsage.estimate(
                            message_text(messages), completion_text(completion)
                        ),
                        pricing=self.pricing,
                    )
                )
                return CompletionOutcome(
                    decision=decision,
                    completion=completion,
                    response_model=None,
                    selected_route=None,
                )

            return complete

        def routed(
            *,
            messages: Sequence[Any],
            tools: Sequence[Any] | None = None,
            tool_choice: Any = None,
            max_completion_tokens: int = AGENT_MAX_COMPLETION_TOKENS,
        ) -> CompletionOutcome:
            start = self.clock()
            outcome = self.executor.execute(
                decision,
                messages,
                tools=tools,
                tool_choice=tool_choice,
                max_completion_tokens=max_completion_tokens,
            )
            latency_ms = (self.clock() - start) * 1000.0
            recorder.add(
                build_call(
                    tier=tier,
                    route=decision.route,
                    requested_model=decision.model,
                    response_model=outcome.response_model,
                    latency_ms=latency_ms,
                    usage=TokenUsage.from_completion(outcome.completion),
                    pricing=self.pricing,
                )
            )
            if outcome.selected_route is not None:
                recorder.selected_route = outcome.selected_route
            return outcome

        return routed


def _reported_model(completion: Any) -> str | None:
    model = getattr(completion, "model", None)
    return model if isinstance(model, str) else None
