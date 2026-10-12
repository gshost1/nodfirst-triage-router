"""Caregiving message classification with an urgent keyword safety net."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from core.llm import LLMClient
from router.config import RouterConfig, RouterMode

MODEL_CLASSIFIER_SOURCE = "model-classifier"
KEYWORD_CLASSIFIER_SOURCE = "keyword-classifier"
SAFETY_NET_SOURCE = "keyword-safety-net"


class MessageClass(str, Enum):
    ROUTINE = "routine"
    LOGISTICS = "logistics"
    URGENT = "urgent"


# The approved urgent keyword categories. The safety net overrides any
# lower-risk classifier result when one of these signals appears.
URGENT_KEYWORD_CATEGORIES: dict[str, tuple[str, ...]] = {
    "falls": (
        "fell",
        "fallen",
        "had a fall",
        "taken a fall",
        "took a fall",
        "has a fall",
        "fall down",
        "fell down",
        "falling down",
        "keeps falling",
        "keep falling",
    ),
    "chest pain": (
        "chest pain",
        "chest hurts",
        "chest is tight",
        "chest feels tight",
        "chest feels heavy",
        "chest pressure",
        "chest discomfort",
    ),
    "confusion": (
        "confused",
        "confusion",
        "disoriented",
        "not making sense",
        "isn't making sense",
        "doesn't recognize",
        "does not recognize",
        "doesn't know where",
        "does not know where",
    ),
    "missed dose": (
        "missed dose",
        "missed a dose",
        "missed her dose",
        "missed his dose",
        "missed medication",
        "missed her medication",
        "missed his medication",
        "skipped dose",
        "forgot her pill",
        "forgot his pill",
        "forgot her medication",
        "forgot his medication",
    ),
    "cannot reach": (
        "can't reach",
        "cannot reach",
        "couldn't reach",
        "no answer",
        "not picking up",
        "isn't answering",
        "is not answering",
        "unreachable",
        "no response",
        "straight to voicemail",
    ),
}

# Deterministic non-urgent keywords used by the canned mock classifier.
LOGISTICS_KEYWORDS: tuple[str, ...] = (
    "appointment",
    "book",
    "schedule",
    "reschedule",
    "cancel",
    "drive",
    "drives",
    "driving",
    "ride",
    "pick up",
    "pickup",
    "drop off",
    "refill",
    "pharmacy",
    "insurance",
    "paperwork",
    "calendar",
    "delivery",
    "order supplies",
)

CLASSIFIER_SYSTEM_PROMPT = (
    "You classify a short message an adult child sends about their elderly "
    "parent into exactly one of three classes.\n"
    "- routine: social updates, general questions, everyday care and well-being.\n"
    "- logistics: appointments, schedules, rides, sibling tasks, supplies, "
    "paperwork.\n"
    "- urgent: a possible health emergency or safety concern.\n"
    "Respond with exactly one word: routine, logistics, or urgent."
)

CLASSIFIER_MAX_COMPLETION_TOKENS = 16


@dataclass(frozen=True)
class ClassificationResult:
    message_class: MessageClass
    source: str
    safety_override: str | None = None

    @property
    def is_urgent(self) -> bool:
        return self.message_class is MessageClass.URGENT


def _normalize(message: str) -> str:
    """Lowercase, flatten whitespace, and normalize apostrophe variants."""
    text = message.lower().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(text.split())


def urgent_keyword_category(message: str) -> str | None:
    """Return the first approved urgent category signaled by the message."""
    normalized = _normalize(message)
    for category, phrases in URGENT_KEYWORD_CATEGORIES.items():
        if any(phrase in normalized for phrase in phrases):
            return category
    return None


def keyword_classify(message: str) -> MessageClass:
    """Deterministic non-urgent classification used for canned mock routing."""
    normalized = _normalize(message)
    if any(keyword in normalized for keyword in LOGISTICS_KEYWORDS):
        return MessageClass.LOGISTICS
    return MessageClass.ROUTINE


_CLASS_WORD_RE = re.compile(r"\b(routine|logistics|urgent)\b")


def parse_message_class(content: str) -> MessageClass | None:
    """Parse the classifier word from model output; None when unparseable."""
    match = _CLASS_WORD_RE.search(content.lower())
    return MessageClass(match.group(1)) if match else None


ModelClassifyFn = Callable[[str], MessageClass]


@dataclass(frozen=True)
class ClassifierCall:
    """One classifier inference call: the class plus the raw completion.

    The completion is retained so the API layer can record the reported model
    and token usage for this call without re-issuing it.
    """

    message_class: MessageClass
    completion: object


def model_classification_call(
    client: LLMClient, config: RouterConfig, message: str
) -> ClassifierCall:
    """Classify with the configured small-tier model and keep the completion.

    An unparseable response falls back to logistics: the mid class keeps the
    request on a stronger model without escalating every ambiguous message to
    the urgent path.
    """
    completion = client.complete(
        [
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        model=config.small_model,
        max_completion_tokens=CLASSIFIER_MAX_COMPLETION_TOKENS,
    )
    try:
        content = completion.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        content = ""
    parsed = parse_message_class(content)
    return ClassifierCall(
        MessageClass.LOGISTICS if parsed is None else parsed, completion
    )


def model_classify(
    client: LLMClient, config: RouterConfig, message: str
) -> MessageClass:
    """Classify with the configured small-tier model."""
    return model_classification_call(client, config, message).message_class


class TriageClassifier:
    """Classify a caregiving message and apply the urgent keyword safety net.

    The base classifier is the model classifier in client/DO modes and a
    deterministic keyword classifier in mock mode (so mock routing never makes
    an inference request). The safety net always wins: an approved urgent
    signal overrides whatever the base classifier said.
    """

    def __init__(
        self,
        client: LLMClient,
        config: RouterConfig,
        *,
        classify_with: ModelClassifyFn | None = None,
    ) -> None:
        self._client = client
        self._config = config
        self._classify_with = classify_with

    def classify(self, message: str) -> ClassificationResult:
        source, base = self._base_classify(message)
        category = urgent_keyword_category(message)
        if category is not None:
            return ClassificationResult(
                message_class=MessageClass.URGENT,
                source=SAFETY_NET_SOURCE,
                safety_override=category,
            )
        return ClassificationResult(message_class=base, source=source)

    def _base_classify(self, message: str) -> tuple[str, MessageClass]:
        if self._classify_with is not None:
            return MODEL_CLASSIFIER_SOURCE, self._classify_with(message)
        if self._config.mode is RouterMode.MOCK or self._client.is_mock:
            return KEYWORD_CLASSIFIER_SOURCE, keyword_classify(message)
        return (
            MODEL_CLASSIFIER_SOURCE,
            model_classify(self._client, self._config, message),
        )
