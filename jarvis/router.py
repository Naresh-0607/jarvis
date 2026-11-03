"""Rule-based routing for the Jarvis assistant.

Phase 1 only needs two tiers and plain rules - the model does not decide
anything about routing yet:

* ``fast``   - small talk, reminders, simple lookups.
* ``strong`` - reasoning, teaching, debugging, planning.

Anything ambiguous stays on ``fast`` so the cheap model is the default.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

_logger = logging.getLogger("jarvis.router")

FAST = "fast"
STRONG = "strong"

# Phrases that reliably mean "think harder about this".
_STRONG_PATTERNS = (
    r"\bwhy\b",
    r"\bhow (does|do|is|are)\b",
    r"\bexplain\b",
    r"\bcompare\b",
    r"\bprove\b",
    r"\bderive\b",
    r"\bdebug\b",
    r"\btraceback\b",
    r"\bstack ?trace\b",
    r"\berror\b",
    r"\bexception\b",
    r"\brefactor\b",
    r"\breview\b",
    r"\bdesign\b",
    r"\barchitect",
    r"\bplan\b",
    r"\bteach\b",
    r"\blearn\b",
    r"\bunderstand\b",
    r"\bstep by step\b",
    r"\btrade-?offs?\b",
    r"\boptimi[sz]e\b",
    r"\bwalk me through\b",
    r"\bwhat does .* mean\b",
)

_STRONG_RE = re.compile("|".join(_STRONG_PATTERNS), re.IGNORECASE)

# Short conversational turns never need the strong model.
_SMALL_TALK_RE = re.compile(
    r"^\s*(hi|hey|hello|thanks|thank you|good (morning|evening|night)|"
    r"how are you|what'?s up|bye|goodbye|ok(ay)?)\b",
    re.IGNORECASE,
)


# Explicit patterns for fast simple lookups (weather, time, math, reminders)
_FAST_LOOKUP_PATTERNS = (
    r"\bweather\b",
    r"\btemperature\b",
    r"\bforecast\b",
    r"\brain(ing)?\b",
    r"\bwhat time\b",
    r"\bcurrent time\b",
    r"\bwhat (date|day) is (it|today)\b",
    r"\btoday'?s date\b",
    r"\bcalculate\b",
    r"\bmath\b",
    r"\bremind(er)?\b",
    r"\bpending\b",
    r"\bwhat did i miss\b",
)
_FAST_LOOKUP_RE = re.compile("|".join(_FAST_LOOKUP_PATTERNS), re.IGNORECASE)


def choose_tier(text: str) -> str:
    """Pick a tier for one user message using rules only and log the decision."""
    message = (text or "").strip()
    if not message:
        return FAST

    if _SMALL_TALK_RE.match(message):
        _logger.debug("Routed to fast tier: small talk")
        return FAST

    # Simple lookups (weather, time, math, reminders) stay fast
    if _FAST_LOOKUP_RE.search(message) and not _STRONG_RE.search(message):
        _logger.debug("Routed to fast tier: simple lookup")
        return FAST

    # Long requests typically contain code or complex instructions
    if len(message) > 280:
        _logger.info("Routed to strong tier: message length (%d chars)", len(message))
        return STRONG

    # High-reasoning and teaching keywords
    match = _STRONG_RE.search(message)
    if match:
        _logger.info("Routed to strong tier: matched reasoning pattern '%s'", match.group(0))
        return STRONG

    _logger.debug("Routed to fast tier: default rule")
    return FAST


def models_for_tier(
    all_models: List[str], tiers: Optional[dict], tier: str
) -> List[str]:
    """Order a provider's models so the requested tier comes first."""
    models = [m for m in (all_models or []) if m]
    if not tiers:
        return models
    preferred = [m for m in tiers.get(tier, []) if m in models]
    return preferred + [m for m in models if m not in preferred]


def route_to_provider(model_name: str) -> str:
    """Legacy helper: guess which tier a model name belongs to."""
    name = (model_name or "").lower()
    if any(token in name for token in ("mini", "small", "8b", "flash", "instant")):
        return FAST
    if any(token in name for token in ("70b", "120b", "large", "sonnet", "pro")):
        return STRONG
    _logger.debug("Unknown model '%s' - defaulting to fast route", model_name)
    return FAST


def get_model_for_provider(provider_name: str, all_models: list) -> Optional[str]:
    """Return the first usable model for a provider."""
    if not all_models:
        return None
    _logger.debug("Provider %s has models: %s", provider_name, all_models)
    return all_models[0]


def get_provider_tier(provider_name: str, model_name: str) -> str:
    """Tier implied by a model name."""
    return route_to_provider(model_name)
