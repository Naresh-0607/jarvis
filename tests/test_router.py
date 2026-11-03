"""Tests for the rule-based tier router."""

import pytest

from jarvis.router import FAST, STRONG, choose_tier, models_for_tier


@pytest.mark.parametrize(
    "message",
    [
        "hi",
        "hey there",
        "thanks!",
        "good morning",
        "what's up",
        "ok",
        "remind me to call mom at 5",
    ],
)
def test_small_talk_and_simple_requests_are_fast(message):
    assert choose_tier(message) == FAST


@pytest.mark.parametrize(
    "message",
    [
        "Why does Python raise KeyError here?",
        "Explain how async generators work",
        "Debug this traceback for me",
        "Compare SQLite and Postgres for this",
        "Walk me through the design of a scheduler",
        "How do I refactor this module?",
    ],
)
def test_reasoning_requests_are_strong(message):
    assert choose_tier(message) == STRONG


def test_empty_message_is_fast():
    assert choose_tier("") == FAST
    assert choose_tier("   ") == FAST


def test_long_message_is_strong():
    assert choose_tier("a " * 200) == STRONG


def test_models_for_tier_puts_requested_tier_first():
    models = ["big-model", "mid-model", "small-model"]
    tiers = {"fast": ["small-model"], "strong": ["big-model"]}

    assert models_for_tier(models, tiers, FAST)[0] == "small-model"
    assert models_for_tier(models, tiers, STRONG)[0] == "big-model"
    # Every model is still present as a fallback.
    assert set(models_for_tier(models, tiers, FAST)) == set(models)


def test_models_for_tier_without_config_keeps_order():
    models = ["a", "b"]
    assert models_for_tier(models, None, STRONG) == ["a", "b"]
    assert models_for_tier([], None, FAST) == []
