"""Tests for streaming, tool-call assembly and chain fallback."""

import json

import httpx
import pytest

from jarvis.providers import Chain, Provider, ProviderConfig


def sse(*chunks) -> bytes:
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
    return (body + "data: [DONE]\n\n").encode()


def text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def make_provider(name: str, models, handler, tiers=None) -> Provider:
    config = ProviderConfig(
        name=name,
        kind="openai_compat",
        base_url="https://example.test/v1",
        api_key_env="",
        models=models,
        connect_timeout_s=5,
        read_timeout_s=10,
        tiers=tiers or {},
    )
    provider = Provider(config)
    provider.session = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return provider


async def collect(chain: Chain, monkeypatch=None, **kwargs):
    events = []
    async for event in chain.chat_completions_stream(
        messages=[{"role": "user", "content": "hello"}], **kwargs
    ):
        events.append(event)
    return events


@pytest.mark.asyncio
async def test_stream_yields_text_incrementally():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(text_chunk("Hello"), text_chunk(" there"), text_chunk("!")),
            headers={"content-type": "text/event-stream"},
        )

    chain = Chain([make_provider("p1", ["m1"], handler)])
    events = await collect(chain)

    assert "".join(e.text for e in events) == "Hello there!"
    assert all(e.tool_calls == [] for e in events)
    assert chain.last_provider == "p1"
    await chain.aclose()


@pytest.mark.asyncio
async def test_stream_assembles_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "function": {"name": "add_task", "arguments": '{"ti'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'tle": "buy milk"}'},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ),
            headers={"content-type": "text/event-stream"},
        )

    chain = Chain([make_provider("p1", ["m1"], handler)])
    events = await collect(chain, tools=[{"type": "function", "function": {"name": "add_task"}}])

    calls = [c for e in events for c in e.tool_calls]
    assert len(calls) == 1
    assert calls[0].name == "add_task"
    assert json.loads(calls[0].arguments) == {"title": "buy milk"}
    await chain.aclose()


@pytest.mark.asyncio
async def test_chain_falls_through_to_next_provider_on_429():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "rate limited"})

    def working(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(text_chunk("backup answered")),
            headers={"content-type": "text/event-stream"},
        )

    first = make_provider("primary", ["m1"], failing)
    second = make_provider("backup", ["m2"], working)
    chain = Chain([first, second], cooldown_s=60)

    events = await collect(chain)

    assert "".join(e.text for e in events) == "backup answered"
    assert chain.last_provider == "backup"
    # A 429 throttles one model, not the whole provider: the other model on
    # "primary" would still be usable.
    assert first.model_available("m1") is False
    assert first.is_available is True
    await chain.aclose()


@pytest.mark.asyncio
async def test_chain_falls_through_to_next_model_on_retired_model():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body["model"] == "retired-model":
            return httpx.Response(400, json={"error": "model does not exist"})
        return httpx.Response(
            200,
            content=sse(text_chunk("second model here")),
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider("p1", ["retired-model", "live-model"], handler)
    chain = Chain([provider], cooldown_s=60)

    events = await collect(chain)

    assert "".join(e.text for e in events) == "second model here"
    assert chain.last_model == "live-model"
    assert provider.model_available("retired-model") is False
    assert provider.is_available is True  # only the model was skipped
    await chain.aclose()


@pytest.mark.asyncio
async def test_tier_changes_model_order():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            content=sse(text_chunk("ok")),
            headers={"content-type": "text/event-stream"},
        )

    provider = make_provider(
        "p1",
        ["big-model", "small-model"],
        handler,
        tiers={"fast": ["small-model"], "strong": ["big-model"]},
    )
    chain = Chain([provider])

    await collect(chain, tier="fast")
    assert seen[-1] == "small-model"

    await collect(chain, tier="strong")
    assert seen[-1] == "big-model"
    await chain.aclose()


@pytest.mark.asyncio
async def test_chain_raises_when_everything_fails():
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    chain = Chain([make_provider("p1", ["m1"], failing)], cooldown_s=60)

    with pytest.raises(RuntimeError):
        await collect(chain)
    await chain.aclose()
