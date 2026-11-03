"""Provider implementations for Jarvis assistant.

Phase 1 scope:
* One OpenAI-compatible transport that works for Groq, Gemini, Cerebras,
  OpenRouter and the local Ollama tunnel.
* Real Server-Sent-Events streaming (text arrives as it is produced).
* Tool-call assembly in both streaming and non-streaming mode.
* Fallback across providers *and* across models, with a short cooldown so a
  throttled or retired model is skipped for a while.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

_logger = logging.getLogger("jarvis.providers")

# Status codes that mean "try somebody else", as opposed to "your config is
# wrong, stop and tell the user".
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class ProviderError(Exception):
    """Provider call failed in a way the chain may recover from."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code

    @property
    def retryable(self) -> bool:
        if self.status_code is None:
            return True  # network/timeout/parse problems
        if self.status_code in RETRYABLE_STATUS:
            return True
        # A retired model usually comes back as 400/404 with the model name in
        # the body, which the caller can only distinguish by message text.
        return self.status_code in {400, 404, 422}


@dataclass
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: str = "{}"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments or "{}"},
        }


@dataclass
class StreamEvent:
    """One unit of a streamed reply.

    ``text`` carries incremental prose; ``tool_calls`` is populated once, after
    the model has finished emitting its tool requests.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    provider: str = ""
    model: str = ""


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    kind: str
    base_url: str
    api_key_env: str
    models: List[str]
    connect_timeout_s: int
    read_timeout_s: int
    options: Dict[str, Any] = field(default_factory=dict)
    tiers: Dict[str, List[str]] = field(default_factory=dict)


class Provider:
    """Base provider class for OpenAI-compatible LLM services."""

    def __init__(self, config: ProviderConfig):
        self.config = config
        timeout_config = httpx.Timeout(
            connect=config.connect_timeout_s,
            read=config.read_timeout_s,
            write=30.0,
            pool=30.0,
        )
        self.session = httpx.AsyncClient(timeout=timeout_config)
        self._last_failed_at: Optional[float] = None
        self._cooldown_s: int = 120
        self._failed_models: Dict[str, float] = {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    def _ensure_session(self) -> httpx.AsyncClient:
        if getattr(self.session, "is_closed", False):
            timeout_config = httpx.Timeout(
                connect=self.config.connect_timeout_s,
                read=self.config.read_timeout_s,
                write=30.0,
                pool=30.0,
            )
            self.session = httpx.AsyncClient(timeout=timeout_config)
        return self.session

    async def aclose(self) -> None:
        await self.session.aclose()

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def is_available(self) -> bool:
        """Check if this provider is currently available (not on cooldown)."""
        if self._last_failed_at is None:
            return True
        return time.time() - self._last_failed_at >= self._cooldown_s

    def mark_failed(self):
        """Mark this provider as failed and put it on cooldown."""
        self._last_failed_at = time.time()

    def mark_model_failed(self, model: str) -> None:
        """Skip one model without cooling down the whole provider."""
        self._failed_models[model] = time.time()
        _logger.warning(
            "Provider %s: model %s skipped for %ss",
            self.name,
            model,
            self._cooldown_s,
        )

    def model_available(self, model: str) -> bool:
        failed_at = self._failed_models.get(model)
        if failed_at is None:
            return True
        return time.time() - failed_at >= self._cooldown_s

    def usable_models(self, tier: Optional[str] = None) -> List[str]:
        """Models for this provider, preferred tier first, unusable ones last."""
        preferred = list(self.config.models or [])
        if tier and self.config.tiers.get(tier):
            tier_models = [m for m in self.config.tiers[tier] if m in preferred]
            preferred = tier_models + [m for m in preferred if m not in tier_models]
        usable = [m for m in preferred if self.model_available(m)]
        cooling = [m for m in preferred if m not in usable]
        return usable + cooling

    async def _get_api_key(self) -> Optional[str]:
        """Get the API key from environment variable."""
        token = os.environ.get(self.config.api_key_env or "")
        return token.strip() if token else None

    async def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        token = await self._get_api_key()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _endpoint(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if self.config.kind == "ollama" and not base.endswith("/v1"):
            base = f"{base}/v1"
        if self.config.kind not in ("ollama", "openai_compat"):
            raise ProviderError(f"Unsupported provider kind: {self.config.kind}")
        return f"{base}/chat/completions"

    def _models_url(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if self.config.kind == "ollama":
            root = base[:-3] if base.endswith("/v1") else base
            return f"{root}/api/tags"
        return f"{base}/models"

    async def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models for this provider."""
        try:
            response = await self.session.get(
                self._models_url(), headers=await self._headers()
            )
            response.raise_for_status()
            data = response.json()
            if self.config.kind == "ollama":
                return {"models": [m["name"] for m in data.get("models", [])]}
            return {"models": [m["id"] for m in data.get("data", [])]}
        except Exception as exc:  # noqa: BLE001 - informational only
            _logger.warning("Failed to list models for %s: %s", self.name, exc)
            return {}

    def _body(
        self,
        messages: List[Dict],
        model: str,
        tools: Optional[List[Dict]],
        stream: bool,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "messages": messages}
        if stream:
            body["stream"] = True
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        # Ollama understands num_ctx under "options"; other providers reject it.
        if self.config.kind == "ollama" and self.config.options:
            body["options"] = self.config.options
        return body

    async def chat_completions_stream(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream a reply, yielding text as it arrives and tool calls at the end."""
        chosen = model or (self.usable_models(tier) or [""])[0]
        if not chosen:
            raise ProviderError(f"{self.name}: no model configured")

        body = self._body(messages, chosen, tools, stream=True)
        calls: Dict[int, ToolCall] = {}

        try:
            async with self._ensure_session().stream(
                "POST", self._endpoint(), json=body, headers=await self._headers()
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread()).decode("utf-8", "replace")
                    raise ProviderError(
                        f"{self.name}/{chosen}: HTTP {response.status_code} {detail[:300]}",
                        status_code=response.status_code,
                    )

                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload in ("[DONE]", ""):
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        _logger.debug("Skipping malformed SSE chunk: %r", payload[:120])
                        continue

                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        text = delta.get("content") or ""
                        if text:
                            yield StreamEvent(text=text, provider=self.name, model=chosen)
                        for item in delta.get("tool_calls") or []:
                            index = item.get("index", 0)
                            current = calls.setdefault(
                                index,
                                ToolCall(
                                    id=item.get("id") or f"call_{index}",
                                    name="",
                                    arguments="",
                                ),
                            )
                            if item.get("id"):
                                current.id = item["id"]
                            function = item.get("function") or {}
                            if function.get("name"):
                                current.name += function["name"]
                            if function.get("arguments"):
                                current.arguments += function["arguments"]

        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}/{chosen}: {type(exc).__name__}: {exc}") from exc

        finished = [c for c in calls.values() if c.name]
        if finished:
            yield StreamEvent(
                tool_calls=finished, provider=self.name, model=chosen
            )

    async def chat_completions(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Non-streaming completion. Returns the raw OpenAI-shaped response."""
        chosen = model or (self.usable_models(tier) or [""])[0]
        if not chosen:
            raise ProviderError(f"{self.name}: no model configured")

        body = self._body(messages, chosen, tools, stream=False)

        try:
            response = await self.session.post(
                self._endpoint(), json=body, headers=await self._headers()
            )
            if response.status_code >= 400:
                raise ProviderError(
                    f"{self.name}/{chosen}: HTTP {response.status_code} "
                    f"{response.text[:300]}",
                    status_code=response.status_code,
                )
            data = response.json()
            data.setdefault("_provider", self.name)
            data.setdefault("_model", chosen)
            return data
        except ProviderError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name}/{chosen}: {type(exc).__name__}: {exc}") from exc

    def __repr__(self):
        return f"Provider(name='{self.config.name}', kind='{self.config.kind}')"


class Chain:
    """Ordered providers/models to try until one answers."""

    def __init__(self, providers: List[Provider], cooldown_s: int = 120):
        self.providers = providers
        self.cooldown_s = cooldown_s
        for provider in self.providers:
            provider._cooldown_s = cooldown_s
        self.last_provider = ""
        self.last_model = ""

    async def aclose(self) -> None:
        for provider in self.providers:
            await provider.aclose()

    def _attempts(self, tier: Optional[str]):
        """Yield (provider, model) pairs in preference order."""
        for provider in self.providers:
            if not provider.is_available:
                _logger.info("Skipping %s (cooling down)", provider.name)
                continue
            for model in provider.usable_models(tier):
                yield provider, model

    async def chat_completions_stream(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream through the chain, moving on when a model or provider fails."""
        attempts = list(self._attempts(tier))
        if not attempts:
            raise RuntimeError("All providers in chain failed")

        errors: List[str] = []
        for provider, candidate in attempts:
            if model and candidate != model:
                continue
            try:
                async for event in provider.chat_completions_stream(
                    messages, tools, candidate, tier
                ):
                    if not self.last_provider:
                        _logger.info(
                            "Answered by %s with model %s", provider.name, candidate
                        )
                    self.last_provider = provider.name
                    self.last_model = candidate
                    yield event
                return
            except ProviderError as exc:
                errors.append(str(exc))
                _logger.warning("Fallback: %s", exc)
                if exc.status_code in (401, 403):
                    provider.mark_failed()
                elif exc.retryable:
                    if exc.status_code in RETRYABLE_STATUS:
                        provider.mark_model_failed(candidate)
                        if exc.status_code != 429:
                            provider.mark_failed()
                    else:
                        provider.mark_model_failed(candidate)
                else:
                    provider.mark_failed()
            except Exception as exc:  # noqa: BLE001 - never crash the chat loop
                errors.append(f"{provider.name}/{candidate}: {exc}")
                _logger.error("Unexpected error with %s: %s", provider.name, exc)
                provider.mark_model_failed(candidate)

        raise RuntimeError(
            "All providers in chain failed: " + "; ".join(errors[-3:])
        )

    async def chat_completions(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        model: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Non-streaming completion through the chain."""
        attempts = list(self._attempts(tier))
        if not attempts:
            raise RuntimeError("All providers in chain failed")

        errors: List[str] = []
        for provider, candidate in attempts:
            if model and candidate != model:
                continue
            try:
                result = await provider.chat_completions(
                    messages, tools, candidate, tier
                )
                self.last_provider = provider.name
                self.last_model = candidate
                _logger.info(
                    "Answered by %s with model %s", provider.name, candidate
                )
                return result
            except ProviderError as exc:
                errors.append(str(exc))
                _logger.warning("Fallback: %s", exc)
                if exc.status_code in (401, 403):
                    provider.mark_failed()
                elif exc.retryable:
                    if exc.status_code in RETRYABLE_STATUS:
                        provider.mark_model_failed(candidate)
                        if exc.status_code != 429:
                            provider.mark_failed()
                    else:
                        provider.mark_model_failed(candidate)
                else:
                    provider.mark_failed()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{provider.name}/{candidate}: {exc}")
                _logger.error("Unexpected error with %s: %s", provider.name, exc)
                provider.mark_model_failed(candidate)

        raise RuntimeError(
            "All providers in chain failed: " + "; ".join(errors[-3:])
        )

    def __repr__(self):
        return f"Chain({', '.join(p.name for p in self.providers)})"
