"""Main assistant logic for Jarvis (programmatic entry point)."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.chat import MAX_TOOL_ROUNDS, ChatSession
from jarvis.providers import ToolCall
from jarvis.router import choose_tier
from jarvis.tools import registry

_logger = logging.getLogger("jarvis.assistant")


class JarvisAssistant:
    """Main assistant class that orchestrates all components."""

    def __init__(self):
        self.session: Optional[ChatSession] = None
        self.current_provider_name: str = ""
        self.current_model_name: str = ""

    def initialize(self, mode: str = "online"):
        """Initialize the assistant with all required components."""
        try:
            self.session = ChatSession()
            self.session.setup_providers(mode)
            _logger.info("Jarvis assistant initialized successfully in %s mode", mode)
        except Exception as e:
            _logger.error("Failed to initialize Jarvis assistant: %s", e)
            raise

    async def process_message(self, user_input: str, tier: Optional[str] = None) -> str:
        """Process one user message, running any tools the model requests."""
        if not self.session or not self.session.provider_chain:
            raise RuntimeError("Assistant not initialized")

        session = self.session
        session.messages.append({"role": "user", "content": user_input})

        chosen_tier = tier or session.tier_override or choose_tier(user_input)
        tools = [
            {"type": "function", "function": schema}
            for schema in registry.get_all_tools().values()
        ] or None

        for round_index in range(MAX_TOOL_ROUNDS + 1):
            result = await session.provider_chain.chat_completions(
                messages=session.messages, tools=tools, tier=chosen_tier
            )

            self.current_provider_name = result.get(
                "_provider", session.provider_chain.last_provider
            )
            self.current_model_name = result.get(
                "_model", session.provider_chain.last_model
            )

            choices = result.get("choices") or []
            if not choices:
                raise RuntimeError("Provider returned no choices")

            message = choices[0].get("message") or {}
            content = message.get("content") or ""
            calls = message.get("tool_calls") or []

            if not calls:
                session.messages.append({"role": "assistant", "content": content})
                return content

            session.messages.append(
                {"role": "assistant", "content": content or None, "tool_calls": calls}
            )

            for call in calls:
                function = call.get("function") or {}
                name = function.get("name", "")
                output = ChatSession._run_tool(
                    ToolCall(
                        id=call.get("id", ""),
                        name=name,
                        arguments=function.get("arguments") or "{}",
                    )
                )

                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "name": name,
                        "content": output,
                    }
                )

        return "I stopped after too many tool rounds without an answer."

    def get_provider_status(self) -> Dict[str, Any]:
        """Get current status of all providers."""
        if not self.session or not self.session.provider_chain:
            return {"status": "not_initialized"}

        return {
            "mode": self.session.mode,
            "last_provider": self.session.current_provider_name,
            "last_model": self.session.current_model_name,
            "providers": [
                {
                    "name": provider.name,
                    "available": provider.is_available,
                    "last_failed_at": provider._last_failed_at,
                    "cooldown_s": provider._cooldown_s,
                    "models": provider.config.models,
                }
                for provider in self.session.provider_chain.providers
            ],
        }

    def force_provider(self, provider_name: str):
        """Force the assistant to use a specific provider."""
        if self.session:
            self.session.force_provider = provider_name

    def restore_auto_provider(self):
        """Restore automatic provider selection."""
        if self.session:
            self.session.force_provider = ""


# Global assistant instance
assistant = JarvisAssistant()
