"""Interactive chat interface for Jarvis assistant (Phase 1)."""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
import re
import yaml

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.providers import Chain, Provider, ProviderConfig, StreamEvent
from jarvis.router import FAST, STRONG, choose_tier, models_for_tier
from jarvis.tools import registry
from jarvis.console_input import read_line
from jarvis import screen_tools  # noqa: F401 - registers the screen tools
from jarvis import task_tools  # noqa: F401 - registers the task tools
from jarvis import project_reader  # noqa: F401 - registers project reader tools
from jarvis import memory  # noqa: F401 - registers memory and docs tools
from jarvis import general_tools  # noqa: F401 - registers general tools (weather, web, calc)
from jarvis.ascii_ui import AsciiUI

_logger = logging.getLogger("jarvis.chat")

MAX_TOOL_ROUNDS = 4


class ChatSession:
    """Manages a chat session with the assistant."""

    def __init__(self):
        self.messages: List[Dict[str, Any]] = []
        self.provider_chain: Optional[Chain] = None
        self.current_provider_name: str = ""
        self.current_model_name: str = ""
        self.force_provider: str = ""  # provider name, or "" for auto
        self.tier_override: str = ""  # "fast" | "strong" | "" for auto
        self.mode: str = ""  # "online" or "local"
        self.cfg: Dict[str, Any] = {}
        self.last_reason: str = ""
        self.scheduler = None
        self.task_service = None
        self.voice = None
        self.transcripts = None
        self.speech = None
        self.screen = None
        self.screen_history = None
        self.screen_tray = None
        self.screen_hotkey = None
        self.recent_errors: List[str] = []
        self.ascii_ui = AsciiUI()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml."""
        cfg_path = project_root / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError("config.yaml not found")

        with cfg_path.open("rt", encoding="utf8") as f:
            cfg = yaml.safe_load(f) or {}

        return cfg

    @staticmethod
    def _provider_mode(provider_config: Dict[str, Any]) -> str:
        """Return a provider's mode, inferring it for older configs."""
        mode = str(provider_config.get("mode", "")).lower()
        if mode in ("online", "local"):
            return mode
        return "local" if provider_config.get("kind") == "ollama" else "online"

    def _mode_hint(self, provider_config: Dict[str, Any]) -> str:
        """Short, offline readiness hint for the startup menu."""
        kind = provider_config.get("kind")
        if kind == "ollama":
            env_name = provider_config.get("base_url_env", "")
            return f"{env_name} set" if os.environ.get(env_name) else f"{env_name} missing"
        env_name = provider_config.get("api_key_env", "")
        if not os.environ.get(env_name):
            return f"{env_name} missing"
        models = [
            m for m in (provider_config.get("models") or [])
            if m and "<" not in str(m)
        ]
        return "model set" if models else "model not set"

    # ------------------------------------------------------------------
    # Startup menu and validation
    # ------------------------------------------------------------------
    def choose_mode(self) -> Optional[str]:
        """Ask whether this run should use online or local providers."""
        if not self.cfg:
            self.cfg = self.load_config()

        groups: Dict[str, List[Dict[str, Any]]] = {}
        for provider_config in self.cfg.get("providers", []):
            groups.setdefault(self._provider_mode(provider_config), []).append(provider_config)

        if not groups:
            print("No providers are configured in config.yaml.")
            return None

        modes = [m for m in ("online", "local") if m in groups]
        modes += [m for m in groups if m not in modes]

        print("\nChoose how Jarvis should think:")
        choices: Dict[str, str] = {}
        for index, mode in enumerate(modes, start=1):
            descriptions = []
            for provider_config in groups[mode]:
                models = provider_config.get("models") or ["no model"]
                descriptions.append(
                    f"{provider_config.get('name')} "
                    f"({models[0]}, {self._mode_hint(provider_config)})"
                )
            print(f"  {index}) {mode.title():<7} - {'; '.join(descriptions)}")
            choices[str(index)] = mode
        print("  q) Quit")
        if "local" in groups:
            print("     Local uses OLLAMA_BASE_URL (your Kaggle tunnel); start that session first.")

        while True:
            try:
                raw = input("\nSelect a mode: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return None
            if raw in ("q", "quit", "exit"):
                return None
            if raw in choices:
                return choices[raw]
            if raw in groups:
                return raw
            print("Please enter a number from the list, or q to quit.")

    def validate_mode(self, mode: str) -> tuple:
        """Check that at least one provider in ``mode`` is usable right now.

        Returns ``(True, "")`` or ``(False, reason)``. Configured models that
        no longer exist are dropped from the runtime config, so the chat uses
        the first model that is actually available.
        """
        providers = [
            p for p in self.cfg.get("providers", [])
            if self._provider_mode(p) == mode
        ]
        if not providers:
            return False, f"no providers are marked as {mode} in config.yaml"

        problems: List[str] = []
        usable_providers = 0
        for provider_config in providers:
            name = provider_config.get("name", "provider")
            kind = provider_config.get("kind")
            models = [
                m for m in (provider_config.get("models") or [])
                if m and "<" not in str(m)
            ]
            if not models:
                problems.append(f"{name}: no real model configured")
                continue

            if kind == "ollama":
                env_name = provider_config.get("base_url_env", "")
                base_url = (os.environ.get(env_name) or "").strip().rstrip("/")
                if not base_url:
                    problems.append(f"{name}: {env_name} is not set in .env")
                    continue
                root_url = base_url[:-3] if base_url.endswith("/v1") else base_url
                url = f"{root_url}/api/tags"

                token_name = provider_config.get("api_key_env")
                token = os.environ.get(token_name) if token_name else None
                headers = {"Authorization": f"Bearer {token}"} if token else {}

                try:
                    response = httpx.get(
                        url,
                        headers=headers,
                        timeout=provider_config.get("connect_timeout_s", 4),
                    )
                except Exception as exc:
                    problems.append(f"{name}: cannot reach {url} ({exc})")
                    continue

                if response.status_code != 200:
                    problems.append(f"{name}: HTTP {response.status_code} from {url}")
                    continue

                try:
                    live_models = [
                        m.get("name", "") for m in response.json().get("models", [])
                    ]
                except Exception:
                    live_models = []

                if not live_models:
                    problems.append(f"{name}: reachable but returned no models")
                    continue

                usable = [m for m in models if m in live_models]
                if not usable:
                    problems.append(
                        f"{name}: configured model(s) {', '.join(models)} not installed "
                        f"(available: {', '.join(live_models[:8])})"
                    )
                    continue
                if len(usable) < len(models):
                    missing = [m for m in models if m not in usable]
                    print(f"  - {name}: skipping missing model(s) {', '.join(missing)}")
                provider_config["models"] = usable
                usable_providers += 1

            elif kind == "openai_compat":
                env_name = provider_config.get("api_key_env", "")
                api_key = (os.environ.get(env_name) or "").strip()
                if not api_key:
                    problems.append(f"{name}: {env_name} is not set in .env")
                    continue

                base_url = (provider_config.get("base_url") or "").rstrip("/")
                if not base_url:
                    problems.append(f"{name}: base_url is missing")
                    continue
                url = f"{base_url}/models"

                try:
                    response = httpx.get(
                        url,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=provider_config.get("connect_timeout_s", 8),
                    )
                except Exception as exc:
                    problems.append(f"{name}: cannot reach {url} ({exc})")
                    continue

                if response.status_code != 200:
                    problems.append(f"{name}: HTTP {response.status_code} from {url}")
                    continue

                try:
                    live_models = [
                        m.get("id", "") for m in response.json().get("data", [])
                    ]
                except Exception:
                    live_models = []

                if not live_models:
                    problems.append(f"{name}: reachable but returned no models")
                    continue

                usable = [m for m in models if m in live_models]
                if not usable:
                    problems.append(
                        f"{name}: none of the configured models exist anymore "
                        f"({', '.join(models)})"
                    )
                    continue
                if len(usable) < len(models):
                    missing = [m for m in models if m not in usable]
                    print(f"  - {name}: skipping retired model(s) {', '.join(missing)}")
                provider_config["models"] = usable
                usable_providers += 1

            else:
                problems.append(f"{name}: unsupported provider kind {kind}")

        if usable_providers == 0:
            return False, "; ".join(problems) or "no provider is ready"
        if problems:
            # Backups can be down; the chain simply skips them.
            print("  - backup provider(s) unavailable: " + "; ".join(problems))
        return True, ""

    # ------------------------------------------------------------------
    # Provider chain
    # ------------------------------------------------------------------
    def initialize(self, mode: Optional[str] = None, voice_preset: Optional[str] = None) -> None:
        """Alias for setup_providers to initialize session providers and persona."""
        self.setup_providers(mode=mode, voice_preset=voice_preset)

    def setup_providers(self, mode: Optional[str] = None, voice_preset: Optional[str] = None):
        """Build a provider chain containing only the selected mode."""
        if not self.cfg:
            self.cfg = self.load_config()

        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")

        # Programmatic callers (for example assistant.py) default to online.
        if mode is None:
            mode = "online"

        providers = []
        for provider_config in self.cfg.get("providers", []):
            if self._provider_mode(provider_config) != mode:
                continue

            kind = provider_config.get("kind")
            if kind == "ollama":
                base_url = os.environ.get(provider_config.get("base_url_env", ""))
                if not base_url:
                    _logger.warning(
                        "Base URL for %s not found in environment",
                        provider_config.get("name"),
                    )
                    continue
                base_url = base_url.rstrip("/")
                if not base_url.endswith("/v1"):
                    base_url = f"{base_url}/v1"
            elif kind == "openai_compat":
                base_url = provider_config.get("base_url", "")
            else:
                _logger.warning("Unsupported provider kind: %s", kind)
                continue

            models = [
                m for m in (provider_config.get("models") or [])
                if m and "<" not in str(m)
            ]
            if not models:
                _logger.warning("No real model configured for %s", provider_config.get("name"))
                continue

            tiers = {
                tier: [
                    m for m in (provider_config.get("tiers", {}).get(tier) or [])
                    if m in models
                ]
                for tier in (FAST, STRONG)
            }

            config = ProviderConfig(
                name=provider_config["name"],
                kind=kind,
                base_url=base_url,
                api_key_env=provider_config.get("api_key_env", ""),
                models=models,
                connect_timeout_s=provider_config.get("connect_timeout_s", 8),
                read_timeout_s=provider_config.get("read_timeout_s", 30),
                options=provider_config.get("options", {}),
                tiers=tiers,
            )
            providers.append(Provider(config))

        if not providers:
            raise RuntimeError(f"No usable providers for {mode} mode")

        self.provider_chain = Chain(
            providers=providers,
            cooldown_s=self.cfg.get("cooldown_s", 120),
        )
        self.mode = mode
        self.voice_preset = voice_preset or self.cfg.get("voice", {}).get("default_preset", "calm")
        try:
            from jarvis.web_search import set_active_provider_chain
            set_active_provider_chain(self.provider_chain)
        except Exception:
            pass

        if not any(m.get("role") == "system" for m in self.messages):
            persona = self._build_persona(self.voice_preset)
            self.messages.append({"role": "system", "content": persona})

    def _build_persona(self, preset_name: Optional[str] = None) -> str:
        name = preset_name or getattr(self, "voice_preset", "calm")
        from jarvis.tts import get_preset_config
        preset_cfg = get_preset_config(self.cfg, name)
        rel_path = preset_cfg.get("persona", "jarvis/persona.md")
        persona_path = project_root / rel_path
        if not persona_path.exists():
            persona_path = project_root / "jarvis" / "persona.md"
        persona = ""
        if persona_path.exists():
            with persona_path.open("rt", encoding="utf8") as f:
                persona = f.read()

        from jarvis.tasks import now as _tz_now
        moment = _tz_now()
        city = self.cfg.get("home_city", "New Delhi")
        tz_name = self.cfg.get("timezone", "Asia/Kolkata")
        persona += (
            f"\n\n## Right now\nIt is {moment.strftime('%A, %d %B %Y, %I:%M %p')}"
            f" in {city} ({tz_name}, UTC+05:30). Use this when converting"
            " 'today', 'tomorrow' and 'next Friday' into deadlines.\n"
        )
        from jarvis.memory import get_service as _get_memory
        startup_ctx = _get_memory().get_startup_context()
        if startup_ctx:
            persona += f"\n\n## User Background & Active Memory\n{startup_ctx}\n"
        return persona

    def set_voice_preset(self, preset_name: str) -> None:
        """Swap voice preset and persona mid-session without losing chat history."""
        self.voice_preset = preset_name
        persona_text = self._build_persona(preset_name)
        for m in self.messages:
            if m.get("role") == "system":
                m["content"] = persona_text
                break
        else:
            self.messages.insert(0, {"role": "system", "content": persona_text})

        if self.speech:
            self.speech.set_preset(preset_name, self.cfg)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def handle_command(self, command: str) -> bool:
        """Handle chat commands. Returns False when the session should end."""
        parts = command.strip().split()
        if not parts:
            return True

        cmd = parts[0].lower()

        if cmd == "/quit":
            print("Goodbye!")
            return False

        if cmd == "/provider":
            if self.current_provider_name:
                print(
                    f"Last provider: {self.current_provider_name} "
                    f"with model {self.current_model_name}"
                )
            else:
                print("No conversation yet")

        elif cmd in ("/fast", "/strong"):
            self.tier_override = cmd[1:]
            print(f"Tier locked to {self.tier_override}. Use /auto to let me choose.")

        elif cmd == "/auto":
            self.tier_override = ""
            print("Tier selection back to automatic (rule-based).")

        elif cmd == "/force":
            names = (
                [p.name for p in self.provider_chain.providers]
                if self.provider_chain else []
            )
            if len(parts) == 2 and parts[1].lower() == "auto":
                self.force_provider = ""
                print("Automatic provider selection restored")
            elif len(parts) > 1:
                provider = parts[1]
                if provider in names:
                    self.force_provider = provider
                    print(f"Force provider set to {provider}")
                else:
                    available = ", ".join(names) if names else "none"
                    print(f"Unknown provider. Available: {available}")
            else:
                available = "|".join(names) if names else "provider"
                print(f"/force [{available}|auto]")

        elif cmd == "/status":
            self.print_provider_status()

        elif cmd == "/clear":
            system = [m for m in self.messages if m.get("role") == "system"]
            self.messages = system
            print("Conversation cleared.")

        elif cmd in ("/mute", "/unmute"):
            if not self.speech:
                print("Voice output is not running.")
            elif cmd == "/mute":
                self.speech.mute()
                print("Voice muted. /unmute to hear me again.")
            else:
                self.speech.unmute()
                print("Voice on.")

        elif cmd == "/screen":
            self.handle_screen_command(parts[1:])

        else:
            print(f"Unknown command: {cmd}")

        return True

    def handle_screen_command(self, args: List[str]) -> None:
        """Show or change the view-only screen access state."""
        if self.screen is None:
            print("Screen reading is not running (screen.enabled is false).")
            return

        action = (args[0].lower() if args else "status")
        if action == "pause":
            self.screen.pause("paused from /screen")
        elif action == "resume":
            self.screen.resume()
        elif action != "status":
            print("Usage: /screen [status|pause|resume]")
            return

        if self.screen_tray is not None:
            self.screen_tray.refresh()
        state = "paused" if self.screen.paused else "active"
        print(f"Screen access {state} (OCR: {self.screen.ocr_name}).")

    def print_provider_status(self):
        """Print the status of all providers in the chain."""
        if not self.provider_chain:
            print("No providers configured")
            return

        tier = self.tier_override or "auto"
        print(f"Mode: {self.mode or 'not selected'}   Tier: {tier}")
        print("Provider Status:")
        for provider in self.provider_chain.providers:
            cooldown_left = 0
            if provider._last_failed_at:
                cooldown_left = max(
                    0,
                    provider._cooldown_s - (int(time.time()) - provider._last_failed_at),
                )
            status = "OK" if provider.is_available else f"cooldown {cooldown_left}s"
            models = ", ".join(provider.usable_models())
            print(f"  {provider.name}: {status} [{models}]")

    # ------------------------------------------------------------------
    # Turn handling
    # ------------------------------------------------------------------
    def _tools_payload(self) -> List[Dict[str, Any]]:
        vision_ready = screen_tools.vision_configured(self.cfg)
        tools: List[Dict[str, Any]] = []
        for name, schema in registry.get_all_tools().items():
            # Do not offer a tool that can only answer "not configured".
            if name == "describe_screen_image" and not vision_ready:
                continue
            tools.append({"type": "function", "function": schema})
        return tools

    async def _maybe_read_screen(self, user_message: str) -> Optional[str]:
        """Rule-based fallback: 'this error' style asks always look at the screen."""
        if not self.screen or not self.screen.enabled:
            return None
        if not screen_tools.wants_screen(user_message):
            return None

        if self.speech:
            self.speech.maybe_filler()  # OCR takes a moment; say so
        result = await asyncio.to_thread(screen_tools.screen_context, user_message)

        if not result.get("ok"):
            reason = result.get("reason") or "screen capture failed"
            print(f"\n[screen] refused: {reason}")
            return (
                f"Screen capture was refused: {reason} Tell the user that plainly, "
                "in one short sentence, and do not guess about their screen."
            )

        text = (result.get("text") or "").strip()
        window = result.get("window") or "the active window"
        print(
            f"\n[screen] {window}: {len(text)} chars in {result.get('total_ms', 0)} ms"
        )
        if not text:
            return (
                f"The screen was captured but no readable text was found in {window}. "
                "Ask the user what they see."
            )

        extra_parts: List[str] = []
        code_ctx = project_reader.extract_code_context_from_ocr(text)
        if code_ctx:
            extra_parts.append(
                f"\nSurrounding project code for error seen on screen:\n{code_ctx}\n"
            )

        rep_note = self._check_repeated_error(f"{text} {user_message}")
        if rep_note:
            extra_parts.append(f"\n{rep_note}\n")

        extra_text = "".join(extra_parts)
        return (
            f"Screen text captured just now from {window} (view-only OCR):\n{text}\n"
            f"{extra_text}"
            "Use this to answer the user's question. Do not say you cannot see the "
            "screen."
        )

    def _check_repeated_error(self, text: str) -> Optional[str]:
        """Check if an error signature was seen earlier in this session."""
        patterns = [
            r'([A-Z][a-zA-Z0-9]*(?:Error|Exception|Warning|Fault))(?::\s*([^\n\r]+))?',
            r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)',
        ]
        found_sig = None
        for pat in patterns:
            match = re.search(pat, text)
            if match:
                found_sig = match.group(0).strip()
                break

        if not found_sig:
            return None

        is_repeated = found_sig in self.recent_errors
        self.recent_errors.append(found_sig)
        if len(self.recent_errors) > 5:
            self.recent_errors.pop(0)

        if is_repeated:
            return (
                f"[Repeated Error Note: The error '{found_sig}' has appeared multiple times "
                "in this session. Say so plainly and suggest a different angle or deeper cause "
                "instead of repeating the same advice.]"
            )
        return None

    def _check_lesson_resume(self, text: str) -> Optional[str]:
        """Check if user is asking to resume an ongoing lesson."""
        lower = text.lower().strip()
        triggers = ("continue", "where were we", "what were we doing", "resume", "resume lesson")
        if any(t in lower for t in triggers):
            from jarvis.memory import get_service as _get_memory
            lesson = _get_memory().get_active_lesson()
            if lesson:
                return (
                    f"[Lesson Resume Context: The user wants to resume their active lesson on "
                    f"'{lesson['topic']}' (currently at step {lesson['current_step']}). "
                    f"Steps completed: '{lesson['steps_completed'] or 'Intro'}'. Recap the last step "
                    f"in one brief, warm sentence and propose the next small step.]"
                )
        return None

    def _extract_raw_tool_calls(self, text: str) -> List[Any]:
        """Extract text-based tool calls emitted by models like Qwen-coder or Ollama."""
        from jarvis.providers import ToolCall
        calls: List[ToolCall] = []

        # 1. XML function/parameter format: <function=name><parameter=key>val</parameter></function>
        func_matches = re.finditer(r"<function=([a-zA-Z0-9_\-]+)>([\s\S]*?)</function>", text)
        for idx, fm in enumerate(func_matches):
            func_name = fm.group(1).strip()
            body = fm.group(2)
            param_matches = re.finditer(r"<parameter=([a-zA-Z0-9_\-]+)>([\s\S]*?)</parameter>", body)
            args = {}
            for pm in param_matches:
                p_name = pm.group(1).strip()
                p_val = pm.group(2).strip()
                args[p_name] = p_val
            calls.append(
                ToolCall(
                    id=f"call_text_{int(time.time())}_{idx}",
                    name=func_name,
                    arguments=json.dumps(args),
                )
            )

        # 2. JSON tool_call format: <tool_call>{"name": ..., "arguments": ...}</tool_call>
        if not calls:
            json_matches = re.finditer(r"<tool_call>([\s\S]*?)</tool_call>", text)
            for idx, jm in enumerate(json_matches):
                raw_json = jm.group(1).strip()
                try:
                    data = json.loads(raw_json)
                    fname = data.get("name") or data.get("function")
                    fargs = data.get("arguments") or data.get("parameters") or {}
                    if isinstance(fargs, dict):
                        fargs = json.dumps(fargs)
                    if fname:
                        calls.append(
                            ToolCall(
                                id=f"call_json_{int(time.time())}_{idx}",
                                name=fname,
                                arguments=str(fargs),
                            )
                        )
                except Exception:
                    pass

        return calls

    def _strip_raw_tool_calls(self, text: str) -> str:
        """Strip raw tool call tags from text before presenting to the user."""
        text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text)
        text = re.sub(r"<function=[a-zA-Z0-9_\-]+>[\s\S]*?</function>", "", text)
        text = re.sub(r"</?tool_call>", "", text)
        return text.strip()

    @staticmethod
    def _run_tool(call) -> str:
        """Execute one tool call and return a JSON string for the model."""
        try:
            tool = registry.get_tool(call.name)
        except ValueError:
            return json.dumps({"error": f"unknown tool {call.name}"})

        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": f"invalid arguments for {call.name}"})

        try:
            result = tool["function"](**args)
        except Exception as exc:  # noqa: BLE001 - report back, do not crash
            _logger.error("Tool %s failed: %s", call.name, exc)
            return json.dumps({"error": str(exc)})

        try:
            return json.dumps(result, default=str)
        except TypeError:
            return json.dumps(str(result))

    async def _stream_once(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        tier: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        if not self.provider_chain:
            raise RuntimeError("Assistant not initialised")

        chain = self.provider_chain
        if self.force_provider:
            chain = Chain(
                [p for p in chain.providers if p.name == self.force_provider],
                cooldown_s=chain.cooldown_s,
            )
            if not chain.providers:
                raise RuntimeError(f"Provider {self.force_provider} is not in this mode")

        async for event in chain.chat_completions_stream(
            messages=messages, tools=tools, tier=tier
        ):
            if event.provider:
                chain.last_provider = event.provider
                chain.last_model = event.model
            yield event

        self.current_provider_name = chain.last_provider or self.current_provider_name
        self.current_model_name = chain.last_model or self.current_model_name

    def _trim_history(self, max_messages: int = 16) -> None:
        """Keep system prompts intact and trim older conversation turns to stay within TPM limits."""
        if len(self.messages) <= max_messages:
            return

        system_msgs = [m for m in self.messages if m.get("role") == "system"]
        non_system = [m for m in self.messages if m.get("role") != "system"]

        keep_count = max(4, max_messages - len(system_msgs))
        if len(non_system) > keep_count:
            trimmed = non_system[-keep_count:]
            while trimmed and trimmed[0].get("role") == "tool":
                trimmed = trimmed[1:]
            self.messages = system_msgs + trimmed

    async def get_response_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """Stream a full turn, running any tools the model asks for.

        A rule-based screen read happens first for phrases like "this error";
        its text is a temporary system message, removed once the turn ends so
        stale screen contents never leak into later turns.
        """
        self._trim_history()
        self.messages.append({"role": "user", "content": user_message})

        note = await self._maybe_read_screen(user_message)
        resume_note = self._check_lesson_resume(user_message)
        if resume_note:
            note = f"{note}\n\n{resume_note}" if note else resume_note

        note_message = None
        if note:
            note_message = {"role": "system", "content": note}
            self.messages.append(note_message)
        else:
            rep_note = self._check_repeated_error(user_message)
            if rep_note:
                note_message = {"role": "system", "content": rep_note}
                self.messages.append(note_message)

        try:
            async for chunk in self._stream_turn(user_message):
                yield chunk
        finally:
            if note_message is not None:
                for index, message in enumerate(self.messages):
                    if message is note_message:
                        del self.messages[index]
                        break

    async def _stream_turn(self, user_message: str) -> AsyncGenerator[str, None]:
        tier = self.tier_override or choose_tier(user_message)
        tools = self._tools_payload() or None
        self.last_reason = tier

        for round_index in range(MAX_TOOL_ROUNDS + 1):
            text_parts: List[str] = []
            calls = []

            try:
                async for event in self._stream_once(self.messages, tools, tier):
                    if event.text:
                        text_parts.append(event.text)
                        yield event.text
                    calls.extend(event.tool_calls)
            except RuntimeError as exc:
                if tools and round_index == 0:
                    # Some endpoints reject tools outright; retry once without.
                    _logger.warning("Retrying without tools: %s", exc)
                    tools = None
                    async for event in self._stream_once(self.messages, None, tier):
                        if event.text:
                            text_parts.append(event.text)
                            yield event.text
                        calls.extend(event.tool_calls)
                else:
                    raise

            text = "".join(text_parts)

            # Fallback: if model emitted tool calls as raw text, extract and strip them
            if not calls:
                raw_parsed_calls = self._extract_raw_tool_calls(text)
                if raw_parsed_calls:
                    calls = raw_parsed_calls
                    text = self._strip_raw_tool_calls(text)

            if not calls:
                self.messages.append({"role": "assistant", "content": text})
                self._trim_history()
                if not text.strip():
                    _logger.warning("Empty reply for tier %s", tier)
                return

            self.messages.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [c.as_dict() for c in calls],
                }
            )

            for call in calls:
                print(f"\n[tool] {call.name}({call.arguments})")
                # Speak a holding line if the tool takes noticeably long.
                tool_task = asyncio.create_task(asyncio.to_thread(self._run_tool, call))
                if getattr(self, "speech", None):
                    done, _pending = await asyncio.wait({tool_task}, timeout=0.7)
                    if not done:
                        self.speech.maybe_filler()
                result = await tool_task
                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result,
                    }
                )

            if round_index == MAX_TOOL_ROUNDS:
                yield "\n(Stopped after too many tool rounds.)"
                return

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self):
        """Run the interactive chat session."""
        # Clear screen and show logo
        self.ascii_ui.clear_screen()
        self.ascii_ui.draw_jarvis_logo()
        print("\nInitializing Jarvis Assistant...")
        print("Commands:")
        print("  /fast | /strong | /auto - choose the thinking tier")
        print("  /provider - show which provider and model answered last")
        print("  /force [name|auto] - restrict to one provider in this mode")
        print("  /status - show mode, tier and provider cooldowns")
        print("  /clear - forget the conversation so far")
        print("  /mute | /unmute - silence or restore spoken replies")
        print("  /screen [status|pause|resume] - view-only screen reading")
        print("  /quit - exit chat")
        print("=" * 50)

        self.cfg = self.load_config()
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")

        while True:
            mode = self.choose_mode()
            if not mode:
                print("Goodbye!")
                return

            ready, reason = self.validate_mode(mode)
            if not ready:
                print(f"\n{mode.title()} mode is not ready: {reason}")
                print("Returning to the mode menu.\n")
                continue

            try:
                self.setup_providers(mode)
            except Exception as e:
                _logger.error("Could not start %s mode: %s", mode, e)
                print(f"\nCould not start {mode} mode: {e}")
                print("Returning to the mode menu.\n")
                continue
            break

        print(f"\nUsing {self.mode} mode. Type /quit to leave.")

        # Voice out first, so barge-in can stop it before the mic opens.
        speech_cfg = self.cfg.get("speech") or {}
        if speech_cfg.get("enabled", True):
            try:
                from jarvis.speech import SpeechOutput

                self.speech = SpeechOutput(
                    self.cfg, preset=getattr(self, "voice_preset", None)
                )
                if not self.speech.enabled:
                    print("Voice output is configured but no engine is available.")
                    self.speech = None
                else:
                    engine = self.speech.engine
                    primary = getattr(engine, "primary", engine)
                    voice = getattr(primary, "voice", "")
                    detail = f" ({voice})" if voice else ""
                    print(f"Voice out: {engine.name}{detail}")
                    self.speech.warm()
            except Exception as exc:  # noqa: BLE001 - never block the chat
                _logger.error("Voice output unavailable: %s", exc)
                print(f"Voice output unavailable: {exc}")
                self.speech = None

        # Screen eyes: view-only, in memory, paused with a hotkey.
        screen_cfg = self.cfg.get("screen") or {}
        if screen_cfg.get("enabled", True):
            try:
                from jarvis.screen import PauseHotkey, ScreenHistory
                from jarvis.tray import ScreenTray

                self.screen = screen_tools.configure(self.cfg)
                hotkey = str(screen_cfg.get("pause_hotkey", "<ctrl>+<alt>+p"))
                print(
                    f"Screen: {self.screen.ocr_name}, "
                    f"{'paused' if self.screen.paused else 'active'} "
                    f"({hotkey} pauses)"
                )

                if screen_cfg.get("warmup", True):
                    self.screen.warm()

                self.screen_hotkey = PauseHotkey(
                    self.screen,
                    hotkey=hotkey,
                    on_change=lambda _paused: (
                        self.screen_tray.refresh()
                        if self.screen_tray is not None
                        else None
                    ),
                )
                self.screen_hotkey.start()

                if screen_cfg.get("tray", True):
                    try:
                        self.screen_tray = ScreenTray(self.screen)
                        self.screen_tray.start()
                    except Exception as exc:  # noqa: BLE001 - tray is optional
                        self.screen_tray = None
                        _logger.warning("Tray icon unavailable: %s", exc)

                self.screen_history = ScreenHistory(
                    self.screen,
                    config={"history": (screen_cfg.get("history") or {})},
                )
                self.screen_history.start()
            except Exception as exc:  # noqa: BLE001 - chat works without eyes
                self.screen = None
                self.screen_history = None
                self.screen_tray = None
                self.screen_hotkey = None
                _logger.error("Screen reading unavailable: %s", exc)
                print(f"Screen reading unavailable: {exc}")

        # Hold-to-talk. Failure here must not stop typed chat.
        voice_cfg = self.cfg.get("voice") or {}
        if voice_cfg.get("enabled", True):
            try:
                from jarvis.voice_input import HoldToTalk, TranscriptQueue

                self.transcripts = TranscriptQueue()
                self.voice = HoldToTalk(
                    self.cfg,
                    on_transcript=self.transcripts.put,
                    on_record_start=(self.speech.stop if self.speech else None),
                )
                self.voice.start()
            except Exception as exc:  # noqa: BLE001 - mic/hotkey problems
                self.voice = None
                self.transcripts = None
                _logger.error("Voice input unavailable: %s", exc)
                print(f"Voice input unavailable: {exc}")

        # Reminders run on plain code: start them even if the model is slow.
        task_cfg = self.cfg.get("tasks") or {}
        if task_cfg.get("enabled", True):
            try:
                from jarvis.scheduler import ReminderScheduler

                self.task_service = task_tools.get_service()
                self.scheduler = ReminderScheduler(self.task_service, task_cfg)
                self.scheduler.start()
                print("Reminders are running (Ctrl+C to stop).")
            except Exception as exc:  # noqa: BLE001 - chat still works without it
                _logger.error("Could not start reminders: %s", exc)
                print(f"Reminders unavailable: {exc}")
        
        try:
            while True:
                # Start listening animation
                self.ascii_ui.start_listening_animation()
                
                user_input, source = read_line("\n> ", self.transcripts)
                
                # Stop listening animation
                self.ascii_ui.stop_listening_animation()
                
                if user_input is None:
                    print("\nGoodbye!")
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.startswith("/"):
                    if not self.handle_command(user_input):
                        break
                    continue

                turn_started = time.perf_counter()
                if self.speech:
                    self.speech.start_turn(turn_started)

                # Start thinking animation during processing
                self.ascii_ui.start_thinking_animation()
                
                print("\nJarvis:", end="", flush=True)

                try:
                    async for chunk in self.get_response_stream(user_input):
                        print(chunk, end="", flush=True)
                        if self.speech:
                            self.speech.feed(chunk)
                    print()

                    # Stop thinking animation
                    self.ascii_ui.stop_thinking_animation()
                    
                    if self.speech:
                        await self.speech.flush()
                        if self.speech.first_audio_ms is not None:
                            _logger.info(
                                "First audio %.0f ms after the request",
                                self.speech.first_audio_ms,
                            )
                    _logger.info(
                        "Turn answered by %s/%s (tier %s)",
                        self.current_provider_name,
                        self.current_model_name,
                        self.last_reason,
                    )
                except Exception as e:
                    # Stop thinking animation on error
                    self.ascii_ui.stop_thinking_animation()
                    _logger.error("Error processing message: %s", e)
                    print(f"\nSorry, I could not answer that: {e}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
        finally:
            self.ascii_ui.stop_listening_animation()
            self.ascii_ui.stop_thinking_animation()
            if self.speech:
                self.speech.stop()
            if self.screen_history:
                self.screen_history.stop()
            if self.screen_hotkey:
                self.screen_hotkey.stop()
            if self.screen_tray:
                self.screen_tray.stop()
            if self.voice:
                self.voice.stop()
            if self.scheduler:
                self.scheduler.shutdown()
            if self.task_service:
                self.task_service.close()
            if self.provider_chain:
                await self.provider_chain.aclose()


def main():
    """Main entry point for the chat."""
    # Windows consoles default to cp1252, which chokes on typographic
    # characters that models produce (non-breaking hyphens, smart quotes).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # Silence background tick and HTTP noise from interrupting terminal prompt
    for noisy_logger in ("apscheduler", "httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    session = ChatSession()
    
    # Show we're using the new ASCII UI
    print("Using enhanced ASCII terminal interface")
    print("=" * 50)

    try:
        asyncio.run(session.run())
    except Exception as e:
        _logger.error("Failed to start chat: %s", e)
        print(f"Failed to start chat: {e}")


if __name__ == "__main__":
    main()
