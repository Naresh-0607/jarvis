# Jarvis - a personal assistant for Windows 11

A calm, spoken-style assistant with an online/local model chain. Phases 1-8 of
the build guide are complete: the brain, tasks/reminders, voice input,
voice output, screen OCR eyes, project folder reader, memory, and general tools (weather, web, time, math).

## Getting started

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. Install the requirements:

   ```powershell
   pip install -r requirements.txt
   ```

3. Fill in `.env` with your keys:

   ```
   OLLAMA_BASE_URL=https://<your-tunnel>/v1
   OLLAMA_AUTH_TOKEN=
   GROQ_API_KEY=...
   NTFY_TOPIC=https://ntfy.sh/<your-topic>
   ```

4. Run the health check:

   ```powershell
   python -m jarvis.doctor
   ```

5. Start the chat:

   ```powershell
   python -m jarvis
   ```

## What the chat can do

| Command | What it does |
| --- | --- |
| `/fast` | Lock the cheap/small model for this session |
| `/strong` | Lock the reasoning model for this session |
| `/auto` | Let the rule-based router pick a tier per message |
| `/provider` | Show which provider and model answered last |
| `/force <name>` | Restrict to one provider (`/force auto` clears it) |
| `/status` | Mode, tier, provider cooldowns, model lists |
| `/clear` | Forget the conversation so far |
| `/quit` | Exit |

Ask things like:

- "Remind me to call the dentist tomorrow at 5." Jarvis reads the deadline
  back before saving when the time was vague.
- "What's pending?", "anything today?", "what did I miss?" - answered from the
  database, never from conversation memory.
- "Delete the dentist task." Jarvis asks first, then deletes.

## How it works

### Provider chain (Phase 1)

- One OpenAI-compatible transport serves Groq, Gemini, Cerebras, OpenRouter and
  the local Ollama tunnel (`kind: openai_compat` / `kind: ollama`).
- Real SSE streaming: text is printed as the model produces it.
- Fallback happens per model and per provider. A 429 cools down that model only;
  a hard 5xx or timeout cools down the provider. Retired models are skipped
  instead of crashing.
- On startup each provider's live model list is fetched and configured models
  that no longer exist are dropped with a warning. A missing backup key is a
  warning, not a dead end.
- Two tiers, `fast` and `strong`, chosen by a small rule table in
  `jarvis/router.py` (no model decides routing). `tiers:` in `config.yaml` says
  which model serves which tier.
- Every turn logs which provider and model answered.

### Tasks and reminders (Phase 2)

- SQLite at `data/jarvis.db` with `tasks` and `reminders_sent` tables.
- Reminder slots: 1 day before, 1 hour before, 10 minutes before, at the
  deadline. Important tasks also get 3 hours before. Slots that are already in
  the past when the task is created are skipped.
- While a task is overdue and open, a nudge repeats every
  `repeat_overdue_minutes` (default 30).
- APScheduler runs a short tick every `tick_seconds` (default 20) and a daily
  morning briefing. Jobs live in `data/scheduler.db` (SQLite job store).
- Anything that came due while the laptop was off is announced once, grouped
  into a single message, on the first tick after startup.
- Delivery is console plus an ntfy.sh push. A failed push is logged and never
  blocks the reminder.
- `reminders_sent` is the single source of truth for "already fired", so
  nothing double-fires across restarts.

### Voice input (Phase 3)

- Hold `Ctrl+Alt+Space` in any window, speak, release. The transcript is fed
  into the same brain as typed text; typing still works.
- Capture is 16 kHz mono, opened only while the key is held, trimmed with an
  adaptive silence gate (no torch dependency).
- Transcription goes to Groq `whisper-large-v3-turbo` through
  `/audio/transcriptions`, with a one-time retry on network errors and a
  plain "could not hear you" message on failure.
- Status lines tell you what happened: too short, only silence with the level
  reading, or the transcript timing.
- Setting `voice.stt_backend` to `local` switches to faster-whisper if you
  install it; it is off by default.
- The hotkey, device, language, prompt vocabulary and thresholds all live
  under `voice:` in `config.yaml`.

### Voice out (Phase 4)

- Replies are spoken sentence by sentence as the model streams, so audio
  starts after the first sentence instead of after the whole answer. The log
  line `first audio after N ms` measures the request start to the first sound.
- `Piper` is the primary engine: local, offline, and in-process via
  `piper-tts`, so there is no network handshake between you and the audio. The
  medium `alba` voice synthesises a sentence in 50-190 ms on CPU. The model
  lives in `models/piper/` and is set with `speech.piper_model`.
- `edge-tts` (`en-GB-SoniaNeural`) stays as the fallback for when the local
  model is missing or fails; `FallbackTTS` switches back to Piper once it
  recovers. `speech.engine` accepts `auto` (Piper first), `edge`, or `piper`.
- The model is loaded and primed with one tiny synthesis at startup
  (`SpeechOutput.warm()`), because the first ONNX inference is ~1.4 s slower
  than every one after it.
- Measured on this machine: Piper warm synthesis 50-190 ms per sentence, first
  audio ~0.9-1.3 s end to end (the rest is the model provider's time to first
  token). The old edge-tts path paid ~1.1 s of WebSocket handshake per
  sentence, which is why Piper replaced it.
- Speech text is cleaned first: code fences become "I've put the code on your
  screen", inline code keeps its words, links become "the link on screen",
  markdown and emoji are dropped, and times, ISO dates, money and percentages
  are expanded (`17:30` → "5:30 PM", `2026-10-29` → "29 October 2026").
- Hold `Ctrl+Alt+Space` while Jarvis is talking to cut it off (barge-in). The
  queued sentences are dropped and the mic opens immediately.
- Tools that take longer than 0.7 s get a short filler line ("One moment.")
  so the silence is not confusing. Set `speech.filler: false` to disable.
- `/mute` and `/unmute` silence spoken replies for a while; typed chat keeps
  working either way.
- Audition local voices with `python scripts\audition_voices.py --piper all`,
  or list them with `--piper-list`. For cloud voices use `--list` and
  `--voice en-GB-SoniaNeural`, then copy the winner into `config.yaml`.

## Tests

```powershell
pip install -r requirements-dev.txt
python -m pytest tests -q
```

Manual checks against the real providers:

```powershell
python scripts\smoke_chat.py          # streaming, tiers, tool calling
python scripts\smoke_tasks.py         # task engine, persistence, grouping
python scripts\smoke_tasks.py --notify  # also sends one push to your phone
python scripts\audition_voices.py     # hear the shortlist, then pick one
```

## Layout

```
jarvis/
  __main__.py     # python -m jarvis  /  python -m jarvis.doctor
  chat.py         # terminal chat loop, tier switches, tool loop
  providers.py    # streaming transport, Cooldown, Chain fallback
  router.py       # fast/strong rules
  tools.py        # tool registry
  task_tools.py   # add/list/complete/snooze/delete tools
  tasks.py        # reminder maths + task service
  db.py           # SQLite schema
  scheduler.py    # APScheduler, missed reminders, briefing
  notify.py       # console + ntfy delivery
  doctor.py       # health check
  speech.py       # text cleaning, sentence splitting, queued playback
  tts.py          # in-process Piper, edge-tts fallback, engine selection
  audio.py        # resample, adaptive silence trim, WAV encoding
  stt.py          # Groq whisper (+ optional local faster-whisper)
  voice_input.py  # hold-to-talk hotkey and capture
  console_input.py # non-blocking typed input next to voice
  project_reader.py # read-only project tools (list_files, read_file, search_text)
  screen.py       # view-only screen capture and RapidOCR
  screen_tools.py # look_at_screen, describe_screen_image
  tray.py         # screen status tray icon
  memory.py       # facts, projects, lessons, and lookup_docs tools
  general_tools.py # get_weather, web_search, current_time, calculate
  persona.md      # system prompt / voice
docs/             # local hardware documentation (ESP32 reference)
config.yaml       # providers, tiers, task, voice and speech settings
models/piper/     # local Piper voices (.onnx + .onnx.json)
scripts/          # manual smoke tests, voice audition
tests/            # pytest suite
data/             # databases (created at runtime)
logs/             # log output
```

## Status

- Phase 0: done (skeleton, doctor, config).
- Phase 1: done (streaming, fallback, tools, tiers, persona, `/fast` `/strong`).
- Phase 2: done (tasks, reminders, briefing, ntfy, tests).
- Phase 3: done (hold-to-talk, 16 kHz capture, Groq whisper, typed chat intact).
- Phase 4: done (streaming TTS, local Piper voice, barge-in, fillers, latency
  logging). Voice is `alba-medium`; audition the others with
  `scripts\audition_voices.py --piper all` if you want a change.
- Phase 5: done (view-only screen capture with mss, RapidOCR, blocklist, pause hotkey,
  tray icon, automatic screen inspection on "this error").
- Phase 6: done (read-only project reader with `list_files`, `read_file` with line slices,
  `search_text`, path traversal jailbreak protection, protected file skipping,
  automatic screen OCR error traceback linking, and repeated-error awareness).
- Phase 7: done (long-term memory with SQLite `facts`, `projects`, `lessons`,
  interactive teaching mode in persona, hardware safety checks, local `docs/`
  reference lookup with `lookup_docs`, and lesson resumption on "continue").
- Phase 8: done (general tools: Open-Meteo weather with `get_weather`, DuckDuckGo
  `web_search` with casual source attribution, safe AST `calculate`, `current_time`,
  and router tuning for fast simple lookups vs strong multi-step reasoning).
