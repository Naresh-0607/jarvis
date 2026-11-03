"""Tests for the Phase 4 speech pipeline: cleaning, splitting, streaming, barge-in."""

import asyncio
import sys
import threading
import time
import types

import numpy as np
import pytest

from jarvis.speech import (
    CODE_NOTICE,
    FILLERS,
    SpeechOutput,
    clean_for_speech,
    normalise_numbers,
    play_audio,
    split_sentences,
    strip_code_blocks,
)
from jarvis.tts import (
    DEFAULT_VOICE,
    EdgeTTSEngine,
    FallbackTTS,
    PiperEngine,
    TTSError,
    build_engine,
)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------
def test_clean_strips_markdown_and_links():
    raw = "## Done\n- **Bold** point with a [link](https://example.com/x)\n- `inline code`"
    cleaned, had_code = clean_for_speech(raw)

    assert "##" not in cleaned
    assert "**" not in cleaned
    assert "https://" not in cleaned
    assert "link" in cleaned
    assert "inline code" in cleaned
    assert had_code is False


def test_clean_replaces_urls_with_a_short_phrase():
    cleaned, _ = clean_for_speech("See https://example.com/very/long?path=1 for that.")
    assert "the link on screen" in cleaned
    assert "http" not in cleaned


def test_clean_removes_emoji_and_tables():
    cleaned, _ = clean_for_speech("Nice work 🎉🚀 | done |")
    assert "🎉" not in cleaned
    assert "🚀" not in cleaned
    assert "|" not in cleaned


def test_strip_code_blocks_reports_and_removes_fences():
    text = "Before.\n```python\nprint('hi')\n```\nAfter."
    stripped, had_code = strip_code_blocks(text)

    assert had_code is True
    assert "print" not in stripped
    assert "```" not in stripped
    assert "Before." in stripped and "After." in stripped


def test_strip_code_blocks_drops_an_unterminated_block():
    stripped, had_code = strip_code_blocks("Talking now.\n```python\nprint('hi')")
    assert had_code is True
    assert "print" not in stripped
    assert "Talking now." in stripped


# ---------------------------------------------------------------------------
# Natural numbers
# ---------------------------------------------------------------------------
def test_times_become_twelve_hour():
    assert normalise_numbers("due at 17:30") == "due at 5:30 PM"
    assert normalise_numbers("starts 09:05") == "starts 9:05 AM"
    assert normalise_numbers("at 12:00") == "at 12 PM"
    assert normalise_numbers("midnight 00:15") == "midnight 12:15 AM"


def test_times_already_marked_am_pm_are_untouched():
    assert normalise_numbers("at 5:30 PM") == "at 5:30 PM"


def test_iso_dates_are_spoken():
    assert normalise_numbers("by 2026-10-29") == "by 29 October 2026"


def test_decimals_are_untouched():
    assert normalise_numbers("3.5 hours") == "3.5 hours"


def test_money_and_percent():
    assert normalise_numbers("₹1,200 only") == "1200 rupees only"
    assert normalise_numbers("Rs. 500") == "500 rupees"
    assert normalise_numbers("$20") == "20 dollars"
    assert normalise_numbers("15% off") == "15 percent off"


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------
def test_split_returns_complete_sentences_and_the_tail():
    sentences, rest = split_sentences("One thing. Two things! A third")
    assert sentences == ["One thing.", "Two things!"]
    assert rest == "A third"


def test_split_keeps_decimals_together():
    sentences, rest = split_sentences("Pi is about 3.14 today")
    assert sentences == []
    assert rest == "Pi is about 3.14 today"


def test_split_waits_for_space_after_terminator():
    sentences, rest = split_sentences("Ready.")
    assert sentences == []
    assert rest == "Ready."


def test_split_does_not_break_on_abbreviations():
    sentences, rest = split_sentences("Ask Dr. Rao about it.")
    assert sentences == []
    assert "Dr. Rao" in rest


def test_split_handles_newlines():
    sentences, rest = split_sentences("First line\nsecond line")
    assert sentences == ["First line"]
    assert rest == "second line"


def test_split_caps_runaway_sentences():
    long_text = "word " * 60
    sentences, rest = split_sentences(long_text, max_chars=40)
    assert sentences
    assert all(len(s) <= 45 for s in sentences)
    assert len(rest) <= 40


# ---------------------------------------------------------------------------
# Streaming pipeline
# ---------------------------------------------------------------------------
class FakeEngine:
    name = "fake"

    def __init__(self, delay: float = 0.0, error: str = ""):
        self.spoken = []
        self.delay = delay
        self.error = error

    async def synth(self, text):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error:
            raise TTSError(self.error)
        self.spoken.append(text)
        return np.zeros(16, dtype=np.float32), 16000


class FakePlayer:
    def __init__(self):
        self.played = []

    def __call__(self, samples, rate, stop_event=None):
        self.played.append(rate)


def make_speech(engine, player=None, **speech_cfg):
    cfg = {"enabled": True}
    cfg.update(speech_cfg)
    return SpeechOutput(
        {"speech": cfg},
        engine=engine,
        player=player or FakePlayer(),
        log=lambda message: None,
    )


def test_sentences_are_spoken_before_the_stream_finishes():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)
        speech.start_turn()

        speech.feed("The first sentence is here. ")
        await asyncio.sleep(0.1)
        midway = list(engine.spoken)

        speech.feed("And the second arrives later.")
        await speech.flush()
        return midway, engine.spoken

    midway, final = asyncio.run(scenario())
    assert midway == ["The first sentence is here."]
    assert final == [
        "The first sentence is here.",
        "And the second arrives later.",
    ]


def test_each_sentence_is_played_in_order():
    async def scenario():
        engine = FakeEngine()
        player = FakePlayer()
        speech = make_speech(engine, player)
        speech.start_turn()
        for chunk in ["One. ", "Two. ", "Three."]:
            speech.feed(chunk)
        await speech.flush()
        return engine.spoken, player.played

    spoken, played = asyncio.run(scenario())
    assert spoken == ["One.", "Two.", "Three."]
    assert played == [16000, 16000, 16000]


def test_streamed_code_is_never_spoken():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)
        speech.start_turn()
        speech.feed("Here is the fix.\n```python\nprint('secret')\n")
        speech.feed("```\nThat should do it.")
        await speech.flush()
        return engine.spoken

    spoken = asyncio.run(scenario())
    assert "print('secret')" not in " ".join(spoken)
    assert "Here is the fix." in spoken
    assert CODE_NOTICE in spoken
    assert spoken[-1] == "That should do it."


def test_code_notice_is_said_only_once_per_turn():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)
        speech.start_turn()
        speech.feed("First.\n```a\n```\n")
        speech.feed("Second.\n```b\n```\n")
        await speech.flush()
        return engine.spoken

    spoken = asyncio.run(scenario())
    assert spoken.count(CODE_NOTICE) == 1


def test_markdown_and_numbers_are_cleaned_before_synthesis():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)
        speech.start_turn()
        speech.feed("**Deadline:** 2026-10-29 at 17:30.")
        await speech.flush()
        return engine.spoken

    spoken = asyncio.run(scenario())
    assert spoken == ["Deadline: 29 October 2026 at 5:30 PM."]


def test_muted_speech_stays_silent_and_recovers():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)

        speech.mute()
        speech.start_turn()
        speech.feed("You should not hear this.")
        await speech.flush()
        muted = list(engine.spoken)

        speech.unmute()
        speech.start_turn()
        speech.feed("Now you can hear this.")
        await speech.flush()
        return muted, engine.spoken

    muted, after = asyncio.run(scenario())
    assert muted == []
    assert after == ["Now you can hear this."]


def test_first_audio_latency_is_recorded():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)
        started = time.perf_counter()
        speech.start_turn(started)
        await asyncio.sleep(0.05)
        speech.feed("Quick reply.")
        await speech.flush()
        return speech.first_audio_ms

    first_audio = asyncio.run(scenario())
    assert first_audio is not None
    assert first_audio >= 40


def test_fillers_are_short_and_not_repeated_back_to_back():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine)
        speech.start_turn()
        for _ in range(3):
            speech.maybe_filler()
            await asyncio.sleep(0.05)
        await speech.flush()
        return engine.spoken

    spoken = asyncio.run(scenario())
    assert len(spoken) == 3
    assert all(line in FILLERS for line in spoken)
    assert len(set(spoken)) == 3


def test_fillers_can_be_disabled():
    async def scenario():
        engine = FakeEngine()
        speech = make_speech(engine, filler=False)
        speech.start_turn()
        speech.maybe_filler()
        await speech.flush()
        return engine.spoken

    assert asyncio.run(scenario()) == []


def test_barge_in_drops_queued_sentences_and_stops_playback():
    async def scenario():
        engine = FakeEngine()
        played = []
        gate = threading.Event()

        def player(samples, rate, stop_event=None):
            played.append(rate)
            if len(played) == 1:
                gate.wait(2.0)

        speech = make_speech(engine, player)
        speech.start_turn()
        speech.feed("First one. Second one. Third one.")

        deadline = time.time() + 2.0
        while not played and time.time() < deadline:
            await asyncio.sleep(0.01)

        speech.stop()  # the hotkey hook does exactly this
        gate.set()
        await speech.flush()
        return played, engine.spoken

    played, spoken = asyncio.run(scenario())
    assert len(played) == 1
    assert spoken == ["First one."]


def test_barge_in_from_a_worker_thread_is_safe():
    async def scenario():
        engine = FakeEngine()
        player = FakePlayer()

        def record(samples, rate, stop_event=None):
            time.sleep(0.03)
            player(samples, rate, stop_event)

        speech = make_speech(engine, record)
        speech.start_turn()
        speech.feed("Hear me out. There is more.")
        await asyncio.sleep(0.05)

        thread = threading.Thread(target=speech.stop)
        thread.start()
        thread.join(2.0)
        await speech.flush()
        return player.played

    # No exception, and the loop is still usable afterwards.
    asyncio.run(scenario())


def test_synthesis_failure_is_reported_and_does_not_raise():
    async def scenario():
        engine = FakeEngine(error="no network")
        logs = []
        speech = SpeechOutput(
            {"speech": {"enabled": True}},
            engine=engine,
            player=FakePlayer(),
            log=logs.append,
        )
        speech.start_turn()
        speech.feed("This will fail.")
        await speech.flush()
        return logs

    logs = asyncio.run(scenario())
    assert any("could not speak" in line for line in logs)


def test_disabled_speech_output_does_nothing():
    speech = SpeechOutput({"speech": {"enabled": False}}, player=FakePlayer())
    assert speech.enabled is False
    speech.start_turn()
    speech.feed("Anything at all.")
    asyncio.run(speech.flush())
    assert speech.first_audio_ms is None


def test_player_with_two_arguments_is_supported():
    async def scenario():
        engine = FakeEngine()
        played = []
        speech = SpeechOutput(
            {"speech": {}},
            engine=engine,
            player=lambda samples, rate: played.append(rate),
            log=lambda message: None,
        )
        speech.start_turn()
        speech.feed("Hello there.")
        await speech.flush()
        return played

    assert asyncio.run(scenario()) == [16000]


# ---------------------------------------------------------------------------
# Engines and fallback
# ---------------------------------------------------------------------------
def test_build_engine_disabled_returns_none():
    assert build_engine({"speech": {"enabled": False}}) is None


def test_build_engine_defaults_to_edge_without_a_local_model():
    engine = build_engine({"speech": {}})
    assert isinstance(engine, EdgeTTSEngine)
    assert engine.voice == DEFAULT_VOICE


def test_build_engine_explicit_edge():
    engine = build_engine({"speech": {"engine": "edge", "voice": "en-GB-SoniaNeural"}})
    assert isinstance(engine, EdgeTTSEngine)
    assert engine.voice == "en-GB-SoniaNeural"


def test_build_engine_auto_prefers_piper_when_installed(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not really a model")

    engine = build_engine(
        {"speech": {"engine": "auto", "piper_model": str(model)}}
    )
    assert isinstance(engine, FallbackTTS)
    assert isinstance(engine.primary, PiperEngine)
    assert isinstance(engine.backup, EdgeTTSEngine)
    assert engine.name == "piper+edge"


def test_build_engine_piper_without_a_model_degrades_to_edge():
    engine = build_engine({"speech": {"engine": "piper"}})
    assert isinstance(engine, EdgeTTSEngine)


def test_piper_engine_reports_availability_and_label(tmp_path):
    model = tmp_path / "alba-medium.onnx"
    engine = PiperEngine(model=str(model))
    assert engine.available is False
    model.write_bytes(b"x")
    assert engine.available is True
    assert engine.voice == "alba-medium"
    assert PiperEngine(model="").available is False


def test_piper_engine_concatenates_chunks(monkeypatch, tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    seen = {}

    class FakeChunk:
        def __init__(self, array):
            self.audio_float_array = array

    class FakeVoice:
        config = types.SimpleNamespace(sample_rate=22050)

        def synthesize(self, text, syn_config=None):
            seen["text"] = text
            seen["length_scale"] = syn_config.length_scale
            yield FakeChunk(np.full(4, 0.25, dtype=np.float32))
            yield FakeChunk(np.full(6, 0.25, dtype=np.float32))

    class FakePiperVoice:
        @classmethod
        def load(cls, path, config_path=None):
            seen["model"] = path
            return FakeVoice()

    class FakeConfig:
        def __init__(self, speaker_id=None, length_scale=None):
            self.length_scale = length_scale

    fake = types.ModuleType("piper")
    fake.PiperVoice = FakePiperVoice
    fake.SynthesisConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "piper", fake)

    engine = PiperEngine(model=str(model), length_scale=1.1)
    samples, rate = asyncio.run(engine.synth("hello there"))

    assert seen["text"] == "hello there"
    assert seen["model"] == str(model)
    assert seen["length_scale"] == pytest.approx(1.1)
    assert rate == 22050
    assert samples.shape == (10,)
    assert 0.24 < float(samples[0]) < 0.26


def test_piper_engine_speaks_nothing_for_blank_text(tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    with pytest.raises(TTSError, match="nothing to say"):
        asyncio.run(PiperEngine(model=str(model)).synth("  "))


def test_piper_warm_loads_once_in_the_background(monkeypatch, tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    loads = {"count": 0}

    class FakeVoice:
        config = types.SimpleNamespace(sample_rate=22050)

        def synthesize(self, text, syn_config=None):
            yield type("C", (), {"audio_float_array": np.zeros(2, dtype=np.float32)})()

    class FakePiperVoice:
        @classmethod
        def load(cls, path, config_path=None):
            loads["count"] += 1
            return FakeVoice()

    fake = types.ModuleType("piper")
    fake.PiperVoice = FakePiperVoice
    fake.SynthesisConfig = lambda **kwargs: None
    monkeypatch.setitem(sys.modules, "piper", fake)

    engine = PiperEngine(model=str(model))
    engine.warm()
    engine._warm_thread.join(timeout=5)

    assert loads["count"] == 1
    engine.warm()  # already loaded, must not reload
    assert loads["count"] == 1


def test_fallback_uses_the_backup_when_primary_fails():
    class Broken:
        name = "broken"

        async def synth(self, text):
            raise TTSError("down")

    class Good:
        name = "good"

        def __init__(self):
            self.said = []

        async def synth(self, text):
            self.said.append(text)
            return np.ones(4, dtype=np.float32), 22050

    good = Good()
    engine = FallbackTTS(Broken(), good)
    samples, rate = asyncio.run(engine.synth("hello"))

    assert good.said == ["hello"]
    assert rate == 22050
    assert samples.shape == (4,)
    assert engine.using_backup is True


def test_fallback_raises_when_there_is_no_backup():
    class Broken:
        name = "broken"

        async def synth(self, text):
            raise TTSError("down")

    with pytest.raises(TTSError):
        asyncio.run(FallbackTTS(Broken()).synth("hello"))


def test_edge_engine_collects_only_audio_chunks(monkeypatch):
    seen = {}

    class FakeCommunicate:
        def __init__(self, text, voice, rate=None, pitch=None, volume=None):
            seen["text"] = text
            seen["voice"] = voice
            seen["rate"] = rate

        async def stream(self):
            yield {"type": "audio", "data": b"abc"}
            yield {"type": "WordBoundary", "offset": 0}
            yield {"type": "audio", "data": b"def"}

    fake_edge = types.ModuleType("edge_tts")
    fake_edge.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge)

    import soundfile

    def fake_read(buffer, dtype=None, always_2d=False):
        assert buffer.read() == b"abcdef"
        return np.zeros(10, dtype=np.float32), 24000

    monkeypatch.setattr(soundfile, "read", fake_read)

    engine = EdgeTTSEngine(voice="en-IN-PrabhatNeural", rate="-5%")
    samples, rate = asyncio.run(engine.synth("hello there"))

    assert seen["text"] == "hello there"
    assert seen["voice"] == "en-IN-PrabhatNeural"
    assert seen["rate"] == "-5%"
    assert rate == 24000
    assert samples.shape == (10,)


def test_edge_engine_reports_empty_audio(monkeypatch):
    class FakeCommunicate:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            yield {"type": "WordBoundary", "offset": 0}

    fake_edge = types.ModuleType("edge_tts")
    fake_edge.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", fake_edge)

    with pytest.raises(TTSError, match="no audio"):
        asyncio.run(EdgeTTSEngine().synth("hello"))


def test_edge_engine_rejects_blank_text():
    with pytest.raises(TTSError, match="nothing to say"):
        asyncio.run(EdgeTTSEngine().synth("   "))


def test_play_audio_returns_immediately_for_empty_samples():
    play_audio(np.zeros(0, dtype=np.float32), 24000)


def test_play_audio_plays_a_silent_clip_without_blocking_forever():
    play_audio(np.zeros(240, dtype=np.float32), 24000, threading.Event())
