"""Audition voices, compare presets, and inspect candidates.

    python scripts/audition_voices.py --list             # edge-tts English voices
    python scripts/audition_voices.py                    # play the edge shortlist
    python scripts/audition_voices.py --piper-list       # local Piper models
    python scripts/audition_voices.py --dark-candidates  # candidate dark male Piper models
    python scripts/audition_voices.py --preset shadow    # render shadow preset WAV
    python scripts/audition_voices.py --preset calm      # render calm preset WAV
    python scripts/audition_voices.py --compare          # render and compare calm vs shadow
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from jarvis.speech import play_audio
from jarvis.tts import (
    AUDITION_VOICES,
    SAMPLE_SENTENCE,
    EdgeTTSEngine,
    PiperEngine,
    build_engine,
    get_preset_config,
)
from jarvis.voice_fx import VoiceFXChain, VoiceFXConfig

PIPER_DIR = project_root / "models" / "piper"
SAMPLES_DIR = project_root / "data" / "voice_samples"

def list_piper() -> None:
    models = local_models()
    if not models:
        print(f"No Piper models in {PIPER_DIR}")
        print("Download a .onnx and .onnx.json pair from rhasspy/piper-voices.")
    else:
        print(f"Piper models in {PIPER_DIR}:")
        for path in models:
            size = path.stat().st_size / 1e6
            print(f"  {path.stem:<20} {size:6.1f} MB")


async def say(voice: str, text: str, rate: str) -> None:
    engine = EdgeTTSEngine(voice=voice, rate=rate)
    print(f"speaking as {voice} ({rate}) ...")
    samples, sample_rate = await engine.synth(text)
    play_audio(samples, sample_rate)


async def audition(rate: str) -> None:
    for voice in AUDITION_VOICES:
        await say(voice, SAMPLE_SENTENCE, rate)
        await asyncio.sleep(0.2)

    print(
        "\nPick one and put it in config.yaml under speech.voice, then run\n"
        "  python -m jarvis\n"
        "and ask a question to hear it in place."
    )


async def say_piper(name: str, text: str) -> None:
    if name == "all":
        targets = local_models()
    else:
        stem = Path(name).stem
        targets = [PIPER_DIR / f"{stem}.onnx"]

    if not targets:
        print(f"No Piper models in {PIPER_DIR}")
        return

    for path in targets:
        if not path.exists():
            print(f"missing: {path}")
            continue
        engine = PiperEngine(model=str(path), length_scale=1.05)
        print(f"speaking as {path.stem} (piper) ...")
        samples, rate = await engine.synth(text)
        play_audio(samples, rate)
        await asyncio.sleep(0.2)


async def synth_preset(
    preset_name: str, text: str, play: bool = False
) -> Tuple[np.ndarray, int, Path]:
    cfg = load_config()
    preset_cfg = get_preset_config(cfg, preset_name)
    engine = build_engine(cfg, preset=preset_name)
    if engine is None:
        raise RuntimeError(f"Could not build TTS engine for preset '{preset_name}'")

    print(f"Synthesising preset '{preset_name}' ({engine.name})...")
    raw_samples, sample_rate = await engine.synth(text)

    # Apply FX
    fx_dict = preset_cfg.get("fx")
    fx = VoiceFXChain(VoiceFXConfig.from_dict(fx_dict))
    processed_samples, processed_rate = fx.process(raw_samples, sample_rate)

    # Save to data/voice_samples/
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLES_DIR / f"{preset_name}_test.wav"

    try:
        import soundfile as sf
        sf.write(str(out_path), processed_samples, processed_rate)
    except ImportError:
        from scipy.io import wavfile
        pcm16 = np.clip(processed_samples * 32767.0, -32768, 32767).astype(np.int16)
        wavfile.write(str(out_path), processed_rate, pcm16)

    print(f"Wrote {processed_samples.size / processed_rate:.2f}s WAV to {out_path}")

    if play:
        play_audio(processed_samples, processed_rate)

    return processed_samples, processed_rate, out_path


async def compare_presets(text: str) -> None:
    cfg = load_config()
    preset_names = list(cfg.get("voice", {}).get("presets", {}).keys()) or ["calm"]
    print(f"\nAuditioning and comparing presets on sample sentence:\n\"{text}\"\n")
    results = {}
    for name in preset_names:
        samples, rate, out_path = await synth_preset(name, text, play=False)
        duration = samples.size / rate
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        results[name] = {
            "samples": samples,
            "rate": rate,
            "path": out_path,
            "duration": duration,
            "peak": peak,
            "rms": rms,
        }

    print("\n" + "=" * 80)
    print(f"{'Preset':<10} {'Duration':<10} {'Peak Amp':<10} {'RMS':<10} {'Pitch':<10} {'High Cut':<12} {'Echo (ms)'}")
    print("-" * 80)
    for name in preset_names:
        r = results[name]
        p_cfg = get_preset_config(cfg, name)
        fx = p_cfg.get("fx") or {}
        pitch = f"{fx.get('semitones', 0.0):+.1f} st" if fx.get("enabled") else "None"
        highcut = f"{fx.get('high_cut_hz', 0):.0f} Hz" if fx.get("enabled") else "None"
        echo = str(fx.get("echo_delays_ms", [])) if fx.get("enabled") else "None"
        print(
            f"{name:<10} {r['duration']:6.2f}s    {r['peak']:6.3f}     {r['rms']:6.3f}    "
            f"{pitch:<10} {highcut:<12} {echo}"
        )
    print("=" * 80 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list edge-tts English voices")
    parser.add_argument("--piper-list", action="store_true", help="list local Piper models")
    parser.add_argument("--piper", help="play a local Piper model, or 'all'")
    parser.add_argument("--voice", help="speak the sample with this edge-tts voice")
    parser.add_argument("--preset", default="calm", help="audition a configured preset")
    parser.add_argument("--compare", action="store_true", help="compare configured presets side-by-side")
    parser.add_argument("--play", action="store_true", help="play generated audio out loud")
    parser.add_argument(
        "--say", default=SAMPLE_SENTENCE, help="text to speak or synthesise"
    )
    parser.add_argument("--rate", default="-10%", help="edge-tts rate, e.g. -10%%")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    if args.compare:
        asyncio.run(compare_presets(args.say))
    elif args.preset and args.preset != "calm":
        asyncio.run(synth_preset(args.preset, args.say, play=args.play))
    elif args.piper_list:
        list_piper()
    elif args.piper:
        asyncio.run(say_piper(args.piper, args.say))
    elif args.list:
        asyncio.run(list_voices())
    elif args.voice:
        asyncio.run(say(args.voice, args.say, args.rate))
    else:
        asyncio.run(synth_preset("calm", args.say, play=args.play))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
