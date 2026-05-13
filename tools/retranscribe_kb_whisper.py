"""
CLI wrapper around ``pipeline.whisper_runner.transcribe`` for one-off
re-transcriptions of a single match half. Use this for ablation studies
or when you want to override the per-language default model in
``pipeline.config.ASR_MODELS``.

The default model per language comes from ``pipeline.config.ASR_MODELS``:
    - sv → KBLab/kb-whisper-large
    - de → primeline/whisper-large-v3-turbo-german
    - en → openai/whisper-large-v3
Override with ``--model``.

Usage:
    # Auto-resolve paths by match substring (Swedish AIK match, half 1):
    python tools/retranscribe_kb_whisper.py --match "AIK" --half 1

    # Explicit paths:
    python tools/retranscribe_kb_whisper.py \\
        --audio path/to/half1.mp3 \\
        --labels path/to/Labels-caption.json \\
        --output path/to/1_asr_kb.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pipeline.config import (  # noqa: E402
    DATASET_ROOT,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
)
from pipeline.whisper_runner import (  # noqa: E402
    build_hotwords_string,
    build_initial_prompt,
    transcribe,
)


def resolve_by_match(match_substring: str, half: int) -> tuple[Path, Path, Path]:
    """Find (audio, labels, output) paths for a given match and half."""
    for asr_path in DATASET_ROOT.rglob(f"{half}_asr.json"):
        match_parent = asr_path.parent
        match_dir = (
            match_parent.parent if match_parent.name == "commentary_data"
            else match_parent
        )
        if match_substring.lower() not in match_dir.name.lower():
            continue
        # Prefer per-match audio under whisper_cache/audio/<match>/halfN.mp3.
        # Falls back to the legacy shared-cache layout for compatibility.
        per_match_audio = (
            _REPO_ROOT / "whisper_cache" / "audio" / match_dir.name / f"half{half}.mp3"
        )
        legacy_audio = _REPO_ROOT / "whisper_cache" / "audio" / f"half{half}.mp3"
        audio = per_match_audio if per_match_audio.exists() else legacy_audio
        labels = match_dir / "Labels-caption.json"
        output = match_parent / f"{half}_asr_kb.json"
        return audio, labels, output
    raise FileNotFoundError(f"No match found matching '{match_substring}'")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--audio", type=Path, help="Path to audio file")
    p.add_argument("--labels", type=Path, help="Path to Labels-caption.json")
    p.add_argument("--output", type=Path, help="Output JSON path")
    p.add_argument("--match", type=str, help="Match name substring (auto-resolves paths)")
    p.add_argument("--half", type=int, default=1, choices=[1, 2])
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "HuggingFace model id. Default picks per --language from "
            "pipeline.config.ASR_MODELS (sv → KBLab/kb-whisper-large)."
        ),
    )
    p.add_argument("--language", type=str, default="sv")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto picks cuda when available",
    )
    p.add_argument("--beam-size", type=int, default=WHISPER_BEAM_SIZE)
    p.add_argument("--best-of", type=int, default=WHISPER_BEST_OF)
    p.add_argument("--vad-filter", action="store_true",
                   help="Enable Silero VAD filtering (default: off — VAD has been "
                        "observed to drop chunks of soft commentary)")
    p.add_argument("--no-prompt", action="store_true",
                   help="Disable initial_prompt biasing")
    p.add_argument("--condition-on-previous", action="store_true",
                   help="Pass previous segment text as Whisper context "
                        "(default off; helps when large-v3 drops fragmented speech).")
    args = p.parse_args()

    if args.match:
        audio, labels, output = resolve_by_match(args.match, args.half)
    else:
        if not (args.audio and args.output):
            print("ERROR: provide --match OR (--audio and --output)", file=sys.stderr)
            sys.exit(2)
        audio = args.audio
        labels = args.labels or Path()
        output = args.output

    if not audio.exists():
        print(f"ERROR: audio file not found: {audio}", file=sys.stderr)
        sys.exit(2)

    prompt = "" if args.no_prompt else build_initial_prompt(labels)
    hotwords = "" if args.no_prompt else build_hotwords_string(labels)
    transcribe(
        audio_path=audio,
        output_path=output,
        initial_prompt=prompt,
        hotwords=hotwords,
        model_name=args.model,
        language=args.language,
        device=args.device,
        beam_size=args.beam_size,
        best_of=args.best_of,
        vad_filter=args.vad_filter,
        condition_on_previous_text=args.condition_on_previous,
    )


if __name__ == "__main__":
    main()
