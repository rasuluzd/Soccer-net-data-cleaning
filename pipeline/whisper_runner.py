"""
pipeline/whisper_runner.py — language-aware Whisper transcription via faster-whisper.

Provides the low-level transcription primitive used by both the CLI
``tools/retranscribe_kb_whisper.py`` and the orchestrator-facing
``pipeline.asr_router.transcribe_match_half``. Centralising it here keeps
``pipeline/`` self-contained (it does not import from ``tools/``).

Per-language model defaults live in ``pipeline.config.ASR_MODELS``; the
``initial_prompt`` builder reads team and player names from
``Labels-caption.json`` and orders them tail-first because Whisper
truncates the prompt to its last 224 tokens.

Output schema (versioned for backward compatibility):

    schema=1: {"segments": {"0": [start, end, text], ...}}
    schema=2: {"schema_version": 2,
               "language": "en",
               "language_probability": 0.99,
               "segments": {
                   "0": {
                       "start": 0.0,
                       "end": 8.0,
                       "text": "And away we go ...",
                       "avg_logprob": -0.18,        # segment-level avg
                       "no_speech_prob": 0.01,
                       "compression_ratio": 1.4,
                       "temperature": 0.0,
                       "words": [
                           {"word": "And", "start": 0.0, "end": 0.2, "prob": 0.99},
                           ...
                       ]
                   }, ...}}

The orchestrator's loader handles both schemas. Schema-2 enables the
SOTA refactor's confidence-gated GER and downstream confidence-aware
event detection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline.config import (
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_COMPUTE_TYPE_CPU,
    WHISPER_COMPUTE_TYPE_GPU,
    get_asr_model,
)


WHISPER_OUTPUT_SCHEMA_VERSION = 2


def _ordered_lineup_names(labels_path: Path) -> list[str]:
    """Read Labels-caption.json and return [teams..., long_names..., short_names...].

    Order matters because Whisper truncates prompts and hotwords at half
    of max_length: highest-value tokens (player short_names) go LAST so
    they survive truncation.
    """
    if not labels_path.exists():
        return []
    with open(labels_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    teams: list[str] = [t for t in labels.get("teams", []) if t]
    short_names: list[str] = []
    long_names: list[str] = []
    for side in ("home", "away"):
        lineup = labels.get("lineup", {}).get(side, {})
        for p in lineup.get("players", []):
            short = (p.get("short_name") or "").strip()
            long_ = (p.get("long_name") or "").strip()
            if short and short not in short_names:
                short_names.append(short)
            if long_ and long_ not in long_names:
                long_names.append(long_)
        for coach in lineup.get("coach", []):
            long_ = (coach.get("long_name") or "").strip()
            if long_ and long_ not in long_names:
                long_names.append(long_)
    seen: set[str] = set()
    deduped: list[str] = []
    for n in teams + long_names + short_names:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    return deduped


def build_initial_prompt(labels_path: Path, max_names: int = 40) -> str:
    """Build an initial_prompt biasing string from Labels-caption.json.

    Whisper truncates the prompt to the *last* 224 tokens, so the highest-
    value tokens (player short_names) go at the end. Used for first-window
    stylistic conditioning; combine with ``build_hotwords_string`` for
    per-window name biasing.
    """
    deduped = _ordered_lineup_names(labels_path)
    if not deduped:
        return ""
    return "Fotbollsmatch. Namn: " + ", ".join(deduped[:max_names]) + "."


def build_hotwords_string(labels_path: Path, max_names: int = 40) -> str:
    """Comma-separated lineup names for faster-whisper's ``hotwords`` arg.

    Unlike ``initial_prompt`` (only consumed for the first decode window),
    ``hotwords`` is prepended to the prompt of *every* window, so name
    tokens stay biasable across the whole match. The string is plain
    "Name1, Name2, Name3" — no header, no period — to maximise the share
    of name tokens within the per-window truncation budget (max_length//2).
    """
    deduped = _ordered_lineup_names(labels_path)
    if not deduped:
        return ""
    return ", ".join(deduped[:max_names])


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _pick_compute_type(device: str) -> str:
    return WHISPER_COMPUTE_TYPE_GPU if device == "cuda" else WHISPER_COMPUTE_TYPE_CPU


def transcribe(
    audio_path: Path,
    output_path: Path,
    initial_prompt: str = "",
    hotwords: str = "",
    model_name: str | None = None,
    language: str = "sv",
    device: str = "auto",
    beam_size: int = WHISPER_BEAM_SIZE,
    best_of: int = WHISPER_BEST_OF,
    vad_filter: bool = False,
    no_speech_threshold: float = 0.95,
    log_prob_threshold: float | None = None,
    compression_ratio_threshold: float = 2.6,
    word_timestamps: bool = True,
    condition_on_previous_text: bool = False,
    temperature: float = 0.0,
) -> Path:
    """Transcribe one audio file with faster-whisper and write Whisper JSON.

    With ``word_timestamps=True`` (default), output uses schema 2 with per-word
    probability and per-segment avg_logprob — required by the SOTA refactor's
    confidence-gated GER (``pipeline/llm_corrector.py``).

    ``initial_prompt`` is consumed by faster-whisper only for the FIRST decode
    window, while ``hotwords`` is prepended to the prompt of every window —
    use the latter to keep player-name tokens biasable across the full match.
    Both are plain text; faster-whisper tokenises them internally. Token budget
    is ~half of max_length per window, so the prompt builders order names
    so the highest-value (short_names) survive truncation.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        print(
            f"ERROR: faster-whisper not installed ({e}). "
            "Run: pip install faster-whisper",
            file=sys.stderr,
        )
        sys.exit(1)

    if model_name is None:
        model_name = get_asr_model(language)
    device = _resolve_device(device)
    compute_type = _pick_compute_type(device)

    print(
        f"Loading {model_name} via faster-whisper "
        f"(device={device}, compute_type={compute_type})..."
    )
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    if initial_prompt:
        print(f"Using initial_prompt biasing ({len(initial_prompt)} chars).")
    if hotwords:
        n_names = hotwords.count(",") + 1 if hotwords else 0
        print(f"Using hotwords biasing ({len(hotwords)} chars, {n_names} names).")

    print(f"Transcribing {audio_path} (temperature={temperature}) ...")
    seg_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        best_of=best_of,
        initial_prompt=initial_prompt or None,
        hotwords=hotwords or None,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,  # 0.95 (vs default 0.6) keeps soft commentary
        log_prob_threshold=log_prob_threshold,    # None disables low-confidence drop
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,                  # 0.0 = greedy; non-zero gives diverse hypotheses for n-best building
    )

    segments: dict[str, dict] = {}
    for i, seg in enumerate(seg_iter):
        text = (seg.text or "").strip()
        if not text:
            continue
        seg_dict: dict = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
            "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
            "compression_ratio": float(getattr(seg, "compression_ratio", 0.0) or 0.0),
            "temperature": float(getattr(seg, "temperature", 0.0) or 0.0),
        }
        if word_timestamps and getattr(seg, "words", None):
            seg_dict["words"] = [
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "prob": float(w.probability),
                }
                for w in seg.words
            ]
        segments[str(i)] = seg_dict

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "schema_version": WHISPER_OUTPUT_SCHEMA_VERSION,
        "language": info.language,
        "language_probability": float(info.language_probability),
        "segments": segments,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(
        f"Wrote {len(segments)} segments to {output_path} "
        f"(detected language: {info.language}, prob {info.language_probability:.2f})"
    )
    return output_path
