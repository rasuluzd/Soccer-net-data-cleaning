"""Language-aware Whisper transcription via faster-whisper.

Output schema=2:
  {"schema_version": 2, "language": "en", "language_probability": 0.99,
   "segments": {"0": {"start", "end", "text", "avg_logprob",
                      "no_speech_prob", "compression_ratio", "temperature",
                      "words": [{"word", "start", "end", "prob"}, ...]}, ...}}

The loader handles both schemas. Per-language defaults live in
pipeline.config.ASR_MODELS."""

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
    """[teams..., long_names..., short_names...] from Labels-caption.json.
    short_names go LAST since Whisper truncates prompts at the back."""
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
    """initial_prompt for the first decode window only. Combine with
    build_hotwords_string for biasing across the whole match."""
    deduped = _ordered_lineup_names(labels_path)
    if not deduped:
        return ""
    return "Fotbollsmatch. Namn: " + ", ".join(deduped[:max_names]) + "."


def build_hotwords_string(labels_path: Path, max_names: int = 40) -> str:
    """Comma-separated names for faster-whisper's hotwords arg. Prepended to
    every decode window, so player names stay biasable across the whole match.
    Plain "Name1, Name2" — no header — maximises name tokens in the budget."""
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
    word_timestamps=True (default) emits schema-2 with per-word prob and
    per-segment avg_logprob — required by Step L."""
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
        temperature=temperature,                  # 0.0 = greedy
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
