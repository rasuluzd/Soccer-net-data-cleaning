"""Language-aware Whisper transcription via faster-whisper.

Output format (simple list-based, no schema versioning):
  {"segments": {"0": [start, end, text], "1": [start, end, text], ...}}

This matches the SoccerNet-Echoes-style layout used by downstream tools
that expect the simpler schema. Per-segment metadata (avg_logprob,
no_speech_prob, etc.) and per-word probabilities are NOT emitted in this
output format — if those are needed for downstream confidence-gated steps,
use the schema-2 variant of this module instead.

Decoder configuration matches the v3 baseline that produced the evaluated
results: lenient filtering thresholds calibrated for stadium-audio (high
no_speech_prob is the norm there), with name biasing handled via
initial_prompt and hotwords from Labels-caption.json.
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
    condition_on_previous_text: bool = False,
    temperature: float = 0.0,
) -> Path:
    """Transcribe one audio file with faster-whisper and write a simple
    list-based JSON: {"segments": {"0": [start, end, text], ...}}.

    Decoder defaults match the v3 baseline (no_speech_threshold=0.95,
    log_prob_threshold=None, vad_filter=False, condition_on_previous_text=False)
    which is calibrated for stadium-audio characteristics.
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
        word_timestamps=False,
        condition_on_previous_text=condition_on_previous_text,
        no_speech_threshold=no_speech_threshold,
        log_prob_threshold=log_prob_threshold,
        compression_ratio_threshold=compression_ratio_threshold,
        temperature=temperature,
    )

    segments: dict[str, list] = {}
    kept = 0
    for seg in seg_iter:
        text = (seg.text or "").strip()
        if not text:
            continue
        segments[str(kept)] = [float(seg.start), float(seg.end), text]
        kept += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {"segments": segments}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(
        f"Wrote {len(segments)} segments to {output_path} "
        f"(detected language: {info.language}, prob {info.language_probability:.2f})"
    )
    return output_path