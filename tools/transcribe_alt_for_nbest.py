"""Transcribe one half at temperature 0.4 with PER-SEGMENT progress logging.

Used to build the n-best alternative transcripts for Step N. The primary
T=0.0 transcript is the existing 1_asr_v3.json — this script only
generates the diverse alternative.

Differences from `pipeline.whisper_runner.transcribe`:

- Streams every decoded segment immediately to stdout with `flush=True`
  so we can follow progress (vs the silent 3+ hour run).
- `beam_size=1` (5x faster than beam=5; T=0.4 already provides the
  diversity we need for the n-best, beam search is redundant).
- No hotwords (we WANT name-spelling diversity for the alternative; the
  primary T=0.0 already has hotword-biased canonical spellings).
- Bypasses the `WhisperModel.transcribe` generator's hidden buffering by
  consuming the iterator one segment at a time and printing
  start/end/text immediately.

Run twice: once per half. Each run prints progress to stdout AND writes
the partial JSON every N segments so a crash doesn't lose all work.

Usage:
    python -u tools/transcribe_alt_for_nbest.py \\
        --audio whisper_cache/audio/chelsea/half1.mp3 \\
        --out  'path/to/.../commentary_data/.nbest_runs/half1_T0.4.json' \\
        --temperature 0.4 --beam 1 --language en
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.config import (  # noqa: E402
    WHISPER_COMPUTE_TYPE_CPU,
    WHISPER_COMPUTE_TYPE_GPU,
    get_asr_model,
)


def _now() -> str:
    return time.strftime("%H:%M:%S")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--temperature", type=float, default=0.4)
    p.add_argument("--beam", type=int, default=1)
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cpu")
    p.add_argument("--save-every", type=int, default=50,
                   help="Persist partial JSON every N segments")
    args = p.parse_args()

    print(f"[{_now()}] launching transcribe", flush=True)
    print(f"  audio: {args.audio} ({args.audio.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"  out:   {args.out}", flush=True)
    print(f"  T={args.temperature}  beam={args.beam}  lang={args.language}  device={args.device}",
          flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{_now()}] importing faster_whisper ...", flush=True)
    from faster_whisper import WhisperModel
    print(f"[{_now()}] faster_whisper imported", flush=True)

    model_name = get_asr_model(args.language)
    compute = (
        WHISPER_COMPUTE_TYPE_GPU if args.device == "cuda"
        else WHISPER_COMPUTE_TYPE_CPU
    )
    print(f"[{_now()}] loading model {model_name} ({compute}) ...", flush=True)
    t0 = time.perf_counter()
    model = WhisperModel(model_name, device=args.device, compute_type=compute)
    print(f"[{_now()}] model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    print(f"[{_now()}] starting transcription (this is where the previous run was silent)",
          flush=True)
    t_start = time.perf_counter()
    seg_iter, info = model.transcribe(
        str(args.audio),
        language=args.language,
        beam_size=args.beam,
        best_of=args.beam,
        # No initial_prompt + no hotwords on this pass — we want diversity
        # for the n-best alternative; the primary T=0.0 already used them.
        initial_prompt=None,
        hotwords=None,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        no_speech_threshold=0.95,
        log_prob_threshold=None,
        compression_ratio_threshold=2.6,
        temperature=args.temperature,
    )
    print(f"[{_now()}] transcribe() returned generator "
          f"(detected lang={info.language}, prob={info.language_probability:.2f}). "
          f"Will print one line per segment as it decodes.", flush=True)

    segments: dict[str, dict] = {}
    last_save = 0
    for i, seg in enumerate(seg_iter):
        text = (seg.text or "").strip()
        if not text:
            continue
        segments[str(i)] = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": text,
            "avg_logprob": float(getattr(seg, "avg_logprob", 0.0) or 0.0),
            "no_speech_prob": float(getattr(seg, "no_speech_prob", 0.0) or 0.0),
            "compression_ratio": float(getattr(seg, "compression_ratio", 0.0) or 0.0),
            "temperature": float(getattr(seg, "temperature", 0.0) or 0.0),
        }
        if getattr(seg, "words", None):
            segments[str(i)]["words"] = [
                {"word": w.word, "start": float(w.start),
                 "end": float(w.end), "prob": float(w.probability)}
                for w in seg.words
            ]

        wall = time.perf_counter() - t_start
        rt_factor = wall / max(seg.end, 0.01)
        print(f"[{_now()}] seg {i:4d} [{seg.start:7.2f}-{seg.end:7.2f}s]"
              f" wall={wall:6.1f}s ({rt_factor:.2f}x rt) | {text[:80]}",
              flush=True)

        # Periodic save
        if i - last_save >= args.save_every:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({
                    "schema_version": 2,
                    "language": info.language,
                    "language_probability": float(info.language_probability),
                    "segments": segments,
                    "in_progress": True,
                }, f, indent=2, ensure_ascii=False)
            last_save = i
            print(f"[{_now()}] partial save → {args.out.name} ({len(segments)} segs)",
                  flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"[{_now()}] decode complete: {len(segments)} segments in {elapsed:.1f}s",
          flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "schema_version": 2,
            "language": info.language,
            "language_probability": float(info.language_probability),
            "segments": segments,
        }, f, indent=2, ensure_ascii=False)
    print(f"[{_now()}] final write → {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
