"""Build a schema-3 ASR JSON with per-segment n-best alternatives.

faster-whisper's API exposes ``beam_size`` for internal beam search but
returns only the 1-best transcription per segment. To get true n-best
hypotheses for ``pipeline/nbest_reranker.py`` we run the model multiple
times with different decoding ``temperature`` values, then time-align
each alternative run against the primary run and persist the
alternative texts in the segment's ``nbest`` field.

Output schema (extends schema-2 with ``schema_version=3`` and ``nbest``):

    {"schema_version": 3,
     "language": "en",
     "segments": {
         "0": {
             "start": 0.0, "end": 8.0,
             "text": "<primary text>",
             "avg_logprob": -0.18, ...
             "words": [...],
             "nbest": ["<alt-1 text>", "<alt-2 text>"]
         }, ...}}

The reranker reads ``nbest`` and picks the alternative with the highest
gazetteer-entity-grounding score (Apple RAG-NEC pattern,
arxiv:2409.06062).

Usage:

    python tools/build_nbest_chelsea.py \\
        --audio-dir whisper_cache/audio/chelsea \\
        --out-dir 'path/to/SoccerNet/.../commentary_data' \\
        --temperatures 0.0 0.4

The --temperatures values control diversity. T=0.0 is greedy (matches
the primary V3 transcription); T>0 introduces sampling diversity. A
two-temperature run gives 2 alternatives per segment, which is
sufficient for the reranker to demonstrate value without doubling
transcription wall time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.whisper_runner import (  # noqa: E402
    transcribe,
    build_initial_prompt,
    build_hotwords_string,
)


def _time_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Fraction of segment a covered by segment b (in [0, 1])."""
    if a_end <= a_start:
        return 0.0
    overlap = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return overlap / (a_end - a_start)


def merge_alt_into_primary(
    primary: dict,
    alt: dict,
    overlap_threshold: float = 0.3,
) -> dict:
    """For each primary segment, attach the best-overlapping alt text to ``nbest``.

    The two transcribe runs almost never produce identical segment
    boundaries — Whisper segments by silence + heuristics that vary with
    temperature. So we time-align: a primary segment "owns" any alt
    segment that overlaps it by ≥ overlap_threshold of its duration.
    Multiple alt segments can attach to one primary; we concatenate them.
    """
    primary_segs = primary["segments"]
    alt_segs = alt["segments"]

    # Pre-extract alt segments as a sorted list for linear scan.
    alt_list = sorted(
        ((s.get("start", 0.0), s.get("end", 0.0), s.get("text", ""))
         for s in alt_segs.values()),
        key=lambda x: x[0],
    )

    for sid, seg in primary_segs.items():
        p_start = seg.get("start", 0.0)
        p_end = seg.get("end", 0.0)
        chunks: list[str] = []

        for a_start, a_end, a_text in alt_list:
            if a_end < p_start:
                continue
            if a_start > p_end:
                break
            if _time_overlap(p_start, p_end, a_start, a_end) >= overlap_threshold:
                if a_text and a_text != seg.get("text"):
                    chunks.append(a_text)

        if chunks:
            existing: list[str] = list(seg.get("nbest") or [])
            merged = " ".join(chunks).strip()
            if merged and merged not in existing:
                existing.append(merged)
            seg["nbest"] = existing

    primary["schema_version"] = 3
    return primary


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-dir", type=Path, required=True,
                   help="Dir containing half1.mp3 and half2.mp3")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Match commentary_data dir for {1,2}_asr_v3_nbest.json")
    p.add_argument("--labels", type=Path, default=None,
                   help="Labels-caption.json for hotwords/initial_prompt")
    p.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.4],
                   help="Temperatures to run; first is the primary 1-best")
    p.add_argument("--language", default="en")
    p.add_argument("--device", default="cpu")
    p.add_argument("--beam-size", type=int, default=5)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.out_dir / ".nbest_runs"
    work_dir.mkdir(exist_ok=True)

    initial_prompt = ""
    hotwords = ""
    if args.labels and args.labels.exists():
        initial_prompt = build_initial_prompt(args.labels)
        hotwords = build_hotwords_string(args.labels)

    halves = {
        1: args.audio_dir / "half1.mp3",
        2: args.audio_dir / "half2.mp3",
    }

    for half, audio in halves.items():
        if not audio.exists():
            print(f"[skip] half {half}: {audio} not found", file=sys.stderr)
            continue

        primary_path = work_dir / f"half{half}_T{args.temperatures[0]:.1f}.json"
        per_temp_paths = [primary_path]

        for i, temp in enumerate(args.temperatures):
            out = work_dir / f"half{half}_T{temp:.1f}.json"
            if out.exists():
                print(f"[half {half}][T={temp}] reusing existing {out}")
                per_temp_paths.append(out) if i > 0 else None
                continue
            t0 = time.perf_counter()
            transcribe(
                audio_path=audio,
                output_path=out,
                initial_prompt=initial_prompt,
                hotwords=hotwords,
                language=args.language,
                device=args.device,
                beam_size=args.beam_size,
                temperature=temp,
            )
            print(f"[half {half}][T={temp}] {time.perf_counter() - t0:.1f}s")
            if i > 0:
                per_temp_paths.append(out)

        # Build the merged schema-3 file
        primary = json.load(open(primary_path, encoding="utf-8"))
        for alt_path in per_temp_paths[1:]:
            alt = json.load(open(alt_path, encoding="utf-8"))
            merge_alt_into_primary(primary, alt)

        merged_out = args.out_dir / f"{half}_asr_v3_nbest.json"
        with open(merged_out, "w", encoding="utf-8") as f:
            json.dump(primary, f, indent=2, ensure_ascii=False)
        n_with_nbest = sum(1 for s in primary["segments"].values() if s.get("nbest"))
        print(f"[half {half}] merged → {merged_out} "
              f"({len(primary['segments'])} segs, {n_with_nbest} with nbest)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
