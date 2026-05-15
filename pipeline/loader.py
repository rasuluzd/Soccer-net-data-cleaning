"""Discover SoccerNet matches and parse their ASR JSON + Labels into Segment / MatchData."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.config import ASR_INPUT_VARIANT, DATASET_ROOT


@dataclass
class Segment:
    """One ASR transcript segment.

    Schema-2 enrichments (words/avg_logprob/etc.) are None when the source
    JSON is the older list-style schema-1. Stages degrade gracefully then."""
    segment_id: str
    start_time: float
    end_time: float
    text: str
    half: int             # 1 or 2
    global_id: str = ""   # set when writing the cleaned output
    # Schema-2 enrichments (None for schema-1).
    words: Optional[list[dict]] = None         # [{word, start, end, prob}, ...]
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None
    nbest: Optional[list[str]] = None          # alternative beam hypotheses
    speaker_id: Optional[str] = None           # set by diarizer.py
    # Indices entity_corrector marked as canonical names. Step L must not edit them.
    frozen_word_indices: Optional[list[int]] = None


@dataclass
class MatchData:
    """One match's data: directory, segments (both halves time-sorted), labels."""
    match_dir: Path
    match_name: str
    league: str
    season: str
    segments: list[Segment]
    labels: Optional[dict] = None


def load_asr_json(filepath: Path, half: int) -> list[Segment]:
    """Parse one {half}_asr.json into Segments (time-sorted).

    Schema-1 (list-style):
      {"segments": {"0": [start, end, "text"], ...}}

    Schema-2 (faster-whisper enriched):
      {"schema_version": 2, "language": "en",
       "segments": {"0": {"start": .., "end": .., "text": ..,
                          "avg_logprob": .., "no_speech_prob": ..,
                          "words": [...], "nbest": [...]} } }
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg_id, values in data.get("segments", {}).items():
        if isinstance(values, list) and len(values) >= 3:
            segments.append(Segment(
                segment_id=seg_id,
                start_time=float(values[0]),
                end_time=float(values[1]),
                text=str(values[2]),
                half=half,
            ))
        elif isinstance(values, dict):
            segments.append(Segment(
                segment_id=seg_id,
                start_time=float(values.get("start", 0.0)),
                end_time=float(values.get("end", 0.0)),
                text=str(values.get("text", "")),
                half=half,
                words=values.get("words"),
                avg_logprob=values.get("avg_logprob"),
                no_speech_prob=values.get("no_speech_prob"),
                nbest=values.get("nbest"),
                speaker_id=values.get("speaker_id"),
            ))

    segments.sort(key=lambda s: (s.half, s.start_time))
    return segments


def load_labels(match_dir: Path) -> Optional[dict]:
    """Parse Labels-caption.json. Returns None when missing."""
    labels_path = match_dir / "Labels-caption.json"
    if not labels_path.exists():
        return None

    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_matches(dataset_root: Path = DATASET_ROOT) -> list[MatchData]:
    """Walk dataset_root/league/season/match/[commentary_data/]{1,2}_asr.json
    and return one MatchData per match with ASR data."""
    matches = []

    if not dataset_root.exists():
        print(f"[WARNING] Dataset root not found: {dataset_root}")
        return matches

    # Walk: league -> season -> match -> commentary_data
    for league_dir in sorted(dataset_root.iterdir()):
        if not league_dir.is_dir():
            continue
        league_name = league_dir.name

        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season_name = season_dir.name

            for match_dir in sorted(season_dir.iterdir()):
                if not match_dir.is_dir():
                    continue

                # Two layouts: standard SoccerNet (commentary_data/) and flat
                # (used for eval matches). ASR_INPUT_VARIANT="_kb" reads
                # KB-Whisper variants when present.
                v = ASR_INPUT_VARIANT
                commentary_dir = match_dir / "commentary_data"
                half1_path = commentary_dir / f"1_asr{v}.json"
                half2_path = commentary_dir / f"2_asr{v}.json"
                if not (half1_path.exists() or half2_path.exists()):
                    half1_path = match_dir / f"1_asr{v}.json"
                    half2_path = match_dir / f"2_asr{v}.json"
                if v and not (half1_path.exists() or half2_path.exists()):
                    # Variant requested but missing — fall back to stock files.
                    half1_path = commentary_dir / "1_asr.json"
                    half2_path = commentary_dir / "2_asr.json"
                    if not (half1_path.exists() or half2_path.exists()):
                        half1_path = match_dir / "1_asr.json"
                        half2_path = match_dir / "2_asr.json"

                if not (half1_path.exists() or half2_path.exists()):
                    continue

                all_segments = []
                if half1_path.exists():
                    all_segments.extend(load_asr_json(half1_path, half=1))
                if half2_path.exists():
                    all_segments.extend(load_asr_json(half2_path, half=2))

                labels = load_labels(match_dir)

                matches.append(MatchData(
                    match_dir=match_dir,
                    match_name=match_dir.name,
                    league=league_name,
                    season=season_name,
                    segments=all_segments,
                    labels=labels,
                ))

    return matches


if __name__ == "__main__":
    # Quick test: discover and summarize all matches
    found = discover_matches()
    print(f"Found {len(found)} match(es) with ASR data:\n")
    for m in found:
        print(f"  [{m.league} / {m.season}] {m.match_name}")
        print(f"    Segments: {len(m.segments)}")
        print(f"    Labels:   {'Yes' if m.labels else 'No'}")
        print()
