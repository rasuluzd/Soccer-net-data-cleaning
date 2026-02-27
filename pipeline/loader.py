"""
ASR Data Loader — discovers and loads all match data from the SoccerNet dataset.

Walks the dataset directory tree, finds ASR JSON files and Labels-caption.json,
and parses them into uniform Python data structures (Segment, MatchData).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pipeline.config import DATASET_ROOT


@dataclass
class Segment:
    """One ASR transcript segment (a chunk of commentary)."""
    segment_id: str       # original key from JSON ("0", "1", ...)
    start_time: float     # start timestamp in seconds
    end_time: float       # end timestamp in seconds
    text: str             # raw ASR transcription text
    half: int             # 1 or 2 (which half of the match)


@dataclass
class MatchData:
    """All data for a single match."""
    match_dir: Path              # absolute path to the match directory
    match_name: str              # human-readable folder name (e.g. "2015-09-19 - ...")
    league: str                  # e.g. "england_epl"
    season: str                  # e.g. "2015-2016"
    segments: list[Segment]      # all ASR segments (both halves, time-sorted)
    labels: Optional[dict] = None  # parsed Labels-caption.json (None if missing)


def load_asr_json(filepath: Path, half: int) -> list[Segment]:
    """
    Parse a single *_asr.json file into a list of Segment objects.

    The JSON format is:
        {"segments": {"0": [start, end, "text"], "1": [start, end, "text"], ...}}

    Args:
        filepath: path to 1_asr.json or 2_asr.json
        half: 1 or 2

    Returns:
        List of Segment objects sorted by start time.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg_id, values in data.get("segments", {}).items():
        # Each value is [start_time, end_time, "text"]
        if len(values) >= 3:
            segments.append(Segment(
                segment_id=seg_id,
                start_time=float(values[0]),
                end_time=float(values[1]),
                text=str(values[2]),
                half=half,
            ))

    # Sort by start time to ensure correct order
    segments.sort(key=lambda s: (s.half, s.start_time))
    return segments


def load_labels(match_dir: Path) -> Optional[dict]:
    """
    Load Labels-caption.json for a match directory.

    Returns:
        Parsed JSON dict, or None if the file doesn't exist.
    """
    labels_path = match_dir / "Labels-caption.json"
    if not labels_path.exists():
        return None

    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_matches(dataset_root: Path = DATASET_ROOT) -> list[MatchData]:
    """
    Walk the dataset directory tree and discover all matches that have
    ASR transcription files.

    Expected directory structure:
        dataset_root/
            league/
                season/
                    match_folder/
                        commentary_data/
                            1_asr.json
                            2_asr.json
                        Labels-caption.json

    Returns:
        List of MatchData objects, one per match that has ASR data.
    """
    matches = []

    if not dataset_root.exists():
        print(f"[WARNING] Dataset root not found: {dataset_root}")
        return matches

    # Walk: league -> season -> match -> commentary_data
    for league_dir in sorted(dataset_root.iterdir()):
        if not league_dir.is_dir():
            continue
        league_name = league_dir.name  # e.g. "england_epl"

        for season_dir in sorted(league_dir.iterdir()):
            if not season_dir.is_dir():
                continue
            season_name = season_dir.name  # e.g. "2015-2016"

            for match_dir in sorted(season_dir.iterdir()):
                if not match_dir.is_dir():
                    continue

                commentary_dir = match_dir / "commentary_data"
                half1_path = commentary_dir / "1_asr.json"
                half2_path = commentary_dir / "2_asr.json"

                # Skip matches without any ASR data
                if not (half1_path.exists() or half2_path.exists()):
                    continue

                # Load segments from both halves
                all_segments = []
                if half1_path.exists():
                    all_segments.extend(load_asr_json(half1_path, half=1))
                if half2_path.exists():
                    all_segments.extend(load_asr_json(half2_path, half=2))

                # Load labels (may be None)
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
