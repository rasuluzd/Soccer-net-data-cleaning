"""
Duplicate Segment Remover — detects and merges consecutive duplicate segments.

Whisper often produces the same text repeated across multiple consecutive
segments (e.g., "De Bruyne" appears 6 times in a row). This module keeps
only the first occurrence and extends its time span to cover all duplicates.
"""

from dataclasses import replace

from rapidfuzz import fuzz

from pipeline.config import DUPLICATE_SIMILARITY_THRESHOLD
from pipeline.loader import Segment


def _normalize_for_dedup(text: str) -> str:
    """Strip surrounding whitespace and trailing punctuation.

    This makes "Hansson.", "Hansson!", "Hansson," all dedup-equivalent —
    they're the same word from Whisper's perspective, just with different
    punctuation. Without this, the fuzz.ratio comparison distinguishes
    them and they slip past dedup individually.
    """
    return text.strip().rstrip(".,!?;:").strip()


def deduplicate_segments(
    segments: list[Segment],
    threshold: int = DUPLICATE_SIMILARITY_THRESHOLD,
) -> tuple[list[Segment], list[dict]]:
    """
    Remove consecutive segments with near-identical text.

    When duplicates are found:
    - Keep the FIRST occurrence
    - Extend its end_time to the last duplicate's end_time
    - Log which segments were removed

    Args:
        segments: list of Segments (should already be time-sorted)
        threshold: similarity score (0–100) above which segments are "duplicates"

    Returns:
        Tuple of:
        - deduped: list of Segment objects with duplicates removed
        - removed: list of dicts with info about removed segments
    """
    if not segments:
        return [], []

    deduped: list[Segment] = []
    removed: list[dict] = []

    # Start with the first segment
    current = replace(segments[0])

    for i in range(1, len(segments)):
        seg = segments[i]

        # Only compare within the same half
        if seg.half != current.half:
            deduped.append(current)
            current = replace(seg)
            continue

        # Compare current segment's text with the previous one
        similarity = fuzz.ratio(
            _normalize_for_dedup(current.text),
            _normalize_for_dedup(seg.text),
        )

        if similarity >= threshold:
            # This is a duplicate — extend the time span but don't keep it
            current.end_time = max(current.end_time, seg.end_time)
            removed.append({
                "segment_id": seg.segment_id,
                "half": seg.half,
                "start_time": seg.start_time,
                "text": seg.text[:80],
                "duplicate_of": current.segment_id,
                "similarity": similarity,
            })
        else:
            # Not a duplicate — save current and move on
            deduped.append(current)
            current = replace(seg)

    # Don't forget the last segment
    deduped.append(current)

    return deduped, removed
