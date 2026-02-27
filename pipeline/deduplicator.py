"""
Duplicate Segment Remover — detects and merges consecutive duplicate segments.

Whisper often produces the same text repeated across multiple consecutive
segments (e.g., "De Bruyne" appears 6 times in a row). This module keeps
only the first occurrence and extends its time span to cover all duplicates.
"""

from rapidfuzz import fuzz

from pipeline.config import DUPLICATE_SIMILARITY_THRESHOLD
from pipeline.loader import Segment


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
    current = Segment(
        segment_id=segments[0].segment_id,
        start_time=segments[0].start_time,
        end_time=segments[0].end_time,
        text=segments[0].text,
        half=segments[0].half,
    )

    for i in range(1, len(segments)):
        seg = segments[i]

        # Only compare within the same half
        if seg.half != current.half:
            deduped.append(current)
            current = Segment(
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                half=seg.half,
            )
            continue

        # Compare current segment's text with the previous one
        similarity = fuzz.ratio(current.text.strip(), seg.text.strip())

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
            current = Segment(
                segment_id=seg.segment_id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                half=seg.half,
            )

    # Don't forget the last segment
    deduped.append(current)

    return deduped, removed
