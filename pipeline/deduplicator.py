"""Merge consecutive segments that Whisper repeated (e.g. "De Bruyne" x6)."""

from dataclasses import replace

from rapidfuzz import fuzz

from pipeline.config import DUPLICATE_SIMILARITY_THRESHOLD
from pipeline.loader import Segment


def _normalize_for_dedup(text: str) -> str:
    # Strip trailing punct so "Hansson.", "Hansson!", "Hansson," compare equal.
    return text.strip().rstrip(".,!?;:").strip()


def deduplicate_segments(
    segments: list[Segment],
    threshold: int = DUPLICATE_SIMILARITY_THRESHOLD,
) -> tuple[list[Segment], list[dict]]:
    """Drop consecutive near-identical segments. Keep the first, extend its end_time."""
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
