"""
Temporal Chunker — creates overlapping time-window documents for Elasticsearch.

Whisper ASR segments are typically 2-3 seconds long. A search for "De Bruyne goal"
might hit a segment that just says "goal!" while "De Bruyne" is in the segment
4 seconds prior. This module concatenates nearby segments into overlapping chunks
so Elasticsearch has complete context to index.

Each chunk contains:
    - Concatenated text from all segments within the time window
    - Start/end timestamps covering the full window
    - A globally unique ID for ES document deduplication
    - Match metadata for filtering
"""

import re
from dataclasses import dataclass

from pipeline.config import ROLLING_WINDOW_SECONDS, ROLLING_WINDOW_OVERLAP_SECONDS
from pipeline.loader import Segment


@dataclass
class TemporalChunk:
    """A time-windowed document ready for Elasticsearch indexing."""
    chunk_id: str         # Globally unique ID: {match_id}__{half}__chunk_{idx}
    match_id: str         # Composite match identifier
    league: str           # e.g. "england_epl"
    season: str           # e.g. "2015-2016"
    half: int             # 1 or 2
    start_time: float     # Window start timestamp
    end_time: float       # Window end timestamp
    text: str             # Concatenated segment text
    segment_ids: list[str]  # Source segment IDs covered
    segment_count: int    # Number of source segments


def generate_match_id(league: str, season: str, match_name: str) -> str:
    """Generate a sanitized composite match identifier.

    Format: {league}__{season}__{match_name}
    Special characters are replaced with underscores for ES compatibility.
    """
    # Sanitize match name: replace spaces, dashes, colons with underscores
    safe_name = re.sub(r"[^a-zA-Z0-9]", "_", match_name)
    # Collapse multiple underscores
    safe_name = re.sub(r"_+", "_", safe_name).strip("_")
    return f"{league}__{season}__{safe_name}"


def generate_segment_global_id(
    match_id: str, half: int, segment_id: str,
) -> str:
    """Generate a globally unique ID for a single segment.

    Format: {match_id}__{half}_{segment_id}
    """
    return f"{match_id}__{half}_{segment_id}"


def create_temporal_chunks(
    segments: list[Segment],
    match_id: str,
    league: str,
    season: str,
    window_seconds: float = ROLLING_WINDOW_SECONDS,
    overlap_seconds: float = ROLLING_WINDOW_OVERLAP_SECONDS,
) -> list[TemporalChunk]:
    """
    Create overlapping temporal chunks from cleaned segments.

    Slides a time window across the segments, concatenating text
    from all segments that fall within each window.

    Args:
        segments: cleaned segments (should be time-sorted within each half)
        match_id: composite match identifier
        league: league name
        season: season string
        window_seconds: duration of each chunk window
        overlap_seconds: overlap between consecutive windows

    Returns:
        List of TemporalChunk objects ready for ES bulk indexing
    """
    if not segments:
        return []

    chunks: list[TemporalChunk] = []
    step = window_seconds - overlap_seconds
    if step <= 0:
        step = window_seconds / 2  # Fallback: 50% overlap

    for half_num in (1, 2):
        half_segs = [s for s in segments if s.half == half_num]
        if not half_segs:
            continue

        # Find the time range for this half
        min_time = min(s.start_time for s in half_segs)
        max_time = max(s.end_time for s in half_segs)

        chunk_idx = 0
        window_start = min_time

        while window_start < max_time:
            window_end = window_start + window_seconds

            # Collect all segments overlapping with this window
            window_segs = [
                s for s in half_segs
                if s.start_time < window_end and s.end_time > window_start
            ]

            if window_segs:
                # Concatenate text
                text = " ".join(s.text.strip() for s in window_segs if s.text.strip())
                seg_ids = [s.segment_id for s in window_segs]

                chunk_id = f"{match_id}__{half_num}__chunk_{chunk_idx}"
                chunks.append(TemporalChunk(
                    chunk_id=chunk_id,
                    match_id=match_id,
                    league=league,
                    season=season,
                    half=half_num,
                    start_time=window_start,
                    end_time=window_end,
                    text=text,
                    segment_ids=seg_ids,
                    segment_count=len(window_segs),
                ))
                chunk_idx += 1

            window_start += step

    return chunks


def chunks_to_es_bulk(chunks: list[TemporalChunk]) -> list[dict]:
    """
    Convert temporal chunks to Elasticsearch bulk API format.

    Each entry is a dict ready for ES bulk indexing with:
        _id: the chunk's global ID
        _source: the document fields

    Args:
        chunks: list of TemporalChunk objects

    Returns:
        List of dicts in ES bulk format
    """
    documents = []
    for chunk in chunks:
        doc = {
            "_id": chunk.chunk_id,
            "_source": {
                "match_id": chunk.match_id,
                "league": chunk.league,
                "season": chunk.season,
                "half": chunk.half,
                "start_time": chunk.start_time,
                "end_time": chunk.end_time,
                "text": chunk.text,
                "segment_ids": chunk.segment_ids,
                "segment_count": chunk.segment_count,
            },
        }
        documents.append(doc)
    return documents
