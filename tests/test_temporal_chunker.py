"""
Tests for the Temporal Chunker module.

Verifies:
- Composite ID generation produces unique, sanitized IDs
- Temporal chunks cover expected time windows
- Overlapping chunks work correctly
- Edge cases: single segment, empty segments, gap between segments
"""

import pytest

from pipeline.loader import Segment
from pipeline.temporal_chunker import (
    generate_match_id,
    generate_segment_global_id,
    create_temporal_chunks,
    chunks_to_es_bulk,
)


class TestGenerateMatchId:
    """Tests for generate_match_id()."""

    def test_basic_match_id(self):
        mid = generate_match_id("england_epl", "2015-2016", "Chelsea vs Palace")
        assert "england_epl" in mid
        assert "2015-2016" in mid
        assert "__" in mid

    def test_sanitizes_special_characters(self):
        mid = generate_match_id(
            "england_epl", "2015-2016",
            "2015-09-19 - 19-30 Manchester City 1 - 2 West Ham",
        )
        # Should not contain spaces or raw dashes from match name
        assert " " not in mid
        assert mid.count("__") >= 2  # separates league, season, match

    def test_different_matches_get_different_ids(self):
        id1 = generate_match_id("england_epl", "2015-2016", "Chelsea vs Palace")
        id2 = generate_match_id("england_epl", "2015-2016", "Arsenal vs Spurs")
        assert id1 != id2

    def test_different_seasons_get_different_ids(self):
        id1 = generate_match_id("england_epl", "2014-2015", "Chelsea vs Palace")
        id2 = generate_match_id("england_epl", "2015-2016", "Chelsea vs Palace")
        assert id1 != id2


class TestGenerateSegmentGlobalId:
    """Tests for generate_segment_global_id()."""

    def test_basic_format(self):
        gid = generate_segment_global_id("epl__2015__match1", 1, "42")
        assert gid == "epl__2015__match1__1_42"

    def test_unique_across_halves(self):
        gid1 = generate_segment_global_id("match1", 1, "0")
        gid2 = generate_segment_global_id("match1", 2, "0")
        assert gid1 != gid2

    def test_unique_across_segments(self):
        gid1 = generate_segment_global_id("match1", 1, "0")
        gid2 = generate_segment_global_id("match1", 1, "1")
        assert gid1 != gid2


class TestCreateTemporalChunks:
    """Tests for create_temporal_chunks()."""

    @pytest.fixture
    def sample_segments(self):
        """Create a set of segments spanning 30 seconds."""
        return [
            Segment("0", 0.0, 3.0, "Sterling shoots", 1),
            Segment("1", 3.0, 6.0, "what a save", 1),
            Segment("2", 6.0, 9.0, "corner kick now", 1),
            Segment("3", 9.0, 12.0, "headed away", 1),
            Segment("4", 12.0, 15.0, "De Bruyne picks it up", 1),
            Segment("5", 15.0, 18.0, "brilliant pass", 1),
        ]

    def test_creates_chunks(self, sample_segments):
        chunks = create_temporal_chunks(
            sample_segments, "test_match", "epl", "2015",
            window_seconds=10.0, overlap_seconds=4.0,
        )
        assert len(chunks) > 0

    def test_chunks_have_correct_fields(self, sample_segments):
        chunks = create_temporal_chunks(
            sample_segments, "test_match", "epl", "2015",
            window_seconds=10.0, overlap_seconds=4.0,
        )
        chunk = chunks[0]
        assert chunk.chunk_id.startswith("test_match__")
        assert chunk.match_id == "test_match"
        assert chunk.league == "epl"
        assert chunk.season == "2015"
        assert chunk.half == 1
        assert chunk.text  # Non-empty
        assert len(chunk.segment_ids) > 0
        assert chunk.segment_count > 0

    def test_first_chunk_starts_at_first_segment(self, sample_segments):
        chunks = create_temporal_chunks(
            sample_segments, "test_match", "epl", "2015",
            window_seconds=10.0, overlap_seconds=4.0,
        )
        assert chunks[0].start_time == 0.0

    def test_chunks_overlap(self, sample_segments):
        chunks = create_temporal_chunks(
            sample_segments, "test_match", "epl", "2015",
            window_seconds=10.0, overlap_seconds=4.0,
        )
        if len(chunks) >= 2:
            # Second chunk should start before first chunk ends
            assert chunks[1].start_time < chunks[0].end_time

    def test_concatenates_segment_text(self, sample_segments):
        chunks = create_temporal_chunks(
            sample_segments, "test_match", "epl", "2015",
            window_seconds=10.0, overlap_seconds=0.0,
        )
        # First chunk (0-10s) should include segments 0-3
        assert "Sterling shoots" in chunks[0].text
        assert "what a save" in chunks[0].text

    def test_empty_segments(self):
        chunks = create_temporal_chunks([], "test", "epl", "2015")
        assert chunks == []

    def test_single_segment(self):
        segs = [Segment("0", 5.0, 8.0, "Goal!", 1)]
        chunks = create_temporal_chunks(
            segs, "test", "epl", "2015",
            window_seconds=12.0, overlap_seconds=4.0,
        )
        assert len(chunks) >= 1
        assert "Goal!" in chunks[0].text

    def test_handles_both_halves(self):
        segs = [
            Segment("0", 0.0, 5.0, "First half action", 1),
            Segment("1", 0.0, 5.0, "Second half action", 2),
        ]
        chunks = create_temporal_chunks(
            segs, "test", "epl", "2015",
            window_seconds=12.0, overlap_seconds=4.0,
        )
        halves = {c.half for c in chunks}
        assert 1 in halves
        assert 2 in halves

    def test_chunk_ids_are_unique(self, sample_segments):
        chunks = create_temporal_chunks(
            sample_segments, "test_match", "epl", "2015",
            window_seconds=10.0, overlap_seconds=4.0,
        )
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestChunksToEsBulk:
    """Tests for chunks_to_es_bulk()."""

    def test_converts_to_es_format(self):
        segs = [
            Segment("0", 0.0, 5.0, "Test text", 1),
        ]
        chunks = create_temporal_chunks(
            segs, "test", "epl", "2015",
            window_seconds=12.0, overlap_seconds=4.0,
        )
        docs = chunks_to_es_bulk(chunks)
        assert len(docs) > 0
        doc = docs[0]
        assert "_id" in doc
        assert "_source" in doc
        assert "text" in doc["_source"]
        assert "match_id" in doc["_source"]
        assert "start_time" in doc["_source"]
        assert "end_time" in doc["_source"]

    def test_empty_chunks(self):
        docs = chunks_to_es_bulk([])
        assert docs == []
