"""
Tests for the Duplicate Segment Remover module.

Verifies that consecutive identical/near-identical segments are merged
correctly, preserving the first occurrence and extending time spans.
"""

import pytest
from pipeline.loader import Segment
from pipeline.deduplicator import deduplicate_segments


def _seg(text: str, start: float, end: float, seg_id: str, half: int = 1) -> Segment:
    """Helper to create a Segment."""
    return Segment(segment_id=seg_id, start_time=start, end_time=end, text=text, half=half)


class TestDeduplicateSegments:
    """Tests for deduplicate_segments()."""

    def test_no_duplicates(self):
        segments = [
            _seg("First half begins", 0, 5, "0"),
            _seg("Sterling receives the ball", 5, 10, "1"),
            _seg("De Bruyne takes a shot", 10, 15, "2"),
        ]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 3
        assert len(removed) == 0

    def test_exact_duplicates(self):
        segments = [
            _seg("the great show is over.", 536, 537, "133"),
            _seg("the great show is over.", 537, 538, "134"),
            _seg("the great show is over.", 538, 539, "135"),
        ]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 1
        assert len(removed) == 2
        # First occurrence is kept
        assert deduped[0].segment_id == "133"
        # Time span should extend to last duplicate
        assert deduped[0].end_time == 539

    def test_near_duplicates(self):
        """Segments with very similar text should be merged."""
        segments = [
            _seg("and he's got a good shot at the ball", 616, 618, "152"),
            _seg("and he's got a good shot at the ball", 618, 620, "153"),
        ]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 1
        assert len(removed) == 1
        assert deduped[0].end_time == 620

    def test_different_segments_not_merged(self):
        """Segments with different text should NOT be merged."""
        segments = [
            _seg("Sterling shoots", 0, 5, "0"),
            _seg("And it's a goal!", 5, 10, "1"),
        ]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 2
        assert len(removed) == 0

    def test_duplicates_across_halves_not_merged(self):
        """Duplicates in different halves should NOT be merged."""
        segments = [
            _seg("The match resumes", 0, 5, "0", half=1),
            _seg("The match resumes", 0, 5, "1", half=2),
        ]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 2
        assert len(removed) == 0

    def test_empty_input(self):
        deduped, removed = deduplicate_segments([])
        assert deduped == []
        assert removed == []

    def test_single_segment(self):
        segments = [_seg("Only one segment", 0, 5, "0")]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 1
        assert len(removed) == 0

    def test_triple_then_different(self):
        """Three duplicates followed by a different segment."""
        segments = [
            _seg("Manchester United", 100, 102, "0"),
            _seg("Manchester United", 102, 104, "1"),
            _seg("Manchester United", 104, 106, "2"),
            _seg("Great save by De Gea", 106, 110, "3"),
        ]
        deduped, removed = deduplicate_segments(segments)
        assert len(deduped) == 2
        assert len(removed) == 2
        assert deduped[0].text == "Manchester United"
        assert deduped[0].end_time == 106
        assert deduped[1].text == "Great save by De Gea"

    def test_removed_log_contains_info(self):
        """Verify removed entries contain correct metadata."""
        segments = [
            _seg("Duplicate text here", 0, 5, "10"),
            _seg("Duplicate text here", 5, 10, "11"),
        ]
        _, removed = deduplicate_segments(segments)
        assert len(removed) == 1
        assert removed[0]["segment_id"] == "11"
        assert removed[0]["duplicate_of"] == "10"
        assert removed[0]["similarity"] == 100.0
