"""
Tests for the Hallucination & Garbage Filter module.

Verifies that garbled/non-English/single-word segments are correctly
flagged while valid English commentary is preserved.
"""

import pytest
from pipeline.loader import Segment
from pipeline.hallucination_filter import (
    filter_segment,
    filter_segments,
    compute_alpha_ratio,
    has_non_latin_characters,
)


def _seg(text: str, seg_id: str = "0") -> Segment:
    """Helper to create a Segment with just text."""
    return Segment(segment_id=seg_id, start_time=0.0, end_time=5.0, text=text, half=1)


class TestComputeAlphaRatio:
    """Tests for compute_alpha_ratio()."""

    def test_all_alpha(self):
        assert compute_alpha_ratio("hello world") > 0.9

    def test_mixed_with_numbers(self):
        ratio = compute_alpha_ratio("3 on 2")
        assert ratio < 0.7  # numbers lower the ratio

    def test_empty_string(self):
        assert compute_alpha_ratio("") == 0.0

    def test_only_spaces(self):
        assert compute_alpha_ratio("   ") == 0.0

    def test_non_latin(self):
        ratio = compute_alpha_ratio("已经就冕复了")
        assert ratio == 0.0


class TestHasNonLatinCharacters:
    """Tests for has_non_latin_characters()."""

    def test_english_text(self):
        assert has_non_latin_characters("De Bruyne scores!") is False

    def test_chinese_characters(self):
        assert has_non_latin_characters("youこちら") is True

    def test_arabic_characters(self):
        assert has_non_latin_characters("حيال مراس") is True

    def test_cyrillic_characters(self):
        assert has_non_latin_characters("It's veryому a side") is True

    def test_korean_characters(self):
        assert has_non_latin_characters("실시") is True

    def test_mixed_valid(self):
        # Accented Latin characters should NOT be flagged
        assert has_non_latin_characters("Agüero scores a goal") is False


class TestFilterSegment:
    """Tests for filter_segment() — the core decision function."""

    def test_valid_commentary(self):
        is_valid, reason = filter_segment(_seg("De Bruyne scores a brilliant goal"))
        assert is_valid is True
        assert reason is None

    def test_empty_segment(self):
        is_valid, reason = filter_segment(_seg(""))
        assert is_valid is False
        assert reason == "empty_segment"

    def test_whitespace_only(self):
        is_valid, reason = filter_segment(_seg("   "))
        assert is_valid is False
        assert reason == "empty_segment"

    def test_non_latin_characters(self):
        is_valid, reason = filter_segment(_seg("youこちら"))
        assert is_valid is False
        assert "non_latin" in reason

    def test_single_word_noise(self):
        is_valid, reason = filter_segment(_seg("transition"))
        assert is_valid is False
        assert "too_few_words" in reason

    def test_two_word_segment_valid(self):
        # Two words should pass the min word count
        is_valid, reason = filter_segment(_seg("Sterling shoots"))
        assert is_valid is True

    def test_garbled_unicode(self):
        is_valid, reason = filter_segment(_seg("已经就冕复了很好的一浪格调子"))
        assert is_valid is False

    def test_valid_short_commentary(self):
        """Short but valid commentary should be kept."""
        is_valid, reason = filter_segment(_seg("stunning goal, absolute stunning goal"))
        assert is_valid is True

    def test_low_alpha_ratio(self):
        is_valid, reason = filter_segment(_seg("3 on 2"))
        assert is_valid is False
        assert "low_alpha_ratio" in reason


class TestFilterSegments:
    """Tests for filter_segments() — batch filtering."""

    def test_mixed_segments(self):
        segments = [
            _seg("De Bruyne passes to Sterling", "0"),
            _seg("youこちら", "1"),
            _seg("transition", "2"),
            _seg("Sterling shoots and scores", "3"),
        ]
        kept, removed = filter_segments(segments)
        assert len(kept) == 2
        assert len(removed) == 2
        # Valid segments should be #0 and #3
        assert kept[0].segment_id == "0"
        assert kept[1].segment_id == "3"

    def test_all_valid(self):
        segments = [
            _seg("First half kicks off", "0"),
            _seg("Great tackle by Kompany", "1"),
        ]
        kept, removed = filter_segments(segments)
        assert len(kept) == 2
        assert len(removed) == 0

    def test_all_garbage(self):
        segments = [
            _seg("", "0"),
            _seg("실시", "1"),
            _seg("x", "2"),
        ]
        kept, removed = filter_segments(segments)
        assert len(kept) == 0
        assert len(removed) == 3
