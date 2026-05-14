"""
Tests for the Hallucination & Garbage Filter module.

Verifies that garbled/non-English/single-word segments are correctly
flagged while valid English commentary is preserved.
"""

from pipeline.loader import Segment
from pipeline.hallucination_filter import (
    filter_segment,
    filter_segments,
    find_repeated_name_hallucinations,
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

    def test_single_word_valid(self):
        # Single-word segments like "GOAL!" are valid commentary.
        # MIN_SEGMENT_WORD_COUNT=1 allows these through.
        is_valid, reason = filter_segment(_seg("GOAL"))
        assert is_valid is True

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
        # "123456" has 0% alpha — clear garbage
        is_valid, reason = filter_segment(_seg("123456"))
        assert is_valid is False
        assert "low_alpha_ratio" in reason

    def test_mixed_alpha_numeric_valid(self):
        # "3 on 2" has 75% alpha — above 0.50 threshold, valid commentary
        is_valid, reason = filter_segment(_seg("3 on 2"))
        assert is_valid is True

    def test_short_english_football_phrase_kept(self):
        """REGRESSION: langdetect on short football phrases (8-14 words)
        misclassifies them as foreign languages.

        Empirical: "get a good tackling, get a good passing" (8 words)
        is detected as 'af' (Afrikaans), and "as Gary was saying no real
        finger-pointing at him Fabregas" (10 words) hits the same trap.
        Both were silently dropped on Chelsea-Liverpool V3 — visible as
        +0.5pp WER each because their content disappeared from the cleaned
        output but stayed in the GT.

        Fix: bump MIN_WORDS_FOR_LANGDETECT from 8 to 15 so we only invoke
        langdetect on text long enough for it to be statistically reliable."""
        is_valid, reason = filter_segment(_seg("get a good tackling, get a good passing"))
        assert is_valid is True, f"valid English football text rejected: reason={reason}"

        is_valid, reason = filter_segment(
            _seg("as Gary was saying no real finger-pointing at him Fabregas")
        )
        assert is_valid is True, f"valid English football text rejected: reason={reason}"

    def test_obviously_foreign_long_text_still_rejected(self):
        """A safety check: bumping the langdetect threshold must not let
        clearly foreign LONG text slip through. 15+ words of Swedish
        commentary should still be filtered when expected_lang='en'."""
        sv = _seg(
            "Spelarna är på sin plats och domaren blåser i visselpipan "
            "för att starta matchen igen efter halvtidspausen i andra halvlek."
        )
        is_valid, reason = filter_segment(sv, expected_lang="en")
        assert is_valid is False
        assert reason == "wrong_language_detected"


class TestFilterSegments:
    """Tests for filter_segments() — batch filtering."""

    def test_mixed_segments(self):
        segments = [
            _seg("De Bruyne passes to Sterling", "0"),
            _seg("youこちら", "1"),                       # non-Latin → removed
            _seg("transition", "2"),                      # 1 word, valid (MIN=1)
            _seg("Sterling shoots and scores", "3"),
        ]
        kept, removed = filter_segments(segments)
        assert len(kept) == 3   # "transition" now kept (MIN_SEGMENT_WORD_COUNT=1)
        assert len(removed) == 1
        assert kept[0].segment_id == "0"
        assert kept[1].segment_id == "2"
        assert kept[2].segment_id == "3"

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
            _seg("", "0"),          # empty → removed
            _seg("실시", "1"),       # non-Latin → removed
            _seg("12345", "2"),     # 0% alpha → removed
        ]
        kept, removed = filter_segments(segments)
        assert len(kept) == 0
        assert len(removed) == 3


def _tseg(text: str, seg_id: str, start: float, end: float) -> Segment:
    """Helper for time-sensitive test segments."""
    return Segment(
        segment_id=seg_id,
        start_time=start,
        end_time=end,
        text=text,
        half=1,
    )


class TestRepeatedNameHallucinations:
    """Rule 6: Whisper 'stuck on a name' pattern (cluster-aware)."""

    def test_finds_clustered_repeated_name(self):
        """Segments 26-31 in AIK raw all said 'Hansson.' within ~30s.
        That's a cluster and should be flagged."""
        segments = [
            _tseg("Hansson.", "0", 0.0, 5.0),
            _tseg("Vänster sida, bollen går ut.", "1", 5.0, 10.0),
            _tseg("Hansson.", "2", 10.0, 15.0),
            _tseg("Corner för AIK.", "3", 15.0, 20.0),
            _tseg("Hansson.", "4", 20.0, 25.0),
        ]
        bad_ids = find_repeated_name_hallucinations(
            segments, min_cluster_size=3, cluster_window_s=60.0,
        )
        # All 3 "Hansson." within 25s — a cluster.
        # Keys are (half, segment_id) tuples so halves don't collide.
        assert bad_ids == {(1, "0"), (1, "2"), (1, "4")}

    def test_spread_out_callouts_not_flagged(self):
        """Real commentator callouts of a player are spread across the match,
        not clustered. E.g. 'Neves!' 16 times over 90 min → not hallucination."""
        segments = [
            _tseg("Neves.", "0", 10.0, 12.0),
            _tseg("Neves.", "1", 600.0, 602.0),    # 10 min later
            _tseg("Neves.", "2", 1200.0, 1202.0),  # 20 min later
            _tseg("Neves.", "3", 1800.0, 1802.0),  # 30 min later
        ]
        bad_ids = find_repeated_name_hallucinations(
            segments, min_cluster_size=3, cluster_window_s=60.0,
        )
        # All 4 are spread widely → not a cluster, not flagged
        assert bad_ids == set()

    def test_does_not_flag_below_threshold(self):
        """Two clustered occurrences is not enough — could be real emphasis."""
        segments = [
            _tseg("Hansson!", "0", 0.0, 5.0),
            _tseg("Other stuff here.", "1", 5.0, 10.0),
            _tseg("Hansson.", "2", 10.0, 15.0),
        ]
        bad_ids = find_repeated_name_hallucinations(segments)
        assert bad_ids == set()

    def test_ignores_multi_word_segments(self):
        """Multi-word segments never count, regardless of clustering."""
        segments = [
            _tseg("Hansson passes forward.", "0", 0.0, 5.0),
            _tseg("Hansson shoots.", "1", 5.0, 10.0),
            _tseg("Hansson scores.", "2", 10.0, 15.0),
        ]
        bad_ids = find_repeated_name_hallucinations(segments)
        assert bad_ids == set()

    def test_ignores_all_caps(self):
        """ALL-CAPS words like 'AIK' are acronyms, not stuck names."""
        segments = [
            _tseg("AIK.", "0", 0.0, 5.0),
            _tseg("AIK!", "1", 5.0, 10.0),
            _tseg("AIK?", "2", 10.0, 15.0),
        ]
        bad_ids = find_repeated_name_hallucinations(segments)
        assert bad_ids == set()

    def test_multiple_clusters_all_flagged(self):
        """Two separate clusters of the same word far apart: flag BOTH clusters."""
        segments = [
            # Cluster 1: at t=0-25
            _tseg("Hansson.", "0", 0.0, 5.0),
            _tseg("Hansson.", "1", 10.0, 15.0),
            _tseg("Hansson.", "2", 20.0, 25.0),
            # Gap with other content
            _tseg("The ball is in play.", "3", 1000.0, 1005.0),
            # Cluster 2: at t=2000-2025
            _tseg("Hansson.", "4", 2000.0, 2005.0),
            _tseg("Hansson.", "5", 2010.0, 2015.0),
            _tseg("Hansson.", "6", 2020.0, 2025.0),
        ]
        bad_ids = find_repeated_name_hallucinations(segments)
        assert bad_ids == {(1, "0"), (1, "1"), (1, "2"), (1, "4"), (1, "5"), (1, "6")}

    def test_filter_segments_removes_clustered_hallucinations(self):
        """End-to-end: filter_segments drops clustered hallucinations only."""
        segments = [
            _tseg("Hansson.", "0", 0.0, 5.0),
            _tseg("The match continues.", "1", 10.0, 20.0),
            _tseg("Hansson.", "2", 20.0, 25.0),
            _tseg("Hansson.", "3", 30.0, 35.0),
            _tseg("Another valid segment.", "4", 100.0, 105.0),
            _tseg("Hansson.", "5", 500.0, 505.0),  # isolated, not flagged
        ]
        kept, removed = filter_segments(segments)
        # 3 clustered Hansson. (0, 2, 3) removed; 1 isolated kept; 2 valid kept
        kept_ids = {s.segment_id for s in kept}
        assert kept_ids == {"1", "4", "5"}
        repeated_removals = [r for r in removed
                             if r["reason"] == "hallucination_repeated_name_segment"]
        assert len(repeated_removals) == 3
