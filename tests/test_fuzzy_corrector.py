"""
Tests for the Fuzzy Corrector module.

Verifies that:
- Known misspellings are corrected to the right canonical name
- Common English words are NOT incorrectly corrected
- Punctuation is preserved during corrections
- Multi-signal scoring works correctly
"""

import pytest
from pipeline.fuzzy_corrector import (
    find_best_match,
    correct_segment_text,
    compute_phonetic_score,
    compute_combined_score,
    COMMON_WORDS_EXCLUDE,
)
from pipeline.ner_extractor import DetectedEntity


# ─── Sample gazetteer for testing ────────────────────────────────────

SAMPLE_GAZETTEER = {
    "Connor Wickham": "Connor Wickham",
    "Wickham": "Connor Wickham",
    "Kurt Zouma": "Kurt Zouma",
    "Zouma": "Kurt Zouma",
    "Wilfried Bony": "Wilfried Bony",
    "Bony": "Wilfried Bony",
    "Raheem Sterling": "Raheem Sterling",
    "Sterling": "Raheem Sterling",
    "Jefferson Montero": "Jefferson Montero",
    "Montero": "Jefferson Montero",
    "Aaron Cresswell": "Aaron Cresswell",
    "Cresswell": "Aaron Cresswell",
    "Matt Targett": "Matt Targett",
    "Targett": "Matt Targett",
    "Kenedy": "Kenedy",
    "Scott Dann": "Scott Dann",
    "Dann": "Scott Dann",
    "Angel Di Maria": "Angel Di Maria",
    "Di Maria": "Angel Di Maria",
    "Brendan Rodgers": "Brendan Rodgers",
    "Rodgers": "Brendan Rodgers",
}


class TestComputePhoneticScore:
    """Tests for compute_phonetic_score()."""

    def test_identical_names(self):
        score = compute_phonetic_score("Sterling", "Sterling")
        assert score == 100.0

    def test_phonetically_similar(self):
        # "Boney" and "Bony" should have similar phonetic codes
        score = compute_phonetic_score("Boney", "Bony")
        assert score >= 50.0

    def test_phonetically_different(self):
        score = compute_phonetic_score("Chelsea", "Liverpool")
        assert score == 0.0


class TestFindBestMatch:
    """Tests for find_best_match() — the core matching function."""

    def test_corrects_conor_to_connor(self):
        """'Conor Wickham' should correct to 'Connor Wickham'."""
        match = find_best_match("Conor Wickham", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Connor Wickham"

    def test_corrects_zuma_to_zouma(self):
        """'Zuma' should correct to 'Zouma' (single word → surname only)."""
        match = find_best_match("Zuma", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Zouma"

    def test_corrects_boney_to_bony(self):
        """'Boney' should correct to 'Bony' (single word → surname only)."""
        match = find_best_match("Boney", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Bony"

    def test_corrects_stirling_to_sterling(self):
        """'Stirling' should correct to 'Sterling' (single word → surname only)."""
        match = find_best_match("Stirling", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Sterling"

    def test_corrects_monteiro_to_montero(self):
        """'Monteiro' should correct to 'Montero' (single word → surname only)."""
        match = find_best_match("Monteiro", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Montero"

    def test_corrects_creswell_to_cresswell(self):
        """'Creswell' should correct to 'Cresswell' (single word → surname only)."""
        match = find_best_match("Creswell", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Cresswell"

    def test_skips_already_correct_name(self):
        """Names already in the gazetteer should NOT be 'corrected'."""
        match = find_best_match("Sterling", SAMPLE_GAZETTEER)
        assert match is None

    def test_skips_already_correct_full_name(self):
        match = find_best_match("Raheem Sterling", SAMPLE_GAZETTEER)
        assert match is None

    def test_skips_common_word_target(self):
        """'target' is a common English word and should NOT be corrected."""
        match = find_best_match("target", SAMPLE_GAZETTEER)
        assert match is None

    def test_skips_common_word_dan(self):
        """'Dan' should NOT be corrected to 'Scott Dann'."""
        match = find_best_match("Dan", SAMPLE_GAZETTEER)
        assert match is None

    def test_skips_common_word_kennedy(self):
        """'Kennedy' should NOT be corrected to 'Kenedy'."""
        match = find_best_match("Kennedy", SAMPLE_GAZETTEER)
        assert match is None

    def test_skips_very_short_entity(self):
        """Entities with 2 or fewer chars should be skipped."""
        match = find_best_match("De", SAMPLE_GAZETTEER)
        assert match is None

    def test_skips_empty_entity(self):
        match = find_best_match("", SAMPLE_GAZETTEER)
        assert match is None

    def test_strips_punctuation_before_matching(self):
        """'Wickham.' (with period) should still match 'Connor Wickham'."""
        match = find_best_match("Wickham.", SAMPLE_GAZETTEER)
        # Should be None because "Wickham" (stripped) IS in the gazetteer
        assert match is None

    def test_corrects_name_with_trailing_punctuation(self):
        """'Conor Wickham.' should correct to 'Connor Wickham'."""
        match = find_best_match("Conor Wickham.", SAMPLE_GAZETTEER)
        assert match is not None
        assert match.corrected == "Connor Wickham"


class TestCorrectSegmentText:
    """Tests for correct_segment_text() — in-place text correction."""

    def test_preserves_trailing_period(self):
        """Correction should preserve sentence-ending punctuation."""
        text = "A great goal from Conor Wickham."
        # "Conor Wickham." starts at index 18, 14 chars long -> end 32
        entities = [
            DetectedEntity(
                text="Conor Wickham.",
                label="PERSON",
                start_char=18,
                end_char=32,
                source="spacy",
            )
        ]
        corrected, corrections = correct_segment_text(
            text, entities, SAMPLE_GAZETTEER, "0"
        )
        assert corrected == "A great goal from Connor Wickham."
        assert len(corrections) == 1

    def test_preserves_exclamation_mark(self):
        """Correction should preserve exclamation marks."""
        text = "What a goal, Conor Wickham!"
        # "Conor Wickham!" starts at index 13, 14 chars long -> end 27
        entities = [
            DetectedEntity(
                text="Conor Wickham!",
                label="PERSON",
                start_char=13,
                end_char=27,
                source="spacy",
            )
        ]
        corrected, corrections = correct_segment_text(
            text, entities, SAMPLE_GAZETTEER, "0"
        )
        assert corrected == "What a goal, Connor Wickham!"
        assert len(corrections) == 1

    def test_no_correction_when_already_correct(self):
        """Already-correct names should not be modified."""
        entities = [
            DetectedEntity(
                text="Sterling",
                label="PERSON",
                start_char=0,
                end_char=8,
                source="spacy",
            )
        ]
        text = "Sterling shoots wide"
        corrected, corrections = correct_segment_text(
            text, entities, SAMPLE_GAZETTEER, "0"
        )
        assert corrected == text  # unchanged
        assert len(corrections) == 0

    def test_multiple_corrections_in_one_segment(self):
        """Multiple entities in one segment should all be corrected."""
        entities = [
            DetectedEntity(
                text="Stirling", label="PERSON",
                start_char=0, end_char=8, source="spacy"
            ),
            DetectedEntity(
                text="Boney", label="PERSON",
                start_char=20, end_char=25, source="spacy"
            ),
        ]
        text = "Stirling passes to Boney"
        corrected, corrections = correct_segment_text(
            text, entities, SAMPLE_GAZETTEER, "0"
        )
        # Both should be corrected to surname-only form
        assert "Sterling" in corrected
        assert "Bony" in corrected
        assert len(corrections) == 2


class TestCommonWordsExclude:
    """Verify the common words exclusion list is comprehensive."""

    def test_target_excluded(self):
        assert "target" in COMMON_WORDS_EXCLUDE

    def test_dan_excluded(self):
        assert "dan" in COMMON_WORDS_EXCLUDE

    def test_kennedy_excluded(self):
        assert "kennedy" in COMMON_WORDS_EXCLUDE

    def test_davies_excluded(self):
        assert "davies" in COMMON_WORDS_EXCLUDE

    def test_pele_excluded(self):
        assert "pele" in COMMON_WORDS_EXCLUDE

    def test_soccer_terms_excluded(self):
        assert "corner" in COMMON_WORDS_EXCLUDE
        assert "penalty" in COMMON_WORDS_EXCLUDE
        assert "referee" in COMMON_WORDS_EXCLUDE
