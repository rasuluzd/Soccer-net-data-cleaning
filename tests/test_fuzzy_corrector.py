"""
Tests for the Fuzzy Corrector module.

Verifies that:
- Known misspellings are corrected to the right canonical name
- Common English words are NOT incorrectly corrected
- Punctuation is preserved during corrections
- Multi-signal scoring works correctly
"""

from pipeline.fuzzy_corrector import (
    find_best_match,
    correct_segment_text,
    compute_phonetic_score,
    extract_and_rebuild_entity,
    extract_entity_core,
    _entity_contains_multiple_gazetteer_names,
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
        """'target' tagged as NOUN should NOT be corrected."""
        match = find_best_match("target", SAMPLE_GAZETTEER, pos="NOUN")
        assert match is None

    def test_skips_common_word_dan(self):
        """'Dan' tagged as NOUN should NOT be corrected to 'Dann'."""
        match = find_best_match("Dan", SAMPLE_GAZETTEER, pos="NOUN")
        assert match is None

    def test_skips_common_word_kennedy(self):
        """'Kennedy' tagged as NOUN should NOT be corrected."""
        match = find_best_match("Kennedy", SAMPLE_GAZETTEER, pos="NOUN")
        assert match is None

    def test_allows_propn_entity(self):
        """A PROPN entity should be allowed through for correction."""
        match = find_best_match("Zuma", SAMPLE_GAZETTEER, pos="PROPN")
        assert match is not None
        assert match.corrected == "Zouma"

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

    def test_accepts_debug_parameter(self):
        """correct_segment_text() should accept a debug parameter without crashing."""
        text = "A great goal from Conor Wickham."
        entities = [
            DetectedEntity(
                text="Conor Wickham.",
                label="PERSON",
                start_char=18,
                end_char=32,
                source="spacy",
            )
        ]
        # Should not raise TypeError
        corrected, corrections = correct_segment_text(
            text, entities, SAMPLE_GAZETTEER, "0", debug=True
        )
        assert corrected == "A great goal from Connor Wickham."
        assert len(corrections) == 1


class TestEntityTextHelpers:
    """Tests for reusable entity strip/rebuild helpers."""

    def test_rebuild_preserves_possessive_before_period(self):
        rebuilt = extract_and_rebuild_entity("Ward's.", "Wickham")
        assert rebuilt == "Wickham's."

    def test_rebuild_preserves_unicode_possessive_before_period(self):
        rebuilt = extract_and_rebuild_entity("Ward’s.", "Wickham")
        assert rebuilt == "Wickham’s."

    def test_extract_core_handles_possessive_with_punctuation(self):
        core = extract_entity_core("Ward's.")
        assert core == "Ward"



class TestCanonicalNotKey:
    """Regression: fuzzy corrector must use canonical value, not gazetteer key."""

    def test_corrects_to_canonical_not_misspelling_key(self):
        """When a learned misspelling is in the gazetteer as a key,
        the corrector should return the canonical value (Bolasie),
        not the key itself (blassie)."""
        gaz = {
            "blassie": "Bolasie",
            "Bolasie": "Yannick Bolasie",
            "Yannick Bolasie": "Yannick Bolasie",
        }
        match = find_best_match("Balassi", gaz)
        assert match is not None
        assert match.corrected == "Bolasie", (
            f"Expected 'Bolasie' but got '{match.corrected}' — "
            f"corrector used gazetteer key instead of canonical value"
        )

    def test_single_word_entity_gets_surname_from_canonical(self):
        """Single-word entity matched to multi-word canonical should
        get the most similar word from the canonical name."""
        gaz = {
            "Kurt Zouma": "Kurt Zouma",
            "Zouma": "Kurt Zouma",
        }
        match = find_best_match("Zuma", gaz)
        assert match is not None
        assert match.corrected == "Zouma"  # surname, not "Kurt"


# ─── Regression: multi-word gazetteer substring guard ─────────────────

class TestMultiWordGazetteerGuard:
    """Fix 1: 'Di Maria Rooney' should NOT be corrected to 'Angel Di Maria'.

    spaCy sometimes merges adjacent names into one entity. When the entity
    contains multiple gazetteer names (including multi-word keys like
    'Di Maria'), the corrector should reject it rather than mapping it
    to a single player.
    """

    def test_di_maria_rooney_rejected(self):
        """'Di Maria Rooney' contains 'Di Maria' + 'Rooney' — two valid names."""
        gaz = {
            "Angel Di Maria": "Angel Di Maria",
            "Di Maria": "Angel Di Maria",
            "Wayne Rooney": "Wayne Rooney",
            "Rooney": "Wayne Rooney",
        }
        match = find_best_match("Di Maria Rooney", gaz)
        assert match is None

    def test_helper_detects_multiword_plus_single(self):
        """'Di Maria Rooney' decomposes into ['Di Maria', 'Rooney']."""
        gazetteer_lower = {"angel di maria", "di maria", "wayne rooney", "rooney"}
        assert _entity_contains_multiple_gazetteer_names(
            "Di Maria Rooney", gazetteer_lower
        ) is True

    def test_helper_rejects_single_name(self):
        """'Di Maria' alone is just one gazetteer entry, not two names."""
        gazetteer_lower = {"angel di maria", "di maria", "wayne rooney", "rooney"}
        assert _entity_contains_multiple_gazetteer_names(
            "Di Maria", gazetteer_lower
        ) is False

    def test_single_misspelled_name_still_corrected(self):
        """'De Maria' (single misspelling) should still get corrected."""
        gaz = {
            "Angel Di Maria": "Angel Di Maria",
            "Di Maria": "Angel Di Maria",
        }
        match = find_best_match("De Maria", gaz)
        assert match is not None
        assert match.corrected == "Di Maria"


# ─── Regression: surname-only multi-word replacement ──────────────────

class TestSurnameOnlyReplacement:
    """Fix 2: 'De Michelis' should correct to 'Demichelis', not 'Martin Demichelis'.

    When a multi-word entity is a surname variant (not first+last), the
    replacement should use just the surname from the canonical name.
    """

    def test_de_michelis_gets_surname_only(self):
        """'De Michelis' → 'Demichelis' (not 'Martin Demichelis')."""
        gaz = {
            "Martin Demichelis": "Martin Demichelis",
            "Demichelis": "Martin Demichelis",
        }
        match = find_best_match("De Michelis", gaz)
        assert match is not None
        assert match.corrected == "Demichelis"

    def test_full_name_entity_keeps_full_canonical(self):
        """'Winston Ritu' → 'Winston Reid' (first name matches, keep full)."""
        gaz = {
            "Winston Reid": "Winston Reid",
            "Reid": "Winston Reid",
        }
        match = find_best_match("Winston Ritu", gaz)
        assert match is not None
        assert match.corrected == "Winston Reid"


# ─── Regression: ASR word-split collapsing ────────────────────────────

class TestASRWordSplitCollapsing:
    """Fix 3: 'John Joe Shelby' should match 'Jonjo Shelvey'.

    ASR splits single names into multiple words ('Jonjo' → 'John Joe').
    The corrector should try collapsing adjacent words before matching.
    """

    def test_john_joe_shelby_matches_jonjo_shelvey(self):
        """'John Joe Shelby' should correct to 'Shelvey' or 'Jonjo Shelvey'."""
        gaz = {
            "Jonjo Shelvey": "Jonjo Shelvey",
            "Shelvey": "Jonjo Shelvey",
        }
        match = find_best_match("John Joe Shelby", gaz)
        assert match is not None
        assert "Shelvey" in match.corrected
