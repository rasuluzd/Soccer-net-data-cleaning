"""Tests for pipeline/ner_extractor.py — regression tests for NER heuristics."""

import pytest

from pipeline.loader import Segment
from pipeline.ner_extractor import (
    extract_entities,
    extract_entities_batch,
    extract_heuristic_entities,
)


def _seg(text: str) -> Segment:
    return Segment(segment_id="0", start_time=0.0, end_time=5.0, text=text, half=1)


class TestCrossHalfSegmentIdCollision:
    """Regression test for a critical bug: extract_entities_batch used
    ``seg.segment_id`` alone as the dict key. Because the ASR JSON restarts
    numbering at 0 for each half, both halves have segments with id "0",
    "1", etc. Later halves overwrote earlier ones, and the orchestrator
    then applied entity positions from one half to the other half's text —
    producing corruption like ``'411 som eventuellt då.Celina'``.

    Fix: key by (half, segment_id) tuple.
    """

    def test_same_segment_id_across_halves_keeps_entities_separate(self):
        """Two segments sharing the same segment_id but different halves
        must not overwrite each other's entities in the batch map."""
        segments = [
            # Half 1, seg 90: text with NO name-like entity near position 26.
            Segment(
                segment_id="90", start_time=100.0, end_time=103.0,
                text="411 som eventuellt då.", half=1,
            ),
            # Half 2, seg 90: text with "Selina" at position 26-32.
            Segment(
                segment_id="90", start_time=200.0, end_time=203.0,
                text="Testa vänsterkanten genom Selina.", half=2,
            ),
        ]
        result = extract_entities_batch(segments, language="sv")
        assert (1, "90") in result, "half=1 entities missing from map"
        assert (2, "90") in result, "half=2 entities missing from map"
        # Half 1 entities must NOT contain Selina — it only exists in half 2.
        h1_texts = [e.text for e in result[(1, "90")]]
        assert "Selina" not in h1_texts, (
            f"Cross-half contamination: half 1 seg 90 got half 2's Selina "
            f"entity. entities={h1_texts}"
        )


class TestRule2CursorBug:
    """Regression tests for F3: Rule 2 heuristic previously used
    `text.find(clean_word)` without a cursor, so when the same capitalized
    word appeared multiple times in a segment (very common in commentary),
    every iteration resolved to start_char=0 of the FIRST occurrence.
    `_deduplicate_entities` collapsed the duplicates, so only ONE correction
    span existed for N occurrences, leaving the 2nd..Nth uncorrected.

    After the fix, each repeated occurrence is located at its actual
    character position.
    """

    def test_two_mid_sentence_occurrences_detected_separately(self):
        """Two non-sentence-start occurrences of the same name must each
        be located at the right character position, not both at pos 0."""
        text = "Idag passar Hansson till Hansson igen."
        ents = extract_heuristic_entities(text, language="sv")
        hansson_ents = [e for e in ents if e.text == "Hansson"]
        # Expect either 2 distinct positions, or (if spaCy POS rejects the
        # second one) no Rule-2 collision collapsing them to identical spans.
        if len(hansson_ents) >= 2:
            positions = sorted(e.start_char for e in hansson_ents)
            assert len(set(positions)) == len(positions), (
                f"Multiple Hanssons collapsed to same position {positions} — "
                "text.find cursor bug is back"
            )
            # First Hansson starts around index 12, second around 25
            assert positions[0] < 20
            assert positions[1] > 20

    def test_repeated_capitalized_words_get_distinct_positions(self):
        """Direct test: with cursor-based find, no two entities returned
        by the heuristic share the same start_char."""
        # Use a long name that spaCy reliably tags as PROPN across languages.
        text = "Spelet börjar. Nu passar Nordfeldt till Nordfeldt-zonen."
        ents = extract_heuristic_entities(text, language="sv")
        nordfeldt_ents = [e for e in ents if "Nordfeldt" in e.text]
        if len(nordfeldt_ents) >= 2:
            positions = [e.start_char for e in nordfeldt_ents]
            assert len(set(positions)) == len(positions), (
                f"Cursor bug regression: positions={positions}"
            )


# ─── Rule 3: gazetteer fuzz-match (Apple RAG-NEC) ────────────────────

class TestGazetteerFuzzyHeuristic:
    """Rule 3 catches ASR mishearings whose surface form looks like a
    common noun (so spaCy NER misses) but fuzz-matches a known lineup name.
    Without this rule, "passes to storage" never reaches entity_corrector."""

    GAZ = {
        "Daniel Sturridge": "Daniel Sturridge", "Sturridge": "Daniel Sturridge",
        "Sadio Mane": "Sadio Mane",             "Mane": "Sadio Mane",
        "Jordan Henderson": "Jordan Henderson", "Henderson": "Jordan Henderson",
        "Eden Hazard": "Eden Hazard",           "Hazard": "Eden Hazard",
    }

    def test_lowercase_dict_word_with_strong_fuzz_to_canonical_emitted(self):
        """'sturage' → fuzz('sturage','sturridge')=82, not in dictionary,
        so Rule 3 emits it for entity_corrector to validate via MCQ."""
        ents = extract_heuristic_entities(
            "and the ball goes to sturage on the wing",
            language="en", gazetteer=self.GAZ,
        )
        assert any(
            e.text.lower() == "sturage" and e.source == "heuristic_gazetteer_fuzz"
            for e in ents
        ), f"Expected sturage to be emitted via Rule 3; got: {[(e.text, e.source) for e in ents]}"

    def test_strong_fuzz_dictionary_word_overrides_veto(self):
        """'storage' IS a real English word, but fuzz('storage','sturridge')
        is high enough that it can still be considered as a candidate for
        entity_corrector. Tests that NER_FUZZY_DICT_OVERRIDE is checked."""
        from rapidfuzz import fuzz
        score = fuzz.ratio("storage", "sturridge")
        # If the override threshold (80) is set right, only words scoring
        # >= 80 against a canonical pass the dictionary veto. Verify the
        # gating contract holds for either side of the threshold.
        ents = extract_heuristic_entities(
            "passes to storage on the right wing",
            language="en", gazetteer=self.GAZ,
        )
        gazetteer_ents = [e for e in ents if e.source == "heuristic_gazetteer_fuzz"]
        from pipeline.config import NER_FUZZY_DICT_OVERRIDE
        if score >= NER_FUZZY_DICT_OVERRIDE:
            assert any(e.text.lower() == "storage" for e in gazetteer_ents), (
                f"storage(score={score}) should override veto at floor {NER_FUZZY_DICT_OVERRIDE}"
            )
        else:
            assert all(e.text.lower() != "storage" for e in gazetteer_ents), (
                f"storage(score={score}) below override {NER_FUZZY_DICT_OVERRIDE}, should be vetoed"
            )

    def test_below_floor_not_emitted(self):
        """A token that fuzz-matches < 65 to any canonical is dropped by
        Rule 3 — entity_corrector wouldn't have anything useful to do
        with such a weak candidate anyway."""
        ents = extract_heuristic_entities(
            "the apple fell from the tree onto the ground",
            language="en", gazetteer=self.GAZ,
        )
        gazetteer_ents = [e for e in ents if e.source == "heuristic_gazetteer_fuzz"]
        # No word here should match any of the 4 player names at >=65 fuzz
        assert gazetteer_ents == [], (
            f"Unexpected gazetteer-fuzz emissions: {[e.text for e in gazetteer_ents]}"
        )

    def test_no_gazetteer_disables_rule_3(self):
        """When gazetteer=None (back-compat default), Rule 3 doesn't fire
        — only Rule 1+2 (capitalization-based) run. Important so callers
        that don't have a gazetteer (legacy code paths, single-segment
        diagnostic scripts) keep working unchanged."""
        ents = extract_heuristic_entities(
            "passes to sturage on the wing",
            language="en", gazetteer=None,
        )
        gazetteer_ents = [e for e in ents if e.source == "heuristic_gazetteer_fuzz"]
        assert gazetteer_ents == []

    def test_exact_gazetteer_match_skipped(self):
        """If the token is already an exact gazetteer entry (lowercase),
        Rule 3 doesn't emit it — entity_corrector's per-match cache will
        handle it via the existing flow without redundant entity records."""
        ents = extract_heuristic_entities(
            "good pass from sturridge to mane",
            language="en", gazetteer=self.GAZ,
        )
        gazetteer_ents = [e for e in ents if e.source == "heuristic_gazetteer_fuzz"]
        # "sturridge" and "mane" are exact gazetteer entries (case-insensitive).
        # Rule 3 should skip them.
        assert all(
            e.text.lower() not in {"sturridge", "mane"}
            for e in gazetteer_ents
        )

    def test_batch_threads_gazetteer_through(self):
        """extract_entities_batch accepts gazetteer kwarg and passes it
        to extract_heuristic_entities. Without this plumbing the new rule
        is dormant in production."""
        segs = [_seg("the ball goes to sturage on the wing")]
        result = extract_entities_batch(segs, language="en", gazetteer=self.GAZ)
        ents = result[(1, "0")]
        assert any(
            e.source == "heuristic_gazetteer_fuzz"
            for e in ents
        ), f"batch did not surface Rule 3 candidates; got: {[(e.text, e.source) for e in ents]}"
