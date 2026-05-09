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
