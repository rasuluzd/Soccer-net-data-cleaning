"""
Tests for the Self-Learning Correction Dictionary module.

Verifies that corrections are stored, confidence grows with repeated
sightings, and high-confidence entries are retrieved correctly.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from pipeline.learned_dictionary import (
    load_learned_dictionary,
    save_learned_dictionary,
    update_learned_dictionary,
    lookup_learned,
)
from pipeline.fuzzy_corrector import Correction


@pytest.fixture
def tmp_dict_path(tmp_path):
    """Provide a temporary path for the learned dictionary."""
    path = tmp_path / "learned_corrections.json"
    return path


@pytest.fixture
def mock_dict_path(tmp_dict_path):
    """Patch the LEARNED_CORRECTIONS_PATH to use a temp file."""
    with patch("pipeline.learned_dictionary.LEARNED_CORRECTIONS_PATH", tmp_dict_path):
        yield tmp_dict_path


def _correction(original: str, corrected: str, score: float = 78.0) -> Correction:
    """Helper to create a Correction object."""
    return Correction(
        original=original,
        corrected=corrected,
        combined_score=score,
        fuzzy_score=score,
        phonetic_match=True,
        context_match=False,
        segment_id="test",
        method="fuzzy+phonetic",
    )


class TestLearnedDictionary:
    """Tests for the learned dictionary module."""

    def test_empty_dictionary_on_first_load(self, mock_dict_path):
        dictionary = load_learned_dictionary()
        assert dictionary == {}

    def test_save_and_load(self, mock_dict_path):
        data = {"boney": {"correct": "Wilfried Bony", "confidence": 0.5,
                          "seen_count": 1, "fuzzy_score_avg": 78.0}}
        save_learned_dictionary(data)
        loaded = load_learned_dictionary()
        assert loaded == data

    def test_update_new_correction(self, mock_dict_path):
        corrections = [_correction("Boney", "Wilfried Bony")]
        dictionary = update_learned_dictionary(corrections)
        assert "boney" in dictionary
        assert dictionary["boney"]["correct"] == "Wilfried Bony"
        assert dictionary["boney"]["seen_count"] == 1
        assert dictionary["boney"]["confidence"] == 0.5

    def test_update_increases_confidence(self, mock_dict_path):
        """Seeing a correction multiple times should increase confidence."""
        c = _correction("Boney", "Wilfried Bony")
        update_learned_dictionary([c])
        update_learned_dictionary([c])
        update_learned_dictionary([c])
        dictionary = load_learned_dictionary()
        assert dictionary["boney"]["seen_count"] == 3
        assert dictionary["boney"]["confidence"] > 0.5

    def test_lookup_returns_none_for_low_confidence(self, mock_dict_path):
        """Single-sighting corrections should NOT be returned by lookup."""
        update_learned_dictionary([_correction("Boney", "Wilfried Bony")])
        result = lookup_learned("Boney")
        assert result is None  # needs at least 2 sightings

    def test_lookup_returns_correction_after_multiple_sightings(self, mock_dict_path):
        """After 2+ sightings, lookup should return the correction."""
        c = _correction("Boney", "Wilfried Bony")
        update_learned_dictionary([c])
        update_learned_dictionary([c])
        result = lookup_learned("Boney")
        assert result == "Wilfried Bony"

    def test_lookup_is_case_insensitive(self, mock_dict_path):
        c = _correction("Boney", "Wilfried Bony")
        update_learned_dictionary([c])
        update_learned_dictionary([c])
        # Should find regardless of case
        assert lookup_learned("boney") == "Wilfried Bony"
        assert lookup_learned("BONEY") == "Wilfried Bony"
