"""
Tests for the Gazetteer Builder module.

Verifies that names are correctly extracted from Labels-caption.json
and that surname variants are properly generated.
"""

import pytest
from pipeline.gazetteer import extract_names_from_labels, build_gazetteer


# ─── Sample Labels fixture (mirrors real Labels-caption.json structure) ───

SAMPLE_LABELS = {
    "gameHomeTeam": "Manchester City",
    "gameAwayTeam": "West Ham",
    "lineup": {
        "home": {
            "players": [
                {"long_name": "Kevin De Bruyne", "short_name": "De Bruyne", "name": "De Bruyne"},
                {"long_name": "Sergio Agüero", "short_name": "Agüero", "name": "Aguero"},
                {"long_name": "Raheem Sterling", "short_name": "Sterling", "name": "Sterling"},
                {"long_name": "Wilfried Bony", "short_name": "Bony", "name": "Bony"},
            ],
            "coach": [
                {"long_name": "Manuel Pellegrini", "short_name": "Pellegrini", "name": "Pellegrini"},
            ],
        },
        "away": {
            "players": [
                {"long_name": "Dimitri Payet", "short_name": "Payet", "name": "Payet"},
                {"long_name": "Aaron Cresswell", "short_name": "Cresswell", "name": "Cresswell"},
                {"long_name": "Winston Reid", "short_name": "Reid", "name": "Reid"},
            ],
            "coach": [
                {"long_name": "Slaven Bilić", "short_name": "Bilić", "name": "Bilic"},
            ],
        },
    },
    "referee_matched": ["Craig Pawson"],
    "venue": ["Etihad Stadium"],
}


class TestExtractNamesFromLabels:
    """Tests for extract_names_from_labels()."""

    def test_extracts_player_long_names(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        assert "Kevin De Bruyne" in gaz
        assert "Sergio Agüero" in gaz
        assert "Dimitri Payet" in gaz

    def test_extracts_player_short_names(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        # Short names should map to the canonical (long) name
        assert gaz.get("De Bruyne") == "Kevin De Bruyne"
        assert gaz.get("Sterling") == "Raheem Sterling"
        assert gaz.get("Payet") == "Dimitri Payet"

    def test_extracts_surnames(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        # Surnames extracted from long_name should also be in gazetteer
        assert gaz.get("Agüero") == "Sergio Agüero"
        assert gaz.get("Reid") == "Winston Reid"

    def test_extracts_coaches(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        assert "Manuel Pellegrini" in gaz
        assert gaz.get("Pellegrini") == "Manuel Pellegrini"
        assert "Slaven Bilić" in gaz

    def test_extracts_referees(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        assert "Craig Pawson" in gaz
        assert gaz.get("Pawson") == "Craig Pawson"

    def test_extracts_teams(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        assert "Manchester City" in gaz
        assert "West Ham" in gaz

    def test_extracts_venue(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        assert "Etihad Stadium" in gaz

    def test_canonical_maps_to_itself(self):
        gaz = extract_names_from_labels(SAMPLE_LABELS)
        assert gaz["Kevin De Bruyne"] == "Kevin De Bruyne"
        assert gaz["Winston Reid"] == "Winston Reid"

    def test_empty_labels(self):
        gaz = extract_names_from_labels({})
        assert gaz == {}


class TestBuildGazetteer:
    """Tests for build_gazetteer()."""

    def test_builds_from_labels(self):
        gaz = build_gazetteer(SAMPLE_LABELS, include_learned=False)
        assert len(gaz) > 0
        assert "Kevin De Bruyne" in gaz

    def test_handles_none_labels(self):
        gaz = build_gazetteer(None, include_learned=False)
        assert gaz == {}
