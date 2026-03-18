"""
Tests for the Gazetteer Builder module.

Verifies that names are correctly extracted from Labels-caption.json
and that surname variants are properly generated.
"""

from unittest.mock import patch

from pipeline.gazetteer import (
    extract_names_from_labels,
    build_gazetteer,
    build_firstname_map,
)


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
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        assert "Kevin De Bruyne" in gaz
        assert "Sergio Agüero" in gaz
        assert "Dimitri Payet" in gaz

    def test_extracts_player_short_names(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        # Short names should map to the canonical (long) name
        assert gaz.get("De Bruyne") == "Kevin De Bruyne"
        assert gaz.get("Sterling") == "Raheem Sterling"
        assert gaz.get("Payet") == "Dimitri Payet"

    def test_extracts_surnames(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        # Surnames extracted from long_name should also be in gazetteer
        assert gaz.get("Agüero") == "Sergio Agüero"
        assert gaz.get("Reid") == "Winston Reid"

    def test_extracts_coaches(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        assert "Manuel Pellegrini" in gaz
        assert gaz.get("Pellegrini") == "Manuel Pellegrini"
        assert "Slaven Bilić" in gaz

    def test_extracts_referees(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        assert "Craig Pawson" in gaz
        assert gaz.get("Pawson") == "Craig Pawson"

    def test_extracts_teams(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        assert "Manchester City" in gaz
        assert "West Ham" in gaz

    def test_extracts_venue(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        assert "Etihad Stadium" in gaz

    def test_canonical_maps_to_itself(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        assert gaz["Kevin De Bruyne"] == "Kevin De Bruyne"
        assert gaz["Winston Reid"] == "Winston Reid"

    def test_empty_labels(self):
        gaz, etypes = extract_names_from_labels({})
        assert gaz == {}
        assert etypes == {}


class TestBuildGazetteer:
    """Tests for build_gazetteer()."""

    def test_builds_from_labels(self):
        gaz, etypes = build_gazetteer(SAMPLE_LABELS, include_learned=False)
        assert len(gaz) > 0
        assert "Kevin De Bruyne" in gaz

    def test_handles_none_labels(self):
        gaz, etypes = build_gazetteer(None, include_learned=False)
        assert gaz == {}
        assert etypes == {}


class TestBuildFirstnameMap:
    """Tests for build_firstname_map()."""

    def test_maps_first_names_to_canonical(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        fmap = build_firstname_map(gaz, etypes)
        # "kevin" (5 chars, >=4) should map to Kevin De Bruyne
        assert "kevin" in fmap
        assert "Kevin De Bruyne" in fmap["kevin"]

    def test_skips_short_first_names(self):
        """First names with <4 characters should be excluded."""
        labels = {
            "lineup": {
                "home": {
                    "players": [
                        {"long_name": "Mo Salah", "short_name": "Salah", "name": "Salah"},
                    ],
                    "coach": [],
                },
                "away": {"players": [], "coach": []},
            },
        }
        gaz, etypes = extract_names_from_labels(labels)
        fmap = build_firstname_map(gaz, etypes)
        assert "mo" not in fmap  # too short (2 chars)

    def test_excludes_teams_and_venues(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        fmap = build_firstname_map(gaz, etypes)
        # Team/venue words should not appear as first name keys
        assert "manchester" not in fmap
        assert "etihad" not in fmap

    def test_includes_coaches(self):
        gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
        fmap = build_firstname_map(gaz, etypes)
        assert "manuel" in fmap
        assert "Manuel Pellegrini" in fmap["manuel"]


class TestLearnedCorrectionFiltering:
    """Tests that low-confidence learned corrections don't pollute the gazetteer."""

    def test_rejects_low_confidence_learned_entries(self):
        """Learned corrections with seen_count=1 should NOT be merged."""
        low_conf_learned = {
            "blassie": {
                "correct": "Bolasie",
                "confidence": 0.5,
                "seen_count": 1,
                "fuzzy_score_avg": 78.6,
            },
        }
        with patch("pipeline.gazetteer.load_learned_corrections", return_value=low_conf_learned):
            gaz, _ = build_gazetteer(SAMPLE_LABELS, include_learned=True)
        assert "blassie" not in gaz

    def test_accepts_high_confidence_learned_entries(self):
        """Learned corrections with seen_count>=2 and confidence>=0.6 should be merged."""
        high_conf_learned = {
            "aspilicueta": {
                "correct": "Azpilicueta",
                "confidence": 0.9,
                "seen_count": 11,
                "fuzzy_score_avg": 96.5,
            },
        }
        with patch("pipeline.gazetteer.load_learned_corrections", return_value=high_conf_learned):
            gaz, _ = build_gazetteer(SAMPLE_LABELS, include_learned=True)
        assert "aspilicueta" in gaz
        assert gaz["aspilicueta"] == "Azpilicueta"
