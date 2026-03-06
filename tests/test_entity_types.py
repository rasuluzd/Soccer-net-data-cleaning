"""
Tests for the Entity-Type Tagging system.

Verifies that:
- Entity types are correctly assigned (player, coach, referee, team, venue)
- Team words are correctly extracted from team/venue names
- Fuzzy corrector rejects corrections targeting team/venue types
- Fuzzy corrector skips entities containing team word fragments
- Tier 3 candidate pool excludes team/venue types
- Learned dictionary filters out team/venue-type corrections
- Valid player corrections still work (no regressions)
"""

import pytest
from pipeline.gazetteer import extract_names_from_labels, get_team_words
from pipeline.fuzzy_corrector import find_best_match, Correction


# ─── Shared test fixtures ────────────────────────────────────────────

SAMPLE_LABELS = {
    "gameHomeTeam": "Chelsea",
    "gameAwayTeam": "Crystal Palace",
    "lineup": {
        "home": {
            "players": [
                {"long_name": "Eden Hazard", "short_name": "Hazard", "name": "Hazard"},
                {"long_name": "Willian Borges da Silva", "short_name": "Willian", "name": "Willian"},
                {"long_name": "Kurt Zouma", "short_name": "Zouma", "name": "Zouma"},
            ],
            "coach": [
                {"long_name": "José Mourinho", "short_name": "Mourinho", "name": "Mourinho"},
            ],
        },
        "away": {
            "players": [
                {"long_name": "Connor Wickham", "short_name": "Wickham", "name": "Wickham"},
                {"long_name": "Yannick Bolasie", "short_name": "Bolasie", "name": "Bolasie"},
                {"long_name": "Bakary Sako", "short_name": "Sako", "name": "Sako"},
            ],
            "coach": [
                {"long_name": "Alan Pardew", "short_name": "Pardew", "name": "Pardew"},
            ],
        },
    },
    "referee_matched": ["Mark Clattenburg"],
    "venue": ["Stamford Bridge"],
}


@pytest.fixture
def gaz_and_types():
    """Build gazetteer and entity types from sample labels."""
    gaz, etypes = extract_names_from_labels(SAMPLE_LABELS)
    return gaz, etypes


@pytest.fixture
def team_words_set(gaz_and_types):
    """Build the team words set."""
    gaz, etypes = gaz_and_types
    return get_team_words(etypes, gaz)


# ─── Entity Type Assignment ──────────────────────────────────────────

class TestEntityTypeAssignment:
    """Verify entity types are correctly assigned."""

    def test_players_typed_as_player(self, gaz_and_types):
        _, etypes = gaz_and_types
        assert etypes.get("Eden Hazard") == "player"
        assert etypes.get("Connor Wickham") == "player"
        assert etypes.get("Yannick Bolasie") == "player"

    def test_coaches_typed_as_coach(self, gaz_and_types):
        _, etypes = gaz_and_types
        assert etypes.get("José Mourinho") == "coach"
        assert etypes.get("Alan Pardew") == "coach"

    def test_referees_typed_as_referee(self, gaz_and_types):
        _, etypes = gaz_and_types
        assert etypes.get("Mark Clattenburg") == "referee"

    def test_teams_typed_as_team(self, gaz_and_types):
        _, etypes = gaz_and_types
        assert etypes.get("Chelsea") == "team"
        assert etypes.get("Crystal Palace") == "team"

    def test_venues_typed_as_venue(self, gaz_and_types):
        _, etypes = gaz_and_types
        assert etypes.get("Stamford Bridge") == "venue"

    def test_empty_labels_no_types(self):
        _, etypes = extract_names_from_labels({})
        assert etypes == {}


# ─── Team Words Extraction ───────────────────────────────────────────

class TestTeamWords:
    """Verify get_team_words extracts correct fragments."""

    def test_extracts_team_name_words(self, team_words_set):
        assert "chelsea" in team_words_set
        assert "crystal" in team_words_set
        assert "palace" in team_words_set

    def test_extracts_venue_words(self, team_words_set):
        assert "stamford" in team_words_set
        assert "bridge" in team_words_set

    def test_excludes_short_words(self):
        """Words shorter than 3 chars should be excluded."""
        labels = {
            "gameHomeTeam": "FC Porto",
            "gameAwayTeam": "AS Roma",
        }
        _, etypes = extract_names_from_labels(labels)
        gaz, _ = extract_names_from_labels(labels)
        tw = get_team_words(etypes, gaz)
        assert "fc" not in tw    # 2 chars
        assert "as" not in tw    # 2 chars
        assert "porto" in tw     # 5 chars
        assert "roma" in tw      # 4 chars


# ─── Fuzzy Corrector Type-Aware Rejection ────────────────────────────

class TestFuzzyTypeRejection:
    """Verify fuzzy corrector rejects type-mismatched corrections."""

    def test_rejects_correction_to_team_name(self, gaz_and_types):
        """An entity should never be 'corrected' to a team name."""
        gaz, etypes = gaz_and_types
        # "Chelsey" is close to "Chelsea" (team) — should be rejected
        match = find_best_match(
            "Chelsey", gaz, entity_types=etypes,
        )
        assert match is None

    def test_skips_entity_with_team_word_fragment(self, gaz_and_types):
        """Multi-word entities containing team words should be skipped."""
        gaz, etypes = gaz_and_types
        tw = get_team_words(etypes, gaz)
        # "Wickham Palace" — "Palace" is a team word
        match = find_best_match(
            "Wickham Palace", gaz,
            entity_types=etypes, team_words=tw,
        )
        assert match is None

    def test_allows_valid_player_correction(self, gaz_and_types):
        """Valid player misspellings should still be corrected."""
        gaz, etypes = gaz_and_types
        tw = get_team_words(etypes, gaz)
        # "Zuma" → "Zouma" should still work
        match = find_best_match(
            "Zuma", gaz, entity_types=etypes, team_words=tw,
        )
        assert match is not None
        assert match.corrected == "Zouma"

    def test_allows_valid_multi_word_correction(self, gaz_and_types):
        """Multi-word player names without team words should correct."""
        gaz, etypes = gaz_and_types
        tw = get_team_words(etypes, gaz)
        # "Conor Wickham" → "Connor Wickham" should still work
        match = find_best_match(
            "Conor Wickham", gaz,
            entity_types=etypes, team_words=tw,
        )
        assert match is not None
        assert match.corrected == "Connor Wickham"


# ─── Tier 3 Candidate Filtering ─────────────────────────────────────

class TestTier3CandidateFiltering:
    """Verify Tier 3 excludes team/venue from candidates."""

    def test_excludes_teams_from_candidates(self, gaz_and_types):
        from pipeline.context_disambiguator import build_candidate_descriptions
        gaz, etypes = gaz_and_types
        descriptions = build_candidate_descriptions(gaz, entity_types=etypes)
        # Team names should not appear as candidates
        assert "Chelsea" not in descriptions
        assert "Crystal Palace" not in descriptions

    def test_excludes_venues_from_candidates(self, gaz_and_types):
        from pipeline.context_disambiguator import build_candidate_descriptions
        gaz, etypes = gaz_and_types
        descriptions = build_candidate_descriptions(gaz, entity_types=etypes)
        assert "Stamford Bridge" not in descriptions

    def test_includes_players_in_candidates(self, gaz_and_types):
        from pipeline.context_disambiguator import build_candidate_descriptions
        gaz, etypes = gaz_and_types
        descriptions = build_candidate_descriptions(gaz, entity_types=etypes)
        assert "Eden Hazard" in descriptions
        assert "Connor Wickham" in descriptions

    def test_includes_coaches_in_candidates(self, gaz_and_types):
        from pipeline.context_disambiguator import build_candidate_descriptions
        gaz, etypes = gaz_and_types
        descriptions = build_candidate_descriptions(gaz, entity_types=etypes)
        assert "José Mourinho" in descriptions
        assert "Alan Pardew" in descriptions


# ─── Learned Dictionary Filtering ────────────────────────────────────

class TestLearnedDictFiltering:
    """Verify learned dict excludes team/venue corrections."""

    def test_filters_team_corrections(self, gaz_and_types, tmp_path, monkeypatch):
        from pipeline.learned_dictionary import update_learned_dictionary
        from pipeline.config import LEARNED_CORRECTIONS_PATH
        import pipeline.config

        # Point the learned corrections to a temp file
        temp_file = tmp_path / "test_learned.json"
        monkeypatch.setattr(pipeline.config, "LEARNED_CORRECTIONS_PATH", temp_file)
        # Also need to patch the module-level import in learned_dictionary
        import pipeline.learned_dictionary
        monkeypatch.setattr(pipeline.learned_dictionary, "LEARNED_CORRECTIONS_PATH", temp_file)

        _, etypes = gaz_and_types

        corrections = [
            Correction(
                original="Chelsey",
                corrected="Chelsea",   # team type — should be filtered
                combined_score=80.0,
                fuzzy_score=80.0,
                phonetic_match=True,
                context_match=False,
                segment_id="0",
                method="fuzzy(80)",
            ),
            Correction(
                original="Zuma",
                corrected="Kurt Zouma",  # player type — should be kept
                combined_score=75.0,
                fuzzy_score=75.0,
                phonetic_match=True,
                context_match=False,
                segment_id="0",
                method="fuzzy(75)",
            ),
        ]

        result = update_learned_dictionary(corrections, entity_types=etypes)

        # Team correction should be filtered out
        assert "chelsey" not in result
        # Player correction should be kept
        assert "zuma" in result
        assert result["zuma"]["correct"] == "Kurt Zouma"
