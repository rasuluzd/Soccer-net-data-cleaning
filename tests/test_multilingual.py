"""
Tests for multilingual support.

Verifies that:
- Language detection correctly identifies Swedish and English commentary
- Swedish text passes the hallucination filter (alpha ratio, language check)
- Entity extraction works with Nordic characters (å, ä, ö)
- Phonetic scoring works for non-English names via accent normalization
- POS-based word filtering works for both English and non-English
"""

import pytest
from pipeline.hallucination_filter import (
    compute_alpha_ratio,
    detect_commentary_language,
    is_valid_commentary,
    filter_segment,
)
from pipeline.fuzzy_corrector import compute_phonetic_score, _strip_accents
from pipeline.ner_extractor import CAPITALIZED_WORD, DetectedEntity
from pipeline.config import (
    get_spacy_model,
    get_context_model,
    get_entity_labels,
    get_scoring_weights,
    LANGUAGE_FAMILIES,
    REJECTED_POS_TAGS,
)
from pipeline.loader import Segment


# ─── Config helpers ──────────────────────────────────────────────────

class TestConfigHelpers:
    """Verify language-conditional config functions."""

    def test_english_spacy_model(self):
        assert get_spacy_model("en") == "en_core_web_sm"

    def test_default_spacy_model(self):
        assert get_spacy_model("sv") == "xx_ent_wiki_sm"

    def test_english_context_model(self):
        assert get_context_model("en") == "all-MiniLM-L6-v2"

    def test_noneng_context_model(self):
        assert get_context_model("sv") == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_english_entity_labels(self):
        labels = get_entity_labels("en")
        assert "PERSON" in labels
        assert "PER" not in labels

    def test_multilingual_entity_labels(self):
        labels = get_entity_labels("sv")
        assert "PER" in labels
        assert "PERSON" not in labels

    def test_english_scoring_weights(self):
        fw, pw, cw = get_scoring_weights("en")
        assert fw == 0.45
        assert pw == 0.40
        assert cw == 0.15

    def test_noneng_scoring_weights(self):
        fw, pw, cw = get_scoring_weights("sv")
        assert fw == 0.55
        assert pw == 0.30
        assert cw == 0.15

    def test_language_families(self):
        assert "sv" in LANGUAGE_FAMILIES
        assert "en" in LANGUAGE_FAMILIES
        assert "no" in LANGUAGE_FAMILIES["sv"]  # Norwegian in Swedish family


# ─── Alpha ratio ─────────────────────────────────────────────────────

class TestAlphaRatioMultilingual:
    """Alpha ratio must accept accented Latin characters."""

    def test_swedish_text_with_diacritics(self):
        ratio = compute_alpha_ratio("Zlatan gör ett fantastiskt mål")
        assert ratio > 0.85

    def test_german_text_with_umlauts(self):
        ratio = compute_alpha_ratio("Müller schießt und trifft")
        assert ratio > 0.85

    def test_french_text_with_accents(self):
        ratio = compute_alpha_ratio("Mbappé accélère sur le côté")
        assert ratio > 0.85

    def test_pure_ascii_english(self):
        ratio = compute_alpha_ratio("Sterling scores a brilliant goal")
        assert ratio > 0.90

    def test_cjk_still_rejected(self):
        """CJK characters should NOT count as alpha."""
        ratio = compute_alpha_ratio("已经就冕复了")
        assert ratio == 0.0


# ─── Language detection ──────────────────────────────────────────────

class TestLanguageDetection:
    """Language detection from commentary segments."""

    def _make_segments(self, texts):
        return [
            Segment(
                segment_id=str(i),
                start_time=float(i * 10),
                end_time=float(i * 10 + 5),
                text=text,
                half=1,
            )
            for i, text in enumerate(texts)
        ]

    def test_english_detected(self):
        segs = self._make_segments([
            "Sterling passes the ball to Aguero on the left side",
            "What a fantastic run from the midfielder, he beats two defenders",
            "The goalkeeper makes an incredible save to deny the striker",
            "Corner kick coming in from the right, headed away by the defense",
        ])
        lang = detect_commentary_language(segs)
        assert lang == "en"

    def test_swedish_detected(self):
        segs = self._make_segments([
            "Zlatan gör ett fantastiskt mål med vänsterfoten från straffområdet",
            "Bollen spelas upp på mittfältet och det blir en fin passning framåt",
            "Domaren blåser för frispark efter en tuff tackling på mittfältet",
            "Målvakten gör en fantastisk räddning och räddar skottet på mållinjen",
        ])
        lang = detect_commentary_language(segs)
        assert lang == "sv"

    def test_empty_segments_default_english(self):
        lang = detect_commentary_language([])
        assert lang == "en"


# ─── Commentary validation ───────────────────────────────────────────

class TestCommentaryValidation:
    """is_valid_commentary must accept expected language families."""

    def test_english_accepted_for_en(self):
        text = "Sterling makes a great pass to Aguero who scores the winning goal"
        assert is_valid_commentary(text, "en") is True

    def test_swedish_accepted_for_sv(self):
        text = "Zlatan gör ett fantastiskt mål och laget leder med två mål till noll"
        assert is_valid_commentary(text, "sv") is True

    def test_swedish_rejected_for_en(self):
        text = "Zlatan gör ett fantastiskt mål och laget leder med två mål till noll"
        assert is_valid_commentary(text, "en") is False

    def test_short_text_always_accepted(self):
        """Texts under 8 words are too short for reliable detection."""
        assert is_valid_commentary("kort text", "en") is True


# ─── Filter segment ─────────────────────────────────────────────────

class TestFilterSegmentMultilingual:
    """filter_segment must work with non-English expected language."""

    def test_swedish_segment_passes_with_sv_lang(self):
        seg = Segment("1", 0.0, 5.0, "Zlatan gör ett fantastiskt mål och laget jublar", 1)
        is_valid, reason = filter_segment(seg, expected_lang="sv")
        assert is_valid is True

    def test_swedish_segment_with_diacritics_passes_alpha(self):
        seg = Segment("1", 0.0, 5.0, "Allbäck skjuter hårt och gör mål", 1)
        is_valid, reason = filter_segment(seg, expected_lang="sv")
        assert is_valid is True


# ─── Phonetic scoring ───────────────────────────────────────────────

class TestPhoneticMultilingual:
    """Phonetic scoring should work for non-English via accent normalization."""

    def test_strip_accents(self):
        assert _strip_accents("Agüero") == "Aguero"
        assert _strip_accents("Özil") == "Ozil"
        assert _strip_accents("Allbäck") == "Allback"
        assert _strip_accents("Ibrahimović") == "Ibrahimovic"

    def test_english_metaphone_still_works(self):
        score = compute_phonetic_score("Sacco", "Sakho", language="en")
        assert score > 0

    def test_noneng_accent_normalized_soundex(self):
        """Swedish: Allbäck vs Allback should match after normalization."""
        score = compute_phonetic_score("Allbäck", "Allback", language="sv")
        assert score >= 75.0

    def test_noneng_phonetic_different_names(self):
        """Completely different names should score 0."""
        score = compute_phonetic_score("Zlatan", "Svensson", language="sv")
        assert score == 0.0


# ─── Capitalized word regex ─────────────────────────────────────────

class TestCapitalizedWordRegex:
    """CAPITALIZED_WORD regex must match Nordic characters."""

    def test_matches_swedish_names(self):
        text = "Allbäck skjuter och Öberg nickar"
        matches = CAPITALIZED_WORD.findall(text)
        assert "Allbäck" in matches
        assert "Öberg" in matches

    def test_matches_norwegian_name(self):
        text = "Ødegaard passes to Ålborg"
        matches = CAPITALIZED_WORD.findall(text)
        # Ødegaard starts with Ø which is in our expanded uppercase class
        assert any("degaard" in m for m in matches)

    def test_still_matches_english(self):
        text = "Sterling passes to Aguero"
        matches = CAPITALIZED_WORD.findall(text)
        assert "Sterling" in matches
        assert "Aguero" in matches


# ─── POS-based rejection ────────────────────────────────────────────

class TestPOSRejection:
    """Verify REJECTED_POS_TAGS contains the right tags."""

    def test_noun_rejected(self):
        assert "NOUN" in REJECTED_POS_TAGS

    def test_verb_rejected(self):
        assert "VERB" in REJECTED_POS_TAGS

    def test_adj_rejected(self):
        assert "ADJ" in REJECTED_POS_TAGS

    def test_propn_not_rejected(self):
        assert "PROPN" not in REJECTED_POS_TAGS

    def test_empty_pos_not_rejected(self):
        """Empty POS (no tag info) should not be rejected."""
        assert "" not in REJECTED_POS_TAGS
