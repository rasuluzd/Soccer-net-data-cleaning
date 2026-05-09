"""
Entity helpers (formerly Tier 2 fuzzy corrector).

After the May 2026 architectural refactor, the heavy Tier 2 fuzzy + phonetic
+ context scoring logic was replaced by ``pipeline/entity_corrector.py``
(TF-IDF retrieval + Qwen MCQ judge + 2-layer cache). This module is now a
slim home for the surface-form helpers that the new architecture still
needs:

  • ``Correction`` dataclass — simple typed record of one applied edit
  • ``extract_entity_core`` / ``extract_and_rebuild_entity`` — strip and
    re-attach punctuation/possessives around an entity span (preserves
    Germanic genitive ``-s`` for sv/de/no/da)
  • ``passes_conservative_gates`` — dictionary veto + C1 fuzzy floor + C2
    length tolerance, applied to every accepted correction regardless of
    which stage proposed it
  • ``compute_phonetic_score`` / ``_strip_accents`` — kept for the
    multilingual test suite; not used in production paths anymore

The DROPPED functions (compute_combined_score, find_best_match,
correct_segment_text) were the Tier 2 fuzzy/phonetic/context cascade. They
are obsolete: the new entity_corrector retrieves with TF-IDF char n-grams
(language-agnostic, no Metaphone) and reranks with the Qwen MCQ judge.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import jellyfish
from rapidfuzz import fuzz

from pipeline.config import (
    CONSERVATIVE_C1_FUZZY_FLOOR,
    CONSERVATIVE_C2_LEN_TOLERANCE,
    DICTIONARY_VETO_ENABLED,
    DICTIONARY_VETO_MIN_LEN,
)


# ─── Correction record (used by entity_corrector + report) ───────────

@dataclass
class Correction:
    """A single entity correction made by any pipeline stage."""
    original: str
    corrected: str
    combined_score: float
    fuzzy_score: float
    phonetic_match: bool
    context_match: bool
    segment_id: str
    method: str
    confidence_band: str = "accepted"
    half: int = 0


# ─── Surface-form parsing (entity span ↔ canonical name) ─────────────

def _split_entity_parts(original_text: str) -> tuple[str, str, str, str, str]:
    """Split entity text into (core, leading_punct, trailing_possessive, trailing_text, trailing_punct)."""
    temp = original_text.strip()

    trailing_punct = ""
    while temp and temp[-1] in ".,!?;:":
        trailing_punct = temp[-1] + trailing_punct
        temp = temp[:-1]

    trailing_text = ""
    lower_temp = temp.lower()
    if lower_temp.endswith(" and co"):
        trailing_text = temp[-7:]
        temp = temp[:-7]

    trailing_possessive = ""
    if temp.endswith("'s") or temp.endswith("’s"):
        trailing_possessive = temp[-2:]
        temp = temp[:-2]

    leading_punct = ""
    while temp and temp[0] in ".,!?;:":
        leading_punct += temp[0]
        temp = temp[1:]

    return temp, leading_punct, trailing_possessive, trailing_text, trailing_punct


def extract_entity_core(original_text: str) -> str:
    """Return the bare entity text (no leading/trailing punct or possessive)."""
    core, _, _, _, _ = _split_entity_parts(original_text)
    return core


def extract_and_rebuild_entity(
    original_text: str,
    corrected_name: str,
    language: str = "en",
) -> str:
    """Strip punct/possessive, swap the name, rebuild the original wrapping.

    For Germanic languages (sv, de, no, da) — preserve a bare trailing ``-s``
    on the original as a genitive marker even when the canonical form drops
    it (ASR transcribes "Guidettis boll" → fuzzy maps to "Guidetti" → we
    re-append the s so the genitive survives).
    """
    core, leading_punct, trailing_possessive, trailing_text, trailing_punct = (
        _split_entity_parts(original_text)
    )
    final_name = corrected_name
    if (
        language in ("sv", "de", "no", "da")
        and core
        and len(core) >= 4
        and core.endswith("s")
        and not corrected_name.endswith("s")
        and not trailing_possessive
    ):
        final_name = corrected_name + "s"
    return leading_punct + final_name + trailing_possessive + trailing_text + trailing_punct


# ─── Phonetic helpers (kept for multilingual test suite) ─────────────

def _strip_accents(text: str) -> str:
    """NFKD-decompose and drop combining marks. Used for accent-neutral
    phonetic comparison across Nordic / Romance / Germanic alphabets."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _phonetic_distance_score(code_a: str, code_b: str) -> float:
    if not code_a or not code_b:
        return 0.0
    max_len = max(len(code_a), len(code_b))
    distance = jellyfish.levenshtein_distance(code_a, code_b)
    return 100.0 * (1.0 - distance / max_len)


def compute_phonetic_score(entity: str, candidate: str, language: str = "en") -> float:
    """Continuous (0-100) phonetic similarity of two strings.

    Used by the multilingual test suite to verify phonetic fallback works
    for non-English. Production entity correction (entity_corrector.py)
    does not use this function — TF-IDF char n-grams replaced it because
    Metaphone+Jaro-Winkler over short codes produces prefix-collision
    artifacts on football-name pairs (Saturday/Sturridge ≈ 0.92).
    """
    try:
        if language == "en":
            scores = []
            entity_code = jellyfish.metaphone(entity)
            candidate_code = jellyfish.metaphone(candidate)
            scores.append(_phonetic_distance_score(entity_code, candidate_code))
            try:
                entity_sx = jellyfish.soundex(entity)
                candidate_sx = jellyfish.soundex(candidate)
                scores.append(_phonetic_distance_score(entity_sx, candidate_sx))
            except Exception:
                pass
            return max(scores) if scores else 0.0
        entity_norm = _strip_accents(entity)
        candidate_norm = _strip_accents(candidate)
        scores = []
        try:
            scores.append(_phonetic_distance_score(
                jellyfish.soundex(entity_norm), jellyfish.soundex(candidate_norm),
            ))
        except Exception:
            pass
        try:
            scores.append(_phonetic_distance_score(
                jellyfish.metaphone(entity_norm), jellyfish.metaphone(candidate_norm),
            ))
        except Exception:
            pass
        return max(scores) if scores else 0.0
    except Exception:
        return 0.0


# ─── Conservative validation gates ───────────────────────────────────

_SPELLCHECKER_CACHE: dict = {}

# Threshold at which a fuzzy ratio is high enough to indicate a genuine
# ASR mishearing rather than a cross-domain accident. Above this we
# trust the fuzzy similarity and skip the dictionary veto.
_DICTIONARY_VETO_FUZZY_TRUST = 75


def _is_dictionary_word(word: str, language: str = "en") -> bool:
    """True iff ``word.lower()`` is in the language's spell-check dictionary.

    Uses pyspellchecker (already a project dep). Cached per language.
    Returns False on any error so a missing dictionary degrades to
    "veto disabled" rather than crashing.
    """
    try:
        sc = _SPELLCHECKER_CACHE.get(language)
        if sc is None:
            from spellchecker import SpellChecker
            sc = SpellChecker(language=language)
            _SPELLCHECKER_CACHE[language] = sc
        return word.lower() in sc
    except Exception:
        return False


def passes_conservative_gates(
    original: str, corrected: str, language: str = "en",
) -> bool:
    """Return True iff the correction passes ALL safety gates.

    Gates (fail-fast order):

      • C1 fuzzy floor: ``fuzz.ratio(orig, corr) ≥ CONSERVATIVE_C1_FUZZY_FLOOR``.
        Rejects wildly-different replacements (Kommer→Kouame ≈ 20).
      • Dictionary veto: if the ORIGINAL is a real ``language`` word
        (length ≥ ``DICTIONARY_VETO_MIN_LEN``) AND the fuzzy ratio to
        the proposed correction is below ``_DICTIONARY_VETO_FUZZY_TRUST``,
        reject. The fuzzy-trust escape lets legitimate ASR mishearings
        like Williams→Willian (ratio 80) and Klein→Clyne (ratio 80)
        through, while still blocking cross-domain accidents like
        Saturday→Sturridge (ratio 59) and Dutchman→Mane (ratio 50).
      • C2 length tolerance: ``|len(corr) − len(orig)| ≤ max(2, tol × len(orig))``.
        Rejects runaway expansions/contractions.
    """
    if not original or not corrected:
        return False
    if original.strip().lower() == corrected.strip().lower():
        return True
    ratio = fuzz.ratio(original.lower(), corrected.lower())
    if ratio < CONSERVATIVE_C1_FUZZY_FLOOR:
        return False
    if (
        DICTIONARY_VETO_ENABLED
        and ratio < _DICTIONARY_VETO_FUZZY_TRUST
    ):
        core = original.strip(" .,!?;:'\"()-")
        if (
            len(core) >= DICTIONARY_VETO_MIN_LEN
            and _is_dictionary_word(core, language)
        ):
            return False
    o_len = len(original)
    c_len = len(corrected)
    tolerance = max(2, int(CONSERVATIVE_C2_LEN_TOLERANCE * o_len))
    if abs(c_len - o_len) > tolerance:
        return False
    return True
