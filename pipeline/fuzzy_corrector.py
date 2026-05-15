"""Surface-form helpers + validation gates used by entity_corrector.
The heavy fuzzy/phonetic scoring lives in entity_corrector.py now;
this module just keeps the boundary-handling and gate logic + the
phonetic helpers the multilingual test suite still imports."""

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
    """One entity correction. Produced by any stage that edits entities."""
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
    """Returns (core, leading_punct, trailing_possessive, trailing_text, trailing_punct)."""
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
    """Bare entity, no surrounding punct or possessive."""
    core, _, _, _, _ = _split_entity_parts(original_text)
    return core


def extract_and_rebuild_entity(
    original_text: str,
    corrected_name: str,
    language: str = "en",
) -> str:
    """Swap the name in-place, keep punctuation/possessive wrapping.
    For sv/de/no/da, re-attach a Germanic genitive -s if the original had one
    ("Guidettis boll" stays a genitive after correcting to "Guidetti")."""
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
    """NFKD-decompose and drop combining marks (accent-neutral comparison)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _phonetic_distance_score(code_a: str, code_b: str) -> float:
    if not code_a or not code_b:
        return 0.0
    max_len = max(len(code_a), len(code_b))
    distance = jellyfish.levenshtein_distance(code_a, code_b)
    return 100.0 * (1.0 - distance / max_len)


def compute_phonetic_score(entity: str, candidate: str, language: str = "en") -> float:
    """Phonetic similarity 0-100. Only used by the multilingual test suite —
    production entity correction uses TF-IDF instead."""
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

# Above this fuzz ratio we treat the pair as a genuine mishearing
# and skip the dictionary veto.
_DICTIONARY_VETO_FUZZY_TRUST = 75


def _is_dictionary_word(word: str, language: str = "en") -> bool:
    """True if word is in the language spell dict. Cached per language.
    Returns False on any error (degrades to 'no veto' rather than crashing)."""
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
    """All-gates check: C1 fuzz floor, dictionary veto, C2 length tolerance.
    The fuzz-trust line lets Williams->Willian (80) through while blocking
    Saturday->Sturridge (59) and similar cross-domain mishaps."""
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
