"""
Fuzzy Corrector — matches detected entities to the gazetteer and corrects them.

Uses a multi-signal scoring approach:
    Score = FUZZY_WEIGHT × fuzzy_ratio + PHONETIC_WEIGHT × phonetic_match + CONTEXT_WEIGHT × context_bonus

This captures both string similarity AND sound similarity, which is critical
for ASR errors where names are phonetically similar but textually different
(e.g., "Sacco" sounds like "Sakho" but has low string similarity).
"""

from dataclasses import dataclass
from typing import Optional

import jellyfish
from rapidfuzz import fuzz, process

from pipeline.config import (
    FUZZY_WEIGHT,
    PHONETIC_WEIGHT,
    CONTEXT_WEIGHT,
    get_fuzzy_threshold,
    TIER2_ACCEPT_THRESHOLD,
)
from pipeline.ner_extractor import DetectedEntity


# ─── Common English words that should NEVER be corrected ────────────
# These are real English words or names that fuzz matching might
# incorrectly "correct" to a player name.
COMMON_WORDS_EXCLUDE = {
    # Common first names (often valid on their own in commentary)
    "target", "dan", "davies", "will", "young", "long", "ward",
    "allen", "paul", "mark", "jones", "parker", "walker", "kennedy",
    "martin", "alex", "jack", "joe", "tom", "nick", "mike", "john",
    "james", "ryan", "adam", "ben", "sam", "matt", "chris", "lee",
    "tony", "gary", "steven", "frank", "henry", "barry", "terry",
    "wayne", "dean", "carl", "dave", "rob", "phil", "gordon",
    "jose", "diego", "alan", "scott", "kurt", "adrian", "pedro",
    "william", "bia", "kael",
    # Soccer terms that might fuzzy-match names
    "corner", "cross", "header", "pass", "shot", "goal", "foul",
    "ball", "match", "kick", "play", "side", "team", "half",
    "free", "throw", "card", "yellow", "red", "penalty",
    "manager", "referee", "striker", "keeper", "defender",
    # Common English words that phonetically resemble player names
    "poor", "poured", "punch", "wall", "kale", "kyle",
    "chile", "marino", "falco", "pele",
    # Geographic names that aren't player misspellings
    "stamford bridge",
}


@dataclass
class Correction:
    """A single entity correction made by the fuzzy matcher."""
    original: str           # the misspelled entity as it appeared in ASR
    corrected: str          # the canonical name from the gazetteer
    combined_score: float   # the multi-signal score (0–100)
    fuzzy_score: float      # RapidFuzz similarity score
    phonetic_match: bool    # whether Metaphone codes matched
    context_match: bool     # whether team context matched
    segment_id: str         # which segment this correction belongs to
    method: str             # description of matching method
    confidence_band: str = "accepted"  # "accepted" (≥75) or "uncertain" (48-74)


def _split_entity_parts(original_text: str) -> tuple[str, str, str, str]:
    """Split entity text into (core, leading_punct, trailing_possessive, trailing_punct)."""
    temp = original_text.strip()

    # 1) Strip trailing punctuation first (important for cases like "Ward's.")
    trailing_punct = ""
    while temp and temp[-1] in ".,!?;:":
        trailing_punct = temp[-1] + trailing_punct
        temp = temp[:-1]

    # 2) Strip possessive
    trailing_possessive = ""
    if temp.endswith("'s") or temp.endswith("\u2019s"):
        trailing_possessive = temp[-2:]
        temp = temp[:-2]

    # 3) Strip leading punctuation
    leading_punct = ""
    while temp and temp[0] in ".,!?;:":
        leading_punct += temp[0]
        temp = temp[1:]

    return temp, leading_punct, trailing_possessive, trailing_punct


def extract_entity_core(original_text: str) -> str:
    """Extract normalized entity text used for lookup/matching."""
    core, _, _, _ = _split_entity_parts(original_text)
    return core


def extract_and_rebuild_entity(original_text: str, corrected_name: str) -> str:
    """Safely strips punctuation/possessives, swaps the name, and rebuilds the string."""
    _, leading_punct, trailing_possessive, trailing_punct = _split_entity_parts(original_text)
    return leading_punct + corrected_name + trailing_possessive + trailing_punct


def compute_phonetic_score(entity: str, candidate: str) -> float:
    """
    Compare two strings phonetically using Double Metaphone.

    ASR errors are fundamentally *sound-based* — Whisper hears phonemes
    and maps them to wrong spellings. Phonetic matching catches cases
    where names sound alike but look very different:
        "Sacco" ≈ "Sakho" (both → "SK")
        "Turi"  ≈ "Touré" (both → "TR")

    Returns:
        100.0 if phonetic codes match, 50.0 for partial match, 0.0 otherwise.
    """
    try:
        # Get Metaphone codes (primary representation of pronunciation)
        entity_code = jellyfish.metaphone(entity)
        candidate_code = jellyfish.metaphone(candidate)

        if entity_code == candidate_code:
            return 100.0

        # Partial match: one is a prefix of the other (e.g., KLRF ⊂ KLRHF)
        # This is very common in ASR errors where extra syllables are added
        if entity_code.startswith(candidate_code) or candidate_code.startswith(entity_code):
            return 75.0

        # Also try Soundex for a second opinion
        try:
            entity_soundex = jellyfish.soundex(entity)
            candidate_soundex = jellyfish.soundex(candidate)
            if entity_soundex == candidate_soundex:
                return 75.0
        except Exception:
            pass

        return 0.0
    except Exception:
        return 0.0


def compute_combined_score(
    entity_text: str,
    candidate: str,
    context_names: Optional[set[str]] = None,
) -> tuple[float, float, bool, bool]:
    """
    Compute the multi-signal score for matching an entity to a candidate.

    Score = FUZZY_WEIGHT × fuzzy + PHONETIC_WEIGHT × phonetic + CONTEXT_WEIGHT × context

    Args:
        entity_text: the detected entity text (potentially misspelled)
        candidate: the gazetteer entry to compare against
        context_names: set of other entity names in the same segment/nearby segments
                      (used for context bonus — if other players from the same team
                      are mentioned, this candidate is more likely correct)

    Returns:
        Tuple of (combined_score, fuzzy_score, phonetic_match, context_match)
    """
    # Signal 1: Fuzzy string similarity (token_sort handles word order)
    fuzzy_score = fuzz.token_sort_ratio(entity_text.lower(), candidate.lower())

    # Signal 2: Phonetic similarity
    phonetic_score = compute_phonetic_score(entity_text, candidate)
    phonetic_match = phonetic_score >= 50.0

    # Signal 3: Context bonus
    context_score = 0.0
    context_match = False
    if context_names:
        # If other known names appear in the same context, boost confidence
        overlap = context_names.intersection({candidate})
        if overlap:
            context_score = 100.0
            context_match = True

    # Combined weighted score
    combined = (
        FUZZY_WEIGHT * fuzzy_score
        + PHONETIC_WEIGHT * phonetic_score
        + CONTEXT_WEIGHT * context_score
    )

    return combined, fuzzy_score, phonetic_match, context_match


def find_best_match(
    entity_text: str,
    gazetteer: dict[str, str],
    context_names: Optional[set[str]] = None,
    entity_types: Optional[dict[str, str]] = None,
    team_words: Optional[set[str]] = None,
) -> Optional[Correction]:
    """
    Find the best matching gazetteer entry for a detected entity.

    First does a rapid pre-filter using RapidFuzz to get top candidates,
    then scores each with the full multi-signal approach.

    Args:
        entity_text: the entity text to correct
        gazetteer: dict mapping name variants to canonical names
        context_names: nearby entity names for context scoring

    Returns:
        A Correction object if a match was found above threshold, or None.
    """
    if not entity_text or not gazetteer:
        return None

    entity_clean = extract_entity_core(entity_text)

    if not entity_clean:
        return None

    # Skip common English words that aren't player names
    if entity_clean.lower() in COMMON_WORDS_EXCLUDE:
        return None

    # Skip very short entities (≤2 chars) — too ambiguous
    if len(entity_clean) <= 2:
        return None

    # Skip if the entity is already in the gazetteer (already correct)
    if entity_clean in gazetteer:
        return None

    # Also skip if lowercase version matches
    entity_lower = entity_clean.lower()
    for variant in gazetteer:
        if variant.lower() == entity_lower:
            return None

    # Skip multi-name entities where each word is already a valid name.
    # spaCy sometimes merges adjacent names into one entity:
    #   "Ivanovic Zouma" → both are correct player surnames
    #   "Wickham Palace" → player name + team name
    # Correcting these would overwrite one valid name with another.
    entity_words = entity_clean.split()
    if len(entity_words) >= 2:
        gazetteer_lower = {k.lower() for k in gazetteer}
        words_in_gazetteer = sum(
            1 for w in entity_words if w.lower() in gazetteer_lower
        )
        if words_in_gazetteer >= 2:
            return None  # Both words are known names — don't "correct"

    # Skip entities that contain a team/venue word fragment.
    # "Wickham Palace" → skip because "palace" is a team word.
    # "pardew gala" → not skipped ("gala" isn't a team word), handled by scoring.
    if team_words and len(entity_words) >= 2:
        entity_lower_words = {w.lower() for w in entity_words}
        if entity_lower_words & team_words:
            return None

    # ── Step 1: Pre-filter with RapidFuzz to get top 5 candidates ────
    all_variants = list(gazetteer.keys())
    top_candidates = process.extract(
        entity_clean,
        all_variants,
        scorer=fuzz.token_sort_ratio,
        limit=5,
    )

    if not top_candidates:
        return None

    # ── Step 2: Score each candidate with multi-signal approach ───────
    best_score = 0.0
    best_correction = None

    for candidate_text, raw_fuzzy, _ in top_candidates:
        combined, fuzzy_score, phonetic_match, context_match = compute_combined_score(
            entity_clean, candidate_text, context_names
        )

        if combined > best_score:
            best_score = combined
            canonical = gazetteer[candidate_text]

            # Decide replacement form: match the "level" of the input.
            # If commentator says "Zuma" (single word), correct to "Zouma"
            # (the matched variant), not "Kurt Zouma" (the full name).
            # If they say "Winston Ritu" (multi-word), use "Winston Reid".
            entity_word_count = len(entity_clean.split())
            if entity_word_count == 1:
                # Single word input → use the closest single-word variant
                # from the gazetteer, not the full canonical name
                if " " not in candidate_text:
                    corrected_name = candidate_text   # e.g. "Zouma"
                else:
                    # Matched a full name variant; find the word from the
                    # canonical name that's MOST SIMILAR to the entity.
                    # This prevents "Stamford"→"Bridge" (venue) while
                    # properly handling "Zuma"→"Zouma" (player surname).
                    canonical_words = canonical.split()
                    best_word = canonical_words[-1]  # default to surname
                    best_word_sim = 0
                    for cw in canonical_words:
                        wsim = fuzz.ratio(entity_clean.lower(), cw.lower())
                        if wsim > best_word_sim:
                            best_word_sim = wsim
                            best_word = cw
                    corrected_name = best_word
            else:
                corrected_name = canonical             # e.g. "Winston Reid"

            best_correction = Correction(
                original=entity_clean,
                corrected=corrected_name,
                combined_score=combined,
                fuzzy_score=fuzzy_score,
                phonetic_match=phonetic_match,
                context_match=context_match,
                segment_id="",  # will be set by caller
                method=_describe_method(fuzzy_score, phonetic_match, context_match),
            )

    # ── Step 3: Check if best score meets the adaptive threshold ─────
    if best_correction:
        threshold = get_fuzzy_threshold(entity_clean)
        if best_correction.combined_score >= threshold:
            # Reject self-corrections (entity → same text)
            if best_correction.corrected.lower() == entity_clean.lower():
                return None
            # Reject corrections that target a team or venue name.
            # A player-entity should never be "corrected" to a team name.
            if entity_types:
                target_type = entity_types.get(canonical, "")
                if target_type in ("team", "venue"):
                    return None
            # Tag confidence band: ≥75 = accepted, <75 = uncertain
            if best_correction.combined_score >= TIER2_ACCEPT_THRESHOLD:
                best_correction.confidence_band = "accepted"
            else:
                best_correction.confidence_band = "uncertain"
            return best_correction

    return None


def correct_segment_text(
    text: str,
    entities: list[DetectedEntity],
    gazetteer: dict[str, str],
    segment_id: str,
    context_names: Optional[set[str]] = None,
    entity_types: Optional[dict[str, str]] = None,
    team_words: Optional[set[str]] = None,
) -> tuple[str, list[Correction]]:
    """
    Correct all detected entities in a segment's text.

    Processes entities from right-to-left (to preserve character offsets
    when replacing text).

    Args:
        text: the original segment text
        entities: list of DetectedEntity found in this segment
        gazetteer: the name lookup dictionary
        segment_id: ID of the segment being corrected
        context_names: nearby names for context scoring

    Returns:
        Tuple of (corrected_text, list_of_corrections)
    """
    corrections = []
    corrected_text = text

    # Sort entities by start position, reversed (so we replace right-to-left)
    sorted_entities = sorted(entities, key=lambda e: e.start_char, reverse=True)

    for entity in sorted_entities:
        match = find_best_match(
            entity.text, gazetteer, context_names,
            entity_types=entity_types, team_words=team_words,
        )
        if match:
            match.segment_id = segment_id

            if match.confidence_band == "accepted":
                replacement = extract_and_rebuild_entity(entity.text, match.corrected)
                before = corrected_text[:entity.start_char]
                after = corrected_text[entity.end_char:]
                corrected_text = before + replacement + after

            corrections.append(match)

    return corrected_text, corrections


def _describe_method(fuzzy_score: float, phonetic: bool, context: bool) -> str:
    """Generate a human-readable description of matching method used."""
    parts = [f"fuzzy({fuzzy_score:.0f})"]
    if phonetic:
        parts.append("phonetic")
    if context:
        parts.append("context")
    return "+".join(parts)
