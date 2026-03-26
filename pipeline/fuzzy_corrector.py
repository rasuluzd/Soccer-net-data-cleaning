"""
Fuzzy Corrector — matches detected entities to the gazetteer and corrects them.

Uses a multi-signal scoring approach:
    Score = FUZZY_WEIGHT × fuzzy_ratio + PHONETIC_WEIGHT × phonetic_match + CONTEXT_WEIGHT × context_bonus

This captures both string similarity AND sound similarity, which is critical
for ASR errors where names are phonetically similar but textually different
(e.g., "Sacco" sounds like "Sakho" but has low string similarity).
"""

import unicodedata
from dataclasses import dataclass
from typing import Optional

import jellyfish
from rapidfuzz import fuzz, process

from pipeline.config import (
    FUZZY_WEIGHT,
    PHONETIC_WEIGHT,
    CONTEXT_WEIGHT,
    REJECTED_POS_TAGS,
    get_fuzzy_threshold,
    get_scoring_weights,
    TIER2_ACCEPT_THRESHOLD,
)
from pipeline.ner_extractor import DetectedEntity


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


def _split_entity_parts(original_text: str) -> tuple[str, str, str, str, str]:
    """Split entity text into (core, leading_punct, trailing_possessive, trailing_text, trailing_punct)."""
    temp = original_text.strip()

    # 1) Strip trailing punctuation first (important for cases like "Ward's.")
    trailing_punct = ""
    while temp and temp[-1] in ".,!?;:":
        trailing_punct = temp[-1] + trailing_punct
        temp = temp[:-1]

    # 2) Handle common NER suffix mis-extractions (e.g. " and co")
    trailing_text = ""
    lower_temp = temp.lower()
    if lower_temp.endswith(" and co"):
        trailing_text = temp[-7:]
        temp = temp[:-7]

    # 3) Strip possessive
    trailing_possessive = ""
    if temp.endswith("'s") or temp.endswith("\u2019s"):
        trailing_possessive = temp[-2:]
        temp = temp[:-2]

    # 4) Strip leading punctuation
    leading_punct = ""
    while temp and temp[0] in ".,!?;:":
        leading_punct += temp[0]
        temp = temp[1:]

    return temp, leading_punct, trailing_possessive, trailing_text, trailing_punct


def extract_entity_core(original_text: str) -> str:
    """Extract normalized entity text used for lookup/matching."""
    core, _, _, _, _ = _split_entity_parts(original_text)
    return core


def extract_and_rebuild_entity(original_text: str, corrected_name: str) -> str:
    """Safely strips punctuation/possessives, swaps the name, and rebuilds the string."""
    _, leading_punct, trailing_possessive, trailing_text, trailing_punct = _split_entity_parts(original_text)
    return leading_punct + corrected_name + trailing_possessive + trailing_text + trailing_punct


def _strip_accents(text: str) -> str:
    """Strip diacritics/accents for language-neutral phonetic comparison.
    Examples: ö→o, å→a, ä→a, é→e, ü→u, ß→ss."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def compute_phonetic_score(
    entity: str,
    candidate: str,
    language: str = "en",
) -> float:
    """
    Compare two strings phonetically.

    For English: uses Metaphone (tuned for English phonemes) + Soundex fallback.
    For other languages: accent-normalizes both strings, then uses Soundex
    (more language-neutral than Metaphone).

    Returns:
        100.0 if phonetic codes match, 75.0 for partial match, 0.0 otherwise.
    """
    try:
        if language == "en":
            # English path: Metaphone (unchanged from original)
            entity_code = jellyfish.metaphone(entity)
            candidate_code = jellyfish.metaphone(candidate)

            if entity_code == candidate_code:
                return 100.0

            if entity_code.startswith(candidate_code) or candidate_code.startswith(entity_code):
                return 75.0

            try:
                if jellyfish.soundex(entity) == jellyfish.soundex(candidate):
                    return 75.0
            except Exception:
                pass

            return 0.0
        else:
            # Non-English: accent-normalize, then Soundex (more language-neutral)
            entity_norm = _strip_accents(entity)
            candidate_norm = _strip_accents(candidate)

            try:
                if jellyfish.soundex(entity_norm) == jellyfish.soundex(candidate_norm):
                    return 100.0
            except Exception:
                pass

            # Also check if accent-stripped forms match via Metaphone
            try:
                entity_code = jellyfish.metaphone(entity_norm)
                candidate_code = jellyfish.metaphone(candidate_norm)
                if entity_code == candidate_code:
                    return 75.0
                if entity_code.startswith(candidate_code) or candidate_code.startswith(entity_code):
                    return 50.0
            except Exception:
                pass

            return 0.0
    except Exception:
        return 0.0


def compute_combined_score(
    entity_text: str,
    candidate: str,
    context_names: Optional[set[str]] = None,
    language: str = "en",
) -> tuple[float, float, bool, bool]:
    """
    Compute the multi-signal score for matching an entity to a candidate.

    Score = fuzzy_w × fuzzy + phonetic_w × phonetic + context_w × context

    Weights are language-conditional: English uses the tuned defaults,
    non-English shifts weight from phonetic (less reliable) to fuzzy.

    Args:
        entity_text: the detected entity text (potentially misspelled)
        candidate: the gazetteer entry to compare against
        context_names: nearby entity names for context scoring
        language: detected commentary language

    Returns:
        Tuple of (combined_score, fuzzy_score, phonetic_match, context_match)
    """
    fuzzy_w, phonetic_w, context_w = get_scoring_weights(language)

    # Signal 1: Fuzzy string similarity (token_sort handles word order)
    fuzzy_score = fuzz.token_sort_ratio(entity_text.lower(), candidate.lower())

    # Signal 2: Phonetic similarity
    phonetic_score = compute_phonetic_score(entity_text, candidate, language=language)
    phonetic_match = phonetic_score >= 50.0

    # Signal 3: Context bonus
    context_score = 0.0
    context_match = False
    if context_names:
        overlap = context_names.intersection({candidate})
        if overlap:
            context_score = 100.0
            context_match = True

    # Combined weighted score
    combined = (
        fuzzy_w * fuzzy_score
        + phonetic_w * phonetic_score
        + context_w * context_score
    )

    return combined, fuzzy_score, phonetic_match, context_match


def _entity_contains_multiple_gazetteer_names(
    entity_text: str,
    gazetteer_lower: set[str],
) -> bool:
    """
    Check if a multi-word entity contains multiple gazetteer names as substrings.

    Handles multi-word gazetteer keys like "Di Maria" inside "Di Maria Rooney":
    finds the longest gazetteer match starting at each position, then checks if
    the remaining words also form a gazetteer entry.

    Returns True if the entity can be decomposed into 2+ gazetteer names.
    """
    words = entity_text.split()
    n = len(words)
    if n < 2:
        return False

    names_found = 0
    i = 0
    while i < n:
        # Try longest match first (e.g., "Di Maria" before "Di")
        matched = False
        for length in range(n - i, 0, -1):
            candidate = " ".join(words[i:i + length]).lower()
            if candidate in gazetteer_lower:
                names_found += 1
                if names_found >= 2:
                    return True
                i += length
                matched = True
                break
        if not matched:
            i += 1

    return False


def _get_collapsed_candidates(
    entity_text: str,
    all_variants: list[str],
) -> list[tuple[str, float, int]]:
    """
    Generate collapsed forms of a multi-word entity and find fuzzy matches.

    ASR sometimes splits a single name into multiple words:
        "Jonjo" → "John Joe" or "Jonjoe"
    This function tries collapsing adjacent word pairs:
        "John Joe Shelby" → queries "JohnJoe Shelby" and "John JoeShelby"
    and returns any new candidates with higher scores.
    """
    words = entity_text.split()
    if len(words) < 2:
        return []

    best_candidates = []
    # Try collapsing each adjacent pair
    for i in range(len(words) - 1):
        collapsed_words = words[:i] + [words[i] + words[i + 1]] + words[i + 2:]
        collapsed = " ".join(collapsed_words)
        results = process.extract(
            collapsed,
            all_variants,
            scorer=fuzz.token_sort_ratio,
            limit=3,
        )
        if results:
            best_candidates.extend(results)

    return best_candidates


def find_best_match(
    entity_text: str,
    gazetteer: dict[str, str],
    context_names: Optional[set[str]] = None,
    entity_types: Optional[dict[str, str]] = None,
    team_words: Optional[set[str]] = None,
    pos: str = "",
    language: str = "en",
) -> Optional[Correction]:
    """
    Find the best matching gazetteer entry for a detected entity.

    First does a rapid pre-filter using RapidFuzz to get top candidates,
    then scores each with the full multi-signal approach.

    Args:
        entity_text: the entity text to correct
        gazetteer: dict mapping name variants to canonical names
        context_names: nearby entity names for context scoring
        entity_types: entity type dictionary for team/venue rejection
        team_words: set of team name fragments for multi-word rejection
        pos: POS tag from spaCy (used instead of static word lists)
        language: detected commentary language

    Returns:
        A Correction object if a match was found above threshold, or None.
    """
    if not entity_text or not gazetteer:
        return None

    entity_clean = extract_entity_core(entity_text)

    if not entity_clean:
        return None

    # Skip words tagged as non-name POS (ADJ, VERB, DET, etc.)
    if pos in REJECTED_POS_TAGS:
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

        # Also check for multi-word gazetteer substrings inside the entity.
        # Example: "Di Maria Rooney" contains "Di Maria" (a gazetteer key)
        # AND "Rooney" (another gazetteer key) — this is two names merged,
        # not a single misspelling.
        if _entity_contains_multiple_gazetteer_names(entity_clean, gazetteer_lower):
            return None

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

    # For multi-word entities, also try collapsed forms to handle ASR
    # word-splitting: "John Joe Shelby" → "Jonjo Shelvey" works better
    # when we also query "JohnJoe Shelby" and "JohnJoeShelby".
    collapsed_form = None
    if len(entity_words) >= 2:
        collapsed_candidates = _get_collapsed_candidates(
            entity_clean, all_variants
        )
        if collapsed_candidates:
            # Merge with original candidates, keeping top 5 overall
            seen = {c[0] for c in top_candidates} if top_candidates else set()
            for cc in collapsed_candidates:
                if cc[0] not in seen:
                    top_candidates.append(cc)
                    seen.add(cc[0])

    if not top_candidates:
        return None

    # ── Step 2: Score each candidate with multi-signal approach ───────
    best_score = 0.0
    best_correction = None

    for candidate_text, raw_fuzzy, _ in top_candidates:
        combined, fuzzy_score, phonetic_match, context_match = compute_combined_score(
            entity_clean, candidate_text, context_names, language=language
        )

        if combined > best_score:
            best_score = combined
            canonical = gazetteer[candidate_text]

            # Decide replacement form: match the "level" of the input.
            # If commentator says "Zuma" (single word), correct to "Zouma"
            # (the surname from canonical), not "Kurt Zouma" (the full name).
            # If they say "Winston Ritu" (multi-word), use "Winston Reid".
            #
            # Always derive from the canonical VALUE, never the matched key.
            # Learned corrections have misspellings as keys (e.g. "blassie"),
            # using the key would produce error-to-error corrections.
            entity_word_count = len(entity_clean.split())
            if entity_word_count == 1:
                canonical_words = canonical.split()
                if len(canonical_words) == 1:
                    corrected_name = canonical  # e.g. "Bolasie"
                else:
                    # Multi-word canonical — pick the word most similar
                    # to the entity. This handles "Zuma"→"Zouma" (surname
                    # from "Kurt Zouma") correctly.
                    best_word = canonical_words[-1]  # default to surname
                    best_word_sim = 0
                    for cw in canonical_words:
                        wsim = fuzz.ratio(entity_clean.lower(), cw.lower())
                        if wsim > best_word_sim:
                            best_word_sim = wsim
                            best_word = cw
                    corrected_name = best_word
            else:
                # Multi-word entity → multi-word canonical.
                # Default: use full canonical (e.g. "Winston Ritu" → "Winston Reid").
                # But if the entity is a surname variant (not first+last), use
                # just the surname portion from the canonical.
                # Detection: the best matched KEY is shorter than the canonical,
                # meaning the entity matched a surname-only variant (e.g.
                # "De Michelis" matched key "Demichelis" → canonical "Martin Demichelis").
                canonical_words = canonical.split()
                candidate_word_count = len(candidate_text.split())
                is_surname_variant = (
                    len(canonical_words) > candidate_word_count
                    or (len(canonical_words) > entity_word_count
                        and entity_word_count <= 2)
                )
                if is_surname_variant:
                    # Check if entity shares a first-name token with canonical
                    entity_first = entity_clean.split()[0].lower()
                    canon_first = canonical_words[0].lower()
                    first_name_sim = fuzz.ratio(entity_first, canon_first)
                    if first_name_sim < 60:
                        # Entity doesn't match the canonical first name, so it's
                        # surname-only. Use the surname portion of canonical.
                        surname_part = " ".join(canonical_words[1:])
                        corrected_name = surname_part
                    else:
                        corrected_name = canonical
                else:
                    corrected_name = canonical

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
    debug: bool = False,
    language: str = "en",
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
        entity_types: entity type dictionary
        team_words: team name fragments
        debug: enable debug output
        language: detected commentary language

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
            pos=entity.pos, language=language,
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
