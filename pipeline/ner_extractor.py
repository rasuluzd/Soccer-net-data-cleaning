"""
NER Entity Extractor — detects named entities in ASR transcripts.

Uses a dual strategy:
1. spaCy NER for automatic entity detection (language-adaptive model)
2. Heuristic rules to catch entities that NER misses (common in ASR text because
   misspellings don't look like known entities to the model)
"""

import re
from dataclasses import dataclass, field

import spacy

from pipeline.config import (
    SPACY_MODEL,
    ENTITY_LABELS_OF_INTEREST,
    REJECTED_POS_TAGS,
    get_spacy_model,
    get_entity_labels,
)
from pipeline.loader import Segment


@dataclass
class DetectedEntity:
    """An entity found in a transcript segment."""
    text: str           # the entity text as it appears in the transcript
    label: str          # entity type: PERSON, ORG, GPE, FAC, PER, etc.
    start_char: int     # character offset within the segment text
    end_char: int       # character offset (exclusive)
    source: str         # "spacy" or "heuristic" — how it was detected
    pos: str = ""       # spaCy POS tag (PROPN, NOUN, ADJ, VERB, etc.)


def _deduplicate_entities(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """
    Remove overlapping entities by keeping the longest one.
    This fixes bugs where multiple heuristic rules (or spaCy + heuristics)
    find the exact same word span and add it multiple times.
    """
    if not entities:
        return []

    # Sort by start_char, then by length (longest first)
    sorted_ents = sorted(entities, key=lambda e: (e.start_char, -(e.end_char - e.start_char)))

    deduped = []
    last_end = -1

    for ent in sorted_ents:
        if ent.start_char >= last_end:
            # No overlap with previous accepted entity
            deduped.append(ent)
            last_end = ent.end_char

    return deduped


# ─── spaCy Model (lazy-loaded, cached per language) ──────────────────
_nlp_cache: dict[str, object] = {}


def get_nlp(language: str = "en"):
    """Load the spaCy model on first use, cached per language."""
    global _nlp_cache
    if language in _nlp_cache:
        return _nlp_cache[language]

    model_name = get_spacy_model(language)
    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"[WARNING] spaCy model '{model_name}' not found.")
        print(f"          Install it with: python -m spacy download {model_name}")
        if language != "en":
            print("          Falling back to en_core_web_sm...")
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    f"No spaCy model found. Install with:\n"
                    f"  python -m spacy download {model_name}\n"
                    f"  or: python -m spacy download en_core_web_sm"
                )
        else:
            print("          Falling back to en_core_web_sm...")
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Install one with:\n"
                    f"  python -m spacy download {model_name}\n"
                    "  or: python -m spacy download en_core_web_sm"
                )

    _nlp_cache[language] = nlp
    return nlp


def _reset_nlp():
    """Reset the cached spaCy models (forces reload on next get_nlp() call).

    Used by the model comparison script to switch between models.
    """
    global _nlp_cache
    _nlp_cache.clear()


# ─── Heuristic Entity Detection ─────────────────────────────────────

# Pattern: A capitalized word (2+ chars) that might be a name.
# Includes Nordic/Germanic diacritics (å, ä, ö, æ, ø, ü, ß).
CAPITALIZED_WORD = re.compile(
    r"\b([A-ZÅÄÖÆØ][a-zà-ÿåäöæøüß]+(?:['-][A-Za-zÅÄÖÆØà-ÿåäöæøüß]+)*)\b"
)


def _get_pos_for_word(doc, word_text: str, char_offset: int) -> str:
    """Look up the POS tag for a word from the spaCy doc by character offset."""
    for token in doc:
        if token.idx <= char_offset < token.idx + len(token.text):
            return token.pos_
    return ""


def extract_heuristic_entities(
    text: str,
    doc=None,
    language: str = "en",
) -> list[DetectedEntity]:
    """
    Use rule-based heuristics to find potential entity names that
    spaCy's NER might miss due to ASR misspellings.

    Heuristic rules:
    1. Lone capitalized words in short segments (likely a player name)
    2. Capitalized words near verbs (detected via POS tagging)
    3. Capitalized words that aren't common non-name POS tags

    Args:
        text: the segment text
        doc: the spaCy Doc object (reuse from NER step to avoid re-processing)
        language: detected commentary language

    Returns:
        List of DetectedEntity objects found by heuristics
    """
    entities = []
    words = text.split()

    # Get spaCy doc for POS tagging (reuse if already available)
    if doc is None:
        nlp = get_nlp(language)
        doc = nlp(text)

    # Rule 1: Short segments (1-3 words) that are just a name
    if len(words) <= 3:
        for match in CAPITALIZED_WORD.finditer(text):
            word = match.group(1)
            if len(word) < 3:
                continue
            pos = _get_pos_for_word(doc, word, match.start())
            # Skip if POS indicates a non-name word
            if pos in REJECTED_POS_TAGS:
                continue
            entities.append(DetectedEntity(
                text=word,
                label="PERSON",
                start_char=match.start(),
                end_char=match.end(),
                source="heuristic_short_segment",
                pos=pos,
            ))

    # Rule 2: Capitalized words near verbs (language-agnostic via POS)
    for i, word in enumerate(words):
        clean_word = word.strip(".,!?;:'\"")
        if len(clean_word) < 3:
            continue
        if not clean_word[0].isupper():
            continue

        # Get POS for this word
        start_pos = text.find(clean_word)
        pos = _get_pos_for_word(doc, clean_word, start_pos) if start_pos >= 0 else ""

        # Skip if POS indicates a non-name word
        if pos in REJECTED_POS_TAGS:
            continue

        # Check if any nearby word (±2 positions) is a verb (POS-based, language-agnostic)
        has_nearby_verb = False
        for j in range(max(0, i - 2), min(len(words), i + 3)):
            if j == i:
                continue
            nearby_word = words[j].strip(".,!?;:'\"")
            nearby_start = text.find(nearby_word)
            if nearby_start >= 0:
                nearby_pos = _get_pos_for_word(doc, nearby_word, nearby_start)
                if nearby_pos == "VERB":
                    has_nearby_verb = True
                    break

        if has_nearby_verb:
            if start_pos >= 0:
                entities.append(DetectedEntity(
                    text=clean_word,
                    label="PERSON",
                    start_char=start_pos,
                    end_char=start_pos + len(clean_word),
                    source="heuristic_near_action_verb",
                    pos=pos,
                ))

    return entities


# ─── Main Extraction Function ───────────────────────────────────────

def extract_entities(segment: Segment, language: str = "en") -> list[DetectedEntity]:
    """
    Extract named entities from a single ASR segment using both
    spaCy NER and heuristic rules.

    The two sources are merged, with spaCy results taking priority
    when there's overlap.

    Args:
        segment: the Segment to analyze
        language: detected commentary language (default "en")

    Returns:
        List of DetectedEntity objects (deduplicated)
    """
    text = segment.text.strip()
    if not text:
        return []

    entities = []
    entity_labels = get_entity_labels(language)

    # ── Step 1: spaCy NER ────────────────────────────────────────────
    nlp = get_nlp(language)
    doc = nlp(text)

    spacy_spans = set()  # track positions to avoid duplicates with heuristics
    for ent in doc.ents:
        if ent.label_ in entity_labels:
            # Get POS of the entity's root/head token
            ent_pos = ent.root.pos_ if ent.root else ""
            entities.append(DetectedEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                source="spacy",
                pos=ent_pos,
            ))
            spacy_spans.add((ent.start_char, ent.end_char))

    # ── Step 2: Heuristic rules ──────────────────────────────────────
    heuristic_entities = extract_heuristic_entities(text, doc=doc, language=language)
    for he in heuristic_entities:
        # Only add if not already covered by a spaCy entity
        overlaps = any(
            he.start_char < sp_end and he.end_char > sp_start
            for sp_start, sp_end in spacy_spans
        )
        if not overlaps:
            entities.append(he)

    return _deduplicate_entities(entities)


def extract_entities_batch(
    segments: list[Segment],
    language: str = "en",
) -> dict[str, list[DetectedEntity]]:
    """
    Extract entities from all segments in a batch.

    Uses spaCy's pipe() for efficient batch processing.

    Args:
        segments: list of Segment objects
        language: detected commentary language (default "en")

    Returns:
        Dict mapping segment_id -> list of DetectedEntity
    """
    nlp = get_nlp(language)
    entity_labels = get_entity_labels(language)

    # Prepare texts and segment references
    texts = [seg.text.strip() for seg in segments]

    results: dict[str, list[DetectedEntity]] = {}

    # Process in batches with spaCy pipe
    for i, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        seg = segments[i]
        entities = []

        # spaCy entities
        spacy_spans = set()
        for ent in doc.ents:
            if ent.label_ in entity_labels:
                ent_pos = ent.root.pos_ if ent.root else ""
                entities.append(DetectedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    source="spacy",
                    pos=ent_pos,
                ))
                spacy_spans.add((ent.start_char, ent.end_char))

        # Heuristic entities (only if not overlapping with spaCy)
        heuristic_entities = extract_heuristic_entities(
            seg.text.strip(), doc=doc, language=language
        )
        for he in heuristic_entities:
            overlaps = any(
                he.start_char < sp_end and he.end_char > sp_start
                for sp_start, sp_end in spacy_spans
            )
            if not overlaps:
                entities.append(he)

        results[seg.segment_id] = _deduplicate_entities(entities)

    return results
