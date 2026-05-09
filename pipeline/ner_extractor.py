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

    # Language-aware POS reject set
    from pipeline.config import get_rejected_pos_tags
    rejected_pos = get_rejected_pos_tags(language)

    # Rule 1: Short segments (1-3 words) that are just a name
    if len(words) <= 3:
        for match in CAPITALIZED_WORD.finditer(text):
            word = match.group(1)
            if len(word) < 3:
                continue
            pos = _get_pos_for_word(doc, word, match.start())
            # Skip if POS indicates a non-name word
            if pos in rejected_pos:
                continue
            entities.append(DetectedEntity(
                text=word,
                label="PERSON",
                start_char=match.start(),
                end_char=match.end(),
                source="heuristic_short_segment",
                pos=pos,
            ))

    # Rule 2: Capitalized words that aren't rejected-POS (proper-noun candidates).
    # In Swedish football commentary, a capitalized non-sentence-start word is
    # almost always a proper noun (name/team/venue). Whether or not it appears
    # near a verb doesn't matter — "till Nordfält." has no nearby verb but
    # Nordfält is clearly an entity. The POS + position signals (not
    # sentence-start, not in rejected POS tags) are enough.
    #
    # F3 FIX (v2): previously used `text.find(clean_word)` which always
    # returns the FIRST occurrence. For repeated-name segments ("Hansson
    # skjuter. Hansson missar.") both iterations hit start=0, got collapsed
    # by _deduplicate_entities, and only the first instance was corrected.
    #
    # Fix: walk word-by-word, consuming from a cursor. Skip tokens simply
    # advance the cursor; entity candidates look up from the cursor. This
    # avoids the off-by-length issues a naive cursor can introduce.
    cursor = 0
    for i, word in enumerate(words):
        # Always advance cursor past this word, whether or not we keep it.
        # Find the word at or after cursor so repeated words get distinct spans.
        word_start = text.find(word, cursor)
        if word_start < 0:
            # Word not found from cursor — reset search to cursor for robustness.
            word_start = text.find(word, max(cursor - 1, 0))
            if word_start < 0:
                continue
        next_cursor = word_start + len(word)

        clean_word = word.strip(".,!?;:'\"")
        if len(clean_word) < 3:
            cursor = next_cursor
            continue
        if not clean_word[0].isupper():
            cursor = next_cursor
            continue
        # Sentence-start capitals may be common words — skip unless spaCy
        # tags them as PROPN.
        is_sentence_start = (i == 0)

        # Find clean_word specifically (not the word-with-punctuation).
        # It must be at or inside [word_start, next_cursor].
        start_pos = text.find(clean_word, word_start)
        if start_pos < 0 or start_pos >= next_cursor:
            cursor = next_cursor
            continue
        pos = _get_pos_for_word(doc, clean_word, start_pos)
        cursor = next_cursor

        # Skip if POS indicates a non-name word (language-aware set)
        if pos in rejected_pos:
            continue

        # At sentence start, only accept if POS is PROPN (to avoid "Det",
        # "Han", "Nu" style capitalized function words).
        if is_sentence_start and pos != "PROPN":
            continue

        if start_pos < 0:
            continue

        entities.append(DetectedEntity(
            text=clean_word,
            label="PERSON",
            start_char=start_pos,
            end_char=start_pos + len(clean_word),
            source="heuristic_capitalized_non_function",
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
) -> dict[tuple[int, str], list[DetectedEntity]]:
    """
    Extract entities from all segments in a batch.

    Uses spaCy's pipe() for efficient batch processing.

    Args:
        segments: list of Segment objects
        language: detected commentary language (default "en")

    Returns:
        Dict mapping (half, segment_id) -> list of DetectedEntity

    NOTE: The map is keyed by ``(seg.half, seg.segment_id)`` because
    ``segment_id`` alone is NOT unique across halves — the JSON format
    restarts numbering at 0 for each half, so half 1 seg "90" and half 2
    seg "90" collide. Keying by segment_id only caused entities (with
    positions valid for half 2 text) to be applied to half 1 text, which
    corrupted segments like '411 som eventuellt då.' → '411 som eventuellt
    då.Celina' (half 2 seg 90 was 'Testa vänsterkanten genom Selina.').
    """
    nlp = get_nlp(language)
    entity_labels = get_entity_labels(language)

    # Prepare texts and segment references
    texts = [seg.text.strip() for seg in segments]

    results: dict[tuple[int, str], list[DetectedEntity]] = {}

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

        results[(seg.half, seg.segment_id)] = _deduplicate_entities(entities)

    return results
