"""
NER Entity Extractor — detects named entities in ASR transcripts.

Uses a dual strategy:
1. spaCy's transformer-based NER (en_core_web_trf) for automatic entity detection
2. Heuristic rules to catch entities that NER misses (common in ASR text because
   misspellings don't look like known entities to the model)
"""

import re
from dataclasses import dataclass

import spacy

from pipeline.config import SPACY_MODEL, ENTITY_LABELS_OF_INTEREST, SOCCER_ACTION_VERBS
from pipeline.loader import Segment


@dataclass
class DetectedEntity:
    """An entity found in a transcript segment."""
    text: str           # the entity text as it appears in the transcript
    label: str          # entity type: PERSON, ORG, GPE, FAC, or HEURISTIC
    start_char: int     # character offset within the segment text
    end_char: int       # character offset (exclusive)
    source: str         # "spacy" or "heuristic" — how it was detected


# ─── spaCy Model (lazy-loaded) ──────────────────────────────────────
_nlp = None

def get_nlp():
    """Load the spaCy model on first use (avoids slow startup if not needed)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load(SPACY_MODEL)
        except OSError:
            print(f"[WARNING] spaCy model '{SPACY_MODEL}' not found.")
            print(f"          Install it with: python -m spacy download {SPACY_MODEL}")
            print(f"          Falling back to en_core_web_sm...")
            try:
                _nlp = spacy.load("en_core_web_sm")
            except OSError:
                raise RuntimeError(
                    "No spaCy model found. Install one with:\n"
                    f"  python -m spacy download {SPACY_MODEL}\n"
                    "  or: python -m spacy download en_core_web_sm"
                )
    return _nlp


# ─── Heuristic Entity Detection ─────────────────────────────────────

# Pattern: A capitalized word (2+ chars) that might be a name
CAPITALIZED_WORD = re.compile(r"\b([A-Z][a-zà-ÿ]+(?:['-][A-Za-zà-ÿ]+)*)\b")

# Common English words that are capitalized at start of sentence but aren't names
COMMON_WORDS_EXCLUDE = {
    "The", "This", "That", "These", "Those", "There", "Here",
    "What", "When", "Where", "Which", "Who", "How", "Why",
    "And", "But", "For", "Not", "Its", "His", "Her",
    "Very", "Just", "Now", "Well", "Good", "Great", "Big",
    "First", "Last", "Next", "New", "Old", "Long", "High",
    "Ball", "Goal", "Game", "Match", "Half", "Side", "Team",
    "Free", "Kick", "Shot", "Pass", "Cross", "Corner", "Throw",
    "Red", "Yellow", "Card", "Foul", "Offside", "Penalty",
    "City", "United", "Palace", "Villa", "Town", "Rovers",
    "They", "Played", "Looking", "Trying", "Coming", "Going",
    "Straight", "Eventually", "Obviously", "Certainly",
    "Nobody", "Someone", "Everyone", "Positive", "Decent",
}


def extract_heuristic_entities(text: str) -> list[DetectedEntity]:
    """
    Use rule-based heuristics to find potential entity names that
    spaCy's NER might miss due to ASR misspellings.

    Heuristic rules:
    1. Lone capitalized words in short segments (likely a player name)
    2. Capitalized words near soccer action verbs
    3. Capitalized words that aren't common English words

    Args:
        text: the segment text

    Returns:
        List of DetectedEntity objects found by heuristics
    """
    entities = []
    words = text.split()

    # Rule 1: Short segments (1-3 words) that are just a name
    if len(words) <= 3:
        # Check if any word is capitalized and not a common word
        for match in CAPITALIZED_WORD.finditer(text):
            word = match.group(1)
            if word not in COMMON_WORDS_EXCLUDE and len(word) >= 3:
                entities.append(DetectedEntity(
                    text=word,
                    label="PERSON",
                    start_char=match.start(),
                    end_char=match.end(),
                    source="heuristic_short_segment",
                ))

    # Rule 2: Capitalized words near soccer action verbs
    words_lower = [w.lower().strip(".,!?;:'\"") for w in words]
    for i, word in enumerate(words):
        clean_word = word.strip(".,!?;:'\"")
        if clean_word in COMMON_WORDS_EXCLUDE or len(clean_word) < 3:
            continue
        if not clean_word[0].isupper():
            continue

        # Check if any nearby word (±2 positions) is a soccer action verb
        nearby_range = range(max(0, i - 2), min(len(words_lower), i + 3))
        has_action_verb = any(words_lower[j] in SOCCER_ACTION_VERBS for j in nearby_range)

        if has_action_verb:
            # Find the character position in the original text
            start = text.find(clean_word)
            if start >= 0:
                entities.append(DetectedEntity(
                    text=clean_word,
                    label="PERSON",
                    start_char=start,
                    end_char=start + len(clean_word),
                    source="heuristic_near_action_verb",
                ))

    return entities


# ─── Main Extraction Function ───────────────────────────────────────

def extract_entities(segment: Segment) -> list[DetectedEntity]:
    """
    Extract named entities from a single ASR segment using both
    spaCy NER and heuristic rules.

    The two sources are merged, with spaCy results taking priority
    when there's overlap.

    Args:
        segment: the Segment to analyze

    Returns:
        List of DetectedEntity objects (deduplicated)
    """
    text = segment.text.strip()
    if not text:
        return []

    entities = []

    # ── Step 1: spaCy NER ────────────────────────────────────────────
    nlp = get_nlp()
    doc = nlp(text)

    spacy_spans = set()  # track positions to avoid duplicates with heuristics
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS_OF_INTEREST:
            entities.append(DetectedEntity(
                text=ent.text,
                label=ent.label_,
                start_char=ent.start_char,
                end_char=ent.end_char,
                source="spacy",
            ))
            spacy_spans.add((ent.start_char, ent.end_char))

    # ── Step 2: Heuristic rules ──────────────────────────────────────
    heuristic_entities = extract_heuristic_entities(text)
    for he in heuristic_entities:
        # Only add if not already covered by a spaCy entity
        overlaps = any(
            he.start_char < sp_end and he.end_char > sp_start
            for sp_start, sp_end in spacy_spans
        )
        if not overlaps:
            entities.append(he)

    return entities


def extract_entities_batch(segments: list[Segment]) -> dict[str, list[DetectedEntity]]:
    """
    Extract entities from all segments in a batch.

    Uses spaCy's pipe() for efficient batch processing.

    Args:
        segments: list of Segment objects

    Returns:
        Dict mapping segment_id -> list of DetectedEntity
    """
    nlp = get_nlp()

    # Prepare texts and segment references
    texts = [seg.text.strip() for seg in segments]
    segment_ids = [seg.segment_id for seg in segments]

    results: dict[str, list[DetectedEntity]] = {}

    # Process in batches with spaCy pipe
    for i, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        seg = segments[i]
        entities = []

        # spaCy entities
        spacy_spans = set()
        for ent in doc.ents:
            if ent.label_ in ENTITY_LABELS_OF_INTEREST:
                entities.append(DetectedEntity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    source="spacy",
                ))
                spacy_spans.add((ent.start_char, ent.end_char))

        # Heuristic entities (only if not overlapping with spaCy)
        heuristic_entities = extract_heuristic_entities(seg.text.strip())
        for he in heuristic_entities:
            overlaps = any(
                he.start_char < sp_end and he.end_char > sp_start
                for sp_start, sp_end in spacy_spans
            )
            if not overlaps:
                entities.append(he)

        results[seg.segment_id] = entities

    return results
