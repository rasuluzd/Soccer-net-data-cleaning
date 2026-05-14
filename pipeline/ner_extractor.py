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
from rapidfuzz import fuzz

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
    gazetteer: dict[str, str] | None = None,
) -> list[DetectedEntity]:
    """
    Use rule-based heuristics to find potential entity names that
    spaCy's NER might miss due to ASR misspellings.

    Heuristic rules:
    1. Lone capitalized words in short segments (likely a player name)
    2. Capitalized words that aren't common non-name POS tags
    3. (Apple RAG-NEC pattern) Any token (incl. lowercased dictionary words
       like "storage" → "Sturridge") that fuzz-matches a gazetteer canonical
       at >= NER_FUZZY_FLOOR. Catches the high-confidence ASR mishearings
       that Step L's logprob-gate misses and spaCy NER doesn't tag.

    Args:
        text: the segment text
        doc: the spaCy Doc object (reuse from NER step to avoid re-processing)
        language: detected commentary language
        gazetteer: optional name → canonical map; enables Rule 3 fuzz-matching

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

    # Rule 3: Gazetteer fuzz-match (Apple RAG-NEC, arxiv 2409.06062).
    # For each multi-char alphabetic token, fuzz-match against the gazetteer
    # canonicals. Catches "storage" → "Sturridge" class: ASR mishearings
    # whose surface form is a real English word, so spaCy doesn't tag them
    # as PROPN, and Step L's logprob-gate doesn't wrap them (Whisper was
    # confident in the wrong word). entity_corrector still validates via
    # MCQ + MLM, so a low-fuzz match here is filtered downstream.
    if gazetteer:
        from pipeline.config import (
            NER_FUZZY_FLOOR, NER_FUZZY_DICT_OVERRIDE, NER_FUZZY_MIN_LEN,
        )
        # Cache the canonical-words list and existing-spans set to make
        # this scan O(N_tokens × N_canonical_words) — both small per match.
        canonical_words: set[str] = set()
        for variant, canonical in gazetteer.items():
            for w in canonical.split():
                if len(w) >= NER_FUZZY_MIN_LEN:
                    canonical_words.add(w)
            for w in variant.split():
                if len(w) >= NER_FUZZY_MIN_LEN:
                    canonical_words.add(w)
        gaz_lower = {w.lower() for w in canonical_words}
        existing_spans = {(e.start_char, e.end_char) for e in entities}

        # Dict veto. Try pyenchant first (Hunspell-backed, fast); fall
        # back to pyspellchecker (pure-Python frequency dict) when the
        # native Hunspell DLLs are missing — common on Windows.
        # Without veto, common English words like "that"/"they"/"been"
        # fuzz-match player surname fragments and flood Stage E with
        # fake entities (empirically: 99 false "that"→"thibaut", 68
        # "they"→"terry", 41 "been"→"eden" on Chelsea-Liverpool 2016).
        _dict = None
        try:
            import enchant  # type: ignore
            _dict_enchant = enchant.Dict("en_US")
            _dict = ("enchant", _dict_enchant)
        except Exception:
            try:
                from spellchecker import SpellChecker  # type: ignore
                _dict_spell = SpellChecker(language="en")
                _dict = ("spellchecker", _dict_spell)
            except Exception:
                _dict = None

        def _is_real_word(w: str) -> bool:
            if _dict is None:
                return False
            kind, obj = _dict
            try:
                if kind == "enchant":
                    return obj.check(w)
                # spellchecker: a word is "known" if its frequency > 0
                return w.lower() in obj
            except Exception:
                return False

        cursor = 0
        for word in text.split():
            word_start = text.find(word, cursor)
            if word_start < 0:
                continue
            next_cursor = word_start + len(word)
            cursor = next_cursor

            clean_word = word.strip(".,!?;:'\"()-—–…")
            if len(clean_word) < NER_FUZZY_MIN_LEN:
                continue
            if not clean_word.replace("'", "").replace("-", "").isalpha():
                continue
            cw_lower = clean_word.lower()
            if cw_lower in gaz_lower:
                continue  # already an exact gazetteer entry; entity_corrector handles via cache

            # Best fuzz against any canonical word
            best_score = 0
            for cand in canonical_words:
                if abs(len(cand) - len(clean_word)) > 4:
                    continue
                s = fuzz.ratio(cw_lower, cand.lower())
                if s > best_score:
                    best_score = s

            if best_score < NER_FUZZY_FLOOR:
                continue
            # Dictionary veto: if it's a real English word, only emit on
            # strong fuzz to gazetteer (override threshold). Skips words
            # like "started", "moves" that randomly fuzz to player names.
            if len(clean_word) >= 4 and _is_real_word(clean_word):
                if best_score < NER_FUZZY_DICT_OVERRIDE:
                    continue

            # Find clean_word's exact start within the word span
            start_pos = text.find(clean_word, word_start)
            if start_pos < 0 or start_pos >= next_cursor:
                continue
            if (start_pos, start_pos + len(clean_word)) in existing_spans:
                continue
            pos = _get_pos_for_word(doc, clean_word, start_pos)
            entities.append(DetectedEntity(
                text=clean_word,
                label="PERSON",
                start_char=start_pos,
                end_char=start_pos + len(clean_word),
                source="heuristic_gazetteer_fuzz",
                pos=pos,
            ))
            existing_spans.add((start_pos, start_pos + len(clean_word)))

    return entities


# ─── Main Extraction Function ───────────────────────────────────────

def extract_entities(
    segment: Segment, language: str = "en",
    gazetteer: dict[str, str] | None = None,
) -> list[DetectedEntity]:
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
    heuristic_entities = extract_heuristic_entities(
        text, doc=doc, language=language, gazetteer=gazetteer,
    )
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
    gazetteer: dict[str, str] | None = None,
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
            seg.text.strip(), doc=doc, language=language, gazetteer=gazetteer,
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
