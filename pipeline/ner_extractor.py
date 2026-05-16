"""Find named entities in ASR segments. spaCy NER + heuristic rules
that catch the misspellings spaCy misses."""

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
    """An entity found in a segment."""
    text: str
    label: str          # PERSON, ORG, GPE, FAC, PER, ...
    start_char: int
    end_char: int       # exclusive
    source: str         # "spacy" or "heuristic_*"
    pos: str = ""       # spaCy POS tag (PROPN, NOUN, ...)


def _deduplicate_entities(entities: list[DetectedEntity]) -> list[DetectedEntity]:
    """Drop overlapping entities, keep the longest at each span."""
    if not entities:
        return []

    sorted_ents = sorted(entities, key=lambda e: (e.start_char, -(e.end_char - e.start_char)))

    deduped = []
    last_end = -1

    for ent in sorted_ents:
        if ent.start_char >= last_end:
            deduped.append(ent)
            last_end = ent.end_char

    return deduped


# ─── spaCy Model (lazy-loaded, cached per language) ──────────────────
_nlp_cache: dict[str, object] = {}


def get_nlp(language: str = "en"):
    """Lazy-load the spaCy model. Cached per language."""
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

# Capitalised word (2+ chars) with Nordic diacritics.
CAPITALIZED_WORD = re.compile(
    r"\b([A-ZÅÄÖÆØ][a-zà-ÿåäöæøüß]+(?:['-][A-Za-zÅÄÖÆØà-ÿåäöæøüß]+)*)\b"
)


def _get_pos_for_word(doc, word_text: str, char_offset: int) -> str:
    """POS tag at this character offset in the spaCy doc, or "" if none."""
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
    """Heuristic entities to supplement spaCy NER:

    1. Lone capitalised word in a short segment.
    2. Capitalised words whose POS isn't in the rejected set.
    3. Any token that fuzz-matches a gazetteer word at >= NER_FUZZY_FLOOR.
       Catches ASR mishearings shaped like real words ("storage" -> "Sturridge")
       that spaCy doesn't tag as PROPN.
    """
    entities = []
    words = text.split()

    if doc is None:
        nlp = get_nlp(language)
        doc = nlp(text)

    from pipeline.config import get_rejected_pos_tags
    rejected_pos = get_rejected_pos_tags(language)

    # Rule 1: 1-3 word segments — likely just a name.
    if len(words) <= 3:
        for match in CAPITALIZED_WORD.finditer(text):
            word = match.group(1)
            if len(word) < 3:
                continue
            pos = _get_pos_for_word(doc, word, match.start())
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

    # Rule 2: capitalised words whose POS isn't in the rejected set.
    # Walk via a cursor so repeated names ("Hansson skjuter. Hansson missar.")
    # produce distinct spans instead of all collapsing to the first occurrence.
    cursor = 0
    for i, word in enumerate(words):
        word_start = text.find(word, cursor)
        if word_start < 0:
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
        is_sentence_start = (i == 0)

        start_pos = text.find(clean_word, word_start)
        if start_pos < 0 or start_pos >= next_cursor:
            cursor = next_cursor
            continue
        pos = _get_pos_for_word(doc, clean_word, start_pos)
        cursor = next_cursor

        if pos in rejected_pos:
            continue

        # At sentence start, only accept PROPN — avoids Det/Han/Nu style
        # capitalised function words.
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

    # Rule 3: gazetteer fuzz-match. Catches mishearings shaped like real words
    # (storage -> Sturridge). Stage E still re-validates with MCQ + MLM.
    if gazetteer:
        from pipeline.config import (
            NER_FUZZY_FLOOR, NER_FUZZY_DICT_OVERRIDE, NER_FUZZY_MIN_LEN,
        )
        # Pre-build canonical-words and existing-spans (both small per match).
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

        # Dict veto. Try pyenchant (Hunspell), fall back to pyspellchecker
        # if Hunspell DLLs are missing (common on Windows). Without this,
        # common words like "that"/"they"/"been" fuzz-match surnames and
        # flood Stage E with junk entities.
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
                continue  # exact gazetteer entry, Stage E cache handles it

            best_score = 0
            for cand in canonical_words:
                if abs(len(cand) - len(clean_word)) > 4:
                    continue
                s = fuzz.ratio(cw_lower, cand.lower())
                if s > best_score:
                    best_score = s

            if best_score < NER_FUZZY_FLOOR:
                continue
            # Real word? Need a higher fuzz to override the dict veto.
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
    """spaCy NER + heuristics on one segment. Heuristics yield only when they
    don't overlap a spaCy span. Returns deduplicated list."""
    text = segment.text.strip()
    if not text:
        return []

    entities = []
    entity_labels = get_entity_labels(language)

    nlp = get_nlp(language)
    doc = nlp(text)

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

    heuristic_entities = extract_heuristic_entities(
        text, doc=doc, language=language, gazetteer=gazetteer,
    )
    for he in heuristic_entities:
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
    """Run NER over a batch via spaCy pipe(). Returns {(half, segment_id): [DetectedEntity]}.

    Keying by (half, segment_id) is required: segment_id restarts at 0 for each
    half, so keying by id alone aliases half-2 segments onto half-1 text and
    corrupts the output."""
    nlp = get_nlp(language)
    entity_labels = get_entity_labels(language)

    texts = [seg.text.strip() for seg in segments]

    results: dict[tuple[int, str], list[DetectedEntity]] = {}

    for i, doc in enumerate(nlp.pipe(texts, batch_size=32)):
        seg = segments[i]
        entities = []

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
