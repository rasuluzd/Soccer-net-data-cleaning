"""
Central configuration for the data cleaning pipeline.

All paths, thresholds, and constants are defined here so you can
easily tune the pipeline without touching any other file.
"""

import os
from pathlib import Path

# ─── Parallelism ─────────────────────────────────────────────────────
# Number of worker processes for parallel match processing.
# 0 = auto-detect (uses all available CPU cores).
# 1 = sequential (no parallelism, useful for debugging).
MAX_WORKERS = int(os.environ.get("PIPELINE_WORKERS", 0))

# ─── Paths ───────────────────────────────────────────────────────────
# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the raw SoccerNet dataset lives
DATASET_ROOT = PROJECT_ROOT / "path" / "to" / "SoccerNet" / "caption-2023"

# Where cleaned output JSONs will be saved (mirrors the dataset structure)
CLEANED_OUTPUT_DIR = PROJECT_ROOT / "cleaned_data"

# Learned corrections dictionary (grows automatically over time)
LEARNED_CORRECTIONS_PATH = PROJECT_ROOT / "data" / "learned_corrections.json"

# ─── Learned Dictionary ────────────────────────────────────────────
# Minimum sightings and confidence before a learned correction is
# trusted enough for instant lookup OR merging into the gazetteer.
LEARNED_MIN_SEEN_COUNT = 2
LEARNED_MIN_CONFIDENCE = 0.6


# ─── Hallucination Filter ───────────────────────────────────────────
# Minimum ratio of ASCII-alpha characters (a-z, A-Z) in a segment's text.
# Segments below this are flagged as garbage (non-Latin hallucinations).
HALLUCINATION_MIN_ALPHA_RATIO = 0.70

# Minimum number of words for a valid segment.
# Single-word segments like "transition" or "penny" are noise.
MIN_SEGMENT_WORD_COUNT = 2

# Maximum allowed repetition ratio: if a short phrase is repeated this
# many times consecutively, the repeats are marked as duplicates.
MAX_CONSECUTIVE_REPEATS = 2


# ─── Deduplication ──────────────────────────────────────────────────
# Similarity threshold (0–100) for merging consecutive segments.
# 95 means nearly identical text gets merged.
DUPLICATE_SIMILARITY_THRESHOLD = 95


# ─── Fuzzy Matching ─────────────────────────────────────────────────
# Combined scoring weights for the multi-signal matcher.
# These sum to 1.0 and determine how much each signal contributes.
#
# ASR errors are fundamentally SOUND-BASED — Whisper hears phonemes
# and maps them to wrong spellings. So phonetic similarity deserves
# the highest weight: names that sound alike are very likely the same
# person even if the spelling looks completely different.
#
# Examples showing why phonetic weight matters:
#   "Kohlerhoff" ≈ "Kolarov"  (fuzzy=59, but phonetically similar)
#   "Sacco" ≈ "Sakho"         (fuzzy=60, but both → "SK" phonetically)
#   "Guanyama" ≈ "Wanyama"    (fuzzy=80, phonetically close)
FUZZY_WEIGHT = 0.45       # RapidFuzz string similarity
PHONETIC_WEIGHT = 0.40    # Metaphone/Soundex phonetic match (primary signal for ASR)
CONTEXT_WEIGHT = 0.15     # Bonus for matching team/match context

# Adaptive threshold: short names get a lower bar because there's
# less string data to compare, making fuzzy scores naturally lower.
#
# These thresholds are calibrated against real ASR errors:
#   "Kohlerhoff"→Kolarov scores ~67 (long, fuzzy=59, phonetic=yes)
#   "Aspilicueta"→Azpilicueta scores ~81 (long, fuzzy=91, phonetic=yes)
#   "Sacco"→Sakho scores ~59 (medium, fuzzy=60, phonetic=yes)
#   "Winston Ritu"→Winston Reid scores ~78 (long, fuzzy=83, phonetic=yes)
def get_fuzzy_threshold(entity_text: str) -> int:
    """Return the minimum combined score needed to accept a correction."""
    length = len(entity_text)
    if length <= 4:
        return 48   # short names like "Turi" → Touré, "Zuma" → Zouma
    elif length <= 8:
        return 55   # medium names like "Sacco" → Sakho
    else:
        return 55   # long names like "Kohlerhoff" → Kolarov


# Confidence band for Tier 2 → Tier 3 routing.
# Corrections scoring >= this threshold are accepted immediately by Tier 2.
# Corrections scoring below this (but above get_fuzzy_threshold) are
# "uncertain" — they are routed to Tier 3 for context validation.
TIER2_ACCEPT_THRESHOLD = 75


# ─── NER ─────────────────────────────────────────────────────────────
# spaCy model to use for Named Entity Recognition.
# en_core_web_trf = transformer-based (most accurate, ~500MB, slow to load)
# en_core_web_sm  = small model (12MB, fast, nearly identical corrections)
# Benchmark: 0 correction differences across 4 matches (3,635 entities).
# Tier 2 fuzzy/phonetic matching + Tier 3 context AI compensate fully.
SPACY_MODEL = "en_core_web_sm"

# Entity labels we care about from spaCy's NER output
ENTITY_LABELS_OF_INTEREST = {"PERSON", "ORG", "GPE", "FAC"}

# ─── Tier 3: AI-Enhanced Correction ─────────────────────────────────
# Sentence-transformer model for contextual disambiguation
# all-MiniLM-L6-v2: 80MB, 384-dim, fast — best speed/quality tradeoff
CONTEXT_MODEL_NAME = "all-MiniLM-L6-v2"

# Minimum cosine similarity to accept a contextual disambiguation
CONTEXT_SIMILARITY_THRESHOLD = 0.50

# Validation threshold for uncertain corrections from Tier 2.
# Lower than CONTEXT_SIMILARITY_THRESHOLD because Tier 2 already verified
# string/phonetic similarity — Tier 3 just confirms the context doesn't
# contradict it.
TIER3_VALIDATION_THRESHOLD = 0.35

# Number of surrounding segments to include as context window
CONTEXT_WINDOW_SIZE = 2

# ─── Elasticsearch / Temporal Chunking ──────────────────────────────
# Rolling window size for creating search-friendly temporal chunks.
# Segments within this window are concatenated into a single document.
ROLLING_WINDOW_SECONDS = 12.0

# Overlap between consecutive windows (ensures context isn't lost at edges).
ROLLING_WINDOW_OVERLAP_SECONDS = 4.0


# ─── Multilingual Support ────────────────────────────────────────────
# Language-specific model selection.
# English uses proven models; other languages use multilingual variants.
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "default": "xx_ent_wiki_sm",
}

CONTEXT_MODELS = {
    "en": "all-MiniLM-L6-v2",
    "default": "paraphrase-multilingual-MiniLM-L12-v2",
}

# Entity label sets differ between spaCy model families.
# en_core_web_* uses OntoNotes labels; xx_ent_wiki_sm uses WikiNER labels.
ENTITY_LABEL_MAP = {
    "en": {"PERSON", "ORG", "GPE", "FAC"},
    "default": {"PER", "ORG", "LOC", "MISC"},
}

# Language families — langdetect may return a related language code.
LANGUAGE_FAMILIES = {
    "en": {"en", "sco", "cy"},
    "sv": {"sv", "no", "da"},
    "de": {"de", "nl", "lb"},
    "fr": {"fr"},
    "es": {"es", "ca", "gl", "pt"},
    "it": {"it"},
}

# POS tags that indicate a word is NOT a player name.
# Includes NOUN because common nouns (goal, target, corner) should not be
# corrected to player names. Actual player names are tagged PROPN by spaCy.
REJECTED_POS_TAGS = {
    "ADJ", "VERB", "DET", "ADP", "ADV", "NOUN",
    "CCONJ", "SCONJ", "AUX", "INTJ", "PART", "PRON",
}

# Football word per language (used in Tier 3 candidate descriptions).
FOOTBALL_WORDS = {
    "en": "football",
    "sv": "fotboll",
    "de": "Fußball",
    "fr": "football",
    "es": "fútbol",
    "it": "calcio",
    "default": "football",
}


def get_spacy_model(lang: str = "en") -> str:
    """Return the spaCy model name for the detected language."""
    return SPACY_MODELS.get(lang, SPACY_MODELS["default"])


def get_context_model(lang: str = "en") -> str:
    """Return the sentence-transformer model name for the detected language."""
    return CONTEXT_MODELS.get(lang, CONTEXT_MODELS["default"])


def get_entity_labels(lang: str = "en") -> set[str]:
    """Return the NER entity label set for the detected language's model."""
    return ENTITY_LABEL_MAP.get(lang, ENTITY_LABEL_MAP["default"])


def get_scoring_weights(lang: str = "en") -> tuple[float, float, float]:
    """Return (fuzzy, phonetic, context) weights for the detected language.

    Non-English languages reduce phonetic weight because jellyfish.metaphone
    is English-specific; the lost weight shifts to fuzzy string matching.
    """
    if lang == "en":
        return (FUZZY_WEIGHT, PHONETIC_WEIGHT, CONTEXT_WEIGHT)
    return (0.55, 0.30, 0.15)
