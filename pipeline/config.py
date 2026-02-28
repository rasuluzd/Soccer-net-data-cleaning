"""
Central configuration for the data cleaning pipeline.

All paths, thresholds, and constants are defined here so you can
easily tune the pipeline without touching any other file.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
# Root of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where the raw SoccerNet dataset lives
DATASET_ROOT = PROJECT_ROOT / "path" / "to" / "SoccerNet" / "caption-2023"

# Where cleaned output JSONs will be saved (mirrors the dataset structure)
CLEANED_OUTPUT_DIR = PROJECT_ROOT / "cleaned_data"

# Learned corrections dictionary (grows automatically over time)
LEARNED_CORRECTIONS_PATH = PROJECT_ROOT / "data" / "learned_corrections.json"


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


# ─── NER ─────────────────────────────────────────────────────────────
# spaCy model to use for Named Entity Recognition.
# en_core_web_trf = transformer-based (most accurate, ~500MB)
# en_core_web_sm  = small model (faster, less accurate)
SPACY_MODEL = "en_core_web_trf"

# Entity labels we care about from spaCy's NER output
ENTITY_LABELS_OF_INTEREST = {"PERSON", "ORG", "GPE", "FAC"}

# Soccer action verbs — words near these are likely player names
SOCCER_ACTION_VERBS = {
    "shoots", "shot", "scores", "scored", "goal", "passes", "passed",
    "crosses", "crossed", "tackles", "tackled", "fouls", "fouled",
    "dribbles", "headers", "headed", "saves", "saved", "clears",
    "cleared", "blocks", "blocked", "intercepts", "intercepted",
    "assists", "assisted", "substituted", "replaced", "booked",
    "kicks", "kicked", "wins", "won", "loses", "lost",
}
