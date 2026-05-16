"""All thresholds, paths, and model names. Tune here, not in module code."""

import os
from pathlib import Path

# ─── Parallelism ─────────────────────────────────────────────────────
# 0 = auto, 1 = sequential.
MAX_WORKERS = int(os.environ.get("PIPELINE_WORKERS", 0))

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "path" / "to" / "SoccerNet" / "caption-2023"
CLEANED_OUTPUT_DIR = PROJECT_ROOT / "cleaned_data"
LEARNED_CORRECTIONS_PATH = PROJECT_ROOT / "data" / "learned_corrections.json"

# Suffix on the ASR input filename. "" reads {half}_asr.json (stock Whisper).
# "_kb" reads {half}_asr_kb.json (re-transcribed via KB-Whisper).
ASR_INPUT_VARIANT = os.environ.get("ASR_INPUT_VARIANT", "")

# ─── Learned Dictionary ────────────────────────────────────────────
# Min sightings + confidence before a learned correction is trusted.
LEARNED_MIN_SEEN_COUNT = 2
LEARNED_MIN_CONFIDENCE = 0.6


# ─── Hallucination Filter ───────────────────────────────────────────
# Min ratio of alphabetic chars in a segment. Below this = garbage.
HALLUCINATION_MIN_ALPHA_RATIO = 0.50

# Used by the deduplicator to flag short phrases repeated this many times.
MAX_CONSECUTIVE_REPEATS = 2


# ─── Deduplication ──────────────────────────────────────────────────
# Similarity (0-100) at which we treat consecutive segments as the same line.
DUPLICATE_SIMILARITY_THRESHOLD = 95


# ─── NER ─────────────────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

ENTITY_LABELS_OF_INTEREST = {"PERSON", "ORG", "GPE", "FAC"}

# ─── NER Rule 3: gazetteer fuzz-match ──────────────────────────────
# Catches ASR mishearings that look like real English words ("storage" -> "Sturridge")
# which spaCy won't tag as PROPN. Stage E still validates downstream.
NER_FUZZY_FLOOR = 75            # min fuzz.ratio to a gazetteer word
NER_FUZZY_DICT_OVERRIDE = 90    # higher bar if the word is in the spell dict
NER_FUZZY_MIN_LEN = 5           # 4-char tokens collide with too many real words

# ─── Elasticsearch / Temporal Chunking ──────────────────────────────
ROLLING_WINDOW_SECONDS = 12.0
ROLLING_WINDOW_OVERLAP_SECONDS = 4.0


# ─── Multilingual Support ────────────────────────────────────────────
# Per-language spaCy models. xx_ent_wiki_sm is a NER-only fallback (no POS).
# Install with:  python -m spacy download sv_core_news_sm
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "sv": "sv_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "default": "xx_ent_wiki_sm",
}

# Per-language ASR model. faster-whisper accepts stock HF checkpoints
# (auto-converted to CT2 on first load) and pre-converted CT2 repos.
# KB-Whisper covers sv/no/da.
ASR_MODELS = {
    "en": "Systran/faster-whisper-large-v3",
    "sv": "KBLab/kb-whisper-large",
    "no": "KBLab/kb-whisper-large",
    "da": "KBLab/kb-whisper-large",
    "default": "Systran/faster-whisper-large-v3",
}

# Decoding params used by faster-whisper. beam_size > 1 stabilises rare names.
WHISPER_BEAM_SIZE = 5
WHISPER_BEST_OF = 5
WHISPER_COMPUTE_TYPE_CPU = "int8"
WHISPER_COMPUTE_TYPE_GPU = "float16"


def get_asr_model(lang: str = "en") -> str:
    return ASR_MODELS.get(lang, ASR_MODELS["default"])


# English spaCy uses OntoNotes labels; the rest use WikiNER (PER/LOC/ORG/MISC).
ENTITY_LABEL_MAP = {
    "en": {"PERSON", "ORG", "GPE", "FAC"},
    "sv": {"PER", "ORG", "LOC", "MISC"},
    "fr": {"PER", "ORG", "LOC", "MISC"},
    "es": {"PER", "ORG", "LOC", "MISC"},
    "it": {"PER", "ORG", "LOC", "MISC"},
    "pt": {"PER", "ORG", "LOC", "MISC"},
    "nl": {"PER", "ORG", "LOC", "MISC"},
    "default": {"PER", "ORG", "LOC", "MISC"},
}

# Treat closely-related languages as the same family for langdetect.
LANGUAGE_FAMILIES = {
    "en": {"en", "sco", "cy"},
    "sv": {"sv", "no", "da"},
    "fr": {"fr"},
    "es": {"es", "ca", "gl", "pt"},
    "it": {"it"},
}

# POS tags that mean "definitely not a player name". English en_core_web_sm
# is reliable enough to drop NOUN/ADJ/VERB; small non-English models mis-tag
# foreign names too often, so we only drop the unambiguous categories there.
REJECTED_POS_TAGS = {
    "ADJ", "VERB", "DET", "ADP", "ADV", "NOUN",
    "CCONJ", "SCONJ", "AUX", "INTJ", "PART", "PRON",
}

REJECTED_POS_TAGS_NON_ENGLISH = {
    "DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART", "PRON", "INTJ",
}


def get_rejected_pos_tags(lang: str = "en") -> set[str]:
    if lang == "en":
        return REJECTED_POS_TAGS
    return REJECTED_POS_TAGS_NON_ENGLISH

def get_spacy_model(lang: str = "en") -> str:
    return SPACY_MODELS.get(lang, SPACY_MODELS["default"])


def get_entity_labels(lang: str = "en") -> set[str]:
    return ENTITY_LABEL_MAP.get(lang, ENTITY_LABEL_MAP["default"])


# ─── Stage Toggles ───────────────────────────────────────────────────

DOMAIN_NORMALIZATION_ENABLED = True
ENTITY_CORRECTION_ENABLED = True

# ─── Validation gates applied to every accepted correction ──────────
# C1: corrected string must share at least this much char similarity with the original.
# Rejects wild swaps like "Kommer" -> "Kouame" (ratio 20).
CONSERVATIVE_C1_FUZZY_FLOOR = 60
# C2: |len(corr) - len(orig)| <= max(2, 0.6 * len(orig)). Catches runaway edits.
CONSERVATIVE_C2_LEN_TOLERANCE = 0.6

# Dictionary veto: if the original is a real word in the commentary language's
# spell dict AND >= DICTIONARY_VETO_MIN_LEN chars long, reject the correction.
# Stops things like Saturday -> Sturridge or Premier -> Milner.
DICTIONARY_VETO_ENABLED = True
DICTIONARY_VETO_MIN_LEN = 4

# If a token shows up >= this many times in the match, it's almost certainly
# a real common word, not a name to correct.
FREQUENCY_HEURISTIC_THRESHOLD = 5

# ─── Step L: confidence-gated GER (llm_corrector.py) ────────────────
# Qwen2.5-1.5B-Instruct (q4_k_m GGUF) via llama-cpp-python. CPU-only.
# Tokens with avg_logprob above LLM_LOGPROB_GATE stay verbatim; only the
# low-confidence ones get wrapped <token> in the prompt and can be edited.
# xlm-roberta vetos any LLM edit where MLM prefers the original.
LLM_CORRECTION_ENABLED = True
LLM_MODEL_PATH = str(PROJECT_ROOT / "whisper_cache" / "models"
                     / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
LLM_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
LLM_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_CTX_WINDOW = 2048
LLM_TEMPERATURE = 0.0
LLM_MAX_NEW_TOKENS = 96
LLM_LOGPROB_GATE = -0.3
LLM_MIN_TOKENS_TO_INVOKE = 2
LLM_CTX_PREVIOUS_SEGMENTS = 2
LLM_CTX_NEXT_SEGMENTS = 1
LLM_NUM_THREADS = 0                   # 0 = llama.cpp picks (uses all cores)
MLM_VETO_ENABLED = True
MLM_VETO_RATIO = 1.5                  # reject LLM edit if MLM(orig)/MLM(prop) >= this
MLM_VETO_MODEL = "xlm-roberta-base"

# ─── Step P: punctuation + casing (punct_restorer.py) ──────────────
PUNCT_RESTORATION_ENABLED = True
PUNCT_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
PUNCT_PRESERVE_EXISTING = True        # only insert; never delete existing punct/casing

# ─── Stage E: TF-IDF + MCQ controls (entity_corrector.py) ──────────
# Don't run MCQ on tokens shorter than this. Short tokens (Kane, Mann, Dante)
# generate too many false positives via TF-IDF char-bigrams.
MCQ_MIN_TOKEN_LEN = 5

# Don't run MCQ unless the top candidate's word-level fuzz to the original
# is at least this. Filters TF-IDF noise like Northampton -> Southampton.
MCQ_MIN_FUZZ_TO_INVOKE = 65

# Short-token exception: only allow MCQ on <5-char tokens if fuzz is very high
# (covers Sakho/Sako). Kane -> Mane (~75) stays blocked.
MCQ_SHORT_TOKEN_MIN_FUZZ = 85

# Number of MCQ samples to draw and majority-vote. With Qwen 1.5B + single-letter
# constrained output the model is fully deterministic, so 1 is plenty. Bump if
# you swap in a less-greedy LLM.
MCQ_SELF_CONSISTENCY_SAMPLES = 1

# After Qwen picks A/B/C, mask the entity and let xlm-roberta veto the pick
# if it strongly prefers the original. Reuses the Step L MLM handle.
MLM_VETO_ON_MCQ_ENABLED = True

# ─── Validated cross-match cache (entity_corrector.py) ─────────────
# A mishearing -> canonical mapping is cached only after MCQ has picked it
# in VALIDATED_CACHE_MIN_CONSENSUS independent matches with high fuzz.
# Stops one bad pick from poisoning all future runs.
VALIDATED_CACHE_PATH = str(PROJECT_ROOT / "data" / "validated_corrections.json")
VALIDATED_CACHE_MIN_CONSENSUS = 1
VALIDATED_CACHE_MIN_FUZZY = 75
VALIDATED_CACHE_ENABLED = True
