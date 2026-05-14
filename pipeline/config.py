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

# Variant suffix on the ASR input filename. Empty (default) = read the
# stock OpenAI Whisper output ``{half}_asr.json``. Set to ``"_kb"`` to read
# the KB-Whisper output ``{half}_asr_kb.json`` produced by
# ``tools/retranscribe_kb_whisper.py``. The orchestrator mirrors the suffix
# on the cleaned output (``{half}_asr{variant}_cleaned.json``) so stock
# and KB-Whisper ablations don't overwrite each other.
ASR_INPUT_VARIANT = os.environ.get("ASR_INPUT_VARIANT", "")

# ─── Learned Dictionary ────────────────────────────────────────────
# Minimum sightings and confidence before a learned correction is
# trusted enough for instant lookup OR merging into the gazetteer.
LEARNED_MIN_SEEN_COUNT = 2
LEARNED_MIN_CONFIDENCE = 0.6


# ─── Hallucination Filter ───────────────────────────────────────────
# Minimum ratio of ASCII-alpha characters (a-z, A-Z) in a segment's text.
# Segments below this are flagged as garbage (non-Latin hallucinations).
HALLUCINATION_MIN_ALPHA_RATIO = 0.50

# (A8/F10) MIN_SEGMENT_WORD_COUNT removed. Was 1, which made Rule 4 in
# hallucination_filter.filter_segment unreachable (Rule 1 already catches
# empty text). Single-word commentary like "GOAL!" is valid. See A8 in
# the plan file.

# Maximum allowed repetition ratio: if a short phrase is repeated this
# many times consecutively, the repeats are marked as duplicates.
MAX_CONSECUTIVE_REPEATS = 2


# ─── Deduplication ──────────────────────────────────────────────────
# Similarity threshold (0–100) for merging consecutive segments.
# 95 means nearly identical text gets merged.
DUPLICATE_SIMILARITY_THRESHOLD = 95


# ─── Entity Correction Scoring (Stage 3C) ──────────────────────────
# Combined scoring weights for the entity name matcher.
# These sum to 1.0 and determine how much each signal contributes.
#
# ASR errors are fundamentally SOUND-BASED — Whisper hears phonemes
# and maps them to wrong spellings. So phonetic similarity deserves
# the highest weight: names that sound alike are very likely the same
# person even if the spelling looks completely different.
#
# Why these weights (May 2026 retune after Chelsea-Liverpool FP audit):
# Phonetic via Metaphone+Jaro-Winkler produces prefix-collision artifacts
# for short/medium tokens — STRT vs STRJ scores 0.92 even though
# Saturday and Sturridge are unrelated. With phonetic at 0.40 the
# composite score for false pairs (Saturday→Sturridge, Manevic→Mane,
# Cueta→Coutinho) climbed into the Tier 3 proposal band. Real ASR
# confusion in football commentary is overwhelmingly orthographic
# (Whisper miss-spells "Sturridge" as "Starridge"/"Sturage", not as
# "Saturday"), so fuzzy similarity is the dominant correct signal.
# Real wins (Sturridge fuzz=89, Mignolet=85, Cahill=82) are unaffected
# by lower phonetic weight; they already clear via fuzzy alone.
FUZZY_WEIGHT = 0.65       # RapidFuzz string similarity (dominant signal)
PHONETIC_WEIGHT = 0.20    # Metaphone/Soundex (secondary; only helps tiebreaks)
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
#
# Tuned 75 → 72 after time-aligned WER evaluation showed multiple real
# name fixes sitting at 71-74:
#   Nordfält → Nordfeldt (74.0) — real fix, correctly capitalized candidate
#   Granat → Granath (71.5) — real fix, already-canonical candidate
# Harmful corrections like Hans→Boman (33) and Man→Boman (55) are safely
# below even this lowered bar. Validated empirically: AIK WER 24.25 → lower.
TIER2_ACCEPT_THRESHOLD = 72


# ─── NER ─────────────────────────────────────────────────────────────
# spaCy model to use for Named Entity Recognition.
# en_core_web_trf = transformer-based (most accurate, ~500MB, slow to load)
# en_core_web_sm  = small model (12MB, fast, nearly identical corrections)
# Benchmark: 0 correction differences across 4 matches (3,635 entities).
# Tier 2 fuzzy/phonetic matching + Tier 3 context AI compensate fully.
SPACY_MODEL = "en_core_web_sm"

# Entity labels we care about from spaCy's NER output
ENTITY_LABELS_OF_INTEREST = {"PERSON", "ORG", "GPE", "FAC"}

# ─── NER Rule 3: gazetteer fuzz-match (Apple RAG-NEC) ───────────────
# Catches ASR mishearings whose surface form is a real English word
# ("storage" → "Sturridge") that spaCy doesn't tag as PROPN and Step L's
# logprob-gate doesn't wrap (Whisper was confident in the wrong word).
# Tokens scoring ≥ NER_FUZZY_FLOOR are emitted to entity_corrector for
# MCQ + gate validation. NER_FUZZY_DICT_OVERRIDE lets a strong fuzz
# match override the dictionary veto (skip pyenchant-known words unless
# they fuzz-match a canonical very strongly).
NER_FUZZY_FLOOR = 75            # min fuzz.ratio to gazetteer canonical word
                                # (raised 65→75 May 2026: 65 caught common-word
                                # FPs like "that"→"thibaut" 73, "they"→"terry" 67,
                                # "been"→"eden" 75, flooding Stage E with noise)
NER_FUZZY_DICT_OVERRIDE = 90    # if word is in dict, need this much (raised 80→90)
NER_FUZZY_MIN_LEN = 5           # skip 4-char fragments — they collide with too
                                # many common English words (eden, kane, mata, etc)

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
#
# Tuning history (Chelsea-Liverpool 2016 GOAL benchmark, May 2026):
# Audit showed Path A at 0.35 was producing FPs because football-domain
# embeddings give 0.36-0.45 cosine to ANY player name vs ANY commentary
# context. Saturday→Sturridge passed at 0.36, Premier→Milner at 0.39,
# Dutchman→Mane at 0.38, Rio Ferdinand→Fernandez at 0.39. Raised to
# 0.55 so Path A only accepts when the proposal is genuinely well-
# anchored in context (not just "any-football-name plausible"). Real
# wins (Sturridge, Mignolet, Cahill) score 0.55+ via Tier 2 directly.
TIER3_VALIDATION_THRESHOLD = 0.55

# Non-English uses paraphrase-multilingual-MiniLM-L12-v2 which is less
# sharply calibrated than all-MiniLM-L6-v2. Empirically, many wrong
# corrections on Swedish land in the 0.35-0.45 range. Raising the bar
# for non-English reduces false positives without losing the strongly-
# contextualized picks (those score 0.50+). This is NOT a Swedish-only
# special case — it's a calibration per embedding-model quality.
TIER3_VALIDATION_THRESHOLD_NON_ENGLISH = 0.45


def get_tier3_validation_threshold(language: str = "en") -> float:
    """Return the Tier 3 validation threshold calibrated to the context model.

    English uses all-MiniLM-L6-v2 (tighter distribution, 0.35 works).
    Other languages use paraphrase-multilingual-MiniLM-L12-v2 (wider
    distribution on non-English, needs 0.45 to be safe).
    """
    return (
        TIER3_VALIDATION_THRESHOLD if language == "en"
        else TIER3_VALIDATION_THRESHOLD_NON_ENGLISH
    )


# MIN_GAP: Tier 3 rejects a winning candidate unless it beats the runner-up
# by at least this margin. Generic commentary can produce multiple ~0.52
# scores for different players; the gap enforces a clear winner.
# See .claude/rules/03-tier3-context-ai.md rule #1.
MIN_GAP = 0.08

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
# English uses proven models; other languages use language-specific models
# when available (they include POS tagging, which xx_ent_wiki_sm lacks).
# xx_ent_wiki_sm is the final fallback — it only does NER, no POS, so
# REJECTED_POS_TAGS filtering won't work for languages using it.
#
# Install a language model:  python -m spacy download sv_core_news_sm
SPACY_MODELS = {
    "en": "en_core_web_sm",
    "sv": "sv_core_news_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm",
    "default": "xx_ent_wiki_sm",
}

CONTEXT_MODELS = {
    "en": "all-MiniLM-L6-v2",
    "default": "paraphrase-multilingual-MiniLM-L12-v2",
}

# Per-language ASR model selection. faster-whisper accepts both stock
# OpenAI HF checkpoints (auto-converted to CTranslate2 on first load,
# slow + extra disk) and pre-converted CT2 repos (load instantly, native
# int8/float16 quantisation). For English we use the official Systran
# CT2-converted version of OpenAI Whisper large-v3 directly. Swedish and
# German use published fine-tunes that cut WER substantially:
#   - KBLab/kb-whisper-large: ~47% rel. WER reduction vs whisper-large-v3
#     on Swedish FLEURS / Common Voice / NST (Interspeech 2025)
#   - primeline/whisper-large-v3-turbo-german: German-fine-tuned turbo
# Override with `--model` flag in tools/retranscribe_kb_whisper.py.
ASR_MODELS = {
    "en": "Systran/faster-whisper-large-v3",
    "sv": "KBLab/kb-whisper-large",
    "no": "KBLab/kb-whisper-large",  # KB-Whisper handles Norwegian decently
    "da": "KBLab/kb-whisper-large",  # and Danish via the Swedish family
    "de": "primeline/whisper-large-v3-turbo-german",
    "default": "Systran/faster-whisper-large-v3",
}

# Decoding parameters used by tools/retranscribe_kb_whisper.py via
# faster-whisper. beam_size > 1 stabilises rare-name recognition.
# WHISPER_NBEST_K is reserved for Phase 2 (RAG entity rerank) and
# Phase 3 (generative error correction over n-best); leave at 1 for now.
WHISPER_BEAM_SIZE = 5
WHISPER_BEST_OF = 5
WHISPER_NBEST_K = 1
WHISPER_COMPUTE_TYPE_CPU = "int8"
WHISPER_COMPUTE_TYPE_GPU = "float16"


def get_asr_model(lang: str = "en") -> str:
    """Return the ASR model name for the detected language."""
    return ASR_MODELS.get(lang, ASR_MODELS["default"])


# Entity label sets differ between spaCy model families.
# en_core_web_* and sv/de/fr/es/it/pt/nl core_news_* models use WikiNER-ish
# labels (PER/LOC/ORG/MISC). xx_ent_wiki_sm also uses those. English uses
# OntoNotes (PERSON/ORG/GPE/FAC).
ENTITY_LABEL_MAP = {
    "en": {"PERSON", "ORG", "GPE", "FAC"},
    "sv": {"PER", "ORG", "LOC", "MISC"},
    "de": {"PER", "ORG", "LOC", "MISC"},
    "fr": {"PER", "ORG", "LOC", "MISC"},
    "es": {"PER", "ORG", "LOC", "MISC"},
    "it": {"PER", "ORG", "LOC", "MISC"},
    "pt": {"PER", "ORG", "LOC", "MISC"},
    "nl": {"PER", "ORG", "LOC", "MISC"},
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
# English: en_core_web_sm reliably tags player names as PROPN, so NOUN is
# safe to reject (common nouns like "goal", "corner" should not become names).
#
# Non-English: small language models (sv_core_news_sm, de_core_news_sm,
# etc.) frequently mis-tag foreign player names as NOUN — they weren't
# trained on sports commentary. So for non-English we keep NOUN out of
# the reject list; we still reject the unambiguous non-name categories
# (pronouns, determiners, auxiliaries, conjunctions, particles).
REJECTED_POS_TAGS = {
    "ADJ", "VERB", "DET", "ADP", "ADV", "NOUN",
    "CCONJ", "SCONJ", "AUX", "INTJ", "PART", "PRON",
}

# Safer set for non-English: drops NOUN, ADJ, VERB (which small language
# models often mis-apply to foreign names) but keeps the unambiguous
# grammatical categories that are NEVER player names.
REJECTED_POS_TAGS_NON_ENGLISH = {
    "DET", "ADP", "CCONJ", "SCONJ", "AUX", "PART", "PRON", "INTJ",
}


def get_rejected_pos_tags(lang: str = "en") -> set[str]:
    """Return the language-appropriate POS tag rejection set.

    English spaCy is confident enough to reject NOUN/ADJ/VERB-tagged words.
    Small language-specific models for Swedish/German/French etc. mis-tag
    foreign names too often to safely reject those categories.
    """
    if lang == "en":
        return REJECTED_POS_TAGS
    return REJECTED_POS_TAGS_NON_ENGLISH

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


# ─── Stage Toggles (for ablation study) ─────────────────────────────
# Each pipeline stage can be independently enabled/disabled.
# This is essential for the evaluation ablation study — shows what
# each stage contributes to overall transcript quality.

# Stage 2A: Domain normalization (scores, times, football compounds)
DOMAIN_NORMALIZATION_ENABLED = True

# Stage 2B (pyspellchecker) + 2C (LanguageTool) removed in May 2026 architectural
# refactor. Both fired ~5 corrections per match, mostly punctuation that Step P
# already handles, with a heavy cold-start cost (LanguageTool downloads JVM rules).
# pyspellchecker is still used internally by entity_corrector's dictionary veto.

# ─── DEPRECATED — kept ON only for ablation comparison vs new SOTA pipeline ──
# After the SOTA refactor (NBEST_RERANK + LLM_CORRECTION + PUNCT_RESTORATION),
# the orchestrator routes around these stages. Set the SOTA flags below to
# False to fall back to this legacy correction cascade.

# Stage 3: Entity name correction (legacy fuzzy+phonetic+context pipeline)
# Re-enabled to run BEFORE the new GER LLM — gazetteer handles obvious
# entity mishearings (Hendrik→Henderson, Starridge→Sturridge); LLM
# polishes the residual segment-level errors. Combined run wins on both
# WER and Entity F1 vs either alone.
ENTITY_CORRECTION_ENABLED = True

# Stage 3 Tier 3: Context disambiguation (sentence-transformer)
TIER3_ENABLED = True

# Stage 4 (mT5 / BERT masked-LM), Stage 5 (Ollama LLMCorrector) and the
# LLM Validator MCQ stage were all removed in the May 2026 cleanup (see
# plans/check-all-files-and-peppy-lemur.md). They were either gated off,
# produced 0 net corrections after their reject filters, or have been
# superseded by Step L (pipeline/llm_corrector.py — confidence-gated GER
# with Qwen + xlm-roberta MLM veto). The model that still uses
# xlm-roberta-base is the MLM veto inside llm_corrector.py.

# ─── Phase C2: Conservative Filtering gates (inference-time safety) ────
# Paper: "Overcorrection Control in ASR Post-Editing" (EMNLP 2024)
# — reduced unwanted corrections from 43.0% → 13.9% by gating accepts
# on two simple checks.
#
# C1 (fuzzy floor): the corrected string must share at least this much
# character similarity with the original. Rejects wildly-different
# replacements (e.g. "Kommer" → "Kouame" at ratio 20).
#
# C2 (length gate): the length delta between original and corrected must
# be within bounds. Rejects runaway expansions/contractions.
#
# Both gates apply uniformly to Tier 2 + Tier 3 + LLM validator corrections.
# Raised from 40 to 60 (May 2026): the lower bar passed every audited FP
# (Saturday/Sturridge=58.8, Cueta/Coutinho=46.2, Dutchman/Mane=50). Real
# wins (Sturridge=89, Mignolet=85, Cahill=82) clear 60 comfortably.
CONSERVATIVE_C1_FUZZY_FLOOR = 60    # rapidfuzz.fuzz.ratio(orig, corr) must be ≥ this
# Tolerance tuned against real Tier 3 outputs on AIK. At 0.4 the gate
# rejected Chong→Csongvai (diff 3, which is a GT-confirmed correction);
# at 0.6 it still catches wild expansions like Man→"Mohammed Salah"
# (diff 11) while letting 3-char adjustments through.
CONSERVATIVE_C2_LEN_TOLERANCE = 0.6 # |len(corr)-len(orig)| ≤ max(2, 0.6·len(orig))

# Dictionary veto (added May 2026 after FP audit):
# If the ORIGINAL token IS a real common word in the commentary
# language's spell-check dictionary AND is at least DICTIONARY_VETO_MIN_LEN
# characters long, reject any correction. Whisper rarely mishears common
# words into player surnames; the observed pattern is fuzzy-match
# accident. Blocks Saturday→*, Premier→*, Dutchman→*, Rio→* by default.
# Player surnames (Sturridge, Mignolet, Cahill, ...) are not in the
# dictionary so corrections targeting them are unaffected.
DICTIONARY_VETO_ENABLED = True
DICTIONARY_VETO_MIN_LEN = 4         # "Rio" is 3 chars and a real name; veto only ≥4-char commons

# Frequency heuristic: if the ORIGINAL word appears ≥ this many times in
# the match AND isn't already a gazetteer name, it's almost certainly a
# common word (verb/noun/article) that ASR transcribed correctly.
# Rejects e.g. Swedish "kommer" (the verb "comes", 17 occurrences in AIK)
# being corrected to player "Kouame". Language-agnostic — purely counts.
FREQUENCY_HEURISTIC_THRESHOLD = 5

# ─── SOTA Refactor (2024-2025) ────────────────────────────────────────
# New pipeline replacing the legacy Tier 2-5 cascade with three coherent
# stages: N-best RAG reranking, confidence-gated GER LLM, punctuation
# restoration. See plans/check-all-files-and-peppy-lemur.md for context
# and research backing (Apple RAG-NEC 2024, Confidence-Guided EC 2025,
# Whispering-LLaMA EMNLP 2023, GER-LoRA ACL Findings 2025).

# Stage R (n-best entity rerank) was removed in May 2026 — Whisper schema-1
# output doesn't expose the n-best alternatives, so the stage was a no-op.
# The architectural slot for "validated entity correction" is now occupied
# by pipeline/entity_corrector.py.

# Stage L: Confidence-gated Generative Error Correction (pipeline/llm_corrector.py)
# Qwen2.5-0.5B-Instruct (q4_k GGUF) via llama-cpp-python. CPU-only.
# Tokens with avg_logprob > LLM_LOGPROB_GATE are kept verbatim; only
# low-confidence tokens are wrapped <token> in the prompt and may be edited.
# After the LLM proposes an edit, xlm-roberta vetos it if MLM(original)
# is more plausible than MLM(proposed) by MLM_VETO_RATIO.
LLM_CORRECTION_ENABLED = True   # re-enabled (Step L gives +0.10 Entity F1 — see ablation thesis/pipeline_run_v5.log)
LLM_MODEL_PATH = str(PROJECT_ROOT / "whisper_cache" / "models"
                     / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf")
LLM_MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"  # for download
LLM_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"  # in the repo
LLM_CTX_WINDOW = 2048                 # n_ctx for llama.cpp
LLM_TEMPERATURE = 0.0                 # deterministic (tweaks only when uncertain)
LLM_MAX_NEW_TOKENS = 96               # plenty for one corrected segment
LLM_LOGPROB_GATE = -0.3               # tokens with logprob > this are NOT touched
LLM_MIN_TOKENS_TO_INVOKE = 2          # skip segments with < this many low-conf tokens
LLM_CTX_PREVIOUS_SEGMENTS = 2         # ±2 segment context window in the prompt
LLM_CTX_NEXT_SEGMENTS = 1
LLM_NUM_THREADS = 0                   # 0 = let llama.cpp pick (uses all cores)
MLM_VETO_ENABLED = True               # only effective when LLM_CORRECTION_ENABLED=True
MLM_VETO_RATIO = 1.5                  # reject LLM edit if MLM(orig)/MLM(prop) >= this
MLM_VETO_MODEL = "xlm-roberta-base"   # multilingual masked-LM used by Step L veto

# Stage P: Punctuation/casing restoration (pipeline/punct_restorer.py)
# oliverguhr/fullstop-punctuation-multilang-large — multilingual,
# Whisper-friendly. Search-friendly output for downstream NER + ES.
PUNCT_RESTORATION_ENABLED = True   # re-enabled (Step P contributes +0.09 Entity F1 via casing — see thesis ablation V5/V6/V3)
PUNCT_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
PUNCT_PRESERVE_EXISTING = True        # only insert; never delete existing punct/casing

# Stage D: Speaker diarization (pipeline/diarizer.py)
# pyannote/speaker-diarization-3.1 — slow on CPU (~2.5x realtime), so off
# by default. Enables per-segment speaker_id in the output JSON for
# downstream multi-commentator handling.
DIARIZATION_ENABLED = False
DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
DIARIZATION_HF_TOKEN_ENV = "HUGGINGFACE_TOKEN"  # env var name; pyannote requires auth

# Legacy learned_dictionary.py was deleted in May 2026. The cross-match
# cache concept is now owned by entity_corrector via a *validated* cache
# that requires consensus across N matches before it short-circuits the
# MCQ judge. See VALIDATED_CACHE_* below.

# ─── Entity Corrector MCQ controls (entity_corrector.py) ───────────────
# Calibrated against Chelsea-Liverpool FP analysis (Kane→Mane, Dante→Kante,
# Northampton→Southampton). Each gate addresses a specific failure mode of
# Qwen 1.5B's MCQ judge for short ambiguous tokens.

# Don't even invoke MCQ for short tokens — TF-IDF char-bigrams produces
# too many false positives below ~5 chars. Empirically: 4-char Kane/Mann
# cleared TF-IDF + Qwen MCQ but were both legitimate non-lineup names.
MCQ_MIN_TOKEN_LEN = 5

# Don't invoke MCQ unless the top retrieved canonical's reduced surface
# form has at least this fuzz.ratio to the original. Filters out
# "Northampton"-style cases where TF-IDF cosine clears 0.40 by sheer
# character-overlap but the actual token similarity is low.
MCQ_MIN_FUZZ_TO_INVOKE = 65

# Exception to the short-token block: 4-char surnames are common in football
# (Sakho/Sako, Doku, Isak, Saka). Allow MCQ for short tokens only when the
# reduced-form fuzz signal is extremely strong; Kane->Mane and Mann->Mane
# sit around 75 and remain blocked.
MCQ_SHORT_TOKEN_MIN_FUZZ = 85

# Self-consistency: run Qwen MCQ N times and majority-vote. The DeRAGEC
# ACL 2025 pattern was originally implemented with N=3 to dampen
# stochasticity at decision boundaries. Empirical probe (.work/probe_
# mcq_consistency.py) on Qwen 1.5B with single-letter MCQ output shows
# that temp=0.3 NEVER diverges from temp=0.0 — sampling adds zero
# diversity and ~30% wall-time overhead. Default is therefore 1.
# Bump to 3 if a larger / less-greedy LLM is wired in later.
MCQ_SELF_CONSISTENCY_SAMPLES = 1

# After Qwen picks A/B/C, use xlm-roberta MLM to second-guess: mask
# the entity in context and compare P(original) vs P(picked). If MLM
# strongly prefers original by MLM_VETO_RATIO, reject the MCQ pick.
# Reuses the MLM handle that llm_corrector already loaded for Step L.
MLM_VETO_ON_MCQ_ENABLED = True

# ─── Validated Cross-Match Cache (entity_corrector) ─────────────────────
# Replaces the old learned_dictionary's "blind cache" with a consensus-
# based cache: a (mishearing → canonical) mapping enters this cache only
# after it has been independently picked by the MCQ judge in
# VALIDATED_CACHE_MIN_CONSENSUS distinct matches with high LLM confidence.
# This eliminates the poison-vector failure mode where one bad correction
# in match A silently re-fired in matches B-Z.
VALIDATED_CACHE_PATH = str(PROJECT_ROOT / "data" / "validated_corrections.json")
# Lowered from 3 → 1 so single-match-validated mappings short-circuit MCQ on
# subsequent runs. Trades some FP risk for the aggressive entity-correction
# behaviour the OLD learned_dictionary delivered. The dictionary itself is
# already curated (96 entries on disk); see thesis discussion for tradeoffs.
VALIDATED_CACHE_MIN_CONSENSUS = 1
VALIDATED_CACHE_MIN_FUZZY = 75         # fuzz.ratio(orig, corr) must be ≥ this
VALIDATED_CACHE_ENABLED = True
