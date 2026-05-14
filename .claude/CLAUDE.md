# SoccerNet ASR Cleaning Pipeline

Multi-stage NLP pipeline that cleans Whisper ASR football commentary and emits
search-friendly JSON for Elasticsearch indexing. Post-May-2026 architecture:
the legacy 3-tier fuzzy/phonetic/context cascade was replaced by a TF-IDF
retrieval + Qwen MCQ-judge entity corrector and a confidence-gated GER LLM.

Multilingual via language-conditional models (English, Swedish, Norwegian,
Danish, German, French, Spanish, Italian, Portuguese, Dutch).

## Commands

```bash
python run_pipeline.py --match "West Ham" --dry-run     # preview one match
python run_pipeline.py                                   # process all matches
python run_pipeline.py --tier 2                          # legacy tier cap (1..3, no longer routes work)
pytest tests/ -v                                         # full test suite
pytest tests/test_entity_corrector.py -v                 # single module
pytest tests/test_multilingual.py -v                     # multilingual tests
python .claude/skills/diagnose/scripts/diagnose_entity.py "stunning" "Sterling"  # debug scoring
```

## Architecture (current, post-May-2026)

The orchestrator runs these steps per match (see `pipeline/orchestrator.py`):

| Step | File | Purpose |
|---|---|---|
| 0  | `hallucination_filter.detect_commentary_language` | langdetect on transcript sample |
| 1  | `gazetteer.build_gazetteer` | Name-variant dict + typed entity map from Labels-caption.json |
| 2  | `hallucination_filter.filter_segments` | Alpha-ratio, langdetect, non-Latin script removal |
| 3  | `deduplicator.deduplicate_segments` | Consecutive near-duplicate merging + collapse repeated words |
| 2A | `domain_normalizer.DomainNormalizer` | Football compounds, disfluency removal (regex) |
| —  | `ner_extractor.extract_entities_batch` | spaCy NER + POS-filtered heuristics |
| E  | `entity_corrector.correct_match` | **TF-IDF retrieve → Qwen MCQ judge** (replaces Tier 2/3) |
| L  | `llm_corrector.correct_match` | **Confidence-gated GER** (Qwen2.5-1.5B + xlm-r MLM veto) |
| P  | `punct_restorer.restore_punctuation_batch` | oliverguhr fullstop multilingual |
| —  | `temporal_chunker.create_temporal_chunks` | Rolling-window ES docs |
| —  | `pipeline.config` | All thresholds, models, toggles |

Supporting modules:
| File | Purpose |
|---|---|
| `pipeline/fuzzy_corrector.py` | Slim helpers: `Correction` dataclass, entity boundary extraction, `passes_conservative_gates` (dict veto + C1 fuzzy floor + C2 length tol) |
| `pipeline/loader.py` | Match discovery, ASR JSON schema-1/schema-2 parsing |
| `pipeline/whisper_runner.py` | faster-whisper transcription, per-word probs, language-aware ASR model selection |
| `pipeline/report.py` | CleaningResult → human-readable summary |

### Removed in the May 2026 refactor

- `learned_dictionary.py` → replaced by **validated cross-match cache** in `entity_corrector` (3-match consensus required before short-circuit)
- `context_disambiguator.py` (sentence-transformer cosine) → replaced by **Qwen MCQ judge**
- Legacy Tier 2 `compute_combined_score` / `find_best_match` (fuzzy + phonetic + context cascade) → replaced by **TF-IDF char-bigram retrieve + MCQ**
- Stage 2B pyspellchecker, Stage 2C LanguageTool, Stage 4 mT5/BERT, Stage 5 Ollama rewriter, Stage R n-best rerank — all deleted (0 net wins after gates / no n-best input available)

## Entity Corrector flow (`pipeline/entity_corrector.py`)

Per detected entity, in order:

1. **Validated cross-match cache** — if mapping has ≥3 independent matches with high fuzz, apply directly
2. **TF-IDF retrieve** — char_wb n-grams (2,4) over gazetteer canonicals, top-K=5, fuzz-rescue lifts high-fuzz typos above reject floor
3. **Per-match cache** — reuse decision for `(entity, top-3 candidates)` tuple
4. **Frequency heuristic** — reject if token appears ≥5× in match (likely common word)
5. **Shortcut-reject** — cosine `< SHORTCUT_REJECT_TFIDF (0.40)` → skip
6. **Shortcut-accept** — cosine `≥ SHORTCUT_ACCEPT_TFIDF (0.90)` AND `gap ≥ 0.10` AND validation gates pass → apply
7. **MCQ pre-gates** — `len < 5 AND top_fuzz < 85` → reject; `top_fuzz < 65` → reject
8. **MCQ judge** — Qwen2.5-1.5B-Instruct GGUF, A/B/C candidates + D=keep + E=unsure, with match-context block (Teams/Players/Referee) + prev/next segment text
9. **MLM veto on MCQ pick** — xlm-roberta-base pseudo-logprob; reject if `lp(orig) - lp(pick) ≥ log(MLM_VETO_RATIO=1.5)`
10. **Validation gates** — dictionary veto (≥4-char common words), C1 fuzzy floor (≥60), C2 length tolerance (≤max(2, 0.6·len))

## Key Thresholds (`pipeline/config.py`)

```
HALLUCINATION_MIN_ALPHA_RATIO = 0.50
DUPLICATE_SIMILARITY_THRESHOLD = 95

# Entity Corrector (Stage E)
SHORTCUT_ACCEPT_TFIDF = 0.90, SHORTCUT_ACCEPT_GAP = 0.10
SHORTCUT_REJECT_TFIDF = 0.40, TOP_K_CANDIDATES = 5, TFIDF_NGRAM_RANGE = (2,4)
MCQ_MIN_TOKEN_LEN = 5, MCQ_MIN_FUZZ_TO_INVOKE = 65, MCQ_SHORT_TOKEN_MIN_FUZZ = 85
MCQ_SELF_CONSISTENCY_SAMPLES = 1
MLM_VETO_ON_MCQ_ENABLED = True, MLM_VETO_RATIO = 1.5, MLM_VETO_MODEL = "xlm-roberta-base"
VALIDATED_CACHE_MIN_CONSENSUS = 3, VALIDATED_CACHE_MIN_FUZZY = 75

# Validation gates (apply to all proposed corrections)
CONSERVATIVE_C1_FUZZY_FLOOR = 60, CONSERVATIVE_C2_LEN_TOLERANCE = 0.6
DICTIONARY_VETO_ENABLED = True, DICTIONARY_VETO_MIN_LEN = 4
FREQUENCY_HEURISTIC_THRESHOLD = 5

# Step L (GER LLM)
LLM_MODEL_FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_LOGPROB_GATE = -0.3, LLM_CTX_WINDOW = 2048, LLM_MAX_NEW_TOKENS = 96
LLM_CTX_PREVIOUS_SEGMENTS = 2, LLM_CTX_NEXT_SEGMENTS = 1

# Step P (punctuation)
PUNCT_MODEL = "oliverguhr/fullstop-punctuation-multilang-large"
PUNCT_PRESERVE_EXISTING = True
```

Legacy weights `FUZZY_WEIGHT=0.65 / PHONETIC_WEIGHT=0.20 / CONTEXT_WEIGHT=0.15`,
`TIER2_ACCEPT_THRESHOLD=72`, `TIER3_VALIDATION_THRESHOLD=0.55`, `MIN_GAP=0.08` still
live in `config.py` but are **only read by tests** and the deprecated `--tier` flag.
Production code does not depend on them.

## Stage Toggles (for ablation)

```
ENTITY_CORRECTION_ENABLED = True   # Stage E (TF-IDF + MCQ)
DOMAIN_NORMALIZATION_ENABLED = True
LLM_CORRECTION_ENABLED = True      # Step L (Qwen GER)
PUNCT_RESTORATION_ENABLED = True   # Step P
MLM_VETO_ENABLED = True            # Step L veto
MLM_VETO_ON_MCQ_ENABLED = True     # Stage E veto on MCQ pick
VALIDATED_CACHE_ENABLED = True
DICTIONARY_VETO_ENABLED = True
DIARIZATION_ENABLED = False        # pyannote, slow on CPU, off by default
TIER3_ENABLED = True               # legacy; only affects tests
```

## Multilingual Support

| Component | English | Non-English |
|---|---|---|
| spaCy model | `en_core_web_sm` | `sv_core_news_sm`, `de_core_news_sm`, ... or `xx_ent_wiki_sm` fallback |
| ASR model | `Systran/faster-whisper-large-v3` | sv/no/da: `KBLab/kb-whisper-large`; de: `primeline/whisper-large-v3-turbo-german` |
| Punctuation | oliverguhr multilang | same (multilingual model) |
| MLM veto | `xlm-roberta-base` | same (multilingual) |
| Entity labels | PERSON, ORG, GPE, FAC | PER, ORG, LOC, MISC |
| POS rejection | adds NOUN/ADJ/VERB | drops NOUN/ADJ/VERB (small models mis-tag foreign names) |
| Language detection | `hallucination_filter.detect_commentary_language` propagates to all stages |

## Development Rules

1. **No static word lists.** Use `token.pos_` filtering (`REJECTED_POS_TAGS` per language), never hardcoded sets
2. **Config-only constants.** All thresholds in `pipeline/config.py`, never inline
3. **Verify every change.** Run `pytest tests/` and `--dry-run` before committing
4. **Every bug fix needs a test** that fails before the fix and passes after

## Development Workflow

```bash
python run_pipeline.py --match "West Ham" --dry-run   # 1. baseline
pytest tests/ -v                                        # 2. confirm green
# ... make change ...
pytest tests/ -v                                        # 3. tests still green
python run_pipeline.py --match "West Ham" --dry-run   # 4. no regression
```

## Test File Map

| Bug location | Test file |
|---|---|
| `hallucination_filter.py` | `tests/test_hallucination_filter.py` |
| `deduplicator.py` | `tests/test_deduplicator.py` |
| `domain_normalizer.py` | `tests/test_domain_normalizer.py` |
| `entity_corrector.py` (TF-IDF + MCQ) | `tests/test_entity_corrector.py` |
| `llm_corrector.py` (Step L) | `tests/test_llm_corrector.py` |
| `punct_restorer.py` (Step P) | `tests/test_punct_restorer.py` |
| `ner_extractor.py` | `tests/test_ner_extractor.py` |
| `gazetteer.py` | `tests/test_gazetteer.py` |
| `whisper_runner.py` | `tests/test_whisper_runner.py` |
| `temporal_chunker.py` | `tests/test_temporal_chunker.py` |
| multilingual support | `tests/test_multilingual.py` |
| WER alignment | `tests/test_evaluate_wer_alignment.py` |
| orchestrator output schema | `tests/test_orchestrator_metadata.py` |

## Commit Format

```
fix(entity_corrector): block Kane→Mane via MCQ_SHORT_TOKEN_MIN_FUZZ gate
feat(llm_corrector): pass language code to xlm-r MLM veto
```
