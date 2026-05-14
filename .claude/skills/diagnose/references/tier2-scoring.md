# Stage E: Gate Breakdown & Routing

Reference for the `/diagnose` skill. See `pipeline/entity_corrector.py` and
`pipeline/config.py` for current values.

## Decision Order

For each (entity, top-K-candidates) pair, `entity_corrector` runs the
following gates in order. The first one that fires terminates the entity:

| # | Gate | Behaviour |
|---|---|---|
| 1 | Validated cross-match cache hit | Apply cached canonical (skip retrieval + MCQ) |
| 2 | TF-IDF retrieve top-K=5 | If no candidates, skip entity |
| 3 | Self-match (best canonical == entity) | Skip — already canonical |
| 4 | Per-match cache hit | Reuse previous in-match decision |
| 5 | Frequency heuristic | If `token_freq ≥ FREQUENCY_HEURISTIC_THRESHOLD=5` → reject |
| 6 | Shortcut-reject | `cosine < SHORTCUT_REJECT_TFIDF=0.40` → reject |
| 7 | Shortcut-accept | `cosine ≥ SHORTCUT_ACCEPT_TFIDF=0.90` AND `gap ≥ SHORTCUT_ACCEPT_GAP=0.10` AND gates pass → apply |
| 8 | MCQ pre-gate A (short token) | `len(entity) < MCQ_MIN_TOKEN_LEN=5` AND `top_fuzz < MCQ_SHORT_TOKEN_MIN_FUZZ=85` → reject |
| 9 | MCQ pre-gate B (reduced-word fuzz) | `top_reduced_fuzz < MCQ_MIN_FUZZ_TO_INVOKE=65` → reject |
| 10 | MCQ judge | Qwen2.5-1.5B-Instruct picks A/B/C/D/E with ±2 segment context + typed match block |
| 11 | MLM veto on MCQ pick | If `MLM(orig)/MLM(pick) ≥ MLM_VETO_RATIO=1.5` → reject pick |
| 12 | Validation gates | Dictionary veto + C1 (`fuzz ≥ 60`) + C2 (`|Δlen| ≤ max(2, 0.6·len)`) → reject if any fail |

## Key Constants (`pipeline/config.py`)

```
# TF-IDF retrieval
SHORTCUT_ACCEPT_TFIDF      = 0.90
SHORTCUT_ACCEPT_GAP        = 0.10
SHORTCUT_REJECT_TFIDF      = 0.40
TOP_K_CANDIDATES           = 5
TFIDF_NGRAM_RANGE          = (2, 4)

# MCQ pre-gates
MCQ_MIN_TOKEN_LEN          = 5
MCQ_MIN_FUZZ_TO_INVOKE     = 65
MCQ_SHORT_TOKEN_MIN_FUZZ   = 85
MCQ_SELF_CONSISTENCY_SAMPLES = 1
MCQ_OPTIONS_SHOWN          = 3   # plus D=keep + E=unsure

# MLM veto
MLM_VETO_ON_MCQ_ENABLED    = True
MLM_VETO_RATIO             = 1.5
MLM_VETO_MODEL             = "xlm-roberta-base"

# Validation
CONSERVATIVE_C1_FUZZY_FLOOR = 60
CONSERVATIVE_C2_LEN_TOLERANCE = 0.6
DICTIONARY_VETO_ENABLED    = True
DICTIONARY_VETO_MIN_LEN    = 4
FREQUENCY_HEURISTIC_THRESHOLD = 5

# Validated cross-match cache
VALIDATED_CACHE_ENABLED       = True
VALIDATED_CACHE_MIN_CONSENSUS = 3
VALIDATED_CACHE_MIN_FUZZY     = 75
```

## Key Details

- **Strip entity boundaries** before scoring — `fuzzy_corrector.extract_entity_core`
  handles possessives (`Ward's`, Germanic `-s`) and trailing punctuation;
  `extract_and_rebuild_entity` re-attaches them after canonical substitution
- **TF-IDF replaces phonetic** — `jellyfish.metaphone` is no longer in the
  production path. Phonetic prefix-collisions (Saturday/Sturridge ≈ 0.92)
  were the original motivation to switch to char n-grams over the gazetteer
- **Reduced-word fuzz** — when the entity is one token and the canonical is
  `"Daniel Sturridge"`, we fuzz-score against the closest canonical word
  (`Sturridge`), not the full multi-word string
- **Validated-cache writes** require `VALIDATED_CACHE_MIN_CONSENSUS=3` independent
  matches with `VALIDATED_CACHE_MIN_FUZZY≥75` — single-match wins do not poison
  the cache
- **Language-conditional dictionary veto** — `passes_conservative_gates(orig, corr, language)`
  consults the per-language spell-check dictionary; degrades to no veto when
  the language is unavailable
