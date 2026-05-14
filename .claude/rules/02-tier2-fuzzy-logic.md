---
paths:
  - "pipeline/entity_corrector.py"
  - "pipeline/fuzzy_corrector.py"
  - "pipeline/config.py"
  - "pipeline/ner_extractor.py"
  - "pipeline/gazetteer.py"
---

# Stage E: Validated Entity Correction

Replaces the legacy Tier 2 fuzzy+phonetic+context cascade (May 2026 refactor).
For each NER-detected entity, `pipeline/entity_corrector.py` runs:

1. Validated cross-match cache lookup (≥3 independent matches with high fuzz)
2. TF-IDF char-bigram retrieval over gazetteer canonicals (`TFIDF_NGRAM_RANGE=(2,4)`, `TOP_K_CANDIDATES=5`)
3. Per-match cache lookup
4. Frequency heuristic (reject if token ≥ `FREQUENCY_HEURISTIC_THRESHOLD=5` in match)
5. Shortcut-reject: cosine < `SHORTCUT_REJECT_TFIDF=0.40` → skip
6. Shortcut-accept: cosine ≥ `SHORTCUT_ACCEPT_TFIDF=0.90` AND gap ≥ `SHORTCUT_ACCEPT_GAP=0.10` AND gates pass → apply
7. MCQ pre-gates: token length < `MCQ_MIN_TOKEN_LEN=5` requires top fuzz ≥ `MCQ_SHORT_TOKEN_MIN_FUZZ=85`; otherwise top fuzz ≥ `MCQ_MIN_FUZZ_TO_INVOKE=65`
8. MCQ judge — Qwen2.5-1.5B-Instruct GGUF picks A/B/C from top candidates, D=keep, E=unsure
9. MLM veto (xlm-roberta-base): reject MCQ pick if `lp(orig) - lp(pick) ≥ log(MLM_VETO_RATIO=1.5)`
10. Validation gates: dictionary veto (≥4-char common word), C1 fuzzy floor (≥`CONSERVATIVE_C1_FUZZY_FLOOR=60`), C2 length tolerance (≤`CONSERVATIVE_C2_LEN_TOLERANCE=0.6`)

No phonetic scoring is in the production path. Fuzzy similarity (`rapidfuzz.fuzz.ratio`) is
only used as a gate, never as primary signal.

## Routing summary

| Signal | Action |
|---|---|
| Validated-cache hit | Apply directly (skip retrieval + MCQ) |
| TF-IDF cosine ≥ 0.90 with clear winner | Auto-accept (after gates) |
| TF-IDF cosine 0.40–0.89 + pre-gates pass | MCQ judge decides |
| MCQ picks A/B/C + MLM doesn't veto + gates pass | Apply |
| Any rejection | Keep original |

## Key Directives

1. **Strip entity boundaries** before retrieval — `fuzzy_corrector.extract_entity_core` / `extract_and_rebuild_entity` handle possessives (`Ward's`, Germanic `-s`) and trailing punctuation
2. **Type-aware** — `entity_types` map flags whether a canonical is a player/team/referee/coach so MCQ context can be specific
3. **Language-conditional gates** — `passes_conservative_gates(original, corrected, language)` reads dictionary veto for the detected language
4. **Tune via gate constants, not weights** — `MCQ_MIN_FUZZ_TO_INVOKE`, `SHORTCUT_ACCEPT_TFIDF`, `CONSERVATIVE_C1_FUZZY_FLOOR` are the right knobs; legacy `FUZZY_WEIGHT`/`PHONETIC_WEIGHT`/`CONTEXT_WEIGHT` are no longer read in production
5. **Validated-cache writes need consensus** — `VALIDATED_CACHE_MIN_CONSENSUS=3` matches with `VALIDATED_CACHE_MIN_FUZZY≥75` before short-circuiting MCQ

## Debugging

```bash
python .claude/skills/diagnose/scripts/diagnose_entity.py "entity" "candidate"
```

The diagnostic shows fuzz/length/dictionary signals + which Stage E gate the pair would hit.
For a real routing decision, the full TF-IDF index needs the match's gazetteer — run
`python run_pipeline.py --match "<match>" --dry-run` and inspect Stage E telemetry.
