---
name: tune-threshold
description: >
  Safely adjust pipeline thresholds or gates in config.py / entity_corrector.py
  with structured before/after verification. Enforces the required protocol:
  capture current value, baseline on two matches, apply change, re-run both
  matches, compare metrics, format commit message. Invoke for "tune", "adjust
  threshold", "change gate", "raise/lower the threshold", or when a specific
  config constant is named.
---

# Tune a Pipeline Threshold or Gate

Safely adjust a threshold or gate in `pipeline/config.py` (or the per-module
constants in `pipeline/entity_corrector.py`) using the required before/after
comparison protocol from the development workflow.

## When to Use

- The user wants to raise or lower a Stage E retrieval/MCQ gate
- Adjusting validation gates (C1 fuzzy floor, C2 length tolerance, dictionary veto)
- Changing Step L LLM gates (`LLM_LOGPROB_GATE`, `MLM_VETO_RATIO`)
- Modifying Stage 1 thresholds (`HALLUCINATION_MIN_ALPHA_RATIO`, `DUPLICATE_SIMILARITY_THRESHOLD`)
- Toggling validated-cache parameters (`VALIDATED_CACHE_MIN_CONSENSUS`, `VALIDATED_CACHE_MIN_FUZZY`)

## Tunable Constants

### Stage 1 (`pipeline/config.py`)

| Constant | Current | Effect |
|---|---|---|
| `HALLUCINATION_MIN_ALPHA_RATIO` | 0.50 | Reject if alpha ratio below; higher = more aggressive |
| `DUPLICATE_SIMILARITY_THRESHOLD` | 95 | Merge consecutive segments above this fuzz |

### Stage E (`pipeline/config.py` + `pipeline/entity_corrector.py` top constants)

| Constant | Current | Where | Effect |
|---|---|---|---|
| `SHORTCUT_ACCEPT_TFIDF` | 0.90 | `entity_corrector.py` | Auto-accept threshold |
| `SHORTCUT_ACCEPT_GAP` | 0.10 | `entity_corrector.py` | Required margin over runner-up |
| `SHORTCUT_REJECT_TFIDF` | 0.40 | `entity_corrector.py` | Below this → no MCQ call |
| `TOP_K_CANDIDATES` | 5 | `entity_corrector.py` | TF-IDF retrieval depth |
| `MCQ_MIN_TOKEN_LEN` | 5 | `config.py` | Below → require strong fuzz |
| `MCQ_MIN_FUZZ_TO_INVOKE` | 65 | `config.py` | Reduced-word fuzz floor for MCQ |
| `MCQ_SHORT_TOKEN_MIN_FUZZ` | 85 | `config.py` | Bypass `MCQ_MIN_TOKEN_LEN` when fuzz this high |
| `MCQ_SELF_CONSISTENCY_SAMPLES` | 1 | `config.py` | Increase only with a non-greedy LLM |
| `MLM_VETO_ON_MCQ_ENABLED` | True | `config.py` | xlm-r veto on MCQ picks |
| `MLM_VETO_RATIO` | 1.5 | `config.py` | Veto if `lp(orig)/lp(pick) ≥ this` |
| `CONSERVATIVE_C1_FUZZY_FLOOR` | 60 | `config.py` | Min `fuzz(orig, corr)` to accept |
| `CONSERVATIVE_C2_LEN_TOLERANCE` | 0.6 | `config.py` | `|Δlen| ≤ max(2, this·len)` |
| `DICTIONARY_VETO_MIN_LEN` | 4 | `config.py` | Veto common-word originals at this length |
| `FREQUENCY_HEURISTIC_THRESHOLD` | 5 | `config.py` | Reject if token appears this many times |
| `VALIDATED_CACHE_MIN_CONSENSUS` | 3 | `config.py` | Matches required to promote into cache |
| `VALIDATED_CACHE_MIN_FUZZY` | 75 | `config.py` | Min fuzz to write cache entry |

### Step L (`pipeline/config.py`)

| Constant | Current | Effect |
|---|---|---|
| `LLM_LOGPROB_GATE` | -0.3 | Tokens with logprob above this are LOCKED (not editable) |
| `LLM_MIN_TOKENS_TO_INVOKE` | 2 | Min low-conf tokens before LLM is called on segment |
| `LLM_CTX_PREVIOUS_SEGMENTS` | 2 | Previous-segment context in prompt |
| `LLM_CTX_NEXT_SEGMENTS` | 1 | Next-segment context in prompt |
| `MLM_VETO_RATIO` | 1.5 | Same xlm-r ratio used by Stage E |

### Legacy (still in config, not read by production code)

These remain in `config.py` for legacy tests and ablation comparison only;
adjusting them no longer affects the production cleaning path:

| Constant | Current |
|---|---|
| `FUZZY_WEIGHT` / `PHONETIC_WEIGHT` / `CONTEXT_WEIGHT` | 0.65 / 0.20 / 0.15 |
| `TIER2_ACCEPT_THRESHOLD` | 72 |
| `TIER3_VALIDATION_THRESHOLD` | 0.55 |
| `CONTEXT_SIMILARITY_THRESHOLD` | 0.50 |
| `MIN_GAP` | 0.08 |

## Execution

1. Identify the constant. Read its current value from `pipeline/config.py` or
   `pipeline/entity_corrector.py`.
2. Run `python run_pipeline.py --match "West Ham" --dry-run` — capture BEFORE metrics.
3. Run `python run_pipeline.py --match "Chelsea" --dry-run` — capture BEFORE metrics.
   For multilingual changes, additionally use `--match "AIK"` (Swedish).
4. Apply the change. Change ONLY the constant value.
5. Run `pytest tests/ -v` — confirm tests pass. If any fail, revert.
6. Run `python run_pipeline.py --match "West Ham" --dry-run` — capture AFTER metrics.
7. Run `python run_pipeline.py --match "Chelsea" --dry-run` — capture AFTER metrics.
8. Compare BEFORE vs AFTER. Report which metrics improved, regressed, and net effect.
9. If beneficial, format commit message:
   ```
   tune(<stage>): <CONSTANT> <old> → <new>

   West Ham: Stage E <before> → <after>, hallucinations <before> → <after>
   Chelsea: Stage E <before> → <after>, hallucinations <before> → <after>
   Reason: <why the new value is better, with the audit evidence>
   ```
10. If regressions occur, revert and explain why.

## Examples

```
/tune-threshold MCQ_MIN_FUZZ_TO_INVOKE to 70
/tune-threshold raise CONSERVATIVE_C1_FUZZY_FLOOR to 65
/tune-threshold lower HALLUCINATION_MIN_ALPHA_RATIO to 0.45
/tune-threshold SHORTCUT_ACCEPT_TFIDF 0.90 0.85
/tune-threshold MLM_VETO_RATIO from 1.5 to 1.8
```

## Tips

- Never inline a threshold in pipeline code. All constants live in `config.py`
  (or the top of `entity_corrector.py` for Stage E-only constants).
- Small changes (1–3 points for integer thresholds, ±0.05 for floats) are safer.
  Large jumps risk cascading regressions across stages.
- Use `/diagnose` on specific entity pairs to verify edge cases before committing.
- When tuning Stage E gates, test against BOTH English (West Ham/Chelsea) and a
  non-English match (AIK) — the multilingual MCQ judge and MLM veto behave
  differently across languages.
- Tightening `MCQ_MIN_FUZZ_TO_INVOKE` is the most-used knob for reducing
  false positives; loosening it is the way to catch missed corrections.
