---
name: fix-bug
description: >
  Test-first bug fix workflow for the ASR pipeline. Enforces the project
  requirement that every bug fix must include a regression test that fails
  before the fix and passes after. Invoke for "fix bug", "this is wrong",
  "pipeline broke", "wrong output", or when the user reports any incorrect
  pipeline behavior.
---

# Fix a Pipeline Bug

Follow the project's mandatory bug-fix protocol: reproduce, test, fix, verify.
Every bug fix MUST include a regression test.

## When to Use

- The user reports an incorrect correction (false positive)
- The user reports a missed correction (false negative)
- A segment was incorrectly filtered or kept by Stage 1
- Pipeline crashes or produces unexpected output
- Any behavioral bug in `pipeline/` code

## Test File Map

| Bug location | Test file |
|---|---|
| `hallucination_filter.py` | `tests/test_hallucination_filter.py` |
| `deduplicator.py` | `tests/test_deduplicator.py` |
| `domain_normalizer.py` | `tests/test_domain_normalizer.py` |
| `entity_corrector.py` (Stage E) | `tests/test_entity_corrector.py` |
| `llm_corrector.py` (Step L) | `tests/test_llm_corrector.py` |
| `punct_restorer.py` (Step P) | `tests/test_punct_restorer.py` |
| `ner_extractor.py` | `tests/test_ner_extractor.py` |
| `gazetteer.py` | `tests/test_gazetteer.py` |
| `whisper_runner.py` | `tests/test_whisper_runner.py` |
| `temporal_chunker.py` | `tests/test_temporal_chunker.py` |
| Multilingual paths | `tests/test_multilingual.py` |
| WER alignment | `tests/test_evaluate_wer_alignment.py` |
| Orchestrator output schema | `tests/test_orchestrator_metadata.py` |

## Execution

1. **Reproduce**: Identify the exact input that triggers the bug.
   - For wrong entity corrections: run `/diagnose <entity> <candidate>` to see Stage E gates
   - For filtering bugs: run `/test-match <match>` to see per-stage output
   - For LLM edits going wrong: re-run with `LLM_CORRECTION_ENABLED=False` in config to confirm Step L is the culprit
   - For punctuation issues: re-run with `PUNCT_RESTORATION_ENABLED=False`
   - For crashes: `python run_pipeline.py --match "<match>" --dry-run`
2. **Locate**: Identify which stage and file contains the bug. Use the test file map.
3. **Write test FIRST**: Add a pytest case that reproduces the bug.
   Run `pytest tests/<test_file>.py -v -k "<new_test_name>"` to confirm it **FAILS**.
4. **Fix**: Apply the minimal fix. Prefer dynamic / config-driven solutions:
   a. POS tagging checks (`get_rejected_pos_tags(lang)`) over word lists
   b. Threshold/gate adjustment in `pipeline/config.py` over inline constants
   c. MCQ prompt edits (examples in `_MCQ_SYSTEM_TEMPLATE`) for systematic LLM bias
   d. Validation-gate tightening (`CONSERVATIVE_C1_FUZZY_FLOOR`, `MCQ_MIN_FUZZ_TO_INVOKE`)
   e. **Never** add to a static word list (project rule)
5. **Verify**: Run the new test again to confirm it **PASSES**.
6. **Regress**: Run `/regress check` to confirm no other tests broke and no metrics regressed.

## Examples

```
/fix-bug "stunning" was corrected to "Sterling"
/fix-bug pipeline crashes on empty segments
/fix-bug "Palace" became a player name via MCQ
/fix-bug hallucination filter removes valid French commentary
/fix-bug Step L over-corrected "the" to "thee"
/fix-bug Stage E missed Kohlerhoff→Kolarov
```

## Tips

- The test must fail BEFORE the fix. If it passes without a fix, the test is wrong.
- For Stage E false positives, always run `/diagnose` first to see which gate
  let the pair through. Most false positives are MCQ-judge bias on short
  ambiguous tokens — tighten `MCQ_SHORT_TOKEN_MIN_FUZZ` or add a `D=keep`
  example to the system prompt.
- For Step L over-corrections, the safest first move is to raise `MLM_VETO_RATIO`
  or lower `LLM_LOGPROB_GATE` (fewer tokens become editable).
- Fix priority: POS check → gate tightening → prompt example → new logic.
- Commit format: `fix(<stage>): <what was fixed>` (e.g. `fix(entity_corrector)`, `fix(llm_corrector)`, `fix(stage1)`).
