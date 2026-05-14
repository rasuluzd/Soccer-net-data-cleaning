---
name: regress
description: >
  Pipeline regression testing with two modes: "baseline" to capture reference
  metrics before changes, "check" to verify no regressions after. Use for
  "establish baseline", "check regressions", "verify changes", "before I
  change", or "did anything break".
---

# Pipeline Regression Testing

Two-mode skill for the baseline-change-verify development cycle.

## Mode: baseline

Capture current pipeline state before making changes. This is the default
mode when no prior baseline exists in the session.

1. Run `pytest tests/ -v`. Count pass/fail. **Stop if any test fails.**
2. Run `python run_pipeline.py --match "$ARGUMENTS" --dry-run`.
   Default to `West Ham` if no match specified.
3. Extract and record these metrics:
   - Total segments in the match
   - Hallucinations removed (Stage 1)
   - Duplicates removed (Stage 1)
   - Stage 2A normalization corrections
   - Entities detected (NER)
   - Stage E entity corrections applied
   - Step L LLM GER corrections (if enabled)
   - Step P punctuation segments restyled
4. State: **"Baseline established. Run `/regress check` after changes to compare."**

## Mode: check

Verify no regressions after a code change by comparing to the baseline.

1. Run `pytest tests/ -v`. Report pass/fail counts.
   **Stop and investigate if any test fails.**
2. Run `python run_pipeline.py --match "West Ham" --dry-run`.
   Report the same metrics as baseline.
3. Run `python run_pipeline.py --match "$ARGUMENTS" --dry-run`.
   If a second match was provided, use it. Otherwise default to `Chelsea`.
   A second match catches data-dependent regressions (different gazetteer
   composition, different language family).
4. Compare all metrics to the baseline established earlier in this session.
5. Produce a verdict:
   - **PASS**: All tests green, no metric regressions
   - **WARN**: All tests green, but some metrics changed (report which and by how much)
   - **FAIL**: Test failures or significant metric regressions (list what broke)

## Example Invocations

```
/regress                          # baseline mode (default), West Ham
/regress baseline Chelsea         # baseline on Chelsea
/regress check                    # compare against baseline, default matches
/regress check Arsenal            # compare using West Ham + Arsenal
/regress check AIK                # second match exercises non-English path
```

## Tips

- Run `/regress` (baseline) before changes and `/regress check` after. They are a pair.
- Two matches are tested in check mode because some regressions only appear with
  specific gazetteer compositions (team names that overlap with common words)
  or different language paths (MCQ judge / MLM veto behave differently on
  non-English text).
- If Stage E metrics changed but tests pass, it may be intentional — document
  in the commit message with before/after values and the constants you changed.
- If Step L is enabled and the only change is in `entity_corrector.py`,
  consider disabling Step L temporarily (`LLM_CORRECTION_ENABLED=False`) for
  faster regression runs, then re-enable before committing.
- Pairs with `/tune-threshold` for config changes.
