---
name: baseline
description: >
  Capture pipeline baseline metrics before making code changes. Runs pytest and
  a dry-run to record segments processed, hallucinations filtered, entities
  corrected (Stage E), and LLM edits applied (Step L) so regressions can be
  detected after changes. Invoke for "establish baseline", "before I change",
  "capture metrics", "current state", or at session start.
---

# Establish Pipeline Baseline

Capture the current pipeline state before making changes, so regressions can
be detected by comparing before/after metrics.

## When to Use

- Before any code change to files in `pipeline/` or `pipeline/config.py`
- When the user says "baseline", "before I start", "capture current state"
- At the start of a development session to confirm the pipeline is healthy
- Before tuning any threshold or gate in `pipeline/config.py`

## Parameters

- `match_name` (optional): match to baseline against (default: `West Ham`)

## Execution

1. Run `pytest tests/ -v`. Count pass/fail. **Stop immediately if any test fails** —
   a broken baseline is meaningless.
2. Run:
   ```bash
   python run_pipeline.py --match "<match_name>" --dry-run
   ```
   Default to `West Ham` if no match name was provided.
3. Extract and report these metrics:
   - Total segments in the match
   - Hallucinations removed (Stage 1)
   - Duplicates removed (Stage 1)
   - Stage 2A normalization corrections (domain normalizer)
   - Entities detected (NER)
   - Stage E entity corrections applied
   - Step L LLM GER corrections (if `LLM_CORRECTION_ENABLED=True`)
   - Step P punctuation segments restyled (if `PUNCT_RESTORATION_ENABLED=True`)
4. State: **"Baseline established. Run `/regress` after changes to compare."**

## Examples

```
/baseline                     # defaults to "West Ham"
/baseline Chelsea             # baseline for Chelsea match
/baseline Manchester City     # baseline for Man City match
```

## Tips

- Keep output factual and compact. No filler text.
- Record the baseline metrics so they can be compared line-by-line after changes.
- If pytest reports failures, diagnose those first — do not proceed with a broken baseline.
- This skill pairs with `/regress` which does the post-change comparison.
- If running with `--tier <N>` (legacy flag), note that production code no longer
  routes work based on tier; the flag is preserved for ablation against the deleted
  legacy cascade.
