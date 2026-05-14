---
paths:
  - "pipeline/**/*.py"
  - "tests/**/*.py"
---

# Development Workflow

## Change Loop (mandatory)

1. `python run_pipeline.py --match "West Ham" --dry-run` — baseline
2. `pytest tests/ -v` — confirm green
3. Make your change
4. `pytest tests/ -v` — still green
5. `python run_pipeline.py --match "West Ham" --dry-run` — compare to step 1

## Bug Fix Requirements

Every bug fix needs a test that fails before the fix and passes after. See CLAUDE.md for test file map.

## Config Change Protocol

When tuning thresholds: record before/after values in commit message, run `--dry-run` on two matches, document why the new value is better.
