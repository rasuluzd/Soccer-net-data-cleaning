---
name: test-match
description: >
  Test the ASR cleaning pipeline on a specific match in dry-run mode with
  per-stage output. Unlike /baseline (which captures metrics for comparison),
  this skill is for exploratory inspection. Invoke for "test on West Ham",
  "run pipeline on Chelsea", "show me what happens with Arsenal", or "try
  Manchester City".
---

# Test Pipeline on a Specific Match

Run the pipeline on a single match with detailed per-stage output for
inspection and exploration. This is the exploratory counterpart to
`/baseline` (which is for before/after comparison).

## When to Use

- The user wants to see how the pipeline handles a specific match
- Exploring pipeline behavior on new or unfamiliar data
- Investigating match-specific issues (e.g., "what happens with the Arsenal match?")
- Verifying that a specific match benefits from a change
- The user provides a match name and wants to see corrections

## Parameters

- `match_name` (required): partial match name, case-insensitive (e.g., `West Ham`, `Chelsea`)
- `--tier N` (optional, legacy): preserved for ablation; production code does not route by tier

## Execution

1. Parse the user arguments. Expected: `<MatchName> [--tier N]`
2. Run:
   ```bash
   python run_pipeline.py --match "<match_name>" --dry-run [--tier N]
   ```
3. Report per-stage statistics in a structured format:
   - **Stage 1 (Hallucination filter)**: segments removed, alpha-ratio failures, non-Latin removals, language-mismatch removals
   - **Stage 1 (Deduplication)**: duplicate segments merged + collapsed repeated words
   - **Stage 2A (Domain normalizer)**: football compound corrections, disfluency removals
   - **NER**: entities detected (spaCy + heuristics)
   - **Stage E (Entity corrector)**: TF-IDF shortcut-accepts, MCQ-decided edits, MLM vetoes, validated-cache hits/promotions
   - **Step L (LLM GER)**: low-confidence segments processed, LLM proposals applied, MLM vetoes
   - **Step P (Punctuation)**: segments restyled
   - **Temporal chunker**: ES chunks generated (live runs only)
4. Show 3–5 sample corrections from Stage E with method:
   Format: `"original" → "corrected" (method=YY, score=XX)`
5. Highlight any warnings or errors in the output

## Examples

```
/test-match West Ham              # full pipeline run
/test-match Chelsea               # full run on Chelsea
/test-match Arsenal               # full run on Arsenal
/test-match Manchester City       # full run on Man City
/test-match AIK                   # Swedish, exercises non-English models
```

## Tips

- Use this skill for exploration. Use `/baseline` + `/regress` for verification.
- If no matches are found, check that the match name substring exists in the dataset.
- Stage E telemetry (validated-cache hits, MCQ invocations, MLM vetoes) is the
  most informative signal for understanding entity-correction behaviour. The
  orchestrator prints a summary; the full telemetry is available via
  `pipeline.entity_corrector.get_last_telemetry()`.
- To exercise non-English models, pick a match in a non-English league (e.g. AIK).
- Step L can be slow (~30–50% of segments × ~500–800 ms LLM call each).
  Use `LLM_CORRECTION_ENABLED=False` in `config.py` for fast exploration runs.
