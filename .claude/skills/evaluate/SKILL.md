---
name: evaluate
description: >
  Compute WER/CER/Entity-F1 of the cleaning pipeline against ground-truth
  transcription (1_asr_corrected.json). Produces an ablation table showing
  raw Whisper vs cleaned output. Invoke for "evaluate", "WER", "compare to
  ground truth", "accuracy", or "how much did we improve".
---

# Pipeline Evaluation Against Ground Truth

Measure how much the cleaning pipeline improved over raw Whisper. Uses jiwer
(industry-standard ASR metrics) and produces a thesis-ready ablation table.

## When to Use

- The user has a ground-truth transcription (`1_asr_corrected.json`) and wants
  to know the WER improvement of the pipeline
- After running the pipeline on a match — to quantify the improvement
- For thesis ablation tables and case studies
- When the user says "evaluate", "WER", "how much did we improve",
  "compare to ground truth", "accuracy"

## Recommended benchmark

GOAL English (Chelsea–Liverpool 2016) is the cleanest evaluation set the
project has. The AIK Swedish match has GT that was contaminated by stock
Whisper output, so do NOT cite raw AIK numbers as a baseline.

## Parameters

- `match_substring` (required): substring of the match name (e.g. `Chelsea`, `GOAL`)
- `half` (optional, default 1): which half to evaluate

## Execution

1. **Pre-check**: verify that the match has `1_asr_corrected.json` alongside
   `1_asr.json`. If not, tell the user: "This match has no ground truth —
   `/evaluate` needs a `1_asr_corrected.json` file in the match folder."

2. **Run pipeline if cleaned output missing**:
   ```bash
   python run_pipeline.py --match "<match_substring>"
   ```
   Only run live (not `--dry-run`) because evaluation needs the written
   `<half>_asr_cleaned.json` files.

3. **Compute WER**:
   ```bash
   python tools/evaluate_wer.py --match "<match_substring>" --half <half>
   ```

4. **Show the per-segment diff** (qualitative examples):
   ```bash
   python tools/per_segment_diff.py --match "<match_substring>" --filter GOOD --limit 5
   python tools/per_segment_diff.py --match "<match_substring>" --filter HARMFUL --limit 5
   ```

5. **Report the headline numbers**: WER, Entity-F1, and Stage E telemetry counts
   (validated-cache hits, MCQ invocations, MLM vetoes) — entity-F1 is the
   thesis's key metric because Stage E's job is name correction.

6. Point the user to `evaluation_results/` for the saved JSON + Markdown table.

## Stage-by-stage ablation

For thesis ablation tables, toggle stages in `pipeline/config.py` and re-run:

| Toggle | Off effect |
|---|---|
| `DOMAIN_NORMALIZATION_ENABLED=False` | Skip Stage 2A football compounds |
| `ENTITY_CORRECTION_ENABLED=False` | Skip Stage E entity correction entirely |
| `MLM_VETO_ON_MCQ_ENABLED=False` | Stage E accepts MCQ picks without xlm-r veto |
| `VALIDATED_CACHE_ENABLED=False` | No cross-match cache short-circuit |
| `LLM_CORRECTION_ENABLED=False` | Skip Step L LLM GER |
| `MLM_VETO_ENABLED=False` | Step L applies LLM edits without veto |
| `PUNCT_RESTORATION_ENABLED=False` | Skip Step P punctuation/casing |
| `DICTIONARY_VETO_ENABLED=False` | No common-word dictionary veto in gates |

## Examples

```
/evaluate Chelsea               # GOAL benchmark, English
/evaluate Liverpool 2           # half 2 if present
/evaluate "AIK Halmstad" 1      # Swedish, with caveat about contaminated GT
```

## Tips

- Expected WER for raw English ASR on broadcast football: 18–30%.
  Swedish raw with stock Whisper: 40–60% (KB-Whisper recovers much of this).
- If cleaned WER is *worse* than raw, the pipeline is over-correcting —
  check `per_segment_diff.py --filter HARMFUL` to see where. Step L is the
  most common culprit; try setting `LLM_CORRECTION_ENABLED=False` for one run.
- Entity F1 is the thesis's key metric — the pipeline's main job is fixing
  player/team names, not generic content words.
- The output schema-2 enrichments (`avg_logprob`, `no_speech_prob`, `words`)
  in cleaned JSONs are useful for downstream confidence-aware search.

## Output Location

- `evaluation_results/<match>_half<N>_wer.md` — markdown ablation table
- `evaluation_results/<match>_half<N>_wer.json` — JSON artifact for plotting
