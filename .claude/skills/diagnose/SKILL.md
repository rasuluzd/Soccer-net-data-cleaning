---
name: diagnose
description: >
  Debug how the pipeline would route an entity-candidate pair through Stage E
  (TF-IDF + MCQ judge) gates. Runs a diagnostic script showing fuzzy/length/
  dictionary signals and the gate the pair would hit. Invoke for "why did it
  change X to Y", "false positive", "wrong correction", "debug scoring", "why
  wasn't X corrected", or when given an entity-candidate pair.
---

# Debug Entity-Candidate Routing

Show the signals Stage E (`pipeline/entity_corrector.py`) uses to gate a
specific entity-candidate pair. Useful for diagnosing false positives
(wrong corrections) and false negatives (missed corrections).

NOTE: A truly accurate routing decision requires the full TF-IDF index built
from the match's gazetteer (because cosine is computed against ALL canonicals
together). This skill computes the per-pair signals — fuzz, reduced-word fuzz,
length tolerance, dictionary veto. For end-to-end routing, also run
`/test-match` and read Stage E telemetry.

## When to Use

- The user reports a wrong correction (e.g., "stunning" → "Sterling")
- The user asks why a known ASR error was not corrected
- The user provides two strings and wants to see which gates pass/fail
- Investigating whether an MCQ gate or validation gate is too strict / too loose
- Sanity-checking before tuning `MCQ_MIN_FUZZ_TO_INVOKE`, `MCQ_SHORT_TOKEN_MIN_FUZZ`,
  `SHORTCUT_ACCEPT_TFIDF`, `CONSERVATIVE_C1_FUZZY_FLOOR`, etc.

## Parameters

- `entity` (required): the ASR-transcribed string (e.g., `stunning`, `Kohlerhoff`)
- `candidate` (required): the gazetteer canonical being considered (e.g., `Sterling`, `Kolarov`)
- `language` (optional, default `en`): ISO 639-1 code for dictionary-veto language

## Execution

1. Run the diagnostic script:
   ```bash
   python .claude/skills/diagnose/scripts/diagnose_entity.py "<entity>" "<candidate>" [language]
   ```
2. Display the signal breakdown:
   - **fuzz_ratio**: `rapidfuzz.fuzz.ratio(entity, candidate)` (0–100)
   - **reduced_word_fuzz**: fuzz against the closest canonical word (handles "Daniel Sturridge" → "Sturridge" matching)
   - **token_length**: matters for `MCQ_MIN_TOKEN_LEN=5` gate
   - **dictionary_veto**: whether `entity` is a real word ≥4 chars in the language's spell-check dict
   - **c1_passes**: `fuzz.ratio(entity, candidate) ≥ CONSERVATIVE_C1_FUZZY_FLOOR=60`
   - **c2_passes**: `|len(candidate) - len(entity)| ≤ max(2, 0.6·len(entity))`
3. Explain the verdict in current Stage E terms:
   - **MCQ-eligible**: pair clears `MCQ_MIN_FUZZ_TO_INVOKE` AND length-or-fuzz gates AND validation; would be sent to Qwen MCQ judge if cosine lands in 0.40–0.89 band
   - **Auto-accept candidate**: would pass `SHORTCUT_ACCEPT_TFIDF=0.90` shortcut IF TF-IDF cosine clears it AND gap ≥ 0.10
   - **Pre-MCQ rejected**: short token + low reduced-word fuzz, OR top fuzz < 65, OR dictionary veto, OR C1/C2 fail
4. If **false positive** (wrong correction applied), suggest fixes in priority order:
   a. Confirm the original token's `token.pos_` via spaCy (`ADJ`/`VERB`/`DET` should already be rejected by `get_rejected_pos_tags`)
   b. Raise `MCQ_MIN_FUZZ_TO_INVOKE` (currently 65) for noisy short pairs
   c. Raise `CONSERVATIVE_C1_FUZZY_FLOOR` (currently 60) to filter low-similarity accepts
   d. Add the original to the MCQ system prompt's "D=keep when ambiguous" examples
   e. Tighten `MCQ_SHORT_TOKEN_MIN_FUZZ` (currently 85) for ≤4-char tokens
   f. **Never** add to a static word list (project rule)
5. If **false negative** (good correction missed), suggest:
   a. Check the entity is being extracted by `ner_extractor` at all
   b. Check the candidate exists in the match gazetteer (`Labels-caption.json`)
   c. Confirm the pair's reduced-word fuzz ≥ `MCQ_MIN_FUZZ_TO_INVOKE`
   d. If short token, confirm fuzz ≥ `MCQ_SHORT_TOKEN_MIN_FUZZ`
   e. Inspect the MCQ judge's actual reply via Stage E telemetry on a real run

## Examples

```
/diagnose stunning Sterling           # false positive investigation
/diagnose Kohlerhoff Kolarov          # real ASR error, long name
/diagnose Sacco Sakho                 # real ASR error, medium name
/diagnose Kommer Kouame sv            # Swedish, dictionary-veto triggers
/diagnose Sturridge Sturridge         # self-match, should be no-op
```

## References

Load `references/tier2-scoring.md` for the full Stage E gate breakdown and
threshold table. Load `references/tier3-acceptance.md` for Step L + Step P
acceptance criteria when the bug is downstream of Stage E.

## Tips

- The script computes per-pair signals only. Real routing also requires
  TF-IDF cosine against ALL gazetteer canonicals; for a full picture run
  `/test-match` and read Stage E telemetry.
- For false positives, always check the entity token's `token.pos_` first
  via spaCy. Most "common-word → player" false positives are tokens that
  should already be rejected by `REJECTED_POS_TAGS` (or the non-English variant).
- Sound-based phonetic scoring is no longer in the production path. Don't
  reach for Metaphone changes; the regressions were caused by Metaphone
  prefix-collisions in the first place.
