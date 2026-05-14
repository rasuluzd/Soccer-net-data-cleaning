# Claude Code Guide — SoccerNet ASR Pipeline

> Full manual for a junior engineer using Claude via GitHub Copilot / Claude Agent SDK.
> Keep this file open in a tab while you work.

---

## Part 1: Understanding Your Setup

### What is Claude Code and How Are You Using It?

Claude Code is a system where the AI assistant is given **project-specific knowledge** to behave like a specialist engineer on your exact codebase.

You are accessing Claude through **GitHub Copilot** with the Claude model, or through the **Claude Agent SDK**. When Claude starts your session, it automatically reads several configuration files from your `.claude/` folder.

**You never run `claude` from the terminal.** You just open your project and start talking to Claude in the chat interface. The configuration loads automatically.

### What Loads Automatically at Startup

When you open the project in your Copilot/Claude environment:

**Always loaded:**
- `.claude/CLAUDE.md` — the main project brief Claude reads first
- `.claude/agents/01-nlp-architect.agent.md` — NLP specialist
- `.claude/agents/02-eval-debugger.agent.md` — debugging specialist

**Loaded when you edit a matching file:**
- `.claude/rules/00-dev-workflow.md` — when editing any `pipeline/*.py` file
- `.claude/rules/01-global-scaling.md` — when editing any `pipeline/*.py` file
- `.claude/rules/02-tier2-fuzzy-logic.md` — when editing `entity_corrector.py`, `fuzzy_corrector.py`, `config.py`, `ner_extractor.py`, `gazetteer.py`
- `.claude/rules/03-tier3-context-ai.md` — when editing `llm_corrector.py` or `punct_restorer.py`

(The rule filenames are historical. The content inside has been rewritten to
reflect the current post-May-2026 architecture: Stage E + Step L + Step P,
not the old 3-tier cascade.)

**Runs automatically as events happen:**
- `.claude/hooks/session-context.sh` — prints context reminder at start of every session
- `.claude/hooks/validate-pipeline-edit.sh` — validates every file edit for common mistakes

### What the Hooks Mean

#### Session Hook
Every time you start a session, you will see this message appear:
```
Pipeline project context: 3-tier ASR cleaning. Run pytest + --dry-run to verify changes. No static word lists.
```
This is the `session-context.sh` hook reminding you of the core rules. The "3-tier" phrasing is a relic of the older naming — the current architecture is multi-stage. CLAUDE.md has the accurate description.

#### Edit Validation Hook
Every time Claude edits a `pipeline/*.py` file, the `validate-pipeline-edit.sh` hook runs. It checks for:

1. **Inline threshold constants** — e.g., `if score >= 0.90` instead of `if score >= SHORTCUT_ACCEPT_TFIDF`
2. **Static word lists** — `COMMON_WORDS_EXCLUDE` or similar
3. **Hardcoded model names** — `"qwen2.5-1.5b-instruct-q4_k_m.gguf"` instead of `config.LLM_MODEL_FILENAME`

If it finds one of these, it outputs a warning message visible in your console. It does NOT block the edit — just warns you.

### What the Permissions Mean

`.claude/settings.json` also sets permissions for what Claude is allowed to do without asking:

**Auto-allowed** (no confirmation):
```
pytest tests/
python run_pipeline.py --dry-run
git status, git diff, git log, git add, git commit
pip install
```

**Blocked entirely:**
```
git push --force   ❌
git reset --hard   ❌
rm -rf             ❌
```

Claude cannot do these even if you accidentally ask.

---

## Part 2: The Pipeline Architecture

The pipeline was restructured in May 2026. The current order is:

```
Input transcript
    ↓
[Stage 1: Hallucination Filter]
    - Drops segments below HALLUCINATION_MIN_ALPHA_RATIO (0.50)
    - Removes non-Latin script
    - Removes segments that don't match the detected language family
    ↓
[Stage 1: Deduplicator]
    - Merges consecutive duplicate segments (DUPLICATE_SIMILARITY_THRESHOLD=95)
    - Collapses immediately repeated words ("Zaha Zaha dribbles" → "Zaha dribbles")
    ↓
[Stage 2A: Domain Normalizer]
    - Regex-based football compound merging ("off side" → "offside")
    - Disfluency removal ("uh", "um")
    - Language-aware per the FOOTBALL_COMPOUNDS dict
    ↓
[NER]
    - spaCy NER + POS-filtered heuristics
    - Language-aware via get_spacy_model(lang)
    ↓
[Stage E: Validated Entity Correction]   ← main correction stage
    - TF-IDF char-bigram retrieval over gazetteer (top-K=5)
    - Shortcut-accept on cosine ≥ 0.90 with gap ≥ 0.10
    - Qwen2.5-1.5B MCQ judge for cosine 0.40–0.89 band (multiple choice + D=keep)
    - xlm-roberta MLM veto on MCQ picks
    - Conservative validation gates (dictionary veto + C1 fuzzy floor + C2 length)
    - Validated cross-match cache promoted at 3-match consensus
    ↓
[Step L: Confidence-Gated GER (LLM)]
    - Qwen2.5-1.5B-Instruct GGUF via llama-cpp-python
    - Only rewrites tokens with logprob ≤ LLM_LOGPROB_GATE=-0.3
    - Same xlm-roberta MLM veto guards each proposed edit
    ↓
[Step P: Punctuation + Casing Restoration]
    - oliverguhr/fullstop-punctuation-multilang-large
    - PUNCT_PRESERVE_EXISTING=True: only INSERTS missing marks/casing
    ↓
[Temporal Chunker]
    - Rolling-window concatenation (12s window, 4s overlap)
    - Produces Elasticsearch-ready chunks
    ↓
Cleaned JSON + ES chunks
```

### Stage E Decision Order (the main correction logic)

For each NER-detected entity:

1. **Validated cross-match cache** — if mapping has ≥3 independent matches at fuzz ≥75 → apply
2. **TF-IDF retrieve** top-K=5 candidates
3. **Per-match cache** — reuse earlier in-match decisions
4. **Frequency heuristic** — reject if token appears ≥5× in the match
5. **Shortcut-reject** — cosine < 0.40 → skip
6. **Shortcut-accept** — cosine ≥ 0.90 AND gap ≥ 0.10 AND gates pass → apply
7. **MCQ pre-gates** — short token requires fuzz ≥ 85; otherwise fuzz ≥ 65
8. **MCQ judge** — Qwen picks A/B/C/D/E with match context + neighbouring segments
9. **MLM veto** — reject pick if `MLM(orig) / MLM(pick) ≥ 1.5`
10. **Validation gates** — dictionary veto, C1 fuzzy floor (≥60), C2 length tolerance (≤0.6)

### Key Configurable Constants

All in `pipeline/config.py` (Stage E retrieval constants live at the top of `pipeline/entity_corrector.py`). Never inline these in code.

| Constant | Value | Effect |
|---|---|---|
| `HALLUCINATION_MIN_ALPHA_RATIO` | 0.50 | Stage 1 alpha-ratio floor |
| `DUPLICATE_SIMILARITY_THRESHOLD` | 95 | Stage 1 dedup merge threshold |
| `SHORTCUT_ACCEPT_TFIDF` | 0.90 | Stage E auto-accept threshold |
| `SHORTCUT_REJECT_TFIDF` | 0.40 | Stage E hard reject below this |
| `MCQ_MIN_FUZZ_TO_INVOKE` | 65 | Stage E MCQ invocation floor |
| `MCQ_SHORT_TOKEN_MIN_FUZZ` | 85 | Stage E short-token override |
| `MLM_VETO_RATIO` | 1.5 | Stage E + Step L veto strictness |
| `CONSERVATIVE_C1_FUZZY_FLOOR` | 60 | Validation gate (fuzz floor) |
| `CONSERVATIVE_C2_LEN_TOLERANCE` | 0.6 | Validation gate (length tolerance) |
| `LLM_LOGPROB_GATE` | -0.3 | Step L: tokens above this are locked |
| `LLM_MIN_TOKENS_TO_INVOKE` | 2 | Step L: min low-conf tokens per segment |
| `VALIDATED_CACHE_MIN_CONSENSUS` | 3 | Cross-match promotion threshold |

**The most important rule:** Never inline these values in pipeline code. Always import from `pipeline.config` (or from the top of `pipeline.entity_corrector` for Stage E retrieval constants).

---

## Part 3: How to Work With Claude

### The 5-Step Loop

Follow this for every change, no exceptions.

```
1. /baseline West Ham        → save current state
2. /diagnose entity name     → understand the problem
3. make change               → Claude writes it
4. /regress                  → did anything break?
5. commit                    → only after step 4 passes
```

#### Step 1: Establish Baseline

**What you type:**
```
/baseline West Ham
```

**What Claude does:**
- Runs `pytest tests/ -v`
- Runs `python run_pipeline.py --match "West Ham" --dry-run`
- Captures: test count, per-stage counts (hallucinations removed, Stage E corrections, Step L edits, Step P restyles)
- Saves these numbers to compare against later

#### Step 2: Debug with Diagnose

**What you type:**
```
/diagnose stunning Sterling
```

**What Claude does:**
- Runs `python .claude/skills/diagnose/scripts/diagnose_entity.py "stunning" "Sterling"`
- Shows full_fuzz, reduced_word_fuzz, length, dictionary-veto status
- Shows MCQ pre-gate result + validation gate result
- Reports the verdict: PRE-MCQ REJECT / VALIDATION REJECT / MCQ-ELIGIBLE
- Notes: true end-to-end routing also depends on TF-IDF cosine over the match gazetteer

#### Step 3: Make Your Fix

You tell Claude what you want to do. The **NLP Pipeline Architect** agent takes over.

It will:
1. Check current code (`sequential-thinking` first)
2. Plan the exact change
3. Write a failing test first
4. Write the fix
5. Run tests
6. Run dry-run

You only need to say "yes, proceed" or "do step 3" to continue.

#### Step 4: Regress

**What you type:**
```
/regress
```

**What Claude does:**
- Runs `pytest tests/ -v` again
- Runs `python run_pipeline.py --match "West Ham" --dry-run` and a second match (default Chelsea)
- Compares per-stage counts to baseline
- Reports: which tests changed, which Stage E / Step L / Step P counts moved

#### Step 5: Commit

Once regress passes, commit using the standard format:
```
fix(entity_corrector): block stunning→Sterling via dictionary veto
feat(stage1): propagate detected language to Stage E
config(stage_e): MCQ_MIN_FUZZ_TO_INVOKE 65 → 70
test(entity_corrector): regression test for MCQ pre-gate
```

Format: `type(stage_or_module): description`

---

## Part 4: The Two Specialist Agents

### NLP Pipeline Architect

**When Claude invokes this agent:** Any request about pipeline code changes, refactoring, threshold/gate tuning, scaling to new languages, swapping models.

**What it does differently from regular Claude:**
- Uses `sequential-thinking` before any file change — forces a structured plan
- Uses `desktop-commander` to actually run `pytest` and dry-run commands
- Uses `context7` to look up exact library documentation before using API
- Uses `github` to check recent commits and open PRs

Priority libraries it consults via Context7:
`spacy`, `rapidfuzz`, `scikit-learn` (TfidfVectorizer char_wb), `llama-cpp-python`,
`transformers` (xlm-roberta + oliverguhr fullstop), `faster-whisper`, `langdetect`.

**Example:**
```
You: I want to add Norwegian support to the pipeline.

Agent: [sequential-thinking activates]
Phase 1: Current state
  - `ASR_MODELS` has "no": "KBLab/kb-whisper-large" (good — Norwegian piggybacks Swedish KB)
  - `SPACY_MODELS` falls back to xx_ent_wiki_sm for "no" (POS tagging unavailable)
  - `LANGUAGE_FAMILIES["sv"]` already includes "no" (Stage 1 won't filter it)
  - Pyspellchecker has no "no" dictionary → dictionary veto degrades to no-op
Phase 2: Impact
  - Stage E's dictionary veto won't fire for Norwegian common words
  - MCQ judge prompts are language-agnostic; should work
  - Step L + Step P models are multilingual; should work
Phase 3: Dynamic solution
  - Acquire a Norwegian spaCy model (`nb_core_news_sm`) and wire into SPACY_MODELS
  - Test pipeline against a Norwegian match and read Stage E telemetry
Phase 4: Exact changes
  - config.py: add "no": "nb_core_news_sm" to SPACY_MODELS
  - tests/test_multilingual.py: add Norwegian-language assertions
  - Baseline + regression on Norwegian match before/after

Want me to proceed with Step 1 (add the model)?
```

### Evaluation & Debugging Specialist

**When Claude invokes this agent:** Any request about wrong corrections, false positives, missed corrections, unexpected behavior.

**What it does differently:**
- Identifies WHICH stage made the bad edit (Stage 1 / 2A / NER / E / L / P)
- Calculates exact gate signals via `/diagnose` (no estimates)
- Reads the cleaned-output JSON's `corrections` (Stage E) and `sota_corrections` (Step L + P) to confirm
- Prioritises fixes: POS check → gate tightening → MCQ prompt example → never word lists

**Example:**
```
You: "fantastic" changed to "Sterling". Investigate.

Agent: Reading cleaned-output for the affected match…
       Found correction: {"original": "fantastic", "corrected": "Sterling", "method": "validated_cache"}

       Root cause: there's a stale entry in data/validated_corrections.json
       Validated cache requires 3-match consensus, so "fantastic"→"Sterling"
       must have been promoted previously after MCQ accepted it three times.

       Two fixes:
       1. Edit data/validated_corrections.json — remove the bad entry
       2. Add a `D=keep` example to _MCQ_SYSTEM_TEMPLATE to make Qwen reject
          "fantastic" cleanly in future matches

       Regression test: assert "fantastic" stays unchanged in test_entity_corrector.py
```

---

## Part 5: The Rules System Explained

### Why Rules Exist

Rules constrain Claude (no shortcuts) and remind you of the project's design philosophy.

Without rules, Claude might:
- Fix a false positive by adding a word to a static exclusion list (quick, but wrong)
- Hardcode a threshold in pipeline code (convenient, but untraceable)
- Skip writing a test (saves time now, causes bugs later)

### How Rules Work

Rules are Markdown files with YAML frontmatter listing the file paths they apply to. When you edit a matching file, the rule loads into Claude's context automatically.

You do not need to "activate" rules. They activate themselves.

### The Four Rules (post-rewrite)

#### Rule 00: Dev Workflow
**Activates for:** all `pipeline/*.py` and `tests/*.py` files

**Core directive:**
- Always run `pytest tests/` + `--dry-run` before AND after changes
- Every bug fix needs a test that fails before the fix and passes after
- When tuning thresholds: record before/after values in commit message

#### Rule 01: Global Scaling
**Activates for:** all `pipeline/*.py` files

**Core directive:**
- No static exclusion lists — use `get_rejected_pos_tags(language)`
- Never hardcode model names — use `get_spacy_model(lang)`, `get_asr_model(lang)`, `MLM_VETO_MODEL`, `LLM_MODEL_FILENAME`, etc.
- Language detection from Stage 1 must propagate to NER, Stage E, Step L, Step P
- Multilingual-by-default models (Qwen, xlm-roberta, oliverguhr fullstop) stay multilingual

#### Rule 02: Stage E (filename `02-tier2-fuzzy-logic.md` — kept for backward path compatibility)
**Activates for:** `entity_corrector.py`, `fuzzy_corrector.py`, `config.py`, `ner_extractor.py`, `gazetteer.py`

**Core directives:**
- Strip entity boundaries via `extract_entity_core` / `extract_and_rebuild_entity`
- Type-aware: `entity_types` flags player/team/referee — MCQ uses this for context
- Validation gates are language-conditional (`passes_conservative_gates(orig, corr, language)`)
- Tune via MCQ / shortcut / validation gate constants, not legacy weights

#### Rule 03: Step L + Step P (filename `03-tier3-context-ai.md` — kept for backward path compatibility)
**Activates for:** `llm_corrector.py`, `punct_restorer.py`

**Core directives:**
- Confidence gate (`LLM_LOGPROB_GATE`) decides which tokens the LLM may edit
- MLM veto (`MLM_VETO_RATIO`) is the safety net for every LLM proposal
- Step P only INSERTS punctuation (`PUNCT_PRESERVE_EXISTING=True`)
- Model identifiers via config only; never hardcode paths

---

## Part 6: Common Scenarios

### Scenario A: Tightening a Stage E Gate

**Starting point:** A false positive landed in Stage E telemetry as `method=mcq_judge` and you suspect `MCQ_MIN_FUZZ_TO_INVOKE` is too loose.

**Conversation:**
```
You: I think MCQ_MIN_FUZZ_TO_INVOKE should be 70 instead of 65.
     Run /baseline West Ham before I change it.

Claude: Running baseline...
        Stage E: 91 corrections (84 tfidf_shortcut, 5 mcq_judge, 2 validated_cache)
        Step L: 14 segment edits. Step P: 412 segments restyled.
        Baseline saved.

You: Change MCQ_MIN_FUZZ_TO_INVOKE to 70 and /regress check.

Claude: Editing config.py...
        Running tests... pass
        Running dry-run on West Ham...
        Stage E: 89 corrections (84 tfidf_shortcut, 3 mcq_judge, 2 validated_cache)
        Running dry-run on Chelsea...
        Stage E: 116 corrections (98 tfidf_shortcut, 9 mcq_judge, 9 validated_cache)

        Comparison:
        - West Ham: -2 MCQ corrections; spot-checked, both were legitimate FPs
        - Chelsea: -3 MCQ corrections; one was a real fix (Hendrik→Henderson at fuzz 67)
        Net: 5 fewer accepts, 1 was a real win lost.
        Should I revert or keep?

You: Keep it. Add an MCQ-prompt example for Hendrik→Henderson so we still get that one.
```

### Scenario B: Investigating a False Positive

**Starting point:** "midfield" became "Midfielders" in a segment.

**Conversation:**
```
You: "midfield" was changed to "Midfielders". That's wrong. Investigate.

Claude: [Evaluation & Debugging Specialist activates]
        Reading cleaned-output for the affected match…
        Correction record:
          {"original": "midfield", "corrected": "Midfielders",
           "method": "tfidf_shortcut", "score": 92.0, "stage": "3"}

        That's Stage E shortcut-accept. Let me re-check the gates.
        Running /diagnose midfield Midfielders…
        full_fuzz=82, reduced_word_fuzz=82, validation_gates_pass=False
        Dictionary veto fired (midfield ≥4 chars, real English word).

        But the cleaned output shows it was applied. That means the cache
        wrote it BEFORE the gate logic was tightened. Check the validated cache:
        data/validated_corrections.json has "midfield": {"correct": "Midfielders",
        "matches_seen": [m1, m2, m3]}. The promotion happened before
        the dictionary veto landed.

        Two fixes:
        1. Remove the cache entry
        2. Add a regression test: midfield must stay unchanged after dictionary-veto pass

        Want me to proceed?
```

### Scenario C: Adding a New Language

**Starting point:** Bundesliga commentary is mishandled.

**Conversation:**
```
You: How do we improve German handling?

Claude: [NLP Pipeline Architect activates]
        Current state for "de":
          - ASR_MODELS["de"] = "primeline/whisper-large-v3-turbo-german" ✅
          - SPACY_MODELS["de"] = "de_core_news_sm" ✅ (has POS)
          - LANGUAGE_FAMILIES["de"] = {"de","nl","lb"} — confirm langdetect's "de" matches
          - dictionary veto: pyspellchecker has "de" → ✅
          - MCQ judge prompt is language-agnostic ✅

        Likely real gap: the MCQ system prompt's few-shot examples are
        all English. Qwen 1.5B may misinterpret D=keep for German common words
        because the examples don't show that case in German.

        Plan: add 1-2 German few-shot examples to _MCQ_SYSTEM_TEMPLATE,
        baseline a Bundesliga match, compare Stage E corrections.

        Want me to draft those examples?
```

---

## Part 7: Technical Debt Guide

These are known limitations. Work through them in order of effort.

### Debt 1: Hardcoded Stage E retrieval constants location

**File:** `pipeline/entity_corrector.py` (top of file)
**Problem:** `SHORTCUT_ACCEPT_TFIDF`, `SHORTCUT_REJECT_TFIDF`, `TOP_K_CANDIDATES`,
`TFIDF_NGRAM_RANGE` live in the module instead of `pipeline/config.py`.
**Fix:** Move them to `pipeline/config.py` so all knobs are in one file.
**Effort:** Low (1 hour with agent help; just relocate + update one import).

### Debt 2: Norwegian spaCy model

**File:** `pipeline/config.py` → `SPACY_MODELS`
**Problem:** No `nb_core_news_sm` entry; "no" falls back to `xx_ent_wiki_sm`
which lacks POS tagging, so `REJECTED_POS_TAGS_NON_ENGLISH` filtering degrades.
**Fix:** Add `"no": "nb_core_news_sm"` to `SPACY_MODELS`. Confirm spaCy install.
**Effort:** Low.

### Debt 3: Pyspellchecker coverage for non-English dictionary veto

**File:** `pipeline/fuzzy_corrector.py` (`passes_conservative_gates`)
**Problem:** Dictionary veto degrades to no-op for languages without a
pyspellchecker dictionary (e.g. Norwegian).
**Fix:** Add a per-language wordfreq fallback or use language-specific
dictionaries via `pyspellchecker.SpellChecker(language=lang)`.
**Effort:** Medium.

### Debt 4: MCQ prompt is English-language-trained

**File:** `pipeline/entity_corrector.py` (`_MCQ_SYSTEM_TEMPLATE`)
**Problem:** All few-shot D=keep examples are English. Qwen 1.5B may
under-use D=keep for German/Swedish/Spanish common words.
**Fix:** Add 1-2 non-English `D=keep` examples or template the few-shots per
detected language.
**Effort:** Medium (requires per-language evaluation to verify).

### Debt 5: Validated cache has no per-language partitioning

**File:** `pipeline/entity_corrector.py` + `data/validated_corrections.json`
**Problem:** Cache keys are `entity_lower` only. A token like "kommer" could
collide across Swedish and German if any of those leagues are introduced.
**Fix:** Key by `(language, entity_lower)`.
**Effort:** Medium.

### Debt 6: Step L prompt context size

**File:** `pipeline/config.py` (`LLM_CTX_WINDOW`)
**Problem:** Qwen2.5-1.5B has 4k context but we use 2048 to stay safe with
the typed gazetteer + ±3 segments. Larger context could let us add lineup
substitutions.
**Fix:** Bump `LLM_CTX_WINDOW` and benchmark llama.cpp latency.
**Effort:** Medium.

---

## Part 8: Quick Reference

### Commands You Can Type in Chat

| Command | Example | What It Does |
|---|---|---|
| `/baseline` | `/baseline West Ham` | Save current test + per-stage counts |
| `/diagnose` | `/diagnose stunning Sterling` | Show Stage E gate signals for a pair |
| `/regress` | `/regress check Arsenal` | Run tests + dry-run on 2 matches, compare |
| `/test-match` | `/test-match Chelsea` | Test pipeline on one match (exploratory) |
| `/tune-threshold` | `/tune-threshold MCQ_MIN_FUZZ_TO_INVOKE to 70` | Before/after protocol around a config change |
| `/evaluate` | `/evaluate Chelsea` | WER + Entity-F1 against ground truth |
| `/fix-bug` | `/fix-bug "stunning" was changed to "Sterling"` | Test-first bug fix workflow |

### Git Commit Format

```
fix(entity_corrector): block stunning→Sterling via dictionary veto
fix(llm_corrector): tighten MLM_VETO_RATIO to 1.8 after over-edit audit
feat(stage1): propagate detected language to Stage E
config(stage_e): MCQ_SHORT_TOKEN_MIN_FUZZ 85 → 88
test(entity_corrector): regression test for MCQ pre-gate
docs(rules): rewrite tier2 rule for Stage E architecture
```

### When to Use Which Agent

| Situation | Agent |
|---|---|
| "I want to change `entity_corrector.py`" | NLP Pipeline Architect |
| "I want to tune a gate or threshold" | NLP Pipeline Architect |
| "I want to add Norwegian support" | NLP Pipeline Architect |
| `"correction X → Y looks wrong"` | Evaluation & Debugging Specialist |
| `"why didn't it correct X to Y?"` | Evaluation & Debugging Specialist |
| `"Step L is over-editing function words"` | Evaluation & Debugging Specialist |

### Test File Map

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

---

## Part 9: Troubleshooting

### "Claude is suggesting a static word list"
Claude is violating Rule 01. Remind it explicitly:
```
Do not add to any static exclusion list.
Use get_rejected_pos_tags(language) for POS-based filtering.
See .claude/rules/01-global-scaling.md.
```

### "A false positive persists after my fix"
The fix may not have covered all code paths. Ask:
```
Re-run /diagnose entity candidate.
Trace which Stage E gate let it through.
Check data/validated_corrections.json for a stale entry.
```

### "Tests pass but Stage E correction count dropped a lot"
A big drop in corrections means you may have over-restricted something. Ask:
```
Which Stage E corrections were lost vs baseline?
Group them by method (tfidf_shortcut vs mcq_judge vs validated_cache).
Are any legitimate fixes that I want to keep?
```

### "Step L is over-correcting"
The LLM may be touching tokens that should be locked. Ask:
```
Run with LLM_CORRECTION_ENABLED=False — does the bad edit disappear?
If yes, lower LLM_LOGPROB_GATE so fewer tokens are exposed,
or raise MLM_VETO_RATIO so the veto is more aggressive.
```

### "The hook warning is appearing incorrectly"
The `validate-pipeline-edit.sh` hook occasionally triggers on commented-out code or test files. If it's annoying:
```
The hook fired on line X. Read .claude/hooks/validate-pipeline-edit.sh
and explain why it triggered. Should we adjust the pattern?
```

### "I'm not sure what a rule is telling me"
Ask Claude to read and explain any rule:
```
Read .claude/rules/02-tier2-fuzzy-logic.md
Explain Stage E's MCQ pre-gate logic in plain English with an example.
```

### "I accidentally committed something wrong"
Don't panic. Ask:
```
I need to undo my last commit without losing my changes.
What is the safest way to do this?
```
Claude will suggest `git revert` (safe) or explain what options exist.

---

## Appendix: Full .claude/ Directory Explained

```
.claude/
│
│  # Main context file — always loaded
├── CLAUDE.md
│     Contains: architecture, commands, thresholds, test file map, commit format
│     Loaded: automatically at every session start
│
│  # Project-wide settings
├── settings.json
│     Contains: permitted commands, hooks configuration
│     Loaded: automatically, enforced by Claude Code runtime
│
│  # Specialist agents
├── agents/
│   ├── 01-nlp-architect.agent.md
│   │     Role: Lead NLP engineer for pipeline changes
│   │     Model: Opus (most capable)
│   │     Tools: sequential-thinking, desktop-commander, context7, github
│   │     Invoked: when editing pipeline code or planning architecture
│   │
│   └── 02-eval-debugger.agent.md
│         Role: Forensic debugger for corrections
│         Model: Opus
│         Tools: sequential-thinking, desktop-commander
│         Invoked: when investigating wrong corrections
│
│  # Path-activated rules (filenames historical; content reflects current arch)
├── rules/
│   ├── 00-dev-workflow.md        activates for: pipeline/**/*.py and tests/**/*.py
│   ├── 01-global-scaling.md      activates for: pipeline/**/*.py
│   ├── 02-tier2-fuzzy-logic.md   activates for: entity_corrector.py, fuzzy_corrector.py, config.py, ner_extractor.py, gazetteer.py
│   └── 03-tier3-context-ai.md    activates for: llm_corrector.py, punct_restorer.py
│
│  # Slash commands (the real implementations)
├── skills/
│   ├── baseline/SKILL.md
│   ├── diagnose/SKILL.md
│   │   └── scripts/diagnose_entity.py   ← core diagnostic tool
│   ├── evaluate/SKILL.md
│   ├── fix-bug/SKILL.md
│   ├── regress/SKILL.md
│   ├── test-match/SKILL.md
│   └── tune-threshold/SKILL.md
│
│  # Slash-command shims (deprecation stubs)
├── commands/
│   ├── baseline.md       → /baseline
│   ├── diagnose.md       → /diagnose
│   ├── fix-bug.md        → /fix-bug
│   ├── regress.md        → /regress
│   ├── test-match.md     → /test-match
│   └── tune-threshold.md → /tune-threshold
│
│  # Automatic hooks
└── hooks/
    ├── session-context.sh          fires: UserPromptSubmit (once per session)
    └── validate-pipeline-edit.sh   fires: PostToolUse on Edit or Write
```

---

End of manual. For a quicker reference, see `CHEATSHEET.md` in this folder.
