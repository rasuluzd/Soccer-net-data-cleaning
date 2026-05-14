# Claude Code Cheat Sheet — SoccerNet Pipeline

> Written for a junior engineer using Claude via GitHub Copilot / Claude Agent SDK.
> The `.claude/` configuration loads automatically when you open the project.

---

## How This Works (Read This First)

You are using Claude models through GitHub Copilot / the Claude Agent SDK.
When you open this project, Claude **automatically** reads:

- `.claude/CLAUDE.md` — project architecture and rules
- `.claude/rules/*.md` — rules that activate based on which file you are editing
- `.claude/agents/*.agent.md` — two specialist agents ready to help
- `.claude/settings.json` — permissions and hooks

**You do not run any setup command.** Just open the project and start talking to Claude.

---

## The 4 Workflow Commands

These are **not terminal commands**. You type them in the chat to Claude:

| Command | What it does |
|---|---|
| `/baseline` | Run tests and capture current correction counts before you change anything |
| `/diagnose entity candidate` | Show whether a (entity → candidate) pair would pass Stage E gates |
| `/regress` | Run tests again and compare to your baseline |
| `/test-match MatchName` | Test the pipeline on a specific match with detailed per-stage output |

**Example:**
```
You:    /baseline West Ham
Claude: Running pytest... all 89 tests pass.
        Running dry-run on West Ham...
        Stage E: 91 entity corrections. Step L: 14 segment edits. Step P: 412 segments restyled.
        Baseline saved.
```

---

## The 5-Step Bug Fix Loop

Every single bug fix follows this exact order. Do not skip steps.

```
Step 1 → /baseline West Ham        (save current state)
Step 2 → /diagnose "word" "Player" (understand the problem)
Step 3 → make your fix             (Claude helps write it)
Step 4 → /regress                  (did you break anything?)
Step 5 → commit                    (only now)
```

**Example conversation:**

```
You:    The pipeline changed "stunning" to "Sterling". That's wrong.
Claude: Let me investigate. Running /diagnose stunning Sterling...
        full_fuzz: 62, reduced_word_fuzz: 62, c1_passes: True (60+)
        validation_gates_pass: False (dictionary veto: "stunning" is a real word)
        Verdict: PRE-MCQ REJECT (reduced fuzz 62 < 65)
        That correction should not have happened. Stage E telemetry next.
        ...
        Want me to write the fix and a failing test?

You:    Yes, write the fix.
Claude: [NLP Pipeline Architect agent activates]
        Step 1: Write failing test...
        Step 2: Tighten the relevant Stage E gate in config.py...
        Step 3: Run tests... all pass.
        Step 4: Dry-run on West Ham... no false positive (was 1, now 0).
        Ready to commit. Suggested message: fix(entity_corrector): raise MCQ_MIN_FUZZ_TO_INVOKE to block stunning→Sterling
```

---

## The 3 Rules You Must Never Break

### Rule 1 — No Static Word Lists
```python
# NEVER DO THIS
COMMON_WORDS_EXCLUDE.add("stunning")

# ALWAYS DO THIS
if token.pos_ in get_rejected_pos_tags(language):
    return None  # works for English AND German AND Spanish AND Swedish
```
The hook warns you automatically if you do this wrong.

### Rule 2 — Constants Only in Config
```python
# NEVER DO THIS
if score >= 0.90:

# ALWAYS DO THIS
if score >= SHORTCUT_ACCEPT_TFIDF:   # imported from entity_corrector top constants
# or from pipeline.config import the cross-pipeline constants
```
All constants live in `pipeline/config.py` (or the top of `entity_corrector.py`
for Stage E-only constants like `SHORTCUT_ACCEPT_TFIDF` / `SHORTCUT_REJECT_TFIDF` /
`TOP_K_CANDIDATES`). The hook warns you automatically if you inline values.

### Rule 3 — Always Test Before Committing
```
Tests pass before change → Tests pass after change → Commit
```
If you skip this, you won't know what you broke.

---

## The 2 Specialist Agents

Claude automatically switches to these when relevant. You don't invoke them.

### NLP Pipeline Architect
**Activates when you ask about:** any code change, threshold/gate tuning,
refactoring, swapping models, scaling to a new language.

What it does for you:
- Plans work step by step
- Writes failing tests first
- Verifies each change with pytest and dry-run
- Suggests commit messages
- Uses Context7 to fetch current docs for spacy / rapidfuzz / scikit-learn /
  llama-cpp-python / transformers / faster-whisper

### Evaluation & Debugging Specialist
**Activates when you ask about:** false positives, missed corrections,
unexpected output, "why did it change X to Y".

What it does for you:
- Traces a correction through the right stage (1 / 2A / NER / E / L / P)
- Calculates exact gate signals via `/diagnose` (no guessing)
- Suggests the safest fix in priority order
- Writes a regression test to prevent recurrence

---

## The 2 Automatic Hooks

Hooks run silently in the background. You never trigger them manually.

### Hook 1 — Validate Pipeline Edits (PostToolUse)
Runs every time Claude edits a `pipeline/*.py` file.
Checks for: hardcoded thresholds, static word lists, hardcoded model names.
Shows a warning in the console. Does not block.

### Hook 2 — Session Context (UserPromptSubmit)
Runs once at the start of every session.
Prints: "Pipeline project context: 3-tier ASR cleaning. Run pytest + --dry-run..."
Reminds Claude (and you) of the core rules.

(Yes, the session-context hook still says "3-tier" — that's a relic of the old
naming. The current pipeline is multi-stage; CLAUDE.md has the accurate map.)

---

## The Auto-Rules System

Rules load **automatically based on which file you are editing**.

| You are editing... | Rule that loads |
|---|---|
| Any `pipeline/*.py` | `01-global-scaling.md` — language-agnostic reminders |
| `entity_corrector.py`, `fuzzy_corrector.py`, `config.py`, `ner_extractor.py`, `gazetteer.py` | `02-tier2-fuzzy-logic.md` — Stage E (TF-IDF + MCQ) rules |
| `llm_corrector.py`, `punct_restorer.py` | `03-tier3-context-ai.md` — Step L / Step P rules |
| Any test or pipeline file | `00-dev-workflow.md` — always test before committing |

(File names are historical — the content inside reflects the current architecture.)

You do not need to ask Claude to read these. They are already in context.

---

## File Map

```
.claude/
├── CLAUDE.md                      ← Always loaded. Architecture + test map.
├── settings.json                  ← Permissions + hooks config.
├── rules/ (auto-loaded by path)
│   ├── 00-dev-workflow.md         ← "run pytest + dry-run"
│   ├── 01-global-scaling.md       ← "no static word lists; multilingual models"
│   ├── 02-tier2-fuzzy-logic.md    ← Stage E (TF-IDF + Qwen MCQ + MLM veto)
│   └── 03-tier3-context-ai.md     ← Step L (LLM GER) + Step P (punctuation)
├── agents/ (auto-invoked)
│   ├── 01-nlp-architect.agent.md  ← pipeline changes
│   └── 02-eval-debugger.agent.md  ← false positive debugging
├── commands/ (deprecated stubs → use skills instead)
└── skills/ (slash commands)
    ├── baseline/SKILL.md
    ├── diagnose/SKILL.md (+ scripts/diagnose_entity.py)
    ├── evaluate/SKILL.md
    ├── fix-bug/SKILL.md
    ├── regress/SKILL.md
    ├── test-match/SKILL.md
    └── tune-threshold/SKILL.md
```

---

## Key Thresholds to Know

All in `pipeline/config.py`, except Stage E retrieval constants which are at
the top of `pipeline/entity_corrector.py`. Never inline these in code.

### Stage 1
| Constant | Value | Meaning |
|---|---|---|
| `HALLUCINATION_MIN_ALPHA_RATIO` | 0.50 | Below this fraction of ASCII letters → drop segment |
| `DUPLICATE_SIMILARITY_THRESHOLD` | 95 | Merge consecutive segments above this fuzz |

### Stage E (Entity Correction)
| Constant | Value | Meaning |
|---|---|---|
| `SHORTCUT_ACCEPT_TFIDF` | 0.90 | TF-IDF cosine ≥ this + gap → auto-accept |
| `SHORTCUT_ACCEPT_GAP` | 0.10 | Required margin over second-best |
| `SHORTCUT_REJECT_TFIDF` | 0.40 | Below this → no MCQ call |
| `MCQ_MIN_TOKEN_LEN` | 5 | Below → require strong fuzz to invoke MCQ |
| `MCQ_MIN_FUZZ_TO_INVOKE` | 65 | Reduced-word fuzz floor for MCQ |
| `MCQ_SHORT_TOKEN_MIN_FUZZ` | 85 | Bypass `MCQ_MIN_TOKEN_LEN` when fuzz this high |
| `MLM_VETO_RATIO` | 1.5 | xlm-r vetos MCQ pick if `lp(orig)/lp(pick) ≥ this` |
| `CONSERVATIVE_C1_FUZZY_FLOOR` | 60 | Min `fuzz(orig, corr)` to accept |
| `CONSERVATIVE_C2_LEN_TOLERANCE` | 0.6 | `|Δlen| ≤ max(2, 0.6·len(orig))` |
| `DICTIONARY_VETO_MIN_LEN` | 4 | Veto common-word originals at this length |
| `FREQUENCY_HEURISTIC_THRESHOLD` | 5 | Reject if token appears this many times in match |
| `VALIDATED_CACHE_MIN_CONSENSUS` | 3 | Matches required to promote into cross-match cache |
| `VALIDATED_CACHE_MIN_FUZZY` | 75 | Min fuzz to write cache entry |

### Step L (LLM GER) + Step P (Punctuation)
| Constant | Value | Meaning |
|---|---|---|
| `LLM_LOGPROB_GATE` | -0.3 | Tokens above this are LOCKED (not editable) |
| `LLM_MIN_TOKENS_TO_INVOKE` | 2 | Min low-conf tokens before LLM is called |
| `LLM_CTX_PREVIOUS_SEGMENTS` | 2 | Previous-segment context in prompt |
| `LLM_CTX_NEXT_SEGMENTS` | 1 | Next-segment context in prompt |
| `MLM_VETO_RATIO` | 1.5 | Step L veto (same xlm-r model as Stage E) |
| `PUNCT_PRESERVE_EXISTING` | True | Step P only INSERTS punctuation; never deletes |

### Legacy (in config but no longer read by production code)
These remain only for tests and ablation:
`FUZZY_WEIGHT=0.65`, `PHONETIC_WEIGHT=0.20`, `CONTEXT_WEIGHT=0.15`,
`TIER2_ACCEPT_THRESHOLD=72`, `TIER3_VALIDATION_THRESHOLD=0.55`, `MIN_GAP=0.08`.

---

## Common Scenarios

### "I found a false positive"
```
You: "outstanding" was changed to "Sterling". That's wrong.
     Investigate and fix it.
```
→ Evaluation & Debugging Specialist activates
→ Identifies which stage (likely Stage E)
→ Runs `/diagnose` to confirm gate behaviour
→ Proposes safest fix + writes regression test

### "I want to tighten a gate"
```
You: I think MCQ_MIN_FUZZ_TO_INVOKE should be 70 instead of 65.
     Let's test this. Run baseline first.
```
→ NLP Architect activates
→ Baseline before, change, baseline after, on two matches
→ Reports Stage E correction counts gained/lost
→ Suggests commit message

### "I want to add Norwegian support"
```
You: Help me wire Norwegian into the language-conditional models.
     Plan this step by step.
```
→ NLP Architect activates
→ Checks `SPACY_MODELS`, `ASR_MODELS`, `LANGUAGE_FAMILIES`, dictionary veto coverage
→ Plans test cases against a Norwegian match
→ Writes a baseline run + the per-config change

---

## If Something Goes Wrong

| Problem | What to do |
|---|---|
| Tests fail | Ask: "Debug this test failure" + paste the error |
| False positive you can't explain | Ask: "Investigate false positive: X → Y in <match>" |
| Regression after change | Ask: "Why did Stage E correction count drop?" |
| Step L over-edits | Try `LLM_CORRECTION_ENABLED=False` to confirm; raise `MLM_VETO_RATIO` |
| Unsure about a gate | Ask: "What does MCQ_MIN_FUZZ_TO_INVOKE control?" |

---

## Commit Message Format

```
fix(entity_corrector): raise MCQ_MIN_FUZZ_TO_INVOKE to block stunning→Sterling
fix(llm_corrector): tighten MLM_VETO_RATIO to 1.8 after over-edit audit
feat(stage1): propagate detected language to entity_corrector
config(stage_e): MCQ_SHORT_TOKEN_MIN_FUZZ 85 → 88
```

Format: `type(stage_or_module): short description`

Types: `fix`, `feat`, `refactor`, `config`, `tune`, `test`, `docs`

---

## Your First 15 Minutes

1. **Open the project** in your Copilot/Claude environment
2. **Establish baseline** — type in chat:
   ```
   /baseline West Ham
   ```
3. **Explore the pipeline** — type in chat:
   ```
   Walk me through what each stage of the pipeline does on the West Ham match.
   ```
4. **Investigate a specific case** — type in chat:
   ```
   /diagnose stunning Sterling
   ```
5. **Watch an agent work** — type in chat:
   ```
   Plan a refactor to add a new language (Norwegian) to the pipeline.
   ```

You are ready. For the full manual, open `CLAUDE_CODE_GUIDE.md`.
