---
paths:
  - "pipeline/**/*.py"
---

# Global Language Scaling

Every change must work for non-English leagues. Supported families today:
English (EPL, Premier League), Swedish/Norwegian/Danish (Allsvenskan, Eliteserien,
Superliga), German (Bundesliga), French (Ligue 1), Spanish (La Liga), Italian
(Serie A), Portuguese, Dutch.

## Directives

1. **No static exclusion lists.** Use `token.pos_` checks (`get_rejected_pos_tags(lang)`), not word lists. English drops NOUN/ADJ/VERB; non-English drops only the unambiguous grammatical categories
2. **Never hardcode model names.** Always reference `pipeline.config.get_spacy_model(lang)`, `get_context_model(lang)`, `get_asr_model(lang)`, `MLM_VETO_MODEL`, `PUNCT_MODEL`, `LLM_MODEL_FILENAME`
3. **Language-informed processing.** The detected language from `hallucination_filter.detect_commentary_language()` propagates to NER (`get_spacy_model`, `get_entity_labels`), Stage E gates (dictionary veto language), Step L prompt, Step P model, and ASR re-transcription
4. **Multilingual-by-default models.** The MCQ judge (Qwen-1.5B), MLM veto (xlm-roberta-base), and punctuation restorer (oliverguhr fullstop multilang) are all multilingual — do not swap them for English-only equivalents
5. **Validate dictionary veto per language.** `DICTIONARY_VETO_MIN_LEN=4` and the pyspellchecker dictionary must be available for the target language; degrade gracefully when not
6. **Validated cache is language-tagged** by being keyed on `(match_id, entity_lower)` — cross-language collisions are prevented by gazetteer scope

## Good vs Bad Patterns

```python
# GOOD: dynamic POS rejection, language-aware
if token.pos_ in get_rejected_pos_tags(language):
    return None

# BAD: static word list
if word.lower() in COMMON_WORDS_EXCLUDE:
    return None
```

```python
# GOOD: language-aware ASR model selection
model = get_asr_model(detected_language)

# BAD: hardcoded English-only model
model = "Systran/faster-whisper-large-v3"
```

```python
# GOOD: language-conditional validation gates
if not passes_conservative_gates(orig, corr, language=lang):
    return None

# BAD: English-assumed dictionary veto
import pyspellchecker
spell = pyspellchecker.SpellChecker(language="en")
```
