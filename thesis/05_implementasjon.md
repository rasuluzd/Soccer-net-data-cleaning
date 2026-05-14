# Kapittel 5 — Implementasjon

## 5.1 Tech stack

| Komponent | Bibliotek | Versjon | Bruk |
|---|---|---|---|
| ASR | `faster-whisper` | 1.0+ | Transkribering med beam_size=5, word_timestamps=True |
| LLM | `llama-cpp-python` | 0.3+ | Qwen2.5-1.5B-Instruct (q4_k_m GGUF) |
| MLM | `transformers` (HF) | 4.40+ | XLM-RoBERTa-base for veto |
| NER + POS | `spaCy` | 3.7+ | `en_core_web_sm`, `xx_ent_wiki_sm` |
| Retrieval | `scikit-learn` | 1.4+ | TfidfVectorizer (char_wb, ngram (2,4)) |
| Fuzzy match | `rapidfuzz` | 3.0+ | Levenshtein + Jaro-Winkler |
| Phonetic | `metaphone`, `pyphonetics` | — | Engelsk og accent-stripped Soundex |
| Eval | `jiwer` | 3.0+ | WER med dynamic programming alignment |

Alle modeller kjører på CPU. Ingen kommersielle API-kall.

## 5.2 Modulkart

```
pipeline/
├── config.py                  # Alle terskler, modellnavn, språkmappinger
├── orchestrator.py            # Coordinator — kobler trinnene sammen
├── loader.py                  # Segment-dataclass, JSON I/O
├── whisper_runner.py          # faster-whisper wrapper
├── hallucination_filter.py    # Tier 1: alpha-ratio + langdetect
├── deduplicator.py            # Tier 1: konsolider repeterte segmenter
├── domain_normalizer.py       # Stage 2A: regelbasert normalisering
├── gazetteer.py               # Bygger navnevariant-dict fra Labels-caption.json
├── ner_extractor.py           # spaCy NER + POS-heuristikker
├── entity_corrector.py        # Tier 2: TF-IDF + MCQ + MLM-veto (hovedbidrag)
├── llm_corrector.py           # Step L: Konfidensgated GER med Qwen
├── fuzzy_corrector.py         # Hjelpefunksjoner (split, normaliser, gates)
├── punct_restorer.py          # (planlagt) Tegnsetting + casing restoration
└── report.py                  # Rapport-generator for kjøremetadata
```

Tester ligger i `tests/`, én testfil per modul. Total testdekning: 100+ tester.

## 5.3 Konfigurasjon

`pipeline/config.py` er **eneste sted** der numeriske terskler defineres. Eksempler:

```python
# Tier 2 vekter (engelsk)
FUZZY_WEIGHT = 0.65
PHONETIC_WEIGHT = 0.20
CONTEXT_WEIGHT = 0.15

# entity_corrector terskler
SHORTCUT_ACCEPT_TFIDF = 0.90
SHORTCUT_ACCEPT_GAP = 0.10
SHORTCUT_REJECT_TFIDF = 0.40
TOP_K_CANDIDATES = 5
MCQ_OPTIONS_SHOWN = 3

# MCQ-eligibility
MCQ_MIN_TOKEN_LEN = 5
MCQ_MIN_FUZZ_TO_INVOKE = 65
MCQ_SELF_CONSISTENCY_SAMPLES = 3

# MLM-veto
MLM_VETO_MODEL = "xlm-roberta-base"
MLM_VETO_ON_MCQ_ENABLED = True

# Validated cache
VALIDATED_CACHE_PATH = "data/validated_corrections.json"
VALIDATED_CACHE_MIN_CONSENSUS = 3
VALIDATED_CACHE_MIN_FUZZY = 75

# Step L (konfidensgated GER)
LLM_LOGPROB_GATE = -0.3
LLM_TEMPERATURE = 0.0
LLM_CTX_WINDOW = 2048
```

Denne disiplinen gjør ablation-studier trivielle: én verdi endres, hele kjøringen reproduserer.

## 5.4 Testing

Hver pipeline-modul har en tilsvarende testfil:

| Modul | Testfil | Antall tester |
|---|---|---|
| `entity_corrector.py` | `test_entity_corrector.py` | 24 |
| `llm_corrector.py` | `test_llm_corrector.py` | 18 |
| `hallucination_filter.py` | `test_hallucination_filter.py` | 12 |
| `deduplicator.py` | `test_deduplicator.py` | 8 |
| `domain_normalizer.py` | `test_domain_normalizer.py` | 14 |
| `gazetteer.py` | `test_gazetteer.py` | 10 |
| `ner_extractor.py` | `test_ner_extractor.py` | 9 |
| `loader.py` (multilingual) | `test_multilingual.py` | 16 |
| `tools/evaluate_wer.py` | `test_evaluate_wer_alignment.py` | 14 |

Tester bruker monkey-patching for å unngå å laste tunge modeller (Qwen, XLM-RoBERTa) i CI. Faktisk LLM-kall valideres i én manuell smoke-test.

Kommando:

```bash
pytest tests/ -v
```

Kjøretid på CPU (uten LLM-laster): ~12 sekunder.

## 5.5 Egne valg vs. bibliotek

For å klargjøre hva som er faktisk **eget bidrag** versus gjenbruk av eksisterende verktøy:

### Egen kode

- `entity_corrector.py` — hele arkitekturen (TF-IDF + MCQ + MLM-veto + valideringscache).
- `orchestrator.py` — pipeline-koordinatoren med språkdeteksjon-betinget modellvalg.
- `gazetteer.py` — kampspesifikk navnevariantgenerator fra `Labels-caption.json`.
- `domain_normalizer.py` — alle regulære uttrykk for tall- og tidsuttrykk.
- `hallucination_filter.py` — alpha-ratio + langdetect-kombinasjonen.
- To-lags valideringscache med consensus-regelen — designvalg som ikke er publisert i denne formen.
- Self-consistency MCQ over Qwen — adaptasjon av DeRAGEC til vår kontekst.
- Frozen word indices mellom Tier 2 og Step L — designvalg for å unngå dobbel-touch.
- `tools/evaluate_wer.py` med både legacy og windowed alignment.
- Hele test-suiten.

### Gjenbruk (med kilde-attribusjon)

- `faster-whisper` for ASR.
- `llama-cpp-python` + Qwen2.5-1.5B GGUF for LLM.
- HuggingFace `transformers` for XLM-RoBERTa.
- `spaCy` for NER og POS.
- `sklearn.TfidfVectorizer` for retrieval.
- `rapidfuzz` for fuzzy matching.
- `jiwer` for WER-beregning.
- Algoritmiske mønstre fra Apple RAG-NEC, DeRAGEC, Confidence-Guided Error Correction (Sun et al. 2024; ACL 2025; Zhang et al. 2025).

### Retningslinje for hva som er "eget bidrag"

Oppgaven hevder ikke å ha oppfunnet TF-IDF, MCQ-prompting eller MLM-pseudo-likelihood. Bidraget er **kombinasjonen og koreografien**: hvilke moduler som gjør hvilke avgjørelser, hvilke som har vetorett, og hvordan systemet lærer over tid uten å arve sine egne feil. Ablation-tabellen i kapittel 6 viser at hver enkelt komponent bidrar til sluttresultatet.

## 5.6 Kjøreprofil

På en moderne bærbar (8-core CPU, 16 GB RAM, ingen GPU):

| Trinn | Tid per halvkamp |
|---|---|
| Whisper transcribe (medium) | 8–12 min |
| Tier 1 + Stage 2A | < 5 sek |
| Tier 2 entity_corrector | 2–4 min (LLM-dominert) |
| Step L konfidensgated GER | 1–2 min |
| **Total** | **~12–18 min** |

Mesteparten av tiden brukes på Whisper-decoding og LLM-kall, som begge er parallelliserbare på flere kjerner.
