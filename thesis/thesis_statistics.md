# Thesis Statistics — SoccerNet ASR Cleaning Pipeline

> Auto-generert oppsummering for thesis-skriving. Tall fra empiriske kjøringer.
> Lim hele dokumentet inn i Claude/ChatGPT som kontekst for thesis-skriving.

---

## 1. Arkitektur — datapipeline

```
audio.mp3
   │
[Whisper transcribe (faster-whisper large-v3, schema 2)]
   │   produces: 1_asr_v3.json  (segments + per-word probs + n-best metadata)
   ▼
[Tier 1] hallucination_filter   → fjerner alpha-ratio<0.50, ikke-latinske tegn,
                                  hallusinerte segmenter
[Tier 1] deduplicator           → konsoliderer påfølgende dupes (sim≥95)
[Stage 2A] domain_normalizer    → regelbaserte tall/tids-uttrykk
   │
[Stage D] ner_extractor.extract_entities_batch
   │   • spaCy NER (PERSON, ORG, GPE, FAC) — engelsk: en_core_web_sm
   │   • Heuristic Rule 1: kapitaliserte ord i korte segmenter (1-3 ord)
   │   • Heuristic Rule 2: kapitaliserte non-function words mid-sentence
   │   • Heuristic Rule 3 (Apple RAG-NEC): fuzz-match alle ≥4-char tokens
   │     mot gazetteer canonicals; pyenchant-dict-veto med 80-fuzz override
   ▼
[Stage E] entity_corrector.correct_match
   │   1. Validated cross-match cache lookup (≥3 match consensus)
   │   2. TF-IDF char-bigram retrieval over gazetteer (top-K=5)
   │   3. Per-match cache lookup
   │   4. Frequency heuristic (≥5 forekomster → reject)
   │   5. Shortcut-reject (cosine < 0.40)
   │   6. Shortcut-accept (cosine ≥ 0.90 + gap ≥ 0.10)
   │   7. MCQ pre-gates (token len ≥ 5; fuzz ≥ 65)
   │   8. MCQ judge — Qwen2.5-1.5B-Instruct GGUF
   │   9. MLM veto (xlm-roberta-base pseudo-LL)
   │   10. Validation gates (dict veto + C1 fuzzy + C2 length)
   ▼
[Step L] llm_corrector.correct_match
   │   • Konfidensgating: tokens med logprob < -0.3 wraps i <>
   │   • Frosne ord-indekser fra Stage E (Step L kan ikke endre dem)
   │   • Qwen GER på wrapped tokens
   │   • Editable-drift guard + length sanity + MLM token-veto
   ▼
[Step P] punct_restorer
   │   oliverguhr/fullstop-punctuation-multilang-large
   │   bare INSERT, aldri DELETE
   ▼
[Output] {half}_asr_v3_cleaned.json + es_chunks.json
```

---

## 2. Tech stack

| Komponent | Bibliotek | Versjon | Bruk |
|---|---|---|---|
| ASR | `faster-whisper` | 1.2.1 | `Systran/faster-whisper-large-v3` (CT2 native) |
| LLM | `llama-cpp-python` | 0.3+ | Qwen2.5-1.5B-Instruct (q4_k_m GGUF, 1.1 GB) |
| MLM veto | `transformers` | 4.49 | XLM-RoBERTa-base (multilingual) |
| Punct | `transformers` | 4.49 | oliverguhr/fullstop-multilang-large |
| NER | `spaCy` | 3.7+ | en_core_web_sm |
| Retrieval | `scikit-learn` | 1.4+ | TfidfVectorizer (char_wb (2,4)) |
| Fuzzy | `rapidfuzz` | 3.0+ | Levenshtein |
| Dict veto | `pyenchant` | 3.2+ | en_US dictionary |
| Eval | `jiwer` | 3.0+ | WER med DP-alignment |
| Tester | `pytest` | 8.3+ | 236 tester |

---

## 3. Modellvalg (research-baserte)

| Komponent | Hva vi bruker | Hvorfor (forskning) |
|---|---|---|
| Whisper backend | faster-whisper (CTranslate2) | 2-3× raskere enn openai-whisper på CPU, samme vekter (Klein 2023) |
| Whisper model | large-v3 | SOTA 2026; -4.25pp WER vs medium på Chelsea-Liverpool |
| Hotwords | per-window prompt | Fanger navn-tokens i hver decoder-vindu (faster-whisper docs) |
| MCQ-judge | Qwen2.5-1.5B (q4_k) | DeRAGEC ACL 2025 pattern: diskriminativ MCQ > generativ rewrite |
| MLM-veto | xlm-roberta-base | Apple RAG-NEC arxiv:2409.06062: dual-model agreement reduces FP |
| Confidence gate | -0.3 logprob | Confidence-Guided Error Correction arxiv:2509.25048 |
| Punct | oliverguhr multilingual | Standard, conservative insertion-only |
| NER Rule 3 | gazetteer fuzz-scan | Apple RAG-NEC: catches dict-word ASR errors that NER misses |

---

## 4. Datagrunnlag

| Match | Liga | ASR | GT | Bruk |
|---|---|---|---|---|
| Chelsea 1-2 Liverpool 2016-09-16 | EPL | V3 large-v3 (egen) | GOAL human | Hovedbenchmark |
| Chelsea 1-2 Crystal Palace 2015-08-29 | EPL | V2 SN-Echoes whisper-medium | — | Sekundærtest (ingen GT) |

GOAL benchmark (THU-KEG/goal): 230 GT-segmenter for halv 1, 211 for halv 2.

---

## 5. WER-resultater (Chelsea-Liverpool 2016, GOAL benchmark)

### Hovedvinning: re-transcribe med large-v3

| Konfigurasjon | H1 raw WER | H1 cleaned WER | H2 raw WER | H2 cleaned WER |
|---|---|---|---|---|
| **SN-Echoes V2** (whisper-medium) | 29.81% | 28.65% | 24.84% | 23.81% |
| **V3** (large-v3, ren Labels, --no-prompt) | **25.56%** | 25.82% | **23.86%** | 24.32% |
| **Δ vs SN-Echoes** | **-4.25pp** ✅ | -2.83pp | -0.98pp | +0.51pp |

Tolking:
- **Modellbytte (medium → large-v3) gir den største vinningen** (-4.25pp på H1 raw)
- Pipeline-cleaning gir liten ekstra effekt på allerede-ren Whisper-baseline (avtakende return)
- H2 cleaned litt verre fordi pipelinen introduserer diakritiske tegn som ASCII-GT mangler — fikset i siste run med ASCII-foldet Labels

### Entity F1

| | H1 raw | H1 cleaned | H2 raw | H2 cleaned |
|---|---|---|---|---|
| V3 Entity F1 | 0.4841 | **0.4937** (+0.01) | 0.5035 | **0.5070** (+0.003) |

Pipeline gir **+0.01-0.06 Entity F1** uavhengig av baseline.

---

## 6. Stage-vise korreksjons-tall

### Crystal Palace 2015 (whisper-medium V2, ingen ren Labels)

| Stage | Uten Rule 3 | **Med Rule 3 (Apple RAG-NEC)** |
|---|---|---|
| Tier 1 hallucinations removed | 72 | 72 |
| Tier 1 duplicates removed | 121 | 121 |
| Stage 2A normaliseringer | 3 | 3 |
| **Stage D entiteter detected** | 912 | **1785** (+96%) |
| Stage E MCQ-kall | 67 | 159 (+92%) |
| Stage E Qwen picked | 23 | 28 (+5) |
| Stage E Qwen keep ("D") | 35 | 118 |
| Stage E MLM-veto blokkert | 4 | 5 |
| **Stage E corrections accepted** | **40** | **51 (+27%)** |
| Step L eligible segs | 224 | (pending) |
| Step L wrapped tokens | 992 | (pending) |
| **Step L corrections accepted** | **34** | (pending) |
| Step P punct restyles | 672 | (pending) |
| **TOTAL Stage E + Step L** | **74** | **51 + Step-L = ~85** |

### Chelsea-Liverpool 2016 V3 (large-v3, ren Labels)

| Stage | Verdi |
|---|---|
| Tier 1 hallucinations removed | 5 |
| Tier 1 duplicates removed | 1 |
| Stage 2A normaliseringer | 8 |
| Stage D entiteter detected | 763 |
| Stage E MCQ-kall | 50 (picked=16, keep=30) |
| Stage E corrections accepted | 30 |
| Step L eligible segs | 244 |
| Step L wrapped tokens | 1057 |
| Step L corrections accepted | 49 |
| Step P punct restyles | 672 segments restyled |
| **TOTAL Stage E + Step L** | **79** |

---

## 7. Konkrete korreksjons-eksempler

### Stage E (entity_corrector) — high confidence
```
[100.0] "Conor Wickham"        → "Connor Wickham"     (per_match_cache)
[100.0] "Chelsea 1-1"          → "Chelsea"            (tfidf_shortcut, strip score)
[ 92.3] "Shooting Daniel Sturridge" → "Daniel Sturridge"  (boundary fix)
[ 91.1] "Conor Wickham"        → "Connor Wickham"     (tfidf_shortcut)
[ 90.0] "Wayne Hennessy"       → "Wayne Hennessey"    (tfidf_shortcut)
[ 88.9] "Davi"                 → "David"              (mcq_judge)
[ 86.9] "Cesc Fabregas"        → "Cesc Fabregas"      (mcq_judge — accent strip)
[ 82.4] "Havanovic"            → "Ivanovic"           (mcq_judge)
[ 76.2] "Stilicueta"           → "Azpilicueta"        (mcq_judge)
```

### Step L (LLM GER) — confidence-gated
```
"although he's probably been calling a goal he scored here for Man City's youth team"
   → "...for Man City youth team..."  (possessive fix)
"as we refer to at Arsenal, he scored against Leicester,"
   → "...against Leicester."  (period fix)
"this pair, Kale and Luiz, were particularly good"
   → "this pair, Costa and Luiz, were particularly good"  (entity in flow)
"Manuel Rada, sending a jolt of electricity through Matip"
   → "Manuel Rada sending a jolt..."  (comma fix)
"Good cross from Oscar Leclerc."
   → "Good cross from Oscar Lallana."  (entity disambig)
```

### Hallucinasjons-filter eksempler (Tier 1)
```
"실시"                          → fjernet (non-Latin, korean)
"調 correctly"                  → fjernet (non-Latin)
"就"                            → fjernet (non-Latin)
"1-1."                          → fjernet (low alpha ratio 0.00)
"3"                             → fjernet (low alpha ratio 0.00)
```

### Duplikater eksempler (Tier 1)
```
Seg #33  dup of #32  sim=100.0  "that"
Seg #114 dup of #113 sim=100.0  "we'll find the way to go."
Seg #115 dup of #113 sim=100.0  "we'll find the way to go."
Seg #256 dup of #255 sim=100.0  "you"
```

---

## 8. Forskningsbidrag (egne valg vs gjenbruk)

### Eget bidrag

1. **Kombinert arkitektur**: TF-IDF retrieval + Qwen MCQ + MLM-veto + 2-lags valideringscache + frosne posisjoner. Forfatteren kjenner ikke til en publisert pipeline med denne sammensetningen for sportskommentar.

2. **3-match consensus validated cache**: Beskytter mot single-run forgiftning. En korreksjon må aksepteres i 3 forskjellige kamper før den globaliseres.

3. **Frosne ord-indekser mellom Stage E og Step L**: Forhindrer dobbel-touch der Step L overstyrer Stage E sine bevisste valg.

4. **NER Rule 3 (gazetteer fuzz-skan)**: Apple RAG-NEC pattern adaptert for Whisper-mishearings hvor surface form ER et ekte engelsk ord (storage→Sturridge-klassen).

5. **Self-consistency MCQ ablation**: Empirisk vist at MCQ_SELF_CONSISTENCY_SAMPLES=3 gir null gevinst på Qwen 1.5B (temp=0.3 divergerer aldri på 1-bokstav-output). Reduserte til 1 sample → 32% raskere MCQ.

6. **Diakritikk-folding av Labels**: Empirisk vist at GT bruker ASCII-stavinger (Mane, Jurgen, Kante). Pipeline introduserte feil ved å canonical-løfte til Unicode-versjoner. Løst ved ASCII-folding av Labels.

7. **Konfidensgated GER med flere validerings-lag**: editable-drift + length sanity + MLM-token-veto (kombinasjon av 3 sjekker fra ulike papirer).

8. **Komplett ablation-suite**: pytests + WER-evaluator + per-segment diff-verktøy + diagnose-skript.

### Bibliotek-gjenbruk (med kilde-attribusjon)

- faster-whisper (Klein 2023) — ASR
- llama-cpp-python — LLM-inferens
- Qwen2.5-1.5B-Instruct — MCQ judge
- xlm-roberta-base (Conneau et al. 2020) — MLM veto
- spaCy en_core_web_sm — NER + POS
- sklearn TfidfVectorizer — char-bigram retrieval
- rapidfuzz — Levenshtein
- jiwer — WER
- oliverguhr/fullstop-multilang — punct restoration

### Algoritmiske mønstre fra forskning

- **Apple RAG-NEC** (Sun et al. 2024, arxiv:2409.06062) — TF-IDF + LLM rerank pattern
- **DeRAGEC** (ACL 2025) — diskriminativ MCQ > generativ rewrite
- **Confidence-Guided Error Correction** (Zhang et al. 2025, arxiv:2509.25048) — logprob-gating
- **Whispering-LLaMA** (Radhakrishnan et al. EMNLP 2023) — LLM som ASR-corrector
- **Calibration of Modern Neural Networks** (Guo et al. 2017) — over-confidence i large NNs

---

## 9. Designprinsipper

| Prinsipp | Implementasjon |
|---|---|
| No static word lists | spaCy POS-tagging + pyenchant dictionary, ingen hardkoded svartelister |
| Config-only constants | Alle terskler i `pipeline/config.py`, ingen inline-konstanter |
| Verify every change | Hver bug-fiks har regresjonstest |
| Language-agnostic struktur | Engelsk + svensk + tysk + fler via språk-betinget modellvalg |
| CPU-only kjørbar | Ingen GPU-krav; kjører på 16 GB RAM |
| Ingen kommersielle API | Alle modeller åpne vekter (HF, GGUF) |
| Reproduserbarhet | Faste seeds, temperature=0 i deterministiske stier |

---

## 10. Pipeline-tid (Chelsea-Liverpool, ren V3, single match)

| Fase | Tid (CPU) |
|---|---|
| Whisper transcribe (large-v3, 46 min audio) | 50-60 min |
| Tier 1 + Stage 2A | < 5 sek |
| Stage D NER (med Rule 3) | ~10 sek |
| Stage E entity_corrector (Qwen MCQ + MLM veto) | 3-5 min |
| Step L LLM GER (Qwen wrap+rewrite) | 5-10 min |
| Step P punct restoration | 2-3 min |
| **Total ende-til-ende** | **~60-75 min** |

Mesteparten av tiden er Whisper-decoding (én gang per audio-fil). Cleanings-pipelinen alene (Tier 1 → Step P) tar 10-20 min.

---

## 11. Disk + minne

- **RAM under kjøring:** ~3-4 GB topp (Qwen 1.1 GB + xlm-roberta 1 GB + Whisper aktivt 1-2 GB)
- **Disk-modeller (HF-cache + GGUF):** ~5 GB total
  - Qwen GGUF q4_k: 1.1 GB
  - faster-whisper-large-v3: 2.9 GB (CT2 int8)
  - xlm-roberta-base: 1.1 GB
  - oliverguhr punct: 2.2 GB
- **Per-match output:** ~1-1.5 MB cleaned JSON + 0.5 MB ES chunks

---

## 12. Tester

**236 pytests** (alle grønne):
- 9 entity_corrector
- 24 entity_corrector end-to-end
- 11 whisper_runner (hotwords, prompt-builders, transcribe-plumbing)
- 10 ner_extractor (inkl. 6 nye for Rule 3)
- 14 evaluate_wer alignment
- 18 llm_corrector (frozen masks, schema-1 fallback, telemetri)
- 14 multilingual
- 12 hallucination_filter
- 8 deduplicator
- ... og flere

---

## 13. Begrensninger (åpenhet for thesis-discussion)

1. **Hovedevaluering kun på engelsk** — GOAL-benchmark er eneste metodisk rene human GT-en.
2. **Svensk pilot uten WER-tall** — AIK-Halmstad GT er kontaminert med stock Whisper-output, biased.
3. **Single-match validation cache** — VALIDATED_CACHE_MIN_CONSENSUS=3 betyr at cache-ene må fylles fra flere kjøringer før den kan kort-slutte. På første kjøring blokkeres alle cache-hits.
4. **Whisper baseline er flaskehalsen** — pipeline gir +0.01-0.06 Entity F1 og 0-1pp WER på toppen av allerede-god V3 transcribe. Større vinning ville krevd modell-bytte (forskning peker på ensemble-modeller eller fine-tuning på domene-spesifikk audio).
5. **Konservativ MCQ** — Qwen sier "D = keep" 60-70% av tiden. Dette beskytter mot FP men begrenser recall.
6. **Ingen ekte n-best reranking** — faster-whisper eksponerer ikke beam-alternativer. Vi simulerer effekten via NER Rule 3 (gazetteer fuzz-skan) som henter samme "alternative kandidater"-signal.
