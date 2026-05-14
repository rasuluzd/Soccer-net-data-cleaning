# Kapittel 3 — Datagrunnlag og evalueringsmetodikk

## 3.1 SoccerNet-v2 og caption-2023

**SoccerNet** (Giancola et al. 2018, Deliège et al. 2021) er et åpent datasett med fullengde fotballkamper fra europeiske toppligaer, primært beregnet på hendelsesdeteksjon (mål, kort, innbytter). I 2023 ble datasettet utvidet med **caption-2023** — automatiske transkripsjoner av kommentarsporet, samt tilhørende `Labels-caption.json`-filer som lister tidsstempler for nevnte hendelser.

Underdatasettet `SN-Echoes` (https://github.com/SoccerNet/sn-echoes) inneholder Whisper-transkripsjoner i to versjoner (`whisper_v1`, `whisper_v2`) for hver kamp. Prosjektet bruker `whisper_v2` som baseline-input til pipelinen.

## 3.2 GOAL-benchmarken

GOAL (Mehdi Mousavi et al. 2024) er en utvidet benchmark der **hovedforbedringen er at GT-en er menneskelig annotert**, ikke generert av Whisper. Dette gjør GOAL det eneste datasettet i prosjektet med metodisk gyldig referanse for WER-evaluering.

GOAL-undersettet for engelsk inkluderer Chelsea-Liverpool 2016-09-16, som er hovedkampen for evaluering i denne oppgaven.

## 3.3 Kampene som brukes

| Kamp | Liga | Språk | Bruk i oppgaven |
|---|---|---|---|
| Chelsea 1–2 Liverpool, 2016-09-16 | EPL | Engelsk | Hovedbenchmark, GOAL human GT |
| Chelsea 1–2 Crystal Palace, 2015-08-29 | EPL | Engelsk | Sekundærtest (generaliserbarhet) |
| AIK 0–2 Halmstad, 2025-11-09 | Allsvenskan | Svensk | Pilot for flerspråklighet (uten WER-tall) |

## 3.4 Begrensninger ved svensk GT

Den svenske kampen brukes som demonstrasjon av at pipelinen er språkagnostisk, men WER-tall rapporteres ikke. Grunnen er at den svenske referansen (`*_corrected.json`) ble bygget ved å starte fra rå Whisper-output og gjøre lette manuelle korreksjoner. Dette betyr at referansen i utgangspunktet er biased mot Whisper-stilte feil — og alle WER-tall mot denne GT-en vil systematisk underestimere reelle forbedringer.

Dette diskuteres åpent i oppgaven som en metodologisk begrensning, og brukes til å motivere hvorfor evalueringen flyttet til GOAL.

## 3.5 Metrikker

### Word Error Rate (WER)

WER beregnes med `jiwer` som:

> WER = (S + D + I) / N

der S = substitutions, D = deletions, I = insertions, og N = antall ord i referansen. Implementasjonen ligger i `tools/evaluate_wer.py`.

### Character Error Rate (CER)

Tilsvarende WER, men på karakter-nivå. Mer sensitiv til mindre stavefeil — særlig nyttig for spillernavn der én bokstav skiller "Klein" fra "Clyne".

### Entity F1

Beregnes ved å:

1. Ekstrahere alle PROPN/PERSON-entiteter fra både hypotese og GT med spaCy.
2. Matche entitetene fuzzy (Levenshtein) på navn-nivå.
3. Beregne presisjon, recall og F1 på det matched settet.

Denne metrikken er den mest direkte indikatoren på om pipelinen gjør jobben sin for nedstrøms søk og hendelsesdeteksjon.

## 3.6 Begrunnelse for legacy 1-til-1 alignment

WER-evaluatoren støtter to alignment-moduser:

- **Legacy (1-til-1)** — Hver hypotese-segment matches mot nærmeste GT-segment etter tidsstempel. Standard.
- **Windowed (mange-til-én)** — Alle hypotese-segmenter som overlapper et GT-vindu konkateneres før WER beregnes.

I tidlige eksperimenter viste det seg at windowed alignment **økte rapportert WER**, ikke reduserte. Dette er fordi `jiwer`s dynamic programming "glatter" over segmenteringsforskjeller i legacy-moduset, mens windowed eksponerer dem som ekstra insertions/deletions.

Legacy-moduset velges som standard fordi:

1. Resultatene er sammenlignbare med online WER-verktøy (https://martin-thoma.com/word-error-rate-calculator/).
2. WER-tallene er direkte sammenliknbare med litteraturen (Apple RAG-NEC bruker også 1-til-1).
3. Segmenteringsforskjeller mellom Whisper og menneske er en separat utfordring (diariseringsproblem) og bør ikke straffes som ASR-feil.

Begge modusene er bevart i koden og kan velges via `--alignment-mode`-argument.

## 3.7 Reproduserbarhet

Alle modeller har faste versjonspinninger i `requirements.txt`. Whisper-decoding bruker `temperature=0.0` og fast `seed`. Qwen-LLM-en kjøres med `temperature=0.0` (deterministisk) for hovedresultatet og `temperature=0.7` for self-consistency-samplingen. XLM-RoBERTa er en frosset, deterministisk modell.

Alle eksperimenter kan reproduseres med:

```bash
python run_pipeline.py --match "Chelsea 1 - 2 Liverpool"
python tools/evaluate_wer.py --match "Chelsea" --half 1
python tools/evaluate_wer.py --match "Chelsea" --half 2
```
