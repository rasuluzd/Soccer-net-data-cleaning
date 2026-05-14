# Kapittel 1 — Innledning

## 1.1 Motivasjon

Fotballkommentar i sanntid er en av de mest språklig krevende domenene for automatisk talegjenkjenning (ASR). Kommentaren inneholder hyppige egennavn (spillere, lag, dommere, stadioner), domenespesifikk terminologi, høy talehastighet i avgjørende øyeblikk, og et lydbilde med publikumsstøy som forstyrrer signalbehandlingen. Selv toppmoderne modeller som OpenAI Whisper produserer transkripsjoner med systematiske feil, der særlig egennavn blir feilstavet eller erstattet med fonetisk like, men semantisk feil ord.

For nedstrøms anvendelser er disse feilene kritiske. Når transkripsjonen skal fôre en søkemotor som Elasticsearch — slik at en bruker kan finne videoklipp ved å søke "Sterling scorer mot Liverpool" — må spillernavnet være korrekt indeksert. På samme måte krever automatisk hendelsesdeteksjon ("Hvilken spiller scoret i 67. minutt?") at navnene i transkripsjonen matcher offisielle laglister. Én bokstav feil betyr at klippet ikke gjenfinnes.

Denne oppgaven utvikler en flertrinnspipeline for automatisk rensing av Whisper-transkripsjoner av fotballkommentar, med særlig fokus på korreksjon av entiteter (spillernavn, lagnavn) i en flerspråklig setting.

## 1.2 Problemstilling

Whisper-transcripsjoner av fotballkommentar har tre dominerende feilkategorier:

1. **Entitetsfeil** — Spillernavn og lagnavn skrives feil, særlig når navnet er fonetisk likt et vanlig engelsk ord eller et annet navn ("Klein" → "Clyne", "Mané" → "Money").
2. **Hallusinasjoner** — Whisper genererer plausibel, men ikke-eksisterende tekst i stille passasjer eller ved publikumsstøy.
3. **Segmenterings-/struktursfeil** — Setninger blir kuttet midt i, eller flere talere blir slått sammen.

En naiv tilnærming — bytte ut alle navn som ligner på et navn i laglisten — produserer flere feil enn den fikser, fordi vanlige engelske ord ofte tilfeldigvis ligner på spillernavn (dokumentert i Apple-papiret RAG-NEC, Sun et al. 2024).

Den tekniske utfordringen er derfor å bygge en korreksjonsmekanisme som er **konservativ nok** til å unngå falske positiver, men **liberal nok** til å fange de ekte navnefeilene — i et språkagnostisk rammeverk som fungerer for engelsk, svensk og tysk kommentar uten håndskrevne ordlister per språk.

## 1.3 Forskningsspørsmål

> **RQ1** — Kan en kombinasjon av retrieval-augmentert oppslag (TF-IDF over en kampspesifikk gazetteer) og diskriminativ MCQ-vurdering med en liten LLM (Qwen2.5-1.5B) redusere ordfeilrate (WER) på fotballkommentar uten å introdusere flere falske positiver enn den korrigerer?

> **RQ2** — Kan konfidens-basert gating av en generativ feilkorreksjonsmodell (GER) — der bare ASR-tokens med lav log-sannsynlighet sendes til LLM-en — gi forbedret WER samtidig som den beskytter mot over-korreksjon?

> **RQ3** — Hvor godt generaliserer en slik arkitektur fra engelsk til andre språk (svensk, tysk) når man bytter underliggende NLP-modeller, men beholder samme algoritmiske rammeverk?

## 1.4 Bidrag

Denne oppgaven leverer:

1. **En komplett, åpen pipeline** for ASR-rensing av fotballkommentar, fra rå MP3-audio til renset JSON klar for Elasticsearch-indeksering.
2. **En egen entitetskorrekturkomponent** (`entity_corrector.py`) som kombinerer TF-IDF-retrieval, MCQ-judge med self-consistency, og MLM-veto med XLM-RoBERTa — en arkitektur som så vidt forfatteren vet ikke er publisert i denne sammensetningen.
3. **En to-lags valideringscache** (per-match + cross-match consensus) som lærer over kjøringer uten å arve falske positiver fra én enkelt feilkjøring.
4. **En metodisk ablation-studie** som isolerer bidraget fra hvert pipeline-trinn på GOAL-benchmarken (Chelsea-Liverpool 2016-09-16).
5. **En flerspråklig design** der språkdeteksjon styrer modellvalg (spaCy, sentence-transformer, fonetisk algoritme) uten håndskrevne ordlister per språk.

## 1.5 Avgrensning

- **Hovedevaluering kun på engelsk.** GOAL-benchmarken er det eneste datasettet i prosjektet med metodisk ren human-annotert GT. Svensk evaluering inkluderes som pilot, men WER-tall rapporteres ikke fordi den eneste tilgjengelige svenske GT-en (AIK-Halmstad 2025-11-09) er kontaminert med stock Whisper-output.
- **CPU-only.** Hele pipelinen kjører uten GPU. Dette begrenser modellvalg (Qwen 1.5B i stedet for 7B+) men gjør løsningen kompatibel med arbeidsstasjoner uten dedikert maskinvare.
- **Ingen kommersielle API-er.** Alle modeller er åpne vekter (HuggingFace, GGUF). Dette utelukker GPT-4 og Claude som korrekturmotor, men sikrer reproduserbarhet og personvern.
- **Ingen håndskrevne ordlister per språk.** Prosjektet følger regelen `No static word lists` — POS-tagging og MLM-veto erstatter manuelle stoppordslister.

## 1.6 Oppgavens struktur

Kapittel 2 går gjennom relevant litteratur innen ASR-rensing, generativ feilkorreksjon og retrieval-augmenterte metoder. Kapittel 3 beskriver datasettene og evalueringsmetodikken. Kapittel 4 presenterer pipeline-arkitekturen i detalj, og kapittel 5 dokumenterer implementasjonen. Kapittel 6 viser eksperimenter og resultater, kapittel 7 diskuterer funnene, og kapittel 8 oppsummerer og peker på videre arbeid.
