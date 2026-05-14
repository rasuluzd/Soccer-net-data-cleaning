# Kapittel 4 — Pipeline-arkitektur

## 4.1 Oversikt

Pipelinen tar rå MP3-audio som input og produserer renset, søkbar JSON som output. Trinnene er:

```
audio.mp3
   ↓
[Whisper]  →  rå transkripsjon (1_asr.json)
   ↓
[Tier 1]   →  hallucination filter + deduplicator + language detection
   ↓
[Stage 2A] →  domain_normalizer (regelbasert renskeløp)
   ↓
[Tier 2]   →  entity_corrector (TF-IDF + MCQ + MLM-veto)
   ↓
[Step L]   →  llm_corrector (konfidensgated GER med Qwen)
   ↓
cleaned.json (med per-token konfidens, korreksjonsmetadata)
```

Hovedprinsippet er **én avgjørelsestaker per bekymring**: Tier 1 håndterer struktur, Tier 2 håndterer entitetsnavn, Step L håndterer fri tekst-rensing. Ingen modul kan overstyre en annens veto.

## 4.2 Tier 1: Deterministiske renselag

### 4.2.1 Hallucination filter (`hallucination_filter.py`)

Tre regler:

1. **Alpha-ratio** — Segmenter med < 70 % bokstaver (resten tegnsetting/tall) markeres som hallusinasjon.
2. **Language detection** — `langdetect` på segmentnivå. Segmenter på "feil" språk (f.eks. fransk i en engelsk kamp) kastes.
3. **Non-Latin removal** — Segmenter dominert av ikke-latinske tegn (kyrillisk, kinesisk) er nesten alltid hallusinerte i en europeisk fotballkontekst.

Tersklene er konfigurerbare i `pipeline/config.py`.

### 4.2.2 Deduplicator (`deduplicator.py`)

Sammenslår påfølgende segmenter med høy tekstlikhet (Whisper repeterer av og til samme setning). Bruker fuzzy ratio fra `rapidfuzz` med terskel 90.

### 4.2.3 Language detection (`orchestrator._detect_commentary_language`)

Bestemmer kampens primærspråk fra rå transkripsjon. Brukes til å:

- Velge spaCy-modell (`en_core_web_sm` for engelsk, `xx_ent_wiki_sm` ellers).
- Velge sentence-transformer (`all-MiniLM-L6-v2` for engelsk, multilingual ellers).
- Velge fonetisk algoritme (Metaphone for engelsk, accent-normalisert Soundex ellers).

## 4.3 Stage 2A: Domenenormalisering (`domain_normalizer.py`)

Ren regelbasert pipeline:

- **Tallnormalisering** — "tjuetre" → "23" (eller omvendt, avhengig av kontekst).
- **Tidsuttrykk** — "i sekstisjuende minuttet" → "i 67. minuttet".
- **Felles uttrykk** — "var ikke det" → "var det ikke" (svensk syntaksrettelse).

Modulen er språkspesifikk (har separate regler for engelsk, svensk, tysk) men bruker **ingen statiske ordlister med personnavn eller domenetermer**. Den jobber kun på syntaktiske mønstre.

## 4.4 Tier 2: Entity correction — vårt hovedbidrag

`entity_corrector.py` (~800 LOC) er pipelinens hovedkomponent. Den korrigerer feilstavede egennavn ved å kombinere fire teknikker.

### 4.4.1 TF-IDF char-bigram retrieval

For hver kamp bygges en gazetteer fra `Labels-caption.json`: spillerlister, lagnavn, dommer, stadion. Hvert navn i gazetteeren indekseres med `TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))`.

For hvert kandidatord i transkripsjonen:

1. Slå opp top-K (default K=5) nærmeste gazetteer-navn etter cosine similarity.
2. Bestem hvilken handling som skal tas basert på score-distribusjon.

### 4.4.2 Shortcut-accept

Hvis topp-kandidaten har cosine ≥ 0.90 OG gap til nest-beste ≥ 0.10, **aksepter umiddelbart** uten å invokere LLM. Dette håndterer åpenbare tilfeller (én bokstav forskjell, klar vinner) raskt.

### 4.4.3 Shortcut-reject

Hvis topp-kandidaten har cosine < 0.40, **forkast umiddelbart**. Ordet er ikke nært nok til noe i gazetteeren — sannsynligvis et vanlig ord som tilfeldigvis ble plukket opp som kandidat.

### 4.4.4 MCQ-judge med Qwen2.5-1.5B-Instruct

Mellom-tilfeller (cosine i området 0.40–0.90) sendes til en LLM for diskriminativ vurdering. Prompten er formet som et flervalgsspørsmål:

> Det opprinnelige ordet er **{word}**. Konteksten er: "{...sentence...}".
> Velg det mest sannsynlige korrekte navnet:
> A) {candidate_1}
> B) {candidate_2}
> C) {candidate_3}
> D) Ingen — behold originalen.

LLM-en (Qwen2.5-1.5B-Instruct via `llama-cpp-python`) returnerer én bokstav. Diskriminativ form unngår at LLM-en finner på et helt nytt ord (en kjent feilmodus i generativ GER).

### 4.4.5 Self-consistency

Hver MCQ kalles 3 ganger med `temperature=0.7`. Majoritetsvinner blir det endelige svaret. Hvis ingen majoritet (3 forskjellige svar), forkast kandidaten. Dette er DeRAGEC-mønsteret (ACL 2025).

### 4.4.6 MLM-veto med XLM-RoBERTa

Etter at MCQ-en har valgt en kandidat, brukes XLM-RoBERTa til en uavhengig validering. Vi maskerer det opprinnelige ordet i kontekst:

> "Beautiful pass from `[MASK]` to Hazard"

og sjekker at LLM-ens valgte kandidat er blant top-K i MLM-fordelingen. Hvis ikke, vetoes korreksjonen. Dette er en cross-model sanity check — to uavhengige modeller må være enige før vi endrer ordet.

### 4.4.7 To-lags valideringscache

For å unngå å invokere LLM-en gjentatte ganger på samme korreksjon over flere kjøringer:

- **Per-match cache** (`data/learned_corrections.json`) — Lagrer accepterte korreksjoner innen en kamp.
- **Cross-match validated cache** (`data/validated_corrections.json`) — En korreksjon promoveres til den globale cachen først etter at den har blitt akseptert i ≥3 forskjellige matcher (consensus rule).

Dette er en designforsiktighet mot **poison-vector failure mode**: hvis én kjøring ved et uhell aksepterer en feil korreksjon, vil ikke den feilen permanent forgifte fremtidige kjøringer — den må gjentas i flere kamper før den blir global.

## 4.5 Step L: Konfidensgated GER (`llm_corrector.py`)

Etter at Tier 2 har fikset entitetsnavn, kjører Step L en generativ feilkorreksjon over hele segmentet — men **bare på lav-konfidens-tokens**.

### 4.5.1 Konfidensgating

Whisper returnerer `avg_logprob` for hvert ord. Tokens med `logprob < -0.3` (konfigurerbart i `LLM_LOGPROB_GATE`) wraps i markører:

> "He passes the ball to <Klein> who shoots from outside the box"

LLM-en (samme Qwen som Tier 2) blir bedt om å **bare endre tokens i markører**, og returnere alt annet uendret. Dette er Confidence-Guided Error Correction (Zhang et al. 2025).

### 4.5.2 Frozen word indices

For å unngå at Step L overstyrer Tier 2s bevisste valg, sender Tier 2 med en liste over indekser den har "berørt". Step L respekterer disse posisjonene som frosne — de wraps aldri, uansett logprob. Dette implementeres via `Segment.frozen_word_indices` i `loader.py`.

## 4.6 Output-format

Pipelinen produserer en utvidet JSON med:

- `text` — Renset transkripsjonstekst.
- `words[].prob` — Bevart per-token Whisper-konfidens.
- `cleaning_metadata.corrections` — Liste over alle endringer (original, corrected, source, confidence).
- `temporal_chunks` — Konsoliderte tidsavgrensede tekstklumper for Elasticsearch-indeksering.

Dette gjør det mulig for nedstrøms søk å vekte resultater etter konfidens, og for utviklere å revidere hver enkelt korreksjon.

## 4.7 Designprinsipper

Tre prinsipper styrer hele arkitekturen:

1. **No static word lists.** Ingen modul har en hardkodet ordliste over personnavn, vanlige ord eller domenetermer. POS-tagging og MLM-veto erstatter manuelle filtre. Dette gjør pipelinen språkagnostisk.
2. **Config-only constants.** Alle terskler er definert i `pipeline/config.py`. Inline-konstanter er forbudt. Dette gjør tuning sporbar og ablasjon mulig.
3. **Verify every change.** Hver bugfiks krever en regresjonstest som feiler før fiksen og passerer etter. Dette er kodifisert i `.claude/skills/fix-bug/SKILL.md`.
