# Bacheloroppgave — Disposisjon (OsloMet-mal)

> Speiler **eksisterende** innholdsfortegnelse fra teamets dokument.
> For hver seksjon: hvilket ASR-innhold (fra `01_*.md` … `05_*.md`) som skal flettes inn.
> Ikke nye kapitler — bare innfletting i seksjonene som allerede finnes.

---

## Forside, abstract, innholdsfortegnelse

- **Forside.** Tittel, gruppenavn, gruppemedlemmer, veileder, OsloMet-logo. ASR bidrar med en figur av pipeline-arkitekturen som blikkfang.
- **Abstract** (1 side, skrives sist). Tre bolker: (i) bakgrunn + problemstilling, (ii) hva er bygget (frontend + backend + ASR-rensepipeline), (iii) hovedresultater (WER fra 29.81 % → 28.65 % på Chelsea-Liverpool, entity F1 0.64). Skriv på samme språk som resten.

---

## 1 Introduksjon

### 1.3 Veileder
Eksisterende tekst beholdes. Ingen ASR-tilskudd.

### 1.4 Bakgrunn og Motivasjon
**Eksisterende tekst: behold om Forzasys, søkbare klipp, kontekst.**
**Tilskudd fra ASR (`01_innledning.md` §1.1):** 1–2 avsnitt om hvorfor rå Whisper-output ikke er søkbar (entitetsfeil bryter Elasticsearch-treff), og hvorfor automatisk rensing er nødvendig før søkemotoren kan brukes.

### 1.5 Prosjektbeskrivelse
- **1.5.1 Prosjektet.** Eksisterende beskrivelse av frontend + backend + søkeflyt. **Tilskudd:** ett avsnitt om at backend består av to lag: Elasticsearch (eksisterende) + ASR-rensepipeline (vårt arbeid).
- **1.5.2 Problemstilling.** **Tilskudd fra ASR (`01_innledning.md` §1.2):** problemstillingen utvides med setningen om at ASR-feil i navn er den dominerende feilkilden for søk.
- **1.5.3 Målgrupper.** Eksisterende. **Tilskudd:** Forzasys (interne brukere som leser cleaned JSON), sluttbrukere (som søker via frontend), forskere/andre studenter (åpen kildekode).

### 1.6 Gruppens mål for prosjektet
**Tilskudd fra ASR (`01_innledning.md` §1.3 + §1.4):** legg til 2–3 mål under produktmål:
- Bygge en pipeline som reduserer WER på engelsk fotballkommentar målbart mot rå Whisper.
- Designe arkitekturen språkagnostisk (engelsk + svensk + tysk uten håndskrevne ordlister).
- Levere ablation-studie som dokumenterer bidraget fra hvert pipeline-trinn.

---

## 2 Begreper og verktøy

### 2.1 Begreper
**Tilskudd fra ASR (`02_bakgrunn.md` §2.1, §2.3, §2.5, §2.7, §2.8):** legg til som korte definisjoner:
- **ASR** (Automatic Speech Recognition).
- **WER** (Word Error Rate) + **CER** + **Entity F1**.
- **GER** (Generative Error Correction).
- **NER** (Named Entity Recognition).
- **MLM** (Masked Language Model).
- **TF-IDF char-bigram retrieval.**
- **MCQ-judge / diskriminativ korreksjon.**
- **Konfidensgating.**

### 2.2 Programmer
- **2.2.1 Frontend.** Eksisterende (10 000 ord beholdes uendret).
- **2.2.2 Backend.** **Tilskudd fra ASR (`05_implementasjon.md` §5.1, §5.2):** beskriv at backend består av Elasticsearch + ASR-rensepipeline. Inkluder modulkartet fra §5.2 her som figur (kort tekst rundt). Pipeline-arkitekturfiguren (fra `04_arkitektur.md` §4.1) kan også henvises herfra.

### 2.3 Rammeverk & verktøy
**Tilskudd fra ASR (`05_implementasjon.md` §5.1):** legg til tabell med:
- `faster-whisper` (ASR)
- `llama-cpp-python` + Qwen2.5-1.5B-Instruct GGUF (LLM)
- `transformers` + XLM-RoBERTa-base (MLM-veto)
- `spaCy` (NER + POS)
- `sklearn` TF-IDF
- `rapidfuzz`, `metaphone`, `pyphonetics`
- `jiwer` (WER-evaluering)
- `pytest` (testing)

For hver: én linje om hva det brukes til.

---

## 3 Prosessdokumentasjon

### 3.1 Planlegging og Metoder
- **3.1.1 Prosessmodellering.** Eksisterende. **Tilskudd:** ett avsnitt om iterativ utvikling av ASR-pipelinen — flere ablation-runder, terskeltuning, regresjonstester etter hver endring.
- **3.1.2 Fremdriftsplan.** Eksisterende. **Tilskudd:** kort om når ASR-arbeidet startet og hvilke milepæler (entity_corrector v1, v2 etter 5 fixes, multilingual pilot).
- **3.1.3 Prosjektstyring.** Eksisterende.

### 3.2 Kravspesifikasjon
- **3.2.1 Interessenter og brukergrupper.** Eksisterende.
- **3.2.2 Funksjonelle krav.** **Tilskudd fra ASR (`01_innledning.md` §1.4):** legg til som funksjonelle krav til backend:
  - Pipelinen skal lese rå Whisper-JSON og produsere renset JSON.
  - Skal korrigere entitetsnavn ved oppslag mot kampens lagliste.
  - Skal filtrere bort hallusinerte segmenter.
  - Skal fungere på engelsk, svensk og tysk uten språkspesifikke ordlister.
- **3.2.3 Ikke-funksjonelle krav.** **Tilskudd:**
  - Skal kjøre på CPU (ingen GPU-krav).
  - Skal ikke kalle kommersielle API-er (alle modeller åpne vekter).
  - Hver bugfiks skal ha tilhørende regresjonstest.
  - Alle terskler i `pipeline/config.py` (ingen inline-konstanter).
- **3.2.4 Tekniske krav og avgrensninger.** **Tilskudd fra ASR (`01_innledning.md` §1.5):**
  - Hovedevaluering kun på engelsk (GOAL benchmark — eneste rene human-GT).
  - Svensk pilot uten WER-tall (GT er kontaminert).
  - Disk-/RAM-begrensning: kun små lokalt-kjørbare modeller.
- **3.2.5 Antakelser og justeringsrom.** Eksisterende.

---

## 4 Systembeskrivelse / Produktdokumentasjon (NY — anbefales lagt til)

> OsloMet-malen anbefaler eget produkt-kapittel mellom prosess og testing.
> Hvis dere allerede har dette innbakt et annet sted, flytt heller dette innholdet dit.

- **4.1 Systemarkitektur.** Hele frontend ↔ backend-flyten. **Tilskudd fra ASR (`04_arkitektur.md` §4.1):** pipeline-figur for ASR-flyten som del av backend.
- **4.2 ASR-rensepipeline (vårt hovedbidrag).** **Innhold fra `04_arkitektur.md` §4.2–4.7:**
  - Tier 1: hallucination filter, deduplicator, language detection.
  - Stage 2A: domain_normalizer.
  - Tier 2: entity_corrector (TF-IDF + MCQ + MLM-veto + valideringscache).
  - Step L: konfidensgated GER.
  - Designprinsipper (no static word lists, config-only, verify every change).
- **4.3 Relaterte løsninger.** **Innhold fra `02_bakgrunn.md` §2.3, §2.4 (kort versjon):** Apple RAG-NEC, DeRAGEC, Confidence-Guided Error Correction. Diskuter hvorfor vår kombinasjon er ny.
- **4.4 Datagrunnlag.** **Innhold fra `03_data_metodikk.md` §3.1–3.4:** SoccerNet-v2, GOAL benchmark, kampene som brukes, begrensninger ved svensk GT.
- **4.5 Designvalg og iterasjoner.** Hvilke arkitekturer som ble forkastet (gammel 3-tier fuzzy + context_disambiguator) og hvorfor. Bytte til entity_corrector. 5 quality fixes etter v1.

---

## 5 Testing, validering og resultater (tidligere kap 4 i deres TOC)

> Deres eksisterende kapittel 4 («Prototypeutvikling og brukertesting») dekker brukertesting av frontend.
> Anbefaler å utvide til å dekke både frontend-brukertesting OG backend/ASR-validering, eller dele i to underseksjoner.

### Eksisterende: Brukertesting (frontend)
Behold som er.

### Tilskudd: Validering av ASR-pipelinen
**Innhold fra `03_data_metodikk.md` §3.5–3.6 + `05_implementasjon.md` §5.4:**
- **5.x.1 Testtyper.** Enhetstest (per modul i `tests/`), integrasjonstest (full pipeline mot Chelsea-kampen), systemtest (full backend ende-til-ende).
- **5.x.2 Metrikker.** WER, CER, Entity F1. Hvorfor disse ble valgt.
- **5.x.3 Alignment-valg.** Legacy 1-til-1 vs windowed mange-til-én. Hvorfor legacy.
- **5.x.4 Resultater.** `[TODO etter siste WER-kjøring]` — ablation-tabell:
  - Rå Whisper baseline.
  - + Tier 1.
  - + Stage 2A.
  - + entity_corrector v1.
  - + 5 quality fixes (v2).
  - + Step L.
- **5.x.5 Eksempler.** Konkrete korreksjoner som lyktes (Klein → Clyne) og som feilet (Saturday → Sturridge før fix). Viser kvalitativ forbedring.

---

## 6 Diskusjon og refleksjon (NY — anbefales)

OsloMet-malen krever dette. **Tilskudd fra ASR:**
- Hvorfor bare ~1 pp WER-reduksjon? Whisper-baselinen er flaskehalsen, ikke pipelinen.
- Trade-off presisjon vs. recall — vi valgte færre, sikrere korreksjoner (relevant for produksjonsbruk).
- Generaliserbarhet til svensk/tysk: design er språkagnostisk, men evaluering mangler ren GT.
- Hva som ville vært gjort annerledes med mer tid (re-transcribe med large-v3, diarisering, punktuasjon).

## 7 Oppsummering og konklusjon (NY — anbefales)
**Tilskudd fra ASR:** kort om at pipelinen oppfyller funksjonelle krav, leverer målbar WER-reduksjon, og demonstrerer flerspråklig design.

## Litteraturliste
**Innhold fra `bibliografi.md`** flettes inn i felles referanseliste.

## Vedlegg
- ASR pipeline kildekode → link til GitHub-repo (anbefalt).
- Full ablation-tabell (alle WER-kjøringer).
- Eksempler på MCQ-prompts (faktiske prompts brukt i `entity_corrector.py`).
- Eksempler på korreksjoner (50–100 fra én halvkamp, med flagg lyktes/feilet).

---

## Innfletting — sammendrag

| Eksisterende seksjon | Hva flettes inn |
|---|---|
| 1.4 Bakgrunn og motivasjon | ASR §1.1 — 1-2 avsnitt om navnefeil og søk |
| 1.5.2 Problemstilling | ASR §1.2 — utvidet problemformulering |
| 1.5.3 Målgrupper | + Forzasys, sluttbrukere, forskere |
| 1.6 Gruppens mål | + 3 ASR-relaterte produktmål |
| 2.1 Begreper | + 8 ASR/NLP-begreper |
| 2.2.2 Backend | ASR-pipelinen som backend-komponent + modulkart-figur |
| 2.3 Rammeverk & verktøy | ASR-verktøytabell |
| 3.1 Planlegging | + iterativ ASR-utviklingsavsnitt |
| 3.2.2 Funksjonelle krav | + 4 ASR-funksjonelle krav |
| 3.2.3 Ikke-funksjonelle krav | + 4 ASR-ikke-funksjonelle krav |
| 3.2.4 Tekniske krav | + ASR-avgrensninger |
| **(ny) 4 Systembeskrivelse** | Hele ASR-arkitekturen + designvalg |
| 4 / (ny) 5 Testing | + ASR-validering, metrikker, ablation-tabell |
| **(ny) 5/6 Diskusjon** | ASR-trade-offs, generaliserbarhet, videre arbeid |
| **(ny) 6/7 Konklusjon** | ASR-bidrag oppsummert |
| Kilder | Slå sammen med `bibliografi.md` |
| Vedlegg | Kode, prompts, eksempler |

---

## Praktiske råd for innflettingen

1. **Ikke flytt 10 000-ords-frontend-teksten.** Den er allerede ferdig — vi føyer ASR-tekst inn i nye underseksjoner ved siden av.
2. **Bruk `01_*.md` … `05_*.md` som råmateriale**, ikke som ferdige kapitler. Klipp ut avsnitt og lim inn på rett sted i hovedfila.
3. **Figurer er viktig** (jf. OsloMet-tipsene). Lag minst:
   - Pipeline-flytdiagram (audio → cleaned JSON, alle trinn).
   - Modulkart (hvilke `pipeline/*.py`-filer som hører til hvilket trinn).
   - WER-resultattabell (når ferdig).
   - Eksempel-flyt: én segment gjennom entity_corrector (TF-IDF → MCQ → MLM-veto).
4. **Konsistens.** Hvis frontend-teksten bruker «vi har valgt», bruk samme stemme i ASR-tekstene. Ikke bytt mellom «forfatteren» og «vi».
5. **Anbefal å legge til Diskusjon + Konklusjon-kapitler** — OsloMet-veiledningen krever det, og deres TOC mangler det foreløpig.
6. **Abstract skrives sist** — når WER-tall er låst.
