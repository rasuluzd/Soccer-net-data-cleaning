# Diskusjon: Pipeline-tuning og overfitting på v2-støy

> **Status:** Lagret 2026-05-16 som diskusjonsmateriale til bacheloroppgaven.
> Empirisk funn fra Chelsea-Liverpool 2016 (GOAL English benchmark).

---

## Det empiriske funnet

Pipelinen forbedrer ASR-transkripsjonen når input er rå SoccerNet-stock Whisper
(her kalt **v2**), men gjør den marginalt verre når input er en bedre
re-transkripsjon med faster-whisper-large-v3 + lineup-biasing (her kalt **v3**).

Måling: corpus-nivå WER mot GOAL human ground truth, normalisert tekst,
ingen tids-aligneringsavhengighet.

| Input | Half 1 raw WER | Half 1 cleaned WER | Δ | Half 2 raw WER | Half 2 cleaned WER | Δ |
|---|---|---|---|---|---|---|
| **v2 (stock)** | 29.71% | **28.16%** | **−1.55pp** ✓ | 24.68% | **23.37%** | **−1.31pp** ✓ |
| **v3 (re-transkr.)** | 25.36% | 25.48% | +0.12pp ✗ | 23.64% | 23.88% | +0.24pp ✗ |

Pipelinen er altså *netto skadelig* når input allerede er av høyere kvalitet.

---

## Hvorfor skjer dette? — diagnose

### 1. Terskelverdier kalibrert mot v2-feilmoder

Stage E sine terskler (`MCQ_MIN_FUZZ_TO_INVOKE=65`, `SHORTCUT_ACCEPT_TFIDF=0.90`,
`CONSERVATIVE_C1_FUZZY_FLOOR=60`) ble valgt under iterativ tuning mot v2-output
hvor typiske feilmoder var:

- `Storage → Sturridge` (fuzz ~50)
- `Kommer → Kouame` (fuzz ~20)
- `Lovern → Lovren` (fuzz ~80)
- `Coutinho → Coutigna` (fuzz ~75)

Disse er reelle mishørings-feil med klare gazetteer-kandidater. Pipelinens
gates fanger dem effektivt.

### 2. v3 har færre faktiske feilstavinger, men flere kandidater

Per-halv telling fra Stage E telemetri:

| Metrikk | v2 input | v3 input (estimat) |
|---|---|---|
| Detected entities (NER) | 875 | ~1100 (mer capitalized pga bedre tegnsetting) |
| MCQ trigger | 24 | 30+ |
| MCQ accepted | 2 | 4-6 |
| Andel kandidater som er ekte feil | ~30 % | ~5-10 % |

v3 har bedre tegnsetting (fra `condition_on_previous_text=True`) → flere
setningsstarter → flere capitalized tokens → flere NER-deteksjoner. Men de
fleste av disse "kandidatene" er allerede korrekte ord. Når true-negative-raten
øker, mens gatene er kalibrert for en false-positive-rate basert på v2,
slipper flere feilkorreksjoner gjennom.

### 3. Step L's "punktuasjons-edits" konflikter med v3 sin tegnsetting

Step L sine 29 edits på v2 var 72 % bare komma↔punktum-bytter. På v3 hvor
tegnsettingen allerede stemmer bedre med ASR-kvalitet, vil tilsvarende edits
oftere bryte med ground truth sin tegnsetting → flere "endrede ord" i
WER-tellingen.

### 4. Step P (oliverguhr fullstop) re-punktuerer på toppen

Step P er konservativ (`PUNCT_PRESERVE_EXISTING=True`, kun innsetting), men
selv konservative casing- og kommainnsettinger kan bryte token-grenser når
GT bruker en annen styling-konvensjon. På v2 var det få konflikter pga få
eksisterende kommaer å bevare; på v3 er det flere.

### 5. Frozen entity protection mister grunnlag uten avg_logprob

Da schema-2-metadata ble fjernet, mistet Step L også per-ord avg_logprob.
Pipelinen kan dermed ikke lenger differensiere mellom "Whisper er sikker på
dette ordet" og "Whisper gjettet". Resultat: Step L behandler alt som
lav-confidence, prøver å redigere alt, og oppfører seg som om input var støyfull.

---

## Konkrete forbedringsstrategier for v3

### Lavt-hengende frukt

1. **Heve `MCQ_MIN_FUZZ_TO_INVOKE` fra 65 til 70-75 når input er v3-kvalitet.**
   Filtrerer ut grenseverdier hvor pipelinen ofte tar feil valg på allerede
   korrekte ord.

2. **Heve `FREQUENCY_HEURISTIC_THRESHOLD` fra 5 til 3.**
   Hvis et token forekommer ≥3× i en match er det nesten alltid et reelt
   vanlig ord, ikke et navn å korrigere.

3. **Aktivere `MLM_VETO_RATIO` strammere (1.5 → 1.2) på v3-input.**
   xlm-roberta får vetorett over Stage E sine MCQ-valg ved svakere preferanse
   for original — flere v3-korrigeringer blir rullet tilbake.

### Strukturelle endringer

4. **Variant-spesifikke konfigurasjoner.**
   ```python
   THRESHOLD_OVERRIDES = {
       "_v3": {
           "MCQ_MIN_FUZZ_TO_INVOKE": 75,
           "SHORTCUT_ACCEPT_TFIDF": 0.92,
           "MLM_VETO_RATIO": 1.2,
       },
   }
   ```
   Lar pipelinen dynamisk justere seg basert på `ASR_INPUT_VARIANT`.

5. **Adaptiv noisy-detection.**
   Beregne en "noise score" på input (mean alpha-ratio, hallucination cluster
   count, duplikat-rate). Hvis lavere enn terskel → konservative gates.
   Hvis høyere → standard gates. Dette gjør pipelinen selv-justerende uten
   per-variant flagg.

6. **Restaurere per-ord avg_logprob bare for Step L.**
   Holde JSON-output enkelt (list-format `[start, end, text]`), men la
   `whisper_runner` skrive en parallell `*_words.json` med per-ord-probs som
   Step L valgfritt kan lese. Da kan v3 dra nytte av confidence-gating uten
   å forurense hoved-pipelinen.

7. **Frozen-token-protection på MCQ-aksepterte navn.**
   Stage E setter allerede `frozen_word_indices` på korrigerte navn. Verifisere
   at Step L respekterer dette FULLT (inkl. for omkringliggende kontekst-edits),
   og utvide til v3-kandidater hvor MCQ valgte "keep original".

### Konseptuelle forbedringer

8. **Quality-aware skipping av Stage E.**
   Hvis input-WER kan estimeres lavt (f.eks. via OOV-rate mot gazetteer), kan
   Stage E *helt skippes*. På Chelsea-Liverpool v3 ville dette spart 39s og
   gitt 0.12pp bedre WER.

9. **Re-tuning av validated_corrections-cache for v3.**
   Den nåværende cachen (80 entries) er kuratert fra v2-kjøringer. Mange
   mappinger (`storage → Sturridge`) er v2-spesifikke ikke-feil i v3.
   Sourcere fra fresh v3-kjøringer ville gitt mindre, mer presis cache.

10. **Ablasjons-tabell per variant.**
    Kjøre per-stage ablasjon (raw → +E → +L → +P) på begge varianter, dokumentere
    hvilke stadier som er netto-positive på v3. Sannsynlig resultat:
    - **E alene:** netto-positiv på v3
    - **+L:** netto-negativ (slå av for v3)
    - **+P:** marginalt på v3

---

## Hva dette sier om generaliserbarhet

Funnet er metodologisk viktig fordi det belyser en sentral begrensning ved
ASR-cleaning pipelines: **terskler tunet på én ASR-kvalitetsnivå generaliserer
ikke automatisk til andre nivåer**. Dette har konsekvenser for:

1. **Cross-language overføring.** Hvis pipelinen tunes på engelsk EPL og
   anvendes på svensk Allsvenskan (med ulik ASR-modell og ulik feilfordeling),
   vil samme overfitting-effekt oppstå. Adaptive terskler er nødvendig.

2. **Cross-time overføring.** Når Whisper-modeller blir bedre (large-v3 →
   v4 → ...), reduseres rom for pipelinens nytte. Funnet vårt er en
   forsmak: jo bedre ASR, desto mer presisjon kreves av cleaning-laget.

3. **Søknad mot annet domene.** Hvis pipelinen brukes på annen sport eller
   ren tale (nyheter, podcasts) hvor entity-mishørings-frekvensen er lavere,
   vil samme effekt oppstå.

---

## Forslag til oppgave-diskusjon

Anbefalt struktur i diskusjonskapitlet:

1. **Presenter funnet ærlig** med tabellen øverst. Ikke skjul at v3 cleaned
   er marginalt verre.
2. **Forklar mekanismen** (tuning mot v2-feilmoder, økt false-positive-rate
   på cleanere input).
3. **Vis at det er forutsigbart** ut fra teorien om overfitting og noisy
   channel models — ikke en bug i implementasjonen, men en designkonsekvens.
4. **Liste 3-5 konkrete forbedringer** (lavt-hengende frukt + adaptive
   terskler + variant-overrides).
5. **Diskuter den bredere lærdommen**: pipeline-validering må gjøres mot
   *input-distribusjonen den vil møte i produksjon*, ikke bare mot det
   som var tilgjengelig under utvikling.

Dette er en god akademisk finding fordi det:
- Er empirisk dokumentert med tall fra reproduserbart benchmark
- Belyser en kjent generell svakhet (overfitting) i et nytt domene
- Gir konkrete forslag til videre arbeid
- Demonstrerer modenhet — du innrømmer en svakhet og analyserer den
  i stedet for å skjule den

---

## Reproduserbar fra kommandolinjen

```bash
# Compute corpus-WER for begge varianter
python -c "
import json, re
from jiwer import wer

def t(p):
    d = json.load(open(p, encoding='utf-8'))
    s = list(d['segments'].values())
    return re.sub(r'\\s+',' ', re.sub(r'[^a-z0-9 ]',' ',
        ' '.join(x[2] if isinstance(x,list) else x['text'] for x in s).lower())).strip()

base = 'path/to/SoccerNet/.../commentary_data'
clean = 'cleaned_data/.../commentary_data'
for half in (1,2):
    gt = t(f'{base}/{half}_asr_corrected.json')
    for label, p in [('v2 raw', f'{base}/{half}_asr.json'),
                     ('v2 cleaned', f'{clean}/{half}_asr_cleaned_withL.json'),
                     ('v3 raw', f'{base}/{half}_asr_v3.json'),
                     ('v3 cleaned', f'{clean}/{half}_asr_v3_cleaned.json')]:
        print(f'h{half} {label}: WER={wer(gt, t(p))*100:.2f}%')
"
```

---

## Relaterte filer

- `evaluation_results/2016-09-16_-_22-00_Chelsea_1_-_2_Liverpool_half*_wer.md`
  — original WER-rapporter (v2 cleaned vs raw)
- `pipeline/config.py` — alle terskler omtalt over
- `pipeline/entity_corrector.py` — Stage E gate-implementering
- `README.md` — Step L ablation-funnet (relatert observasjon)
