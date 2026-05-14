# Kapittel 2 — Bakgrunn og relatert arbeid

## 2.1 Automatisk talegjenkjenning (ASR) og Whisper

Moderne ASR er dominert av encoder-decoder Transformer-modeller trent på store mengder lyd-tekst-par. **OpenAI Whisper** (Radford et al. 2023) ble trent på 680 000 timer flerspråklig lyd og er åpent tilgjengelig i flere størrelser (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`). For dette prosjektet brukes `faster-whisper` (Klein 2023), en CTranslate2-basert reimplementasjon som er 4–8× raskere på CPU enn referanseimplementasjonen.

Whisper har kjent svake sider på domener med høy egennavntetthet. Modellen er trent primært på generelle webdata, der ord som "Sterling" oftere refererer til valuta enn til fotballspilleren Raheem Sterling. Den genererer derfor systematisk ord som er statistisk vanlige i treningsdataene, selv når lydsignalet entydig peker på en mindre vanlig egennavn.

I tillegg har Whisper en kjent **hallusinasjonstendens** (Koenecke et al. 2024): i passasjer med stille eller publikumsstøy genererer modellen plausibel tekst som ikke er sagt. Dette håndteres i prosjektet med en deterministisk filtreringsmodul (`pipeline/hallucination_filter.py`).

## 2.2 Domeneadaptasjon for ASR

Tre hovedstrategier for å forbedre ASR på et spesifikt domene:

1. **Initial prompt** — Whisper aksepterer en prompt-streng som conditioner dekoderen. Dette brukes i prosjektet med kampspesifikke data (lagnavn, spillerliste).
2. **N-best rescoring** — Beam search produserer flere kandidat-transkripsjoner; en nedstrøms modul velger den som best passer domenet. Diskutert som videre arbeid.
3. **Etterfølgende korreksjon** — Den valgte hovedstrategien i denne oppgaven: la Whisper gjøre 1-best dekoding, korrigér feil etterpå.

## 2.3 Generativ feilkorreksjon (GER)

GER er den dominerende paradigmen for ASR-rensing siden 2023. Ideen er å bruke en stor språkmodell til å lese Whisper-outputen og foreslå korreksjoner basert på språklig plausibilitet.

- **Whispering-LLaMA** (Radhakrishnan et al. 2023, EMNLP). Bruker LLaMA fine-tunet på par av (ASR-output, korrekt transkripsjon). Demonstrerer betydelig WER-reduksjon, men krever LLM-fine-tuning og store modeller.
- **Apple RAG-NEC** (Sun et al. 2024, arxiv:2409.06062). Kombinerer TF-IDF retrieval med LLM-rerangering. Rapporterer 33–39 % WER-reduksjon på entitetstunge spørringer. Sentral inspirasjon for entity_corrector i denne oppgaven.
- **Confidence-Guided Error Correction** (Zhang et al. 2025, arxiv:2509.25048). Bruker per-token log-sannsynligheter fra Whisper til å markere bare lav-konfidens-tokens for LLM-korreksjon. Rapporterer 68 % relativ WER-reduksjon. Implementert i prosjektet som Step L.

## 2.4 Diskriminative korreksjonsmønstre

Tradisjonell GER er **generativ**: LLM-en blir bedt om å skrive om hele setningen. Dette er kraftig, men risikerer over-korreksjon når LLM-en omformulerer korrekt tekst.

**DeRAGEC** (ACL 2025) introduserer det diskriminative mønsteret: i stedet for å la LLM-en generere fritt, gir man modellen et flervalgsspørsmål — "Var det opprinnelige ordet `Klein` eller én av disse kandidatene: A) Clyne, B) Klein, C) Klint?" — og lar modellen velge. Dette begrenser handlingsrommet og reduserer hallusinasjon.

Dette mønsteret er kjernen i `entity_corrector.py` (Steg 4.4).

## 2.5 Retrieval-augmentert oppslag

For at LLM-en skal kunne velge korrekt navn, må den ha tilgang til kandidatlisten. Denne hentes via retrieval over en kampspesifikk gazetteer (lagliste + dommer + stadion).

To familier av retrievers er relevante:

- **Sparse (TF-IDF, BM25)** — Effektive for staving og morfologisk likhet. `sklearn.feature_extraction.text.TfidfVectorizer` med `analyzer="char_wb"` og `ngram_range=(2, 4)` brukes i prosjektet.
- **Dense (sentence-transformers)** — Embeddings fra modeller som `all-MiniLM-L6-v2` eller `paraphrase-multilingual-MiniLM-L12-v2`. Bedre for semantisk likhet, men dårligere for ren stavelikhet.

Prosjektet bruker sparse TF-IDF over karakter-bigrammer/quadgram fordi det er stavelikhet (ikke semantisk likhet) som dominerer feiltypen.

## 2.6 Entitetsoppslag og NER

Før man kan korrigere et ord, må man identifisere at det er et navn. Dette gjøres i `pipeline/ner_extractor.py` med to komplementære teknikker:

- **spaCy NER** — `en_core_web_sm` for engelsk, `xx_ent_wiki_sm` for andre språk. Henter entiteter med tag PERSON, ORG, GPE, FAC.
- **POS-baserte heuristikker** — Token som er PROPN (proper noun) ifølge spaCy POS-tagger, men som ikke ble fanget av NER-modellen, behandles også som kandidatentiteter.

Begge metoder brukes side om side fordi NER-modellene som er små nok til å kjøre på CPU har begrenset rekkevidde, særlig for navn fra ikke-engelske kontekster.

## 2.7 MLM-veto (XLM-RoBERTa)

Selv etter at LLM-en har valgt en kandidat, kan valget være feil. En andre opinion fra en uavhengig modell — en masked language model — gir billig validering.

**XLM-RoBERTa-base** (Conneau et al. 2020) er en flerspråklig MLM som returnerer en sannsynlighetsfordeling over hele vokabularet for et maskert token. I `entity_corrector.py` brukes pseudo-likelihood: vi maskerer det aktuelle ordet i kontekst og spør "hvilket ord passer best her?". Hvis MLM-ens topp-K ikke inkluderer LLM-ens valg, vetoer vi korreksjonen og beholder originalen.

Denne mekanismen er språkagnostisk og krever ingen håndskrevne ordlister — den lærte sannsynlighetsfordelingen fungerer som en plausibilitetssjekk på alle språk modellen ble trent på (100+ språk).

## 2.8 Evaluering av ASR-rensing

Standardmetrikken er **Word Error Rate (WER)**, definert som Levenshtein-avstand på ord-nivå normalisert med antall ord i referansen. WER beregnes med biblioteket `jiwer`, som implementerer dynamisk programmering for optimal alignment mellom hypotese og referanse.

For dette prosjektet rapporteres også:

- **Character Error Rate (CER)** — Mer sensitiv til mindre stavefeil i navn.
- **Entity F1** — Andelen korrekt-stavede navn i hypotesen som matcher navn i GT, beregnet både på presisjon og recall.

WER alene er ikke tilstrekkelig for dette domenet, fordi én bokstav i et navn kan være forskjellen mellom et søkbart og et usøkbart klipp — men den teller som ett ord i WER, samme vekt som et hvilket som helst funksjonsord.

---

## Posisjonering av oppgaven

Sammenliknet med litteraturen kombinerer denne oppgaven elementer fra:

- Apple RAG-NEC (TF-IDF + LLM rerank)
- DeRAGEC (diskriminativ MCQ)
- Confidence-Guided Error Correction (logprob-gating)

— men i en arkitektur der hver komponent har én klart definert rolle, hvor ingen komponent kan overstyre en annens veto, og hvor en valideringscache lærer fra konsensus over tid. Forfatteren kjenner ikke til en publisert pipeline med akkurat denne sammensetningen, særlig ikke spesifikt for fotballkommentar.
