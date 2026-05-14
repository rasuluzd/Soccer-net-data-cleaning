# Snippet — Kapittel 5 (Testdokumentasjon): ASR-rensepipelinen

> Limes inn som underseksjon i Kapittel 5 (Testdokumentasjon).
> Dekker testplanlegging, testtyper, evalueringsmetrikker og generelt opplegg for resultatanalyse.
> Selve resultattabellen står som plassholder fordi tallene endres når pipelinen kjøres på nytt.

---

## 5.x Testing av ASR-rensepipelinen

Testingen av ASR-rensepipelinen skiller seg fra brukertestingen av frontend ved at det er **transkripsjonens kvalitet**, ikke brukerens opplevelse, som skal valideres. Vi bruker derfor andre teststrategier og målemetrikker. Testingen er delt inn i tre nivåer som speiler den vanlige inndelingen i programvareutvikling.

### 5.x.1 Testtyper

**Enhetstesting.** Hver modul i pipelinen har en tilhørende testfil som validerer modulens oppførsel isolert. Dette inkluderer hallusinerte segmenter (skal filtreres), nesten-identiske segmenter (skal slås sammen), feilstavede spillernavn (skal korrigeres når kandidaten er klar nok) og kandidater som er like nok til vanlige engelske ord til at de ikke bør korrigeres (skal ikke korrigeres). Hver testfil har mellom 8 og 24 testtilfeller, og hele suiten kjøres på under et halvt minutt på CPU. Tester gjør bruk av "monkey-patching" for å unngå å laste tunge språkmodeller i testkjøringene.

**Integrasjonstesting.** Hele pipelinen kjøres ende-til-ende på en hel halvkamp av Chelsea-Liverpool 2016-09-16, og utdataen sammenliknes mot referansen fra GOAL-benchmarken. Dette validerer at modulene faktisk fungerer sammen og at ingen modul forstyrrer en annen sin avgjørelse.

**Systemtesting.** Den endelige rensede transkripsjonen indekseres i Elasticsearch, og noen forhåndsdefinerte søkespørringer kjøres mot indeksen. Dette validerer at hele bakgrunnsflyten fra audio til søkbart resultat fungerer som forventet.

### 5.x.2 Evalueringsmetrikker

For å måle kvaliteten på pipelinens arbeid bruker vi tre komplementære metrikker:

**Word Error Rate (WER).** Standardmetrikken for ASR-evaluering. Beregnes som summen av substitusjoner, slettinger og innsettinger som trengs for å gjøre hypotesen lik referansen, normalisert med antall ord i referansen. Vi bruker biblioteket `jiwer` for denne beregningen, som implementerer dynamisk programmering for optimal alignment.

**Character Error Rate (CER).** Tilsvarende WER, men på karakter-nivå. Mer sensitiv til mindre stavefeil i navn — én bokstav forskjell mellom "Klein" og "Clyne" gir 12 % CER-bidrag, men 100 % WER-bidrag for det enkelte ordet.

**Entity F1.** Direkte måling av om egennavnene i transkripsjonen matcher dem i referansen. Beregnes ved å ekstrahere alle PROPN/PERSON-entiteter fra både hypotese og referanse, fuzzy-matche dem på navn-nivå, og rapportere presisjon, recall og F1. Denne metrikken er mest direkte relevant for søkbarheten i Elasticsearch.

WER alene er ikke tilstrekkelig for dette domenet, fordi én bokstav i et navn kan være forskjellen mellom et søkbart og et usøkbart klipp — men teller som ett ord i WER, samme vekt som et hvilket som helst funksjonsord. Derfor rapporterer vi alle tre metrikkene parallelt.

### 5.x.3 Datagrunnlag for testene

Hovedtesten kjøres mot **GOAL-benchmarken** for kampen Chelsea–Liverpool 16. september 2016. GOAL-benchmarken har menneskelig annoterte referansetranskripsjoner, og er det eneste datasettet i prosjektet vårt som gir metodisk pålitelige WER-tall. For den svenske kampen AIK–Halmstad 9. november 2025 viste det seg at den tilgjengelige referansen var generert ved å rette stock Whisper-output manuelt, noe som gjør den biased mot Whisper-feil. Vi rapporterer derfor ikke WER-tall for svensk, men bruker den svenske kampen til å demonstrere at den språkagnostiske strukturen i pipelinen fungerer i praksis.

### 5.x.4 Alignment mellom hypotese og referanse

Et metodisk poeng som er verdt å nevne: når Whisper segmenterer kampen annerledes enn den menneskelige annotatoren, kan WER-beregningen straffe segmenteringsforskjellen som om det var en innholdsfeil. Vi har derfor implementert to alignment-metoder i evalueringsverktøyet vårt — én tradisjonell 1-til-1-alignment og én vindubasert mange-til-én-alignment. Den tradisjonelle 1-til-1-alignmenten brukes som standard, fordi den gir tall som er sammenliknbare med både online WER-verktøy og litteraturen.

### 5.x.5 Resultater (plassholder for endelige tall)

> Tabell 5.x.1 fylles ut etter at siste pipeline-kjøring og WER-evaluering er gjennomført.
> Tabellen skal vise raw Whisper baseline, hvert pipeline-trinns bidrag (ablation), og det endelige resultatet, for både halvdel 1 og halvdel 2 av Chelsea-Liverpool-kampen.
> Nåværende status (foreløpige tall): WER reduseres fra 29,81 % til ca. 28,65 % på halvdel 1, og fra 24,84 % til ca. 23,81 % på halvdel 2, med Entity F1 rundt 0,64.

### 5.x.6 Hva som ikke ble testet

Vi har ikke gjennomført formell brukertesting av selve den rensede teksten — for eksempel hvor mye lettere det er for en sluttbruker å gjenfinne en spesifikk kamphendelse i Elasticsearch med renset tekst kontra rå Whisper-output. Dette er en naturlig oppfølgingstest, men ble nedprioritert fordi vi ikke hadde tilgang til et representativt utvalg av Forzasys-brukere i prosjektperioden.

Vi har heller ikke testet pipelinen på flere enn tre kamper. Generaliserbarheten til hele SoccerNet-datasettet er sannsynlig basert på arkitekturvalget (kampspesifikk gazetteer, ingen håndskrevne regler), men ikke empirisk verifisert i denne rapporten.

### 5.x.7 Tilbakemelding fra oppdragsgiver

> Plassholder — fyll ut etter siste demo-/oppsummeringsmøte med Forzasys.
