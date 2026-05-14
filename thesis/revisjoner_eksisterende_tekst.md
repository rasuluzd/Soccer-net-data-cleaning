# Mindre fikser i eksisterende tekst

> Småting jeg la merke til mens jeg leste igjennom dokumentet.
> Dette er ren korrektur — ikke nytt innhold.

---

## 1. Duplisert avsnitt om transkripsjonsfeil (i Whisper-seksjonen i Kap 3)

Det står to nesten-identiske avsnitt etter hverandre:

> Selv med direkte transkribering fra orginalspråket oppstår det likevel feil i dataen, navn på spillere, klubber og stadioner får problemer, men den generelle teksten var mye bedre. Dette gjorde det nødvendig å utvikle egne metoder for datarensing og feilretting av tekstdataen, slik at søkefunksjonaliteten kunne gi riktige resultater
> Selv med direkte transkribering fra originalspråket oppstår det likevel feil i ASR-utdata, særlig i navn på spillere, klubber og stadioner. Dette gjorde det nødvendig å utvikle egne metoder for datarensing og feilretting av tekstdataen, slik at søkefunksjonaliteten kunne gi pålitelige resultater.

**Foreslått fiks:** Behold bare den andre versjonen (bedre språkflyt, korrekt staving av "originalspråket", "pålitelige" framfor "riktige"). Den nye ASR-arkitektur-seksjonen kan kobles direkte etter den.

## 2. Duplisert avsnitt om ForzaSearch-bruk (i Kap 3)

To nesten-identiske avsnitt om innlogging og søkebar:

> Ved bruk av systemet starter brukeren med å logge inn. Etter innlogging presenteres hovedgrensesnittet, som består av en sentral søkebar og et sidefelt. Søkebaren fungerer som en åpen input hvor brukeren kan skrive inn forespørsler i naturlig språk, for eksempel «takling av Isherwood», «Granats mål mot AIK» eller «rødt kort i andre omgang». Systemet er designet slik at brukeren kan uttrykke seg på samme måte som i en vanlig samtale.
> I møte med systemet starter brukeren med å logge inn. Etter innlogging presenteres hovedgrensesnittet, som består av en sentral søkebar og et sidefelt. Søkebaren fungerer som en åpen felt for inndata hvor brukeren kan skrive inn forespørsler, for eksempel «takling av Isherwood», «Granaths mål mot AIK» eller «rødt kort i kampen Degerfors mot Brommapojkarna». Systemet er designet slik at brukeren kan utrykke seg på samme måte som vanlig samtale, så lenge det er presist mot en hendelse i kampen.

**Foreslått fiks:** Slå sammen til én versjon. Den andre er litt mer presis ("åpen felt for inndata" og "presist mot en hendelse"), men har stavefeil ("utrykke" → "uttrykke"). Pass også på navnestavinger: "Granats" eller "Granaths" — velg én.

## 3. Duplisert avsnitt om produksjonssetting

> I en produksjonssetting vil ForzaSearch fungere som en kontinuerlig oppdatert søketjeneste. Nye kamper behandles gjennom en datarørledning (pipeline) som inkluderer transkripsjon av lyd (Whisper), datarensing og indeksering i Elasticsearch. Når nye data er indeksert, blir de automatisk tilgjengelige for søk i systemet.
> I en produksjonssetting vil ForzaSearch fungere som en kontinuerlig oppdatert søketjeneste. Nye kamper behandles gjennom en datarøreledning(pipeline) som inkluderer transkripsjon av lyd via Whisper, datarensing og indeksering i ElasticSearch. Når nye data er indeksert, blir de automatisk tilgjengelig for søk i systemet.

**Foreslått fiks:** Behold den første. Den andre har "datarøreledning" (skrivefeil) og inkonsekvent "ElasticSearch" vs "Elasticsearch".

## 4. Mindre staveting / typografi

| Side | Problem | Fiks |
|---|---|---|
| 1.4 | "spisset og gitt oss praktisk erfaring med teknologi som er utbredt i IT og teknologibransjen, noe vi ser på verdifullt" | "noe vi ser på som verdifullt" |
| 1.5.2 | "Tredimensjonale konvolusjonsnevrale nettverk (3D CNN) har vist seg effektive for å gange opp ulike egenskaper" | "for å fange opp" (ikke "gange") |
| 1.6 | "Whsiper" | "Whisper" |
| 2.2.1 | "TypeScripts" | "TypeScript" |
| 2.2.1 | "HTLM-koden" | "HTML-koden" |
| 1.3 | E-post for Pål Halvorsen er oppgitt som "paalh@forzasys.com" — sjekk om dette er rett (hørtes ut som intern OsloMet-veileder med @oslomet.no?) |
| Forord | "Forazsys" | "Forzasys" |
| Thomas' rollebeskrivelse | "navn p spillere" | "navn på spillere" |

## 5. Inkonsekvent navngivning

- Veilederen heter "Mehdi H. Sarkhoosh" på forsiden, "Mehdi H. Sarkoosh Houshmand" i forordet og "Mehdi H. Sarkoosh" i 1.2. Velg én staving og bruk den konsekvent.
- "ForzaSys" / "Forzasys" / "Forazsys" forekommer alle tre — velg én skrivemåte.
- "ElasticSearch" / "Elasticsearch" — den offisielle skrivemåten er "Elasticsearch".

## 6. Innholdsfortegnelse mangler kapitler

Den limte innholdsfortegnelsen stopper ved "4 Prototypeutvikling og brukertesting", men resten av dokumentet inneholder utkast til Kapittel 5 (Testdokumentasjon), Kapittel 6 (Resultater), Kapittel 8 (Oppsummering og konklusjon), Kapittel 9 (Referanser) og Kapittel 10 (Vedlegg). Innholdsfortegnelsen bør oppdateres når disse skrives ferdig — og merk at det står Kapittel 6 og Kapittel 7 begge med tittel "Resultater" lenger ned i dokumentet (et av dem skal nok være "Diskusjon og refleksjon").
