# Snippet — Kapittel 3 (System/produktdokumentasjon): ASR-rensepipelinens arkitektur

> Limes inn som ny underseksjon i Kapittel 3, etter Whisper-avsnittet og før (eller integrert med) Elasticsearch-avsnittet.
> Erstatter den eksisterende, korte setningen om "egne metoder for datarensing og feilretting".
> Forklarer arkitekturen til ASR-rensepipelinen på et detaljnivå som passer en bachelorrapport — uten å gå i kildekoden.

---

## ASR-rensepipelinen

Selv om Whisper gir et godt utgangspunkt for transkripsjon av kommentatorlyd, viste det seg gjennom prosjektet at rå Whisper-output har systematiske feil som direkte rammer søkefunksjonaliteten. Tre feiltyper dominerer: feilstavede egennavn (særlig spillernavn), hallusinerte segmenter i passasjer med stille eller publikumsstøy, og påfølgende segmenter som gjentar samme setning. Vi utviklet derfor en flertrinns pipeline som tar rå Whisper-JSON som input og produserer en renset versjon klar for indeksering i Elasticsearch.

Pipelinen er bygget rundt prinsippet om at **én komponent skal ha ansvaret for én bekymring**. I stedet for én stor monolittisk modul som forsøker å rette alt på en gang, deler vi behandlingen opp i selvstendige trinn som hver kan utvikles, testes og evalueres uavhengig. Ingen komponent kan overstyre en annen sin avgjørelse — i stedet bygger pipelinen videre på det forrige trinnet, og senere trinn respekterer hva tidligere trinn har bestemt.

### Trinnoversikt

Pipelinen består av fire hovedtrinn etter Whisper-transkriberingen:

1. **Strukturell renseing.** Først filtreres åpenbart hallusinerte segmenter bort, og påfølgende, nesten-identiske segmenter slås sammen. Dette gjøres med deterministiske regler — for eksempel at segmenter med svært lav andel bokstavtegn, eller på et språk som ikke er kampens hovedspråk, sannsynligvis er hallusinasjoner.
2. **Domenenormalisering.** Deretter normaliseres tall- og tidsuttrykk ("i sekstisjuende minutt" → "i 67. minutt"). Dette er rene mønstertilpasninger og bruker ingen håndskrevne ordlister med personnavn.
3. **Entitetskorrigering.** Hovedtrinnet i pipelinen. Her identifiseres mulige egennavn i transkripsjonen, og hvert kandidatord vurderes mot en kampspesifikk gazetteer (lagliste, dommer, stadion). Korrigeringer som kommer gjennom flere uavhengige sjekker, anvendes; de andre kastes.
4. **Konfidensbasert språkmodellrensing.** Til slutt brukes per-token konfidensverdier fra Whisper til å markere bare lav-konfidens-tokens for fri-tekst-rensing av en mindre språkmodell. Tokens som tidligere trinn allerede har bestemt seg for, fryses og kan ikke endres her — slik unngår vi at språkmodellen overskriver gode entitetskorrigeringer.

### Entitetskorrigering i detalj

Trinn 3 er det mest kritiske, og fortjener en grundigere beskrivelse. Egennavnsfeil er den enkeltfaktoren som har størst negativ effekt på søkbarheten i Elasticsearch — én feil bokstav i "Sterling" kan gjøre at klippet ikke gjenfinnes når en bruker søker på spillerens navn. Samtidig er en naiv tilnærming farlig: hvis vi bare bytter ut alle ord som ligner på et navn i laglisten, vil vi rette mange ord som faktisk er korrekte.

For å unngå dette kombinerer entitetskorrigeringen fire teknikker som må være enige før en korreksjon anvendes:

- **Kandidatoppslag.** Hvert ord som kan være et navn (basert på syntaktisk rolle i setningen) slås opp i den kampspesifikke gazetteeren. Oppslaget gjøres med en TF-IDF-indeks over karakter-bigrammer, som er effektiv for stavefeil. Topp fem nærmeste navn returneres.
- **Snarveiavgjørelse.** Hvis det øverste treffet er svært likt (over 90 % cosinus-likhet) og klart bedre enn det nest beste, aksepteres korreksjonen umiddelbart. Hvis det øverste treffet er svært forskjellig (under 40 %), forkastes hele kandidaten umiddelbart. Dette håndterer de åpenbare tilfellene raskt og uten språkmodellkall.
- **Diskriminativ vurdering.** For mellomtilfellene formuleres et flervalgsspørsmål til en liten språkmodell: "Det opprinnelige ordet er X. Konteksten er Y. Velg det riktige navnet: A, B, C, eller behold originalen." Modellen returnerer én bokstav. Dette mønsteret er hentet fra forskning som viser at diskriminative formuleringer gir mindre overretting enn å la modellen generere fritt.
- **Kryssverifisering.** Etter at språkmodellen har valgt en kandidat, bes en uavhengig modell (en flerspråklig masked language model) om å vurdere om det valgte navnet er språklig plausibelt i konteksten. Hvis denne uavhengige modellen ikke har det valgte navnet blant sine topp-kandidater, vetoes korreksjonen og originalen beholdes.

I tillegg holder pipelinen en valideringscache som lærer over flere kjøringer. En korreksjon må aksepteres på minst tre forskjellige kamper før den promoteres til en global, kjøring-uavhengig cache. Dette beskytter mot at en enkeltfeilkjøring permanent forgifter fremtidige korrigeringer.

### Datastrukturen i utdataen

Pipelinen produserer en utvidet JSON med samme grunnstruktur som Whisper-output, men med tilleggsfelter:

- `text` — den endelige rensede transkripsjonsteksten.
- `words` — per-token informasjon med opprinnelig konfidens fra Whisper bevart.
- `cleaning_metadata.corrections` — en logg over alle endringer som ble gjort, inkludert originalord, foreslått korreksjon, kilde (hvilket trinn som foreslo den) og hvor sikker pipelinen var.
- `temporal_chunks` — sammenslåtte tidsavgrensede tekstklumper som er klare for indeksering i Elasticsearch.

Denne utvidede strukturen gjør det mulig for både søkemotoren og en menneskelig revisor å forstå akkurat hvilke endringer pipelinen har gjort og hvor sikker den var i hver enkelt avgjørelse.

### Designvalg som ikke ble valgt

Underveis i utviklingen vurderte vi flere alternative arkitekturer som ble forkastet:

- **En stor LLM som rensesteg.** Vi vurderte å la en stor språkmodell lese hele segmentet og foreslå korreksjoner fritt. Dette gir kraftig korrigering, men har en kjent feilmodus der modellen omformulerer korrekt tekst og introduserer nye feil. Vi valgte heller den diskriminative MCQ-tilnærmingen, som begrenser modellens handlingsrom.
- **Manuelle regler per spillernavn.** Et alternativ var å skrive om håndspesifikke regler for de hyppigste navnefeilene. Dette ville fungert for én kamp, men ikke skalert til andre kamper eller språk. Den retrieval-baserte tilnærmingen henter automatisk gazetteeren fra hver enkelt kamps lagliste.
- **Statiske ordlister med "vanlige ord" som ikke skal korrigeres.** En vanlig løsning i ASR-rensesystemer er å vedlikeholde en svarteliste over vanlige ord (the, og, is) som aldri skal korrigeres. Vi forkastet dette fordi det ikke skalerer over flere språk. I stedet bruker vi språkmodellens lærte sannsynlighetsfordeling som plausibilitetssjekk.
