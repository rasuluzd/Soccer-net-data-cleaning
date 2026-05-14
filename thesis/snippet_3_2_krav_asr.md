# Snippet — 3.2.2 / 3.2.3 / 3.2.4 (krav knyttet til ASR-rensepipelinen)

> Limes inn som tillegg eller punktlister i de eksisterende kravseksjonene.
> Holdes på samme overordnede nivå som de andre kravene i 3.2.2.

---

## Tillegg til 3.2.2 Funksjonelle krav

I tillegg til de fem områdene som er beskrevet over, definerte vi en egen gruppe funksjonelle krav for ASR-rensepipelinen. Disse kravene gjelder hva pipelinen skal levere mellom rå Whisper-output og indeksering i søkemotoren:

- Pipelinen skal lese rå transkripsjoner fra Whisper og produsere en renset versjon i samme JSON-format, slik at integrasjonen mot Elasticsearch ikke krever endringer i søkemotoren.
- Pipelinen skal automatisk filtrere bort segmenter som er åpenbart hallusinerte (for eksempel segmenter dominert av ikke-latinske tegn eller på "feil" språk i forhold til kampens hovedspråk).
- Pipelinen skal slå sammen påfølgende segmenter som er nesten identiske, da Whisper av og til gjentar samme setning to eller tre ganger på rad.
- Pipelinen skal korrigere feilstavelser av spillernavn, lagnavn og dommernavn ved oppslag mot kampens offisielle laglister, og bare gjennomføre korreksjoner når flere uavhengige sjekker er enige om at korreksjonen er riktig.
- Pipelinen skal bevare per-token konfidensverdier fra Whisper i utdataen, slik at nedstrøms søk kan vekte resultater etter hvor sikker ASR-modellen var.
- Pipelinen skal logge hver enkelt korreksjon (originalord, foreslått ord, kilde, score) til en metadata-blokk i utdataen, for sporbarhet og senere ablation-analyse.

## Tillegg til 3.2.3 Ikke-funksjonelle krav

For ASR-rensepipelinen definerte vi i tillegg følgende ikke-funksjonelle krav:

- Pipelinen skal kunne kjøres på CPU uten krav om dedikert GPU. Dette er nødvendig for at Forzasys og andre brukere skal kunne kjøre løsningen på vanlige arbeidsstasjoner.
- Pipelinen skal ikke avhenge av kommersielle eksterne API-er (for eksempel OpenAI eller Anthropic). Alle modeller som brukes skal være åpent tilgjengelige med vekter som kan lastes ned, slik at løsningen er reproduserbar og personvernet ivaretatt.
- Alle terskelverdier og parametre skal samles på ett sted i en sentral konfigurasjonsfil. Inline-konstanter i forretningslogikken er ikke tillatt. Dette gjør det mulig å gjennomføre ablation-studier og å justere atferd uten å endre selve koden.
- Pipelinen skal være språkagnostisk i strukturen sin. Den skal ikke avhenge av håndskrevne ordlister per språk (for eksempel manuelle "vanlige ord"-filtre). Språkspesifikke valg skal styres av byttbare modeller, ikke av hardkodede lister.
- Hver feilretting i pipelinen skal følges av en automatisk regresjonstest som feiler før rettingen og passerer etter. Hele test-suiten skal kunne kjøres på under et halvt minutt på CPU.

## Tillegg til 3.2.4 Tekniske krav og avgrensninger

For ASR-rensepipelinen gjelder følgende tekniske avgrensninger som det er viktig å gjøre eksplisitt:

- Hovedevalueringen av rensekvalitet gjøres på engelsk fotballkommentar mot et utvalg fra GOAL-benchmarken, fordi dette er det eneste datasettet i prosjektet med menneskelig annotert referanse-transkripsjon. Andre språk inngår som demonstrasjon av at strukturen er språkagnostisk, men WER-tall rapporteres bare for engelsk.
- Pipelinen er optimalisert for fotballkommentar med tydelige spillernavn og lagnavn. Generaliserbarhet til andre sportsgrener (basketball, ishockey) er ikke testet og ligger utenfor prosjektets omfang.
- Lokal disk- og minnebruk er begrenset til det som passer på en vanlig bærbar (16 GB RAM, et titalls GB ledig disk). Dette begrenser hvilke språkmodeller som kan brukes, og motiverer valget av en liten LLM (Qwen 1.5B) framfor større alternativer.
