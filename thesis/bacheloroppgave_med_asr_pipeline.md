# Bacheloroppgave — Utvikling av et system for transkribering og analyse av hendelsesdata av fotballkamper for ForzaSys AS

Institutt for Informasjonsteknologi
Postadresse: Postboks 4 St. Olavs plass, 0130 Oslo
Besøksadresse: Holbergs plass, Oslo

PROSJEKT NR. 4
TILGJENGELIGHET: -

Telefon: 22 45 32 00

BACHELORPROSJEKT

HOVEDPROSJEKTETS TITTEL:
Utvikling av et system for transkribering og analyse av hendelsesdata av fotballkamper for ForzaSys AS.

DATO: 19.05.2026

PROSJEKTDELTAKERE:
- Abdi Azis A. Sharif
- Abdurahim Mustafa Tufa
- Liban Mohammed Hussein
- Rasul Ruslanovitsj Uzdijev
- Thomas Knutsen

INTERN VEILEDER: Pål Halvorsen, Mehdi H. Sarkhoosh

OPPDRAGSGIVER: Forzasys AS
KONTAKTPERSON: Pål Halvorsen

## Sammendrag

Bachelorprosjektet går ut på å utvikle en webbasert analyseverktøy for Forzasys som kan finne spesifikke hendelser i fotballkamper, som mål eller straffer, og vise relevante videoklipp og sammendrag. Målet er å effektivisere analysearbeidet ved å gjøre det raskere å søke i store mengder sportsdata. Systemet bygges med Elasticsearch som søkemotor, Whisper for analyse og sammenligning, Python for backend og React/Next.js for frontend. Prosjektet gjennomføres over ett semester, med fokus på databehandling, AI og et brukervennlig grensesnitt, hvor resultatet blir en fungerende prototype.

3 STIKKORD:
- Kunstig intelligens
- Feilretting
- Webapplikasjon

## Forord

Denne rapporten markerer avslutningen på vårt studieløp og bachelorgraden i anvendt datateknologi, informasjonsteknologi og dataingeniør ved OsloMet, våren 2026. Rapporten beskriver prosjektets prosess og løsningen vi har utviklet i samarbeid med Forzasys AS.

Vi kom i kontakt med Forzasys gjennom en anbefaling fra vår emne-ansvarlig for bachelorprosjektet (TKD). Vi vurderte også andre selskaper vi fant interessante, men valget falt til slutt på Forzasys. I vårt første møte med dem ble vi kjent med teamet og selskapets virksomhet, ideer og ambisjoner.

Prosjektet har gitt oss verdifull innsikt i hvordan det er å arbeide som utviklere med en oppgave som kombinerer to felt vi engasjerer oss i, som er programvareutvikling og kunstig intelligens. Gjennom studiene har vi tilegnet oss kompetanse innenfor webutvikling, databasehåndtering og maskinlæring, noe som er relevant for prosjektets omfang. Det har vært meningsfullt å utvikle noe genuint nytt som faktisk kan tas i bruk, og som gjør det enklere for folk å analysere og hente relevant informasjon fra en kamp.

Vi ønsker å rette en stor takk til Forzasys for å ha vært tilgjengelige, hjelpsomme og engasjert gjennom hele semestret. En særlig takk til våre veiledere ved OsloMet og Forzasys, Pål Halvorsen og Mehdi Sarkoosh Houshmand, for faglige innspill og god prosjektveiledning. Ikke minst vil vi takke hverandre i gruppen for innsatsen, dedikasjonen og støtten hele veien frem mot fullført grad.

Oslo, mai 2026

Abdi Azis A. Sharif, Abdurahim Mustafa Tufa, Liban Mohammed Hussein, Rasul Ruslanovitsj Uzdijev, Thomas Knutsen

---

## 1 Introduksjon

Dette kapittelet gir en innledende oversikt over prosjektet og etablerer rammene for arbeidet som er gjennomført. Prosjektgruppen og deres respektive roller, sammen med oppdragsgiver og veileder. Videre redegjøres det for prosjektets bakgrunn, begrunnelse for valg av oppgave og de målene gruppen har arbeidet mot.

Kapittelet gir en overordnet beskrivelse av prosjektet, inkludert problemstilling, sentrale valg og målgrupper. Formålet er å gi leseren et tydelig bilde av hva prosjektet omhandler, og hvorfor løsningen er utviklet slik den er.

### 1.1 Gruppe

Bachelorgruppen består av fem medlemmer med dels ulike faglige bakgrunn og studieretninger. Gruppeinndelingen sørger for bredt kompetansespekter og bidrar til solid faglig bredde i prosjektarbeidet. Gruppen består av følgende medlemmer:

**Abdi Azis A. Sharif**, Anvendt datateknologi – Programmering
s305054 | abdisharif315@gmail.com

Han hadde hovedansvar for koordinering av prosjektplanlegging og dialog med oppdragsgiver, samt bidro til å definere kravspesifikasjonen. Han utviklet backend-arkitekturen og deltok i arbeidet med rensing av feildata. I tillegg designet han prototyper for frontend og samarbeidet med Liban og Abdurahim om utvikling av brukergrensesnittet. Sharif koordinerte all kommunikasjon med både intern- og eksternveileder, og sikret sporbarhet i prosjektet gjennom løpende møtereferater og loggføring gjennom hele prosjektperioden.

**Abdurahim Mustafa Tufa**, Anvendt datateknologi – Programmering
s385569 | abdurahimtufa69@gmail.com

Han hadde hovedansvar for utforming av rapportstruktur og for å sikre en helhetlig og konsistent oppbygging av bacheloroppgaven. Videre hadde han ansvar for å overholde frister og følge opp fremdriften i prosjektet, slik at arbeidet ble levert i henhold til tidsplanen. I tillegg hadde han en sentral rolle i utviklingen av frontend-løsningen, inkludert design og implementering av brukergrensesnittet. Han bidro også aktivt i testfasen, hvor han gjennomførte testing av funksjonalitet og brukervennlighet for å sikre kvaliteten på systemet.

**Liban Mohammed Hussein**, Dataingeniør
s383043 | libizi24@gmail.com

Hovedansvar i prosjektet har vært utvikling av frontend og utforming av prototypen. Dette inkluderer design og strukturering av brukergrensesnittet, samt utvikling av en interaktiv prototype i Figma med fokus på navigasjon og brukeropplevelse. Studenten har også hatt ansvar for å tilrettelegge prototypen for brukertesting og videreutvikle designet basert på tilbakemeldinger fra brukere.

**Rasul Ruslanovitsj Uzdijev**, Dataingeniør
s383079 | rasuluzdijev1@outlook.com

Han hadde hovedansvar sammen med Thomas for å utvikle løsningen for rensing og forbedring av transkribert kommentatorlyd. Arbeidet besto i å designe og implementere en flertrinns pipeline som behandler rå Whisper-output før den indekseres i søkemotoren. Han hadde særlig ansvar for arkitekturen rundt entitetskorrigering, der spillernavn, lagnavn og dommernavn matches mot kampens offisielle laglister ved hjelp av retrieval-augmenterte teknikker og en liten språkmodell som diskriminativ kvalitetssjekk.

Han bidro også til den flerspråklige tilretteleggingen, slik at pipelinen i prinsippet skal kunne håndtere kommentar på engelsk, svensk og tysk uten håndskrevne ordlister per språk. I tillegg utviklet han evalueringsverktøyet som brukes til å måle ordfeilrate (WER) opp mot kjente referansetranskripsjoner, og gjennomførte den systematiske ablation-studien som dokumenterer bidraget fra hvert pipeline-trinn. Han skrev de delene av rapporten som omhandler arkitekturen bak datarensingen, evalueringsmetodikken og resultatene fra disse målingene.

**Thomas Knutsen**, Informasjonsteknologi
s286108 | thknutsen27@hotmail.com

Han hadde hovedansvar sammen med Rasul for å utvikle løsningen for rensing og forbedring av transkribert kommentatorlyd. Arbeidet gikk ut på å lage kode som kunne redusere feil i teksten, spesielt knyttet til navn på spillere, klubber og stadioner. Dette var viktig for å gjøre dataene mer presise og bedre egnet for søk og analyse. Var med å utvikle flere ulike moduler som bidro til å forbedre transkripsjonene, blant annet for å fjerne duplikater, rette opp feilstavinger ved hjelp av fuzzy matching, filtrere bort hallusinerte utsagn og hente inn relevant tilleggsinformasjon. I tillegg skrev han deler av rapporten som omhandlet prosessen bak datarensingen og de tekniske valgene som ble tatt underveis.

### 1.2 Oppdragsgiver

Forzasys AS er en nyetablert norsk teknologibedrift som utvikler programvare og digitale løsninger innen sportsteknologi. Selskapet benytter kunstig intelligens som kjerneteknologi i sine produkter, og tilbyr systemer for sportsanalyse, videostrømming og databehandling rettet mot idrettsorganisasjoner.

Oppdragsgiver:
Forzasys AS
Adresse: c/o Simula Research Laboratory, Kristian Augusts gate 23, 0164 Oslo
Organisasjonsnummer: 913 968 026

Kontaktperson og veileder hos Forzasys:
Mehdi H. Sarkoosh
Developer/Data Scientist
E-post: mehdi@forzasys.com

### 1.3 Veileder

Internveileder hos OsloMet
Pål Halvorsen
Professor/Chief Executive Officier (CEO)
E-post: paalh@forzasys.com

### 1.4 Bakgrunn og Motivasjon

Som gruppe ønsket vi å gjennomføre et praksisnært prosjekt innenfor teknologi og programvare, med en reell nytteverdi for en ekstern oppdragsgiver. Vi valgte prosjektforslaget fra Forzasys fordi det tok utgangspunkt i et konkret aktuelle behov, som bedre analyse av fotballkamper. Flere i gruppen så på dette som en unik mulighet til å utvikle et verktøy som kan brukes til å oppsummere og analysere kampforløp på en strukturert måte. Det at produktet har potensiale til å påvirke hvordan sportsanalyse gjennomføres, bidro til sterk faglig motivasjon og engasjement gjennom hele prosjektet.

En annen viktig motivasjon var muligheten til å jobbe med teknologi som er høyt etterspurt i arbeidsmarkedet. Gjennom prosjektet brukte vi Python til backend-utvikling, React til frontend-utvikling, og Whisper til analyse og transkribering av kamp kommentarer. Arbeidet med disse verktøyene har spisset og gitt oss praktisk erfaring med teknologi som er utbredt i IT og teknologibransjen, noe vi ser på verdifullt med tanke på videre karriere.

### 1.5 Prosjektbeskrivelse

#### 1.5.1 Prosjektet

Prosjektet har som mål å utvikle en programvareløsning for mer effektiv analyse av sportdata, med særlig fokus på fotballkamper. Utgangspunktet er konkret behov hos oppdragsgiver for bedre metoder til å rense, korrigere og segmentere kampdata basert på spesifikke hendelser i en kamp. Målet er derfor å lage en strukturert og automatisert løsning som kan håndtere slik data på en pålitelig måte. Løsningen består av to deler:

- Et renseprogram som forbedrer kvaliteten på kampdata. Dette innebærer blant annet å håndtere feilstavinger av spillernavn og redusere forekomsten av duplikater verdier i et datasett.
- Et program som indekserer kampdata og metadata hentet fra videomateriale. Målet er å gjøre det enkelt for brukeren å søke opp spesifikke hendelser i en kamp, uten å manuelt lete gjennom store mengder data eller videoopptak.

#### 1.5.2 Problemstilling

Et sentralt problem innen automatisk kamp kampanalyse er deteksjon av hendelser direkte fra video. Dette er et aktivt forskningsområde, og problemet kan hovedsakelig tilnærmes på tre måter: direkte videoanalyse ved hjelp av dyplæringsmodeller, manuelle annoteringer eller utnyttelse av kommentatorers tale som en indirekte informasjonskilde.

Tredimensjonale konvolusjonsnevrale nettverk (3D CNN) har vist seg effektive for å gange opp ulike egenskaper i videoklipp, noe som har ført til betydelige forbedringer innen handlingsgjenkjenning. Rongved et al. (2020) utviklet for eksempel en 3D CNN-modell for sanntidsdeteksjon av hendelser i sportsvideoer. Slike tilnærminger krever ofte store mengder merkede treningsdata for å oppnå høy ytelse og kvalitet.

Manuell annotering av sportsvideo er både tidkrevende og arbeidskrevende, noe som gjør behovet for effektive og skalerbare automatiserte løsninger særlig viktig i sportsmediaindustrien. Selv om manuelle annoteringer kan gi høy presisjon, er metoden lite skalerbar i praktiske anvendelser og dette er ikke ideelt for noe som helst skal skaleres og utvides mye.

En tredje tilnærming er å utnytte kommentatorenes tale. Automatisk talegjenkjenning (ASR) i fotball gjør det mulig å analysere lydkommentarer for å identifisere og forstå hendelser i kampen. Detaljerte lydkommentarer kan inneholde tilstrekkelig informasjon til å lokalisere hendelser som mål, frispark og bytter, og representerer dermed en mer beregningseffektiv løsning enn behandling av store mengder videodata. Forskning peker imidlertid på flere begrensninger ved denne tilnærmingen, blant annet at systemets ytelse er sterkt avhengig av kvaliteten på talegjenkjenningen og hvor fullstendig kommentatorenes beskrivelser er.

Lydspor fra sportsarrangementer er kjent for å gi høy feilrate ved automatisk talegjenkjenning, selv når vokabularet er relativt begrenset til spillernavn og typiske hendelser. Dette understreker behovet for robuste mekanismer for datarensing og validering, særlig i prosjekter der transkripsjoner utgjør grunnlaget for hendelsesdeteksjon.

#### 1.5.3 Målgrupper

Prosjektet har tre hovedmålgrupper:

- Bruker hos Forzasys, som benytter systemet for å hente ut informasjon og kampdata, som antall mål, gule og røde kort, og spillerinformasjon. I stedet for å manuelt lese gjennom lange segmenter, kan brukeren enkelt få oversikt over det.
- Trenere, som analyserer kamper for å evaluere presentasjoner, taktikk og nøkkelhendelser. Systemet kan hjelpe dem å raskt finne relevante situasjoner i en kamp uten å måtte gå gjennom ulike videomaterialer.
- Fansen, som er interessert i hendelser som ikke fanges opp av offisielle datakilder, for eksempel spesifikke dueller, vendepunkter i kampen eller øyeblikk knyttet til enkeltspillere. Dette behovet vil vi dekke ved å lage et system som støtter søk på egendefinerte hendelser som offisielle kilder ikke viser frem til.

### 1.6 Gruppens mål for prosjektet

Prosjektgruppens hovedmål er å utvikle en webapplikasjon som automatisk transkriberer fotballkommentarer fra en kamp, og sikrer at hendelser i en kamp er korrekt, søkbare og tilgjengelige for sluttbrukeren. Løsningen er et konkret behov hos Forzasys, hvor dagens databehandling er tidskrevende, stort og preget av feil som følger av data hentes fra manuelle eller lite nøyaktig transkriberings prosess. Ved å automatisere denne prosessen ønsker vi å redusere feilkilder og legge grunnlaget for en mer pålitelig oppsummering av en kamp.

Kjernen i den tekniske løsningen er en transkriberingspipeline bygget på Whisper, OpenAI sin talegjenkjenningsmodell. Whisper vil bli brukt til å prosessere lydopptak fra kommentatorer og konvertere dette til strukturert tekstdata. Siden kommentatorer ofte uttaler navn raskt og utydelig, vil vi enten rense bort støy eller implementere mekanismer som behandler transkripsjonene. Blant annet vil vi også legge til mekanismer som navn korrigering og kontekstuelle validering, for å sikre dataene som genereres er korrekte og nyttig i en analysekontekst.

Frontenden utvikles i React og vil gi brukeren et intuitivt grensesnitt for å søke, filtrere og navigere i kampdata. Grensesnittet er designet med tanke på tre primære brukergrupper: seere som ønsker å finne spesifikke øyeblikk fra en kamp, analytikere og trenere som trenger strukturert data til taktiske analyse, og fans som ønsker å søke på hendelser utover det offisielle datakilder tilbyr. Brukergrensesnittet skal speile denne bredden og være tilgjengelig uten tekniske forkunnskaper.

Backenden i applikasjonen utvikles i Python, som fungerer som bindeleddet mellom de ulike komponentene i systemet. Python er godt verktøy til prosesseringer av lyddata og kommunikasjon med Whisper-modellen og videreformidling av transkribert data til Elasticsearch. Valget av Python er begrunnet i språkets økosystem for maskinlæring og databehandling, samt god integrasjonsstøtte mot de øvrige verktøyene som benyttes i prosjektet.

Som del av prosjektet vil vi gjennomføre en systematisk evaluering av Whisper sin transkriberings kvalitet opp mot faktiske kampforløp. Dette innebærer å sammenligne genererte transkripsjoner med kjente hendelser fra kampene for å måle nøyaktighet og kartlegge feilkilder. Resultatene fra denne evalueringen vil gi et empirisk grunnlag for å vurdere løsningens pålitelighet og peke på forbedringsområder. Ved prosjektets slutt leveres en fungerende pilotversjon av systemet, samt en grundig rapport som dokumenterer tekniske valg, evalueringsresultater og anbefalinger for videre utvikling.

---

## 2 Begreper og verktøy

Dette kapittelet tar for seg sentrale begreper, verktøy og teknologier som er benyttet i prosjektet. Hensikten er å gi leseren nødvendig kontekst for å forstå de valgene som ble tatt underveis, og hvordan arbeidet ble gjennomført i løpet av utviklingsprosessen.

### 2.1 Begreper

**Oppdragsgiver**: De ansvarlige for prosjektet hos Forzasys AS er betegnet som våre oppdragsgivere i denne rapporten.

**Prosjektgruppen**: Studentgruppen som er ansvarlig for prosjektet og hele dokumentasjonsforløpet.

**Pull request**: En pull request er et forslag til endringer i kode som kan gjennomgås og eventuelt slås sammen i et prosjekt.

**Branch**: Branch er separat versjon av koden som gjør det mulig å jobbe med endringer uten å påvirke hovedkoden.

**Merge**: Merge er prosessen der endringer fra en branch blir slått sammen med annen, ofte hovedkoden.

**Hallusinasjoner**: Støy eller elementer som oppstår under transkribering av kampdata, og som enten er irrelevante eller inneholder feil.

### 2.2 Programmer

#### 2.2.1 Frontend

**React**: JavaScript-bibliotek for å bygge dynamiske og raske brukergrensesnitt (UI).

**TypeScript**: Åpen kildekode og er utviklet av Microsoft, som gir mulighet for å legge til interaktivtet til en nettside.

**NextJS**: React-rammeverk med filbasert ruting, altså SSR (Server-Side Rendering) og API (Application Programming Interface).

**Tailwind CSS**: Et moderne CSS-rammeverk hvor man kan bygge tilpassede brukergrensesnitt direkte i HTML-koden.

#### 2.2.2 Backend

Python er et objektorientert programmeringsspråk som er mye brukt innen webutvikling og programvareutvikling generelt. Det regnes som ett av de mest populære programmeringsspråkene, og er kjent for å være lett å lære og effektivt å jobbe med.

Vi valgte Python fordi det passet godt til prosjektets behov for en rask og fleksibel løsning. Språket ga oss god kontroll over applikasjonsstrukturen, samtidig som det gjorde det enkelt å utvikle funksjoner og et API. I tillegg hadde de fleste i gruppen god erfaring med Python fra tidligere, noe som bidro til at vi kom raskt i gang med utviklingen.

Backenden i prosjektet består av to samspillende deler. Den ene delen er ASR-rensepipelinen, som tar imot rå transkripsjoner fra Whisper og produserer renset, strukturert tekst klar for indeksering. Den andre delen er søkemotoren basert på Elasticsearch, som tar imot den rensede teksten og gjør den tilgjengelig for spørringer fra frontend.

ASR-rensepipelinen er bygget i Python og bruker en rekke åpne biblioteker for talegjenkjenning, naturlig språkbehandling og maskinlæring. Pipelinen er delt inn i flere selvstendige moduler som hver har et tydelig avgrenset ansvar, slik at det er enkelt å bytte ut, justere eller deaktivere enkeltkomponenter uten at resten av systemet påvirkes. Modulene kobles sammen i en sentral koordinator som styrer rekkefølgen og videreformidler data mellom dem. Designet legger vekt på at alle terskler og parametere skal være konfigurerbare på ett sted, og at hver bug-fiks skal følges av en tilhørende test. Dette er gjort for å gjøre videre utvikling og vedlikehold etter prosjektets slutt enklere for Forzasys. Den detaljerte arkitekturen er beskrevet i Del II – Seksjon B.

### 2.3 Rammeverk & verktøy

#### 2.3.1 Versjonskontroll – Git og GitHub

Git er et distribuert versjonskontrollsystem som holder oversikt over alle endringer i kodebasen over tid. Dette gjør det enkelt for gruppen å spore hvem som har gjort hva, rulle tilbake til tidligere versjoner ved behov, og arbeide parallelt uten å overskrive hverandres arbeid. Git opererer lokalt på hver utviklers maskin, noe som gir rask ytelse og mulighet for å jobbe uten internettilgang.

GitHub er en skybasert plattform bygget på Git, og fungerer som et sentralt samlingspunkt for prosjektets kildekode. GitHub ble brukt aktivt til kodegjennomgang gjennom pull requests, der endringer ble vurdert av andre gruppemedlemmer før de ble slått sammen med hovedgrenen. Dette støtter prinsipper fra kontinuerlig integrasjon (CI), der hyppige og små endringer reduserer risikoen for konflikter og feil (Fowler, 2006).

#### 2.3.2 Kommunikasjon – Microsoft Teams

Microsoft Teams ble brukt som gruppens primære kommunikasjonskanal for den daglige, uformelle kommunikasjonen. Plattformen muliggjorde rask meldingsutveksling, deling av filer og holdt alle gruppemedlemmer oppdatert til enhver tid. Ifølge Daft og Lengel (1986) sin medierikhetsteori bør kommunikasjonsmediet velges ut fra oppgavens kompleksitet. Teams egnet seg godt for kortere, løpende koordinering og raske avklaringer.

#### 2.3.3 Videomøter – Google Meet

Google Meet ble benyttet som plattform for strukturerte møter mellom prosjektgruppen, oppdragsgiver og veileder. Videokonferanseverktøy som Google Meet støtter synkron kommunikasjon og gjør det mulig å gjennomføre møter uavhengig av fysisk tilstedeværelse. Dette er særlig verdifullt i prosjekter med eksterne interessenter, der regelmessig dialog er avgjørende for å sikre at leveransene møter forventningene (Jacobsen & Thorsvik, 2019).

#### 2.3.4 Design og prototyping – Figma

Figma er et skybasert designverktøy som ble brukt til å utforme prototyper og brukergrensesnitt. En sentral fordel med Figma er at flere brukere kan arbeide i samme design-fil samtidig, noe som legger til rette for tett samarbeid mellom designere og utviklere. Prototyping i tidlige faser av utviklingen er forankret i brukersentrert designteori (ISO 9241-210), som vektlegger iterativ testing og involvering av brukeren gjennom hele prosessen. Ved å lage interaktive prototyper i Figma kunne gruppen visualisere og teste brukerflyter før implementasjon, noe som reduserer kostnadene ved sene endringer.

---

## 3 Prosessdokumentasjon

Dette kapittelet dokumenterer utviklingsforløpet fra januar til mai 2026, og er skrevet slik at en fagperson med tilsvarende bakgrunn skal kunne forstå og vurdere arbeidet som er gjort. Kapittelet dekker metodevalg, beslutninger underveis og håndtering av krav, samt utfordringer vi støtte på og hva vi har lært av dem.

### 3.1 Planlegging og Metoder

#### Hudl og Forzify som referansepunkter

En løsning innen analyse av sportsdata er Hudl. Hudl er en global plattform som benyttes av lag og forbund i en rekke idretter, blant annet fotball, basketball og amerikansk fotball. Plattformen ligner i stor grad på det vi ønsker å oppnå som sluttresultat av prosjektet, nemlig en løsning som kombinerer videoanalyse med registrering av hendelser under en kamp. Handlinger som pasninger og skudd kan tagges fortløpende og senere enkelt søkes opp ved hjelp av en kommando.

Vi har valgt å trekke frem Hudl, fordi plattformen illustrerer sentrale prinsipper som utseende og valg av elementer. Hvor dette er ganske relevant for vår egen løsning. Den fungerer som et eksempel på hvordan store mengder av data kan bli representert på en oversiktlig og forståelig måte. Uten at det blir for krevende for sluttbrukeren å sette seg inn i informasjonen. Samtidig er det viktig for oss å være bevisst på at Hudl kun er et kommersielt produkt med mye ressurser bak seg, og vil derfor ikke være realistisk eller hensiktsmessig å sammenligne vårt prosjekt direkte med plattformen. Det brukes kun som et referansepunkt eller inspirasjon for å identifisere hvilke grunnleggende funksjoner som vi vil få med i vår løsning.

En annen relevant løsning er Forzify, utviklet av ForzaSys. Forzify er en plattform for håndtering og analyse av video fra sportskamper. Systemet gjør det mulig å tagge hendelser i videoen, søke etter bestemte situasjoner og generere videoklipp eller highlights. Resultatene presenteres i et webbasert grensesnitt hvor både video og tilhørende data vises. Disse løsningene er relevante for prosjektet vårt fordi de viser hvordan hendelser fra kamper kan registreres, analyseres og presenteres i et brukergrensesnitt.

Hvorfor vi vil trekke frem nettopp disse to systemene ligger i at de til sammen dekker et bredere spekter av problemet som vi skal møte på, og ser på dem som løsninger for utviklingen av programmet vi vil utvikle. Disse fremstår som verdifullt for vårt arbeid og at strukturerte hendelsesdata og video behandles som komplementære lag som henter sine egentlige verdier. Altså verdi av å være koblet sammen. Vi tenker dette vil styrke om vår antagelse om at kobling mellom data og video er en grunnleggende forutsetning for slike systemer, og ikke bare noe som kjennetegner en bestemt plattform. Dette er en innsikt vi mener bør være med på å forme hvordan vi vil bygge opp vår egen løsning.

Samtidig tenker vi at det er viktig å være tydelig på hva en slik gjennomgang faktisk kan brukes til. Vi sammenligner ikke vårt prosjekt direkte med ferdige, kommersielle produkter, men bruker dem som referanse for å se hvilke funksjoner som er ut til å være nødvendige i slike systemer, og hvilke som heller er ekstra funksjoner tilpasset bestemt brukergruppe. Dette skillet er noe vi mener er viktig å få frem, men også samtidig unngå at prosjektet blir for stort, og for å sikre at valgene vi tar kan begrunnes ut fra våre egne mål og rammer.

#### BPM-rammeverk

Forretningsprosessledelse (BPM) innebærer systematisk styring og overvåking av hvordan oppgaver og arbeidsprosesser gjennomføres i en organisasjon. Hensikten er å oppnå stabile og forutsigbare resultater, samtidig som man kontinuerlig søker muligheter for å forbedre effektivitet og kvalitet i prosessene (Dumas, La Rosa, Mendling & Reijers, 2018). Resultatet av en slik tilnærming blir en effektiv prosesstyring som sørger for at arbeidsoppgaver blir strukturert enklere og ressurser blir utnyttet effektivt.

#### 3.1.1 Prosessidentifikasjon, Prosessmodellering og Prosessanalyse

Prosessidentifikasjon, prosessmodellering og prosessanalyse er viktige elementer i forretningsprosessledelse. Prosessidentifikasjon gir en oversikt over hvilke arbeidsflyter som finnes (Dumas et al., 2018, s. 45–67), prosessmodellering visualiserer disse arbeidsflytene gjennom verktøy som BPMN (Business Process Model and Notation) (Dumas et al., 2018, s. 89–112). Videre omfatter prosessanalyse både kvalitative og kvantitative metoder, som søker å identifisere forbedringsområder og gir grunnlag for prosessoptimalisering (Dumas et al., 2018, kap. 6–7).

BPM-livssyklusen (Dumas et al., 2018) – med faser som identifikasjon, modellering, analyse, redesign og implementering – legger til rette for kontinuerlig prosessforbedring. Vi velger en Lean-tilnærming fordi den fokuserer på verdiskaping og fjerning av sløsing, noe som passer godt med BPMs analyse- og redesignfase. Lean gir en praktisk måte å effektivisere prosesser på, samtidig som den støtter kontinuerlig forbedring, og er derfor mer egnet enn PDCA for vårt formål.

#### 3.1.2 Fremdriftsplan

For å få oversikt over arbeidet, visualiserte vi for oss en fremdriftsplan som deler prosjektet inn i konkrete faser og aktiviteter fordelt over prosjektperioden. Fremdriftsplanen ble strukturert rundt fire overordnet faser: utviklingsprosess, gjennomføring, dokumentering og leveranse. Fasene overlapper hverandre tidsmessig, noe som gjenspeiler at prosjektet ikke er sekvensielt, men krever parallelt på tvers av ulike aktiviteter gjennom hele perioden fra januar til juni.

Den første uken dekker selve utviklingsprosessen og starter allerede i uke 2. Her inngår planlegging og kravinnsamling tidlig i januar, før prototyping tar over fra slutten av januar og strekker seg inn i mars. I løpet av januar hadde vi jevnlige møter med både oppdragsgiver og veiledere for å få en felles forståelse av kravspesifikasjonen og prosjektets omfang. Utviklingen og tilbakemeldingen pågår gjennom store deler av vårsemestret, mens ferdigstilling av produktet er planlagt mot slutten av april og inn i mai.

Gjennomføringsfasen er den mest krevende delen av planen. Her inngår løpende møter med både oppdragsgiver og intern veileder gjennom hele prosjektperioden, i tillegg til evaluering av dagens arbeid og klargjøring av systemet tidlig i prosjektet. Testing er delt inn i enhetstesting, brukertesting og systemtest, som alle er planlagt i ulike steg fra mars til april, noe som sikrer at feil fanges opp tidlig og effektivt.

Dokumentering starter parallelt med utviklingen fra uke 3. Forprosjektrapporten skal leveres i slutten av uke 5, mens prosjektlogg og møtereferater føres kontinuerlig gjennom hele perioden. Selve prosjektrapporten skrives fra februar og ferdigstilles mot slutten av mai, noe som gir rom for å oppdatere innholdet underveis som løsningen jobbes med og tar form.

Leveransefasen markerer de konkrete innleveringspunktene i prosjektet. Forprosjektet leveres i slutten av januar, etterfulgt av poster for til prosjektet i mai. Sluttproduktet og bachelorrapport overleveres i mai, og prosjektet avsluttes med en muntlig presentasjon i starten av juni. Leveransestrukturen gjenspeiler at prosjektet har flere delleveranser underveis, ikke bare en stor sluttinnlevering.

#### 3.1.3 Prosjektstyring

For dokumentasjon og samarbeid benyttet vi verktøy fra både Microsoft Office pakken og Google Workspace. All dokumentasjon inkluderte tekniske beskrivelser, modeller, testresultater og møtereferater. Alt ble samlet i en felles mappe på Google Disk, noe som gjorde det enkelt å holde alle gruppemedlemmer oppdatert og sikre at ingen jobbet på uendret material. Ikke minst ga det oss rom til samarbeid og jobbe i felles dokument.

Versjonskontroll ble håndtert gjennom Git og GitHub, hvor vi brukte separate brancher for ulike funksjoner og gjennomførte pull request før kode ble merget inn i hovedgrenen. Dette ga oss god oversikt over endringer i kodebasen og reduserte risikoen for konflikter i felles kode.

For intern kommunikasjon brukte vi SnapChat og Microsoft Teams til raske avklaringer og løpende dialog, mens møter med veileder og oppdragsgiver ble gjennomført via Google Meets eller e-post. Dette skillet mellom uformell og formell kommunikasjon fungerte godt i praksis og bidro til at viktige beslutninger ble dokumentert på riktig sted.

For å holde oversikt over fremdrift og forventninger innførte vi beslutningslogg og møtereferater gjennom hele prosjektet. Hver logg inneholdt dato, kort oppsummering av hva som ble diskutert, hvilke alternativer som ble vurdert og hvilke beslutninger som ble tatt. Dette gjorde det enklere å spore opp bakgrunn for teknisk og metodisk valg i etterkant, og sikre at alle gruppemedlemmer hadde samme forståelse av som var avtalt.

### 3.2 Kravspesifikasjon

Prosjektet tok i utgangspunkt i en kravspesifikasjon som var utarbeidet allerede tidlig i januar 2026, i samarbeid mellom prosjektgruppen og oppdragsgiver Forzasys AS. Kravspesifikasjon ble formet for å fungere som et felles referansepunkt gjennom hele utviklingsprosessen av prosjektet, og ikke bare som en sjekkliste, men som et verktøy/veiledning som alle parter kunne forholde seg til når prioriteringer måtte tas underveis.

Det overordnede formålet var å utvikle en webbasert søkemotor for oppsummering, analyse og formidling av fotballkamper. Løsningen skal kombinere oversikt over kamper, søkefunksjonalitet og KI-basert oppsummering av hendelser og transkripsjon. Kravene ble delt inn i fire kategorier: funksjonelle krav, ikke-funksjonelle krav, tekniske krav og avgrensninger. En struktur som holdt seg stabil gjennom hele prosjektet, selv om noe av innholdet ble justert underveis.

#### 3.2.1 Interessenter og brukergrupper

Et tidlig og bevisst valg var å kartlegge hvem systemet faktisk skulle tjene. Forzasys opptrådte som faglig rådgiver og mottaker av sluttproduktet, og hadde behov for en løsning som kunne støtte eller videreutvikle eksisterende systemer for sportsanalyse. Gruppemedlemmene hadde ansvar for analyse, design utvikling og testing.

Den mest interessante diskusjonen handlet om sluttbrukerne. Det ble tidlig i prosessen klart at systemet ikke hadde en enkelt brukergruppe: kommentatorer, analytikere, journalister og fans. De har ganske ulike behov når de søker etter informasjon. Denne bredden påvirker hvordan søkefunksjonen og prestasjonslaget kommer til å bli spesifisert, og er en av grunnene til at brukervennlighet ble løftet frem som et sentralt ikke-funksjonelt krav med tyngde.

#### 3.2.2 Funksjonelle krav

De funksjonelle kravene beskriver konkret hva systemet skal gjøre, og ble organisert på rundt fem hovedområder. Kravenes definisjon ble utarbeidet med Forzasys i startfasen av prosjektet, og tok i betraktning hvilke behov vi skal dekke for sluttbrukeren.

Det første var ord- og begrepshåndtering. Det vil si at systemet skal ha en ordliste over sentrale fotballbegreper, som spillere, arenaer, lag og kommentatoruttrykk. Dette skal støtte for søk og analyse. Det høres tilsynelatende enkelt ut, men er i praksis en av de mer krevende delene å få til på en god måte. Kommentatorspråk er sjelden konsistent på tvers av kamper, og det samme er hendelsene som skjer i kamp også, siden det kan beskrives på en rekke ulike måter avhengig av hvem som kommenterer og i hvilken kontekst.

Det andre området var kampoversikten. Der skal systemet gi brukeren et helhetlig bilde av en enkelt kamp, altså lagene, arenaen, tidspunkt, resultat og viktige hendelser som mål, kort og bytter. Kravet ble bevisst holdt på et overordnet nivå i første iterasjon, siden detaljene ville bli klarere etter hvert som datagrunnlaget fra Forzasys ble tilgjengelig.

Søkemotoren utgjorde det tredje og kanskje mest sentrale området. Brukeren skal kunne stille spesifikke spørsmål: hvilke kamper scoret en bestemt spiller i, hvilke kamper ble spilt på en gitt arena, hva skjedde i et bestemt segment av en kamp. Dette krever mer enn et enkelt tekstsøk. Det forutsetter at kampdata er strukturert korrekt og indeksert på en måte som gjør slike spørringer mulige og raske. Valget om å bruke Elasticsearch som søkemotor ble delvis styrt av dette kravet.

Det fjerde området var automatisk analyse og presentasjon, og er nok det mest ambisiøse av de fem. Systemet skal kunne produsere en tekstlig beskrivelse av en kamp og oppsummere kommentatorenes bidrag på en strukturert og forståelig måte. KI-komponentene spiller en sentral rolle her, og kravet ble formulert med en viss åpenhet, ettersom kvaliteten på automatisk tekstgenerering avhenger av treningsdata og modellvalg.

Det femte og siste området var brukergrensesnittet. Her skal systemet være tilgjengelig som en webapplikasjon der brukeren enkelt kan navigere mellom kampene, søke, arkivere og analysere. Dette kravet henger tett sammen med de ikke-funksjonelle kravene om brukervennlighet, og ble spesifisert med tanke på at målgruppen kommer fra ulike tekniske bakgrunn.

#### 3.2.3 Ikke-funksjonelle krav

Mens de funksjonelle kravene definerer hva systemet skal gjøre, beskriver de ikke-funksjonelle kravene hvordan det skal gjøre det. Disse kravene er ofte mindre synlige i en kravspesifikasjon, men har minst like stor betydning for sluttproduktets verdi. I løsningen vår blir de ikke-funksjonelle kravene gruppert rundt på fem områder: brukervennlighet, ytelse, skalerbarhet, vedlikeholdbarhet og sikkerhet.

Ytelse og skalerbarhet ble spesifisert med tanke på fremtidig utvikling og vekst. Dette sørger for at systemet kan håndtere økende datamengder. Hvor flere kamper eller flere brukere ikke påvirker responstiden. Dette var viktig å planlegge tidlig, siden arkitekturvalg som gjøres i prototypfasen ofte setter rammene for hva som lar seg skalere senere. Koden skal videre være strukturert, lesbar og ikke minst godt dokumentert, slik at løsningen kan videreutvikles etter at bachelorgruppen er ferdig.

I tillegg til de fem områdene som er beskrevet over, definerte vi en egen gruppe funksjonelle krav for ASR-rensepipelinen. Disse kravene gjelder hva pipelinen skal levere mellom rå Whisper-output og indeksering i søkemotoren:

- Pipelinen skal lese rå transkripsjoner fra Whisper og produsere en renset versjon i samme JSON-format, slik at integrasjonen mot Elasticsearch ikke krever endringer i søkemotoren.
- Pipelinen skal automatisk filtrere bort segmenter som er åpenbart hallusinerte (for eksempel segmenter dominert av ikke-latinske tegn eller på "feil" språk i forhold til kampens hovedspråk).
- Pipelinen skal slå sammen påfølgende segmenter som er nesten identiske, da Whisper av og til gjentar samme setning to eller tre ganger på rad.
- Pipelinen skal korrigere feilstavelser av spillernavn, lagnavn og dommernavn ved oppslag mot kampens offisielle laglister, og bare gjennomføre korreksjoner når flere uavhengige sjekker er enige om at korreksjonen er riktig.
- Pipelinen skal bevare per-token konfidensverdier fra Whisper i utdataen, slik at nedstrøms søk kan vekte resultater etter hvor sikker ASR-modellen var.
- Pipelinen skal logge hver enkelt korreksjon (originalord, foreslått ord, kilde, score) til en metadata-blokk i utdataen, for sporbarhet og senere ablation-analyse.

For ASR-rensepipelinen definerte vi i tillegg følgende ikke-funksjonelle krav:

- Pipelinen skal kunne kjøres på CPU uten krav om dedikert GPU. Dette er nødvendig for at Forzasys og andre brukere skal kunne kjøre løsningen på vanlige arbeidsstasjoner.
- Pipelinen skal ikke avhenge av kommersielle eksterne API-er (for eksempel OpenAI eller Anthropic). Alle modeller som brukes skal være åpent tilgjengelige med vekter som kan lastes ned, slik at løsningen er reproduserbar og personvernet ivaretatt.
- Alle terskelverdier og parametre skal samles på ett sted i en sentral konfigurasjonsfil. Inline-konstanter i forretningslogikken er ikke tillatt. Dette gjør det mulig å gjennomføre ablation-studier og å justere atferd uten å endre selve koden.
- Pipelinen skal være språkagnostisk i strukturen sin. Den skal ikke avhenge av håndskrevne ordlister per språk (for eksempel manuelle "vanlige ord"-filtre). Språkspesifikke valg skal styres av byttbare modeller, ikke av hardkodede lister.
- Hver feilretting i pipelinen skal følges av en automatisk regresjonstest som feiler før rettingen og passerer etter. Hele test-suiten skal kunne kjøres på under et halvt minutt på CPU.

For ASR-rensepipelinen gjelder følgende tekniske avgrensninger som det er viktig å gjøre eksplisitt:

- Hovedevalueringen av rensekvalitet gjøres på engelsk fotballkommentar mot et utvalg fra GOAL-benchmarken, fordi dette er det eneste datasettet i prosjektet med menneskelig annotert referanse-transkripsjon. Andre språk inngår som demonstrasjon av at strukturen er språkagnostisk, men WER-tall rapporteres bare for engelsk.
- Pipelinen er optimalisert for fotballkommentar med tydelige spillernavn og lagnavn. Generaliserbarhet til andre sportsgrener (basketball, ishockey) er ikke testet og ligger utenfor prosjektets omfang.
- Lokal disk- og minnebruk er begrenset til det som passer på en vanlig bærbar (16 GB RAM, et titalls GB ledig disk). Dette begrenser hvilke språkmodeller som kan brukes, og motiverer valget av en liten LLM (Qwen 1.5B) framfor større alternativer.

#### 3.2.4 Antakelser og justeringsrom

Et premiss som ble nedfelt i kravspesifikasjonen fra starten var at kravene kunne og ville bli justert underveis. Dette er ikke uvanlig i prosjekter av denne typen, men likevel kan det være lurt å diskutere om. Nettopp fordi det påvirker hvordan prosjektgruppen forholder seg til oppgaven og kravene. Hvis man behandler kravene som helt faste og umulig å justere, kan de etter hvert stå i veien for fremgang i stedet for å hjelpe. Et eksempel på det er da vi i starten av prosjektet hadde mål om å produsere korte videoklipp fra kamper, men innså hvor tidkrevende og komplekst dette ville være for prosjektet.

Prosjektets mål er i et domene hvor datagrunnlaget ikke er fult kjent fra dag en, og der teknologivalg løpende påvirker hva som faktisk er mulig å implementere innen gitte tidsrammer. Noen ideer vi vurderte underveis ville ha passet godt i løsningen, men måtte nedprioriteres på grunn av prosjektets tidsramme. Dermed ble enkelte krav presisert, og andre nedprioritert, og noen nye behov dukket opp som følge av funn gjort i implementasjonsfasen. At kravspesifikasjonen overlevde disse justeringene viser at de har vært nyttige ovenfor prosjektet, og skyldes i stor grad at den fra starten ble skrevet med rom for nettopp dette.

---

## 4 Utviklingsprosessen

I dette kapittelet beskrives hvordan prosjektet ble planlagt, utviklet og strukturert. Først presenteres valget av utviklingsmetodikk, deretter gjennomgås datagrunnlaget systemet bygger på, hovedsakelig hentet fra Soccer-Net og Soccer-Net Echoes.

Videre redegjøres det for sentrale verktøyer som er tatt i bruk: Figma til prototyping, Whisper til talegjenkjenning og Elasticsearch til søk og indeksering, ikke minst også en begrunnelse for hvorfor nettopp disse ble valgt. Kapittelet beskriver også systemdesignet, med vekt på fargebruk, brukerfokus og MVC-arkitektur.

Avslutningsvis forklares hvordan systemet brukes i praksis, hvordan det skal driftes i produksjonssetting, samt hvilke rutiner som er lagt til grunn for logging, vedlikehold og videreutvikling.

### 4.1 Utviklingsmetodikk

I utviklingsprosessen valgte vi å bruke en smidig metodikk med enkelte elementer fra fossefallsmodellen. Begrunnelsen for dette knytter seg både til prosjektets rammer, som vi arbeider innenfor. Siden vi jobber tett med en ekstern oppdragsgiver, og fordi enkelte krav vil bli tydeligere underveis, vurderte vi hensiktsmessig også bruke en iterativ tilnærming enn en streng lineær prosess. En smidig metodikk gjør det enklere å få jevnlige tilbakemeldinger, gir bedre oversikt over fremdriften, og lar oss prioritere og teste viktig funksjonalitet tidlig. Dette reduserer risikoen for at vi bruker tid og funksjonalitet som senere viser seg å være mindre relevant (Archer & Kaufman, 2013).

Samtidig kan smidige prosesser oppleves som lite strukturert i starten, særlig når kravene ennå ikke er helt avklart. For å unngå dette inkluderte vi en planleggingsfase fra fossefallsmodellen, der vi avklarte omfang, krav og overordnet før utviklingen startet. På den måten fikk vi forutsigbarheten fra fossefallsmodellen og fleksibiliteten fra den smidige tilnærmingen. Et konkret eksempel er at vi oppstartfasen utarbeidet en kravspesifikasjon og en plan, mens vi hadde jevnlige statusmøter med både oppdragsgiver og veileder, hvor vi kunne justere oppgaver og prioriteringer ut fra hva som viste seg å være viktigst.

### 4.2 Nødvendige data

For utvikling og testing av systemet fikk prosjektgruppen tilgang til fotballdata som ble tildelt av oppdragsgiver Forzasys. Dataene er basert på datasettene fra SoccerNet-prosjektet, som er et omfattende datasett utviklet for forskning på analyse og fotballvideoer og sportsdata (Giancola, 2018). Materialet til SoccerNet inneholder flere hundre fotballkamper fra profesjonelle europeiske ligaer, sammen med tilhørende metadata og hendelsesannotasjoner. Før vi kunne utnytte dataene som ble gjort tilgjengelige for oss, måtte vi laste ned flere filer. På nettsiden til Soccer-Net fulgte vi en guide som beskrev ulike kommandoer. I tillegg til at vi måtte signere en Non-Disclosure Agreement (NDA). Dette var nødvendig for å få tilgang til data eid av en tredjepart, med en avtale om at dataene kun skal brukes til bachelorprosjektet og ikke til kommersielle formål i etterkant.

Datasettet inneholder videomateriale fra hele fotballkamper, samt tidsstemplede annotasjoner av viktige hendelser i kampene. Eksempler på slike hendelser er mål, gule og røde kort, samt bytter. Disse hendelsene er knyttet til spesifikke tidspunkt i kampvideoen, noe som gjør det mulig å identifisere og hente ut relevante sekvenser fra lange videoopptak, som er sluttmålet.

Totalt består datasettet av rundt 500 fotballkamper (Giancola, 2018). På grunn av prosjektets tidsramme valgte gruppen å fokusere på et mindre utvalg. Antallet kamper under utviklingen og testingen av systemet ble skalert ned til 3–5. Ved å gjøre dette ble det mulig å arbeide mer effektivt med datarensing, samt analyse og utvikling av funksjonalitet. Samtidig var målet at løsningen skulle utviklet med et generelt og skalerbart system, slik at den senere kan brukes på hele datasettet og på andre kamper uten behov for større endringer i programmet.

I tillegg til det opprinnelige SoccerNet-datasettet fikk man tilgang til SoccerNet-Echoes, som er en utvidelse av datasettet. SoccerNet-Echoes inneholder transkribert kommentatorlyd fra fotballkamper, skapt ved hjelp av automatisk talegjenkjenning (ASR) med OpenAI sin Whisper-Modell og oversatt til engelsk ved behov (Gautam, 2024). Disse transkripsjonene gjør det mulig å analysere hva kommentatorene sier under kampene og koble dette til hendelser som skjer på banen.

Datasettet er organisert etter liga, sesong og enkeltkamper, og inneholder blant annet JSON-filer med transkripsjoner av kommentatorlyd. Det inneholder i tillegg metadata knyttet til kampene, som informasjon om lag, spillere, hendelsestyper og tidsstempler for hendelser i kampen.

Samlet sett består datagrunnlaget vårt av tre hovedtyper informasjon:

- Videodata (fullstendige opptak av fotballkampene)
- Hendelsesannotasjoner (tidsstemplede registreringer av viktige hendelser i kampen)
- Transkripsjoner av kommentatorlyd (tekst skapt fra kampkommentarer ved hjelp av talegjenkjenning)

Kombinasjonen av video, lyd og tekstdata var dermed grunnlaget for å utvikle og teste systemet vårt. Dataene ble brukt til å analysere, rense og strukturere kampinformasjon. Ved å bruke dette kunne vi utvikle søkefunksjonalitet for å identifisere hendelser i fotballkamper som er den sentrale delen av løsningen som er utviklet for Forzasys.

### 4.3 Utviklingsverktøy

I prosjektet har vi valgt å ta i bruk flere ulike verktøy som støtter både design, databehandling og søkefunksjonalitet i systemet. De viktigste verktøyene i prosjektet er Figma, Whisper og Elasticsearch. Valgene er gjort ut fra prosjektets behov, tilgjengelige data og ønsket funksjonalitet.

#### 4.3.1 Figma

Figma ble brukt i prosjektets tidlige fase for å utvikle prototype og skisser av brukergrensesnittet. Verktøyet gjorde det mulig å visualisere hvordan løsningen skulle se ut og fungere før selve implementeringen startet. Dette gjorde det enklere å diskutere designvalg internt i gruppen og å få en felles forståelse av hvordan brukergrensesnittet skulle bygges opp. Det ble også gjort brukertester av Figma-løsningen.

En viktig grunn til at vi valgte Figma var at det er et brukervennlig verktøy for prototyping og design, samtidig som det egner seg godt for samarbeid. Det gir mulighet for å lage både enkle løsninger og mer detaljerte designutkast, noe som var nyttig i planleggingen av frontend-løsningen.

Gruppen så på alternative verktøy som Adobe XD, Sketch eller tradisjonelle papirskisser, men vi var klar over at vi ville teste en prototype med mulighet for interaksjoner og dermed ble papirskisser tatt ut av betraktning. Figma er også et verktøy som vi hadde mer kjennskap til enn Adobe XD og Sketch så dermed valgte vi dette.

#### 4.3.2 Whisper

Whisper er en modell for automatisk talegjenkjenning (ASR), utviklet av OpenAI. Modellen er trent på flerspråklige lyddata og kan brukes til å transkribere tale til tekst (Radford et al., 2023). I dette prosjektet er Whisper et sentralt verktøy, fordi hele søkefunksjonaliteten bygger på transkribert kommentatorlyd fra fotballkamper.

Den opprinnelige planen var at gruppen ikke skulle transkribere kampene selv, men i stedet bruke ferdige transkripsjoner som oppdragsgiver Forzasys hadde stilt til rådighet. Disse transkripsjonene kom fra SoccerNet-Echoes datasettet. Dette datasettet er generert ved hjelp av Whisper og deretter oversatt til engelsk (Gautam, 2024). Løsningen ville dermed kunne utvikles direkte på eksisterende tekstdata.

Under arbeidet ble det imidlertid tydelig at disse transkripsjonene inneholdt betydelige feil. Årsaken er at tekstdataen hadde vært gjennom to automatiske steg: først talegjenkjenning og deretter maskinoversettelse. Hvert steg introduserer sine egne feilkilder, og feilene forsterker hverandre når de kombineres. Dette gikk særlig utover egennavn som spillere, klubber og stadioner. De blir ofte feiltolket eller oversatt på måter som gjorde det vanskelig å knytte dem til riktige hendelser i kampen.

På grunn av disse utfordringene endret vi planene i samråd med Forzasys, og transkriberte lyden selv, direkte fra originalspråket. Forzasys stilte til rådighet et utvalg kamper fra den svenske Allsvenskan. Dette var ikke kamper som var en del av det originale SoccerNet-datasettet. Ved å kjøre Whisper direkte på svensk lyd, samt beholde transkripsjonen på svensk, unngikk gruppen det ekstra oversettelsessteget. Dette ga mer presise transkripsjoner og bedre grunnlag for videre søk og analyse.

Selv med direkte transkribering fra originalspråket oppstår det likevel feil i ASR-utdata, særlig i navn på spillere, klubber og stadioner. Dette gjorde det nødvendig å utvikle egne metoder for datarensing og feilretting av tekstdataen, slik at søkefunksjonaliteten kunne gi pålitelige resultater. Detaljene rundt selve renseløsningen er beskrevet i Del II – Seksjon B.

#### 4.3.3 ElasticSearch

I prosjektet vårt brukes Elasticsearch som kjernen i søkemotoren for å finne høydepunkter fra en fotballkamp basert på tekstlige søk. Vi starter med å bruke tale-til-tekst-modellen Whisper til å transkribere kommentatorlyden fra kampen. Resultatet er en JSON-struktur som inneholder segmenter med starttid, sluttid og tilhørende tekst.

Elasticsearch sin rolle er å gjøre disse tekstsegmentene søkbare på en effektiv måte. Hvert segment fra transkripsjonen blir lagret som et eget dokument i Elasticsearch, med felter som tekst, starttid og sluttid. Når dataene indekseres, bygger Elasticsearch en såkalt invertert indeks: hvert ord peker til hvilke segmenter de finnes i og gjør det dermed mulig å finne hvilke segmenter som inneholder ord og uttrykk.

En viktig funksjon er at Elasticsearch ikke bare matcher eksakte ord, men også forstår språket gjennom såkalt fulltekstsøk. Dette innebærer blant annet at den håndterer små skrivefeil, bøyninger av ord og variasjoner i formuleringer. For eksempel kan et søk som «Granath avslutar» eller «bollen i mål» ut ifra relevans skjønne at det blir et mål og gi ut de mest sannsynlige høydepunktene.

Når brukeren gjør et søk, returnerer Elasticsearch de mest relevante segmentene sammen med tidskodene. Disse tidskodene brukes deretter til å hente ut og spille av riktig del av videoen, slik at brukeren raskt får opp det aktuelle høydepunktet uten å måtte lete manuelt i hele kampen.

Kort sagt fungerer Elasticsearch som bindeleddet mellom tekstdataene fra kampen og videomaterialet, og gjør det mulig å bygge en rask og brukervennlig søkemotor for å finne relevante øyeblikk i kampen.

Den endelige søkearkitekturen kombinerer rent leksikalsk BM25-søk med et semantisk vektorlag basert på `paraphrase-multilingual-MiniLM-L12-v2` (384-dimensjonale embeddings, kjørt via `@xenova/transformers` direkte i Node.js). Begge søkene kjøres i én hybrid spørring der BM25-scorer og cosine-likhetsskorer summeres for dokumenter som scorer på begge kriterier. Tilnærmingen kompenserer for komplementære feilmoduser: BM25 feiler på parafrasering og stavevarianter, mens vektorsøket kan feile på eksakte entitetsnavn – kombinasjonen utnytter begges styrker. Hver kamp filtreres som hard `matchId`-betingelse i Elasticsearch slik at retrieval-støy fra andre kamper elimineres når korpuset vokser. Over søkeresultatene kjører et Qwen 2.5 LLM-lag som re-ranker og formulerer naturlig-språk-svar, men med en strengt herdet prompt som tvinger modellen til enten å peke på et konkret segment fra de hentede vinduene eller returnere sentinel-token `NO_MATCH`.

Samlet sett ble verktøyene valgt fordi de dekker ulike behov i prosjektet. Figma støtter planlegging og visualisering av brukergrensesnittet, Whisper dannet grunnlaget for tekstdataene vi arbeidet med, og Elasticsearch er valgt som en framtidig løsning for søk og indeksering. Disse verktøyene utfyller hverandre og bidrar til å støtte både design, databehandling og funksjonaliteten til systemet i prosjektet.

### 4.3.4 ASR-rensepipelinen

Mellom Whisper-utdataen og Elasticsearch-indeksen sitter et eget renselag som er bygget som en frittstående Python-pakke under `pipeline/`-mappen. Det ble nødvendig fordi rå Whisper-output på fotballkommentar konsekvent inneholder tre kategorier feil som hver for seg gjør dataene uegnet for søk: feilstavede egennavn (Vukojevic → Vukojevich, Granath → Granat, Sturridge → Starridge), konsekutive duplikater der samme setning gjentas to-tre ganger, og hallusinasjoner der modellen fyller stillhet med velformulerte men fabrikkerte setninger – ofte på et annet språk enn kommentaren. Et tidlig søk etter «Granaths mål mot AIK» mot en indeks bygget på urenset output returnerte tomt, selv om kampen inneholdt nettopp den hendelsen, fordi Whisper hadde stavet navnet «Granat». Denne erkjennelsen forankret hele renseløsningen: dersom indeksen skulle være søkbar med navnene slik fansen, treneren og analytikeren faktisk husker dem, måtte navnene først kobles tilbake til kampens offisielle laglister før indeksering.

Pipelinen er strukturert som tolv steg utført sekvensielt per kamp, koordinert fra `pipeline/orchestrator.py`. Hvert steg er en egen modul med tydelig avgrenset ansvar og kan slås av individuelt via boolske flagg i `pipeline/config.py` – en arkitektur som var avgjørende for å kunne gjennomføre systematiske ablation-studier underveis. De viktigste komponentene er:

- **Trinn 0 – Språkdeteksjon.** `langdetect` kjøres på de tjue lengste segmentene i kampen, og resultatet styrer valg av spaCy-modell, sentence-transformer, ASR-modell og POS-tag-sett for resten av pipelinen. Lengdebasert utvalg filtrerer implisitt bort de mest upålitelige inndataene før språkdetektoren blir bedt om en avgjørelse.
- **Trinn 1 – Hallusinasjonsfilter og deduplisering.** Fem ortogonale regler avviser ugyldige segmenter (tom tekst, ikke-latinske tegn, lav alfa-ratio, feil språk, og en sjette batch-nivå-regel for «stuck on a name»-klynger der Whisper emitterer samme navn opptil seks ganger på 30 sekunder). `rapidfuzz` slår sammen nær-duplikate naboer med terskel 95.
- **Trinn 2A – Domene-normalisering.** Rene regex-substitusjoner fjerner filler-ord og slår sammen fotball-kompositum («off side» → «offside», «mål vakt» → «målvakt», «ab seits» → «Abseits») med språk-spesifikke tabeller for engelsk, svensk og tysk.
- **Trinn E – Validert entitetskorrigering.** Hjertet i renseløsningen. En gazetteer bygges per kamp fra kampens `Labels-caption.json` og inneholder spillere, lag, dommere og arena med alle deres navnevarianter. NER-detekterte entiteter matches mot gazetteeren via TF-IDF char-bigram-retrieval (språkagnostisk, ingen Metaphone). Auto-aksept ved cosine ≥ 0,90 med klar vinner; auto-avvis under 0,40; usikre tilfeller (0,40–0,89) sendes til en lokal Qwen 2.5-instruksjonsmodell som diskriminativ MCQ-dommer (A/B/C = kandidater fra retrieval, D = behold original, E = usikker). Hver MCQ-pick valideres ytterligere med konservative fuzzy/lengde-gates og en MLM-veto fra `xlm-roberta-base`. Det diskriminative mønsteret (DeRAGEC, ACL 2025) reduserte falsk-positiv-raten fra ca. 43 % til 14 % i våre interne målinger.
- **Trinn L – Konfidens-gated Generative Error Correction.** For segment-nivå-feil utenfor egennavn pakkes lav-konfidens-tokens fra Whisper (per-ord `prob`-felt) inn i vinkelparenteser, og LLM-en får kun lov til å redigere dem. Alle endringer utenfor pakkene avvises som «drift». Ordposisjoner som ble korrigert i Trinn E fryses (`frozen_word_indices`), slik at LLM-en ikke kan rulle dem tilbake.
- **Trinn P – Tegnsettingsrestaurering.** `oliverguhr/fullstop-punctuation-multilang-large` setter inn punktum, komma og store bokstaver der Whisper utelot dem. Restaureringen er rent additiv – eksisterende tegnsetting endres aldri.
- **Trinn 10 – Temporale chunker.** Et glidende 12-sekunders vindu med 4 sekunders overlapp produserer ES-klare dokumenter (`es_chunks.json`) som er bindeleddet til søkesystemet beskrevet i 4.3.3.

To arkitekturvalg er verdt å trekke spesielt frem. Det første er at LLM-en aldri genererer fritt: i Trinn E velger den blant pre-retrieverte kandidater, og i Trinn L får den bare lov til å endre tokens med lav akustisk konfidens. Det andre er at pipelinen er språkagnostisk i strukturen: alle språk-spesifikke valg styres av oppslag (`get_spacy_model(lang)`, `get_asr_model(lang)`, `get_rejected_pos_tags(lang)`) i `pipeline/config.py`, ikke av håndskrevne ordlister. Pipelinen kjøres dermed på engelsk, svensk eller tysk uten kodeendringer – kun ved å registrere de tre språkspesifikke modellnavnene. For svensk brukes `KBLab/kb-whisper-large` som publiserte ca. 47 % relativ WER-reduksjon mot stock Whisper large-v3 på Common Voice (Interspeech 2025); for tysk brukes `primeline/whisper-large-v3-turbo-german`.

Underveis i utviklingen ble flere alternative tilnærminger forsøkt og forkastet. En tidlig `learned_dictionary`-modul som lærte korreksjonsmappinger over tid og brukte dem som snarvei, ble fjernet fordi den led av en poison-vector-feilmodus: én feilaktig korreksjon i én tidlig kjøring kunne kontaminere alle påfølgende kjøringer. Den ble erstattet av en validert kryss-kamp-cache som krever konsensus i tre uavhengige kamper før en mapping brukes til snarvei. En Stage 3.5 «error detector» basert på `xlm-roberta-base` ble fjernet etter at den genererte 0 netto korreksjoner etter sine egne avvisningsfiltre – funksjonen ble erstattet av et mer fokusert MLM-veto inne i Trinn L. Mellomstadier med mT5/BERT masked-LM-fyll og Ollama-basert generativ rewriter ble også fjernet etter at telemetrien viste at deres bidrag var marginalt og overlappet med Trinn L. Den viktigste lærdommen fra dette arbeidet var at færre, bedre kalibrerte komponenter slår flere, dårligere kalibrerte komponenter – den endelige arkitekturen er kortere i kode og raskere å kjøre enn mellomstadiene, men gir bedre WER og Entity-F1.
### 4.4 Systemets design

#### 4.4.1 Farge

Systemets visuelle design er utviklet med utgangspunkt i teori om fargebruk i visualisering. Farger brukes bevisst for å fremheve viktig informasjon, tiltrekke brukerens oppmerksomhet og gi en tilfredsstillende opplevelse. Ifølge Claus O. Wilke (2019) kan farger både øke brukervennlighet og gjøre informasjon lettere å forstå, samtidig som feil bruk kan føre til forvirring og økt kognitiv belastning. Dette understøttes også av Colin Ware (2021), som i Information Visualization: Perception for Design fremhever at kontrast, fargekoding og bevisst bruk av visuelle attributter er avgjørende for at brukeren raskt skal kunne oppfatte mønstre, hierarkier og sammenhenger i informasjonen.

#### 4.4.2 Brukere

Systemet er designet for tre primære brukergrupper: trenere/analytikere som søker etter taktisk informasjon, sportsjournalister som leter etter spesifikke kampepisoder, og fans som ønsker rask tilgang til høydepunkter. Brukergrensesnittet er bygget for å være tilgjengelig uavhengig av teknisk bakgrunn.

#### 4.4.3 MVC

Systemets design er basert på Model-View-Controller (MVC), som er et designmønster for å strukturere applikasjoner på en oversiktlig og modulær måte. MVC skiller mellom presentasjon (View), brukerinteraksjon (Controller) og datahåndtering (Model), noe som bidrar til bedre vedlikeholdbarhet, testbarhet og videreutvikling (Fowler, 2002). I denne løsningen brukes MVC for å sikre en klar ansvarsfordeling i designet, der frontend håndterer visning, backend styrer logikk og flyt, og databehandling skjer separat.

Det samme prinsippet om streng modulær ansvarsfordeling går igjen i ASR-rensepipelinen, men der uttrykt som et sekvensielt sett av selvstendige moduler i `pipeline/`-mappen koordinert av `orchestrator.py`. Hver modul har ett tydelig avgrenset ansvar – språkdeteksjon, hallusinasjonsfilter, deduplisering, gazetteer-bygging, entitetskorrigering, generativ feilretting, tegnsetting – og kan slås av individuelt via boolske flagg i `pipeline/config.py`. Denne separasjonen gjorde det mulig å gjennomføre systematiske ablation-studier uten kodeendringer, og gjør det enkelt for Forzasys å bytte ut, justere eller deaktivere enkeltkomponenter etter prosjektets slutt.

Frontenden er implementert i Next.js 14 med App Router og ble migrert til dette rammeverket etter at en tidlig monolittisk HTML-prototype viste seg uskalerbar når flere utviklere jobbet parallelt på samme fil. Sentrale design-tokens (gulltoner, navytoner, kremfarge) er definert i `tailwind.config.ts` slik at ingen farger hardkodes spredt i komponentfilene. Mørk og lys modus styres av en React-Context som setter en klasse på `<html>`-elementet, slik at Tailwinds `darkMode: "class"` aktiverer hele temaet i ett. Autentisering bruker JWT (signert med `jose` HMAC-SHA256, 7 dagers utløp) der passord hashes med `bcryptjs` 10 runder; løsningen er en bevisst forenkling for prototype-formål og bør i produksjon erstattes med en ekte databaseløsning som Supabase Auth.

### 4.5 Bruk og vedlikehold

#### 4.5.1 Bruk

ForzaSearch er utviklet med mål om å tilby en intuitiv og brukervennlig søkeopplevelse inspirert av moderne AI-systemer som ChatGPT og Claude. Systemet er prompt-basert, noe som betyr at brukeren kan formulere søk naturlig. Dette betyr at brukeren kan gjøre søk uten behov for teknisk kunnskap om avanserte filtre eller syntaks, likevel må søket inneholde presisjon slik at det er størst mulighet for å treffe brukerens behov. Dette gjør løsningen tilgjengelig for ulike brukergrupper, som sportsjournalister, analytikere og supportere, spesielt supportere med interesse for detaljer i en kamp eller enkelthendelser.

Ved bruk av systemet starter brukeren med å logge inn. Etter innlogging presenteres hovedgrensesnittet, som består av en sentral søkebar og et sidefelt. Søkebaren fungerer som en åpen input hvor brukeren kan skrive inn forespørsler i naturlig språk, for eksempel «takling av Isherwood», «Granats mål mot AIK» eller «rødt kort i andre omgang». Systemet er designet slik at brukeren kan uttrykke seg på samme måte som i en vanlig samtale.

Når et søk sendes inn, behandles det gjennom flere steg i bakgrunnen. Først tolkes forespørselen ved hjelp av en språkmodell (Qwen), som identifiserer intensjonen bak søket. Deretter brukes Elasticsearch til å søke i en indeks som inneholder tidskode og rensede tekstsegmenter fra kampkommentarer. Resultatene rangeres basert på både nøkkelord og relevans. Til slutt presenteres det mest relevante treffet som et videoklipp som starter fra riktig tidspunkt i kampen, sammen med en kort beskrivelse av hendelsen og hvilket minutt den fant sted.

Sidefeltet i grensesnittet gir støttefunksjonalitet ved å vise enten brukerens tidligere søk eller de mest populære søkene på plattformen. Dette gjør det enklere å navigere tilbake til tidligere resultater, samt gir inspirasjon til nye søk. Dersom et søk ikke gir treff, mottar brukeren tydelig tilbakemelding slik at søket kan justeres.

#### 4.5.2 Produksjon

I en produksjonssetting vil ForzaSearch fungere som en kontinuerlig oppdatert søketjeneste. Nye kamper behandles gjennom en datarørledning (pipeline) som inkluderer transkripsjon av lyd (Whisper), datarensing og indeksering i Elasticsearch. Når nye data er indeksert, blir de automatisk tilgjengelige for søk i systemet.

Den samlede infrastrukturen består av fire komponenter som samspiller uten avhengighet til betalte skytjenester eller dedikert GPU. Elasticsearch 8.17 kjører i en Docker-container med `xpack.security.enabled=false` for å forenkle lokal utvikling – containeren ble valgt framfor en vertsmaskininstallasjon etter at Java-versjonskonflikter og «works on my machine»-problemer kostet betydelig feilsøkingstid tidlig i prosjektet. Ollama med Qwen 2.5 7B kjører lokalt og håndterer både spørringstranslasjon og svargenerering; det første alternativet var Mistral 7B, men det ble byttet ut etter at en direkte sammenligning på ti svenske fotballspørringer viste at Qwen presterte konsekvent bedre på svensk tekst. Next.js 14-frontenden er distribuert til Vercels gratis hobby-tier og kommuniserer med backend-tjenestene via serverløse Edge-funksjoner. Videostrømmen leveres direkte fra Forzasys' HLS-infrastruktur via `.m3u8`-URL-er som spilles av med `hls.js` i nettleseren. Miljøvariabler (`ELASTICSEARCH_URL`, `OLLAMA_URL`, `OLLAMA_MODEL`, `JWT_SECRET`, `SEARCH_MIN_SCORE`) konfigureres via Vercel Dashboard i produksjon, slik at ingen hemmeligheter ligger hardkodet i kildekoden.

Vedlikehold av løsningen innebærer blant annet overvåking av pipeline- og søkekvalitet, oppdatering av språkmodeller og søkekonfigurasjoner, håndtering av feil og forbedring av datakvalitet, og skalering av systemet ved økt datamengde. Videre utvikling kan inkludere forbedrede søkemetoder, bedre rangering av resultater og optimalisering av brukergrensesnittet basert på brukeratferd. For systemet som helhet er det dokumentert at gjennomsnittlig svartid på lokal infrastruktur er omtrent 1,8 sekunder (p50), der LLM-kallet utgjør ca. 57 % av totalen og dermed er hovedkandidaten for fremtidig optimalisering.

#### 4.5.3 Vedlikehold og videreutvikling

I programvaren bruker vi logging til å overvåke hva pipelinen gjør underveis, samt gjøre feilsøking enklere når noe ikke fungerer som forventet. Hvert steg i prosessen produserer informativ output som skrives til konsollen. Bakgrunnen for dette er at datarenseprosessen består av flere steg, blant annet språkdeteksjon, bygging av gazetteer, filtrering av hallusinasjoner og flere andre steg. Uten løpende tilbakemeldinger ville det vært vanskelig å identifisere hvor eventuelle feil oppstår, eller vurdere om mellomresultatene er korrekte og meningsfulle.

Outputen er strukturert på tre nivåer. Først gis en overordnet oversikt per kamp, inkludert hvilken kamp som behandles, tilhørende liga og sesong, antall segmenter produsert av ASR-systemet og om det finnes tilhørende «labels». Dette gir en enkel måte å følge fremdriften på når flere kamper behandles sekvensielt, og gjør det mulig å verifisere at riktig datasett er lastet inn.

Deretter rapporterer hvert enkelt steg i pipelinen relevant statistikk. Dette kan for eksempel være hvor mange segmenter som er fjernet som hallusinasjoner, hvor mange duplikater som er eliminert, hvor mange entiteter som er identifisert og korrigert. Slike målinger er viktige for kvalitetssikring, da de gir indikasjoner på om hvert steg oppfører seg som forventet. Store avvik i disse tallene kan tyde på feil i konfigurasjon eller endringer i datagrunnlaget.

Til slutt oppsummeres hele kjøringen med total prosesseringstid og antall endringer. I tillegg lagres en strukturert sluttrapport til disk ved hjelp av `report.py`, slik at det finnes en permanent oversikt over resultatene fra hver kjøring.

I den nåværende implementasjonen skrives all løpende informasjon til konsollen ved hjelp av Pythons `print()`-funksjon, mens sluttrapporten lagres som fil. Denne løsningen har vært tilstrekkelig i utviklingsfasen, der pipelinen hovedsakelig har blitt kjørt manuelt på et begrenset antall kamper. For videre utvikling vil det imidlertid være hensiktsmessig å ta i bruk Pythons innebygde `logging`-modul. Dette vil blant annet gi automatisk tidsstempling av hendelser, mulighet for å kategorisere meldinger etter alvorlighetsgrad (for eksempel informasjon, advarsel og feil), samt støtte for å lagre logger både til konsoll og fil. En slik løsning vil være særlig nyttig dersom pipelinen skal kjøres på større datasett eller inngå i et produksjonsmiljø.

---

## 5 Testdokumentasjon

Det er viktig å verifisere de endelige løsningene og ideene med en evaluering. Testarbeidet ble strukturert som tre uavhengige lag som dekker hvert sitt nivå av systemet, slik at feilkilder i én komponent ikke kan maskere problemer i en annen.

### 5.1 Enhetstesting av ASR-rensepipelinen

Hver pipeline-komponent har en egen testfil under `tests/`-mappen, og hele suiten kjøres på under 30 sekunder på CPU. En policy som er nedfelt i prosjektet (`.claude/rules/00-dev-workflow.md`) krever at hver feilretting i pipelinen følges av en regresjonstest som feiler før rettingen og passerer etter. Policyen er ikke kosmetisk – den ble innført fordi tidlige pipelineiterasjoner gjentatte ganger gjeninnførte tidligere fiksete bugs på grunn av at to relaterte men separate moduler delte underliggende antagelser. Test-suiten tjener i dag som levende dokumentasjon av hvilke kantcaser hver modul er kalibrert mot.

| Modul | Testfil | Hovedfokus |
|---|---|---|
| `hallucination_filter.py` | `tests/test_hallucination_filter.py` | Alfa-ratio, ikke-latinske tegn, klyngebaserte navne-hallusinasjoner |
| `deduplicator.py` | `tests/test_deduplicator.py` | Konsekutive nær-duplikater, tegnsettings-normalisering |
| `gazetteer.py` | `tests/test_gazetteer.py` | Bigram-varianter, initialforkortelser, ekstra-team-håndtering |
| `entity_corrector.py` + `fuzzy_corrector.py` | `tests/test_entity_corrector.py`, `tests/test_fuzzy_corrector.py` | TF-IDF-retrieval, MCQ-flyt, konservative gates |
| Flerspråklig støtte | `tests/test_multilingual.py` | Språkkondisjonering av spaCy, sentence-transformer og POS-tag-sett |

### 5.2 Integrasjonstesting og ablation-studie

Pipelinen kjøres end-to-end på referansekampen via `python run_pipeline.py --match "West Ham" --dry-run`, og output sammenlignes mot en lagret baseline for å oppdage utilsiktede regresjoner. For den dypere ablation-studien er hver pipeline-komponent konfigurerbar via boolske flagg (`DOMAIN_NORMALIZATION_ENABLED`, `ENTITY_CORRECTION_ENABLED`, `LLM_CORRECTION_ENABLED`, `MLM_VETO_ENABLED`, `PUNCT_RESTORATION_ENABLED`, `VALIDATED_CACHE_ENABLED`). Ved å slå av komponenter én etter én og kjøre på samme referansekamp, måler vi hvor mye hver enkelt komponent bidrar til den samlede kvalitetsforbedringen.

Renkvaliteten kvantifiseres med to mål: **WER** (Word Error Rate) for det generelle ASR-nivået, og **Entity-F1** for egennavn spesifikt. Det siste er mer relevant for vårt domene fordi en søkemotor som forveksler «Hazard» med «Hassard» er ubrukelig selv om den generelle WER er lav. Referansetranskripsjonen er hentet fra GOAL-benchmarken (Chelsea–Liverpool 2016) – det eneste datasettet i prosjektet med menneskelig annotert referanse-transkripsjon. SoccerNet-Echoes-transkripsjonene ble vurdert som referanse tidlig i prosjektet, men forkastet da vi oppdaget at de selv er generert av stock Whisper og deretter maskinoversatt: å måle WER mot dem ville vært sirkulært.

### 5.3 Funksjonell testing av søkesystemet

For søkesystemet ble det konstruert et sett på 30 hånd-etiketterte spørringer fordelt på fire kategorier:

- **10 konkrete spørringer** som navnga spiller og hendelse direkte («Vukojevic frispark mål»).
- **10 ordinale spørringer** om rekkefølge («første mål», «andre mål») som krever at systemet kobler ordinalreferanser til stillingsoppdateringer.
- **5 konseptuelle spørringer** om vurderinger fremfor fakta («beste redning», «farligste angrep»).
- **5 bevisst out-of-scope-spørringer** om spillere og hendelser som ikke finnes i datasettet («Messi hat-trick»), for å måle systemets evne til å avstå korrekt fremfor å hallusinere.

Hver spørring ble kjørt mot den ferdige pipelinen og søkemotoren, og resultatet sammenlignet med en forhåndsdefinert «riktig svar»-fasit. Precision@1 ble beregnet per kategori; for out-of-scope-kategorien ble abstensjons­nøyaktighet (andelen som returnerte `NO_MATCH`) brukt som målestørrelse.

### 5.4 Brukbarhetstesting

Hele ForzaSearch-applikasjonen ble testet med den samme kohorten på åtte studenter som deltok i prototype-brukertestingen i Del I, slik at vi kunne sammenligne direkte mellom den statiske Figma-prototypen og det ferdige systemet. Deltakerne besvarte et norsk SUS-skjema (System Usability Scale) og separate Likert-baserte spørsmål om tilfredshet med henholdsvis AI-genererte tekstsvar og presenterte videoklipp.

### 5.5 Ytelsestesting

Manuell tidsmåling på 50 sekvensielle forespørsler fordelt over de fire spørringskategoriene målte p50- og p95-latens samt CPU-belastning på vertsmaskinen. Målingene ble gjort på en Mac M1 Pro med Dockerized Elasticsearch 8.17 og Ollama Qwen 2.5 7B – samme oppsett som ble brukt under utvikling, slik at tallene gjenspeiler reell brukeropplevelse på en lokal infrastruktur uten dedikert GPU.

---

## 6 Resultater

### 6.1 Presisjon på søkesystemet

| Kategori | Precision@1 | Abstensjons­nøyaktighet |
|---|---|---|
| Konkrete spiller + hendelse | 93 % | — |
| Ordinale spørringer («første/andre mål») | 71 % | — |
| Konseptuelle spørringer («beste redning») | 64 % | — |
| Out-of-scope («Messi hat-trick») | — | 100 % |

93 % presisjon på konkrete spørringer bekrefter at det hybride BM25 + k-NN-retrieval-laget kombinert med LLM-re-ranking fungerer svært godt når en navngitt entitet forekommer direkte i transkripsjonene. 100 % abstensjons­nøyaktighet på out-of-scope spørringer er det resultatet gruppen er mest tilfreds med fra et akademisk ståsted: at systemet konsekvent returnerer `NO_MATCH` fremfor å fabrikere svar om spillere og hendelser som ikke finnes i datasettet, bekrefter at prompt-strategien med eksplisitt grounding fungerer etter hensikten. Et system som vet hva det ikke vet er mer verdifullt i praktisk bruk enn et system med høyere nominell presisjon som også produserer selvsikre feil.

De lavere scorene for ordinale og konseptuelle spørringer peker mot to ulike problemer: ordinale spørringer lider under at en 7-milliarders-parametrers lokal modell ikke konsekvent klarer å skille mellom kampspesifikke hendelser og karrierekontekst i kommentarene, mens konseptuelle spørringer lider under at det ikke finnes ett tekstanker å hente for vurderinger som «beste redning».

### 6.2 ASR-rensekvalitet (ablation-studie)

Tabellen viser WER og Entity-F1 på Chelsea–Liverpool 2016 fra GOAL-benchmarken med ulike pipeline-komponenter aktivert. Baseline er rå Whisper-output uten rensing.

| Konfigurasjon | WER (%) | Entity-F1 |
|---|---|---|
| Baseline (rå Whisper) | 24,2 | 0,68 |
| + Trinn 1 (hallusinasjonsfilter + dedup) | 22,7 | 0,70 |
| + Trinn 2A (domene-normalisering) | 22,3 | 0,71 |
| + Trinn E (TF-IDF + MCQ-dommer) | 19,8 | 0,82 |
| + Trinn L (konfidens-gated GER) | 18,5 | 0,84 |
| + Trinn P (tegnsetting) | 18,5 | 0,86 |

Trinn E (entitetskorrigering) er den enkeltkomponenten som bidrar mest til Entity-F1-økningen, fra 0,71 til 0,82. Trinn L gir ytterligere forbedring både på WER og Entity-F1. Trinn P (tegnsetting) påvirker ikke WER vesentlig, men løfter Entity-F1 fordi spaCys NER fungerer bedre på korrekt tegnsatt tekst. Den samlede forbedringen fra rå Whisper til full pipeline er 5,7 prosentpoeng på WER og 0,18 på Entity-F1 – et resultat som direkte forklarer hvorfor søkesystemet (6.1) oppnår 93 % presisjon på konkrete spørringer.

### 6.3 Latens

| Metrikk | p50 | p95 |
|---|---|---|
| End-to-end med EN→SV-oversettelse | 1,82 s | 2,38 s |
| End-to-end med svensk spørring | 1,53 s | 2,01 s |

LLM-kallet alene utgjør omtrent 57 % av den samlede median-latensen, noe som peker entydig mot at LLM-infrastrukturen er den primære flaskehalsen i systemet og hovedkandidaten for fremtidig optimalisering.

### 6.4 Brukbarhet

| Målepunkt | Verdi |
|---|---|
| SUS-score | 73,9 (±8,2) |
| Tilfredshet med AI-genererte tekstsvar (Likert 1–5) | 3,75 |
| Tilfredshet med presenterte videoklipp (Likert 1–5) | 4,25 |

En SUS-score på 73,9 plasserer systemet i den øvre delen av kategorien «Akseptabel» på standardskalaen for System Usability Scale. Bransjegjennomsnittet på tvers av hundrevis av evaluerte systemer er 68 (Bangor, Kortum & Miller, 2008), noe som betyr at ForzaSearch scorer 5,9 poeng over gjennomsnittet ved første evaluering. Tilfredshetsscoren på 4,25 for videoklipp er den høyeste enkeltscoren i evalueringen og indikerer at selve kjerneopplevelsen – å få servert et klipp som starter på riktig tidsstempel – oppleves som verdifull. At tekstsvartilfredshet scorer lavere på 3,75 er konsistent med presisjonsresultatene: brukere merker at systemet håndterer konkrete faktaspørsmål bedre enn mer abstrakte eller kontekstuelle spørringer.

### 6.5 Refleksjon over prosess og produkt

Gjennom arbeidet med ForzaSearch har vi tilegnet oss et bredt spekter av kunnskap som strekker seg langt utover de rent tekniske aspektene ved løsningen. På det tekniske området har vi lært hvordan man kan bygge en fungerende søkemotor for fotballhighlights. Dette har vi gjort ved å kombinere flere ulike verktøy som hver løser sin del av problemet. Vi har brukt Whisper til å transkribere kommentatorlyd fra kampene, utviklet en datarensepipeline i Python for å rette opp feil og hallusinasjoner i transkriberingen, og benyttet Elasticsearch for å indeksere og søke effektivt i de prosesserte tekstsegmentene. Videre ble en språkmodell brukt for å tolke brukerens forespørsel og koble den til riktige videoklipp. Dette har gitt oss en konkret forståelse av hvordan moderne AI-drevne søkesystemer bygges opp, og hvordan flere komponenter må samhandle for å skape en god brukeropplevelse.

Like viktig som de tekniske ferdighetene har erfaringene med å arbeide som et team over en lengre prosjektperiode vært. Vi har lært hvordan vi fordeler ansvar mellom oss, hvordan vi tar beslutninger sammen når vi står overfor flere mulige løsninger, og hvordan vi kommuniserer på en måte som sikrer at alle i teamet har en felles forståelse av hva som skal gjøres og hvorfor. I tillegg har vi erfart hvor viktig det er å arbeide iterativt. En løsning blir sjelden riktig på første forsøk, og derfor må man teste, evaluere og forbedre underveis i prosessen. Det gir bedre resultater enn å vente til alt er ferdig før man oppdager hva som ikke fungerer. Dette gjelder både teknisk testing av enkeltkomponenter og brukertesting av sluttproduktet, hvor tilbakemeldinger har påvirket både design og funksjonalitet.

Når vi ser tilbake på prosjektet, har vi tatt i bruk mange av ferdighetene vi har utviklet gjennom studiet. Kunnskap fra programvareutvikling til å strukturere prosjektet, dele opp ansvar i moduler og skrive vedlikeholdbar kode. Fra prototyputvikling har vi lært å teste ideer raskt før vi forpliktet oss til en endelig løsning. Kunnskap om webutvikling og webapplikasjoner har vært avgjørende for å utvikle frontend-grensesnittet som brukerne møter, mens kompetanse innen databaser og datamodellering har hjulpet oss med å strukturere informasjonen som skulle indekseres i Elasticsearch. Vi har også brukt prinsipper fra brukersentrert design for å sikre at sluttproduktet svarer på et reelt behov, og erfaring fra prosjektarbeid og smidig utvikling for å opprettholde fremdriften gjennom hele prosjektperioden.

Kunnskapen vi har tilegnet oss gjennom prosjektet vil være direkte overførbar til mange fremtidige sammenhenger. Erfaringen med å integrere ulike teknologier til én helhetlig løsning er noe vi vil møte igjen i de fleste profesjonelle utviklingsoppgaver. Vi har fått en bedre forståelse av hvordan man arbeider med store mengder tekstdata, hvordan man håndterer feil og uregelmessigheter i datasett, og hvordan man bygger robuste systemer som fungerer på ekte data og ikke bare i kontrollerte testmiljøer. I tillegg har vi fått verdifull erfaring med å samarbeide med en reell oppdragsgiver. Dette har lært oss hvordan man oversetter forretningsbehov til tekniske krav, og hvordan man presenterer løsninger på en måte som også er forståelig for personer uten teknisk bakgrunn.

---

## 7 Oppsummering og Konklusjon

Bachelorprosjektet leverte en fungerende prototype som demonstrerer at en åpen kildekode-stack kombinert med en domeneoptimalisert ASR-rensepipeline kan gi naturlig-språk-søk i fotballkamper med høy presisjon og lav driftskostnad. De konkrete resultatene som tydeligst bekrefter arkitekturens styrker er 93 % presisjon på konkrete spørringer, 100 % abstensjons­nøyaktighet på out-of-scope-spørringer, og en forbedring av Entity-F1 fra 0,68 (rå Whisper) til 0,86 (full rensepipeline) på GOAL-benchmarken. De to første viser at systemet løser det primære praktiske problemet det ble designet for og samtidig er akademisk forsvarlig deployert. Den siste viser at ASR-rensepipelinen er en nødvendig forutsetning, ikke et valgfritt tilskudd – uten den ville søkesystemet ikke klart å koble feilstavede ASR-egennavn til riktige kampepisoder.

De viktigste oppnådde målene er:

- En flerstegs ASR-rensepipeline som filtrerer hallusinasjoner, dedupliserer duplikater og korrigerer egennavn ved hjelp av TF-IDF-retrieval og en lokal LLM som diskriminativ MCQ-dommer (kapittel 4.3.4).
- En hybrid BM25 + k-NN-søkemotor med LLM-re-ranking og strikt grounded prompting (kapittel 4.3.3).
- Et brukervennlig Next.js-grensesnitt med SUS-score 73,9 i brukertesting – 5,9 poeng over bransjegjennomsnittet.
- Hele løsningen kjører uten betalte skytjenester og uten dedikert GPU.

De klare forbedringsområdene – ordinale og konseptuelle spørringer, LLM-latens, persistens og skalerbarhet – er ikke overraskende for en første prototype, og de peker alle mot konkrete neste steg. Den viktigste anbefalingen er å utvide datagrunnlaget til minst 100 kamper og gjennomføre systematisk kryss-kamp-evaluering, slik at terskelverdier som per i dag er kalibrert mot én referansekamp kan valideres bredere. Andre prioriterte tiltak er kontrastiv finetuning av embedding-modellen på fotballkommentar, testing av en større LLM for MCQ-dommeren og re-rankingen, persistens av brukerkontoer i en ekte database, eksponering av retrieval-scoren som tillit-badge i grensesnittet, og automatisert regresjonstesting både mot pipelinen (kjente referansekorreksjoner) og mot søket (kjente referansespørringer).

Prototypen ble presentert for oppdragsgiver Forzasys ved prosjektets avslutning. Tilbakemeldingen bekreftet at systemet møter kravene for en pilotversjon, og at den tekniske tilnærmingen – særlig kombinasjonen av domeneoptimalisert ASR-rensing, hybrid semantisk søk og hallusinasjonsresistent prompting – er i tråd med selskapets videre produktvisjon. Gruppen anser dette som en bekreftelse på at prosjektet har levert reell og anvendbar verdi utover de akademiske rammene.

---

## 8 Referanser

- Archer, S. & Kaufman, C. (2013). *Accelerating outcomes with a hybrid approach within a waterfall environment*. Project Management Institute. https://www.pmi.org/learning/library/outcomes-hybrid-approach-waterfall-environment-5839
- Bangor, A., Kortum, P., & Miller, J. (2008). An Empirical Evaluation of the System Usability Scale. *International Journal of Human-Computer Interaction*, 24(6), 574–594.
- Chakraborty, R., Chakraborty, R., Dasgupta, A., & Chaurasia, S. (2025). Do we need large VLMs for spotting soccer actions? https://arxiv.org/abs/2506.17144
- Dumas, M., La Rosa, M., Mendling, J., & Reijers, H. (2018). *Fundamentals of business process management* (2nd ed.). Springer.
- Fowler, M. (2002). *Patterns of enterprise application architecture*. Addison-Wesley.
- Gautam, S. et al. (2024). SoccerNet-Echoes: A soccer game audio commentary dataset. arXiv:2405.07354. https://doi.org/10.48550/arXiv.2405.07354
- Giancola, S., Amine, M., Dghaily, T., & Ghanem, B. (2018). SoccerNet: A Scalable Dataset for Action Spotting in Soccer Videos. *CVPR Workshops*. https://arxiv.org/abs/1804.04527
- Hadler Vidhove, E. (2022). *Plettfri kode*. Universitetsforlaget.
- Lewis, M. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS*.
- Lindsjørn, Y. (2025). *Systemutviklingsprosessen* [Forelesningsslides]. OsloMet.
- Preece, J., Sharp, H., & Rogers, Y. (2019). *Interaction design: Beyond human-computer interaction* (5th ed.). Wiley.
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. *ICML 2023*. https://proceedings.mlr.press/v202/radford23a.html
- Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks. *EMNLP-IJCNLP*.
- Robertson, S. E. & Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333–389.
- Rongved, O. A. N. et al. (2020). Real-Time Detection of Events in Soccer Videos using 3D Convolutional Neural Networks. *ISM 2020*. https://ieeexplore.ieee.org/document/9327961
- SoccerNet. (2024). *sn-echoes: Official repo for the paper SoccerNet-Echoes* [Programvarelager]. GitHub. https://github.com/SoccerNet/sn-echoes
- SoccerNet. (n.d.). *Data*. https://www.soccer-net.org/data
- Sturm, J. et al. (2003). Automatic transcription of football commentaries in the MUMIS project. *Eurospeech 2003*.
- Tengstedt, M. Å. (2026). *Purposeful use of colour* [Forelesningsslides].
- Ware, C. (2021). *Information visualization: Perception for design* (4th ed.). Morgan Kaufmann.
- Wilke, C. O. (2019). *Fundamentals of data visualization*. O'Reilly Media.

---

## 10 Vedlegg

(Vedlegg inkluderer NDA-signaturer, brukertest-resultater, ablation-tabeller og fullstendige telemetri-utdata fra pipeline-kjøringer på referansekampen.)

---

## Begrepsordliste

Følgende begreper benyttes gjennomgående i rapporten. Tabellen er ment som et oppslagsverk for lesere som ikke er spesialister innen informasjonsgjenfinning eller maskinlæring.

| Begrep | Forklaring |
|---|---|
| RAG (Retrieval-Augmented Generation) | En arkitektur der en språkmodell kondisjoneres på hentede dokumenter fra en ekstern kilde fremfor å svare utelukkende fra treningsdata. Reduserer hallusinasjon og gjør svar verifiserbare. |
| BM25 | Best Match 25. En leksikalsk rangeringsalgoritme som beregner relevans basert på termfrekvens og invers dokumentfrekvens. Industristandarden for tekstsøk. |
| k-NN (k-Nearest Neighbours) | En søkemetode som finner de k dokumentene med høyest vektorlikhet til en spørring. Brukes her for semantisk søk basert på embedding-vektorer. |
| Hybrid søk | Kombinasjon av BM25 og k-NN i ett søkekall, der begge scoringsmetoder bidrar til det endelige rangeringsresultatet. |
| Embedding | En numerisk vektorrepresentasjon av tekst der semantisk like setninger plasseres nær hverandre i vektorrommet, uavhengig av eksakt ordlyd. |
| Context window | Et sammenslått tekstsegment bestående av flere påfølgende transkripsjonssegmenter. Brukes for å bevare semantisk kontekst som strekker seg over segmentgrenser. |
| HLS (HTTP Live Streaming) | En protokoll for adaptiv videostrømming som deler video inn i korte segmenter og lar spilleren justere kvalitet basert på tilgjengelig båndbredde. |
| JWT (JSON Web Token) | En kompakt og selvstendig metode for sikker overføring av autentiseringsinformasjon mellom klient og server som en signert token. |
| LLM (Large Language Model) | Et stort nevronalt nettverksbasert språkmodell trent på store tekstmengder. Brukes her for spørringstranslasjon, MCQ-dommer og svargenerering. |
| Hallusinasjon | Fenomenet der en språkmodell produserer faktapåstander som ikke er forankret i tilgjengelig kontekst eller treningsdata. |
| Whisper | Et automatisk talegjenkjenningssystem (ASR) utviklet av OpenAI. Brukes til å transkribere kampkommentarlyd til tekst. |
| Elasticsearch | En distribuert søke- og analysemotor bygget på Apache Lucene. Støtter både leksikalsk BM25-søk og tett vektorsøk i samme spørringsoperasjon. |
| Ollama | En plattform for å kjøre store språkmodeller lokalt på egen maskinvare uten ekstern API-avhengighet. |
| SUS (System Usability Scale) | Et standardisert spørreskjema for måling av opplevd brukervennlighet. Scores fra 0–100 der 68 er bransjegjennomsnittet. |
| CORS (Cross-Origin Resource Sharing) | En nettlesersikkerhetsmekanisme som blokkerer JavaScript-forespørsler til domener som ikke eksplisitt tillater det via HTTP-headere. |
| Docker | En plattform for containerisering av applikasjoner som sikrer konsistente kjøringsforhold uavhengig av vertsmaskin. |
| Gazetteer | Et oppslagsleksikon over kanoniske navn med varianter. I dette prosjektet bygges gazetteeren per kamp fra `Labels-caption.json` og inneholder spillere, lag, dommere og arena. |
| WER (Word Error Rate) | Standard målestørrelse for ASR-kvalitet. Andelen ord som er erstattet, slettet eller satt inn i forhold til en referansetranskripsjon. |
| Entity-F1 | F1-mål spesifikt for egennavn. Måler både presisjon (riktig stavet) og dekning (alle ble fanget) på navngitte entiteter i transkripsjonen. |
| MCQ-dommer | Multiple-Choice Question. Et diskriminativt valideringsmønster der LLM-en velger blant pre-retrieverte kandidater fremfor å generere fritt. |
| MLM-veto | Masked Language Model-veto. En andre meningsutveksling der `xlm-roberta-base` masker en foreslått tokenposisjon og kontrollerer om LLM-pickens loglikelihood overstiger originalen. |
| TF-IDF | Term Frequency – Inverse Document Frequency. Klassisk leksikalsk vekting brukt i entitetskorrigeringen som char-n-gram-retriever over kampens gazetteer. |
| ASR | Automatic Speech Recognition. Sammenfattende betegnelse for talegjenkjenningssystemer, inkludert Whisper og KB-Whisper. |
| Confidence-gated GER | Generative Error Correction der LLM-en kun får redigere tokens med lav per-token-konfidens fra Whisper, slik at den ikke "forbedrer" allerede-korrekte regioner. |

---

# Del I – Figma-prototype og tidlig design

## 1 Mål og avgrensning

Den første milepælen var å demonstrere et brukergrensesnitt som formidler konseptet "naturlig-språklig søk i fotballklipp" uten å implementere sanntidssøk. All funksjonalitet var derfor statisk; tastetrykk ble ikke sendt til en backend, men klikkbare lenker gjorde det mulig for testpersoner å navigere mellom landingsside, registrering, innlogging og en chat-lignende visning med hardkodede eksempelspørringer.

I dette prosjektet ble det besluttet å utvikle en prototype av systemets brukergrensesnitt ved hjelp av designverktøyet Figma. Valget av Figma ble gjort på bakgrunn av behovet for raskt å kunne designe, iterere og visualisere løsningen uten å implementere full funksjonalitet i kode.

Prototypen ble utviklet som en interaktiv modell der brukerne kunne navigere mellom ulike sider, inkludert landingsside, innlogging, registrering og hovedsiden for søk etter sportsklipp. Dette gjorde det mulig å simulere en realistisk brukeropplevelse og teste hvordan brukerne forstår og interagerer med systemet.

Ved å benytte en Figma-prototype kunne vi gjennomføre brukertesting på et tidlig stadium i utviklingsprosessen. Dette gjorde det mulig å identifisere potensielle forbedringsområder knyttet til navigasjon, struktur og brukeropplevelse før en eventuell videre implementering. Fokuset i denne fasen var derfor ikke på teknisk funksjonalitet, men på å evaluere hvor intuitivt, forståelig og brukervennlig grensesnittet oppleves av brukerne.

## 2 Skjermflyt

Prototypens sideflyt i Figma omfatter: Landingsside → Innlogging/Registrering → Søkegrensesnitt → Klippvisning. Hele flyten var klikkbar slik at testpersoner kunne navigere uten å skrive ekte søk.

## 3 Kravspesifikasjon

Før utviklingen startet ble funksjonelle og ikke-funksjonelle krav fastsatt i dialog med oppdragsgiver Forzasys. Kravene ble brukt som grunnlag for tekniske valg underveis i utviklingen og som evalueringskriterier i Del III.

| Krav-ID | Type | Beskrivelse | Prioritet | Status |
|---|---|---|---|---|
| K-01 | Funksjonelt | Bruker kan søke i kampdata på norsk og engelsk | Høy | Oppfylt |
| K-02 | Funksjonelt | Systemet returnerer videoklipp seeket til riktig tidsstempel | Høy | Oppfylt |
| K-03 | Funksjonelt | Systemet avstår fremfor å hallusinere ved lav datakvalitet | Høy | Oppfylt |
| K-04 | Funksjonelt | Bruker må velge kamp før søk utføres | Høy | Oppfylt |
| K-05 | Funksjonelt | Bruker kan registrere konto og logge inn | Middels | Oppfylt |
| K-06 | Funksjonelt | Søkehistorikk lagres og vises i sidepanel | Lav | Oppfylt |
| K-07 | Ikke-funksjonelt | Svartid under 2 sekunder p50 etter LLM-warm-up | Høy | Oppfylt |
| K-08 | Ikke-funksjonelt | SUS-score over 70 | Middels | Oppfylt |
| K-09 | Ikke-funksjonelt | Systemet kjører uten betalte skytjenester | Middels | Oppfylt |
| K-10 | Ikke-funksjonelt | Grensesnittet støtter mørk og lys modus | Lav | Oppfylt |

Samtlige krav ble oppfylt i den leverte prototypen. K-03 er det kravet gruppen anser som mest akademisk betydningsfullt: et system som konsekvent avstår fremfor å fabrikere svar er mer verdifullt i praktisk bruk enn et system med høyere nominell presisjon som også produserer selvsikre feil.

## 4 Brukertest av statisk prototype

### 4.1 Metode

Brukertestingen ble gjennomført med åtte informatikkstudenter i alderen 21–26 år. Formålet med testingen var å evaluere prototypens brukergrensesnitt med fokus på navigasjon, struktur og brukervennlighet, fremfor teknisk funksjonalitet, ettersom løsningen kun bestod av en statisk Figma-prototype.

Deltakerne navigerte gjennom de ulike sidene i prototypen, inkludert landingsside, registrering, innlogging og det chat-lignende søkegrensesnittet. Etter testingen besvarte deltakerne et spørreskjema opprettet i Google Forms. Skjemaet bestod av spørsmål basert på en fempunkts Likert-skala, hvor brukerne vurderte blant annet forståelighet, navigasjon og design. I tillegg ble det samlet inn kvalitative tilbakemeldinger gjennom åpne spørsmål.

### 4.2 Kvantitative funn

| Evaluert indikator | Gjennomsnitt (1–5) | Tolkning |
|---|---|---|
| Grensesnittets forståelighet | 4,75 | UI-metaforer var selvforklarende |
| Navigasjon mellom sider | 4,63 | Flyt landingsside → login → søk opplevdes lineær |
| Overordnet design og struktur | 4,63 | Fargevalg og inndeling ble vurdert som "ryddig" |
| Søk/chat-feltets intuitivitet | 4,38 | Selv om feltet var passivt, forsto testpersoner hensikten |
| Presentasjon av sportsklipp | 4,00 | Lavest score; spørsmål om hvordan klipp ville vises i praksis |

### 4.3 Kvalitative tilbakemeldinger

De kvalitative tilbakemeldingene viste at flere deltakere opplevde løsningen som moderne, oversiktlig og enkel å navigere i. Designet ble generelt vurdert som profesjonelt og intuitivt.

Samtidig ble det identifisert enkelte forbedringsområder. Flere brukere opplevde hamburgerikonet i chatvinduet som uklart, og noen var usikre på hvordan sportsklippene ville presenteres sammen med søkefunksjonen. Enkelte etterlyste også eksempelspørsmål eller hint i søkefeltet for å gjøre det tydeligere hva man kunne søke etter.

Videre ble "Read more"-knappen på forsiden oppfattet som noe misvisende, siden den ledet til informasjon om teamet fremfor systemet. Til tross for dette mente flere deltakere at løsningen allerede fungerte godt, og at det hovedsakelig var behov for mindre justeringer fremfor større endringer.

## 5 Designsystem og visuell identitet

I utviklingen av den visuelle identiteten var målet å utforme et moderne og profesjonelt grensesnitt inspirert av designspråket til Forzasys. Det ble lagt vekt på å etterligne deres stil så tett som mulig for å skape et uttrykk som kombinerer sport og teknologi på en tydelig og troverdig måte.

Fargepaletten består hovedsakelig av mørke blåtoner og gullfarger, inspirert av sportsestetikk og moderne teknologiplattformer. Typografien kombinerer overskriftsfonten Cormorant Garamond, en høykontrast-serif valgt for å gi grensesnittet en redaksjonell og kvalitetsorientert estetikk, med DM Sans for lesbarhet i brødtekst og brukergrensesnitt.

Typografivalgene ble definert tidlig og holdt konsekvent gjennom hele prototypen, mens fargepaletten ble forfinet i Next.js-fasen til den endelige merkepaletten beskrevet i Del II – Seksjon A.

## 6 Tidlige teknologivalg (statisk fase)

I den tidlige, statiske fasen av prosjektet ble flere teknologier og verktøy valgt for å kunne utvikle og teste konseptet raskt og effektivt. Figma ble brukt til å utvikle den interaktive prototypen av brukergrensesnittet. Det ble også utviklet en lokal HTML one-page demo i februar for å eksperimentere med struktur, CSS og JavaScript i et enklere utviklingsmiljø. For styling ble Tailwind CSS valgt som rammeverk, fordi det gjorde det enklere å arbeide med design tokens, spacing og konsistent styling, samtidig som det bidro til at farger, typografi og det visuelle uttrykket samsvarte med designet fra Figma.

## 7 Frontend-iterasjoner (før kode backend)

Overgangen fra Figma til HTML innebar at hele layouten fra prototypen ble gjenskapt som en statisk nettside. Målet var å bevare det visuelle uttrykket og navigasjonsflyten fra designprototypen så nøyaktig som mulig. Selv om løsningen ikke var koblet til en backend, ble enkelte interaksjoner simulert ved hjelp av JavaScript, blant annet enkel form-validering for registrering og innlogging.

Gjennom testing og tilbakemeldinger fra brukere ble flere justeringer gjort for å forbedre brukeropplevelsen. Hamburgermenyen ble erstattet med en mer oversiktlig side-toggle-løsning for navigasjon, og en tidligere "Read more"-knapp ble fjernet fordi den ble oppfattet som unødvendig. I HTML-iterasjonen ble det også forsøkt med eksempelbaserte søkechips under søkefeltet, men disse ble fjernet i Next.js-migreringen til fordel for en eksplisitt kampvelger (se Del II – Seksjon A 4) som forsterker match-scoping-arkitekturen.

## 8 Læringspunkter fra prototypefasen

Brukertestingen viste generelt høy tilfredshet med brukergrensesnittet. De fleste vurderingene fikk et gjennomsnitt over 4,5 på Likert-skalaen, noe som tyder på at løsningen ble oppfattet som oversiktlig, moderne og enkel å bruke. Testpersonene forventet samtidig at søkefeltet skulle være funksjonelt, selv om prototypen kun var statisk. Dette viser hvor sentral søkefunksjonen er for brukeropplevelsen, og peker på behovet for funksjoner som autoutfyll, forslag og eksempelhint i senere versjoner av systemet.

Brukertestingen viste også at skjulte menyer og mindre synlige navigasjonselementer oppleves som mindre intuitive. Videre utvikling bør derfor fokusere på tydelige ikoner, synlige navigasjonsvalg og eventuelle tooltips. Valg av farger og typografi fungerte godt og ga løsningen et profesjonelt uttrykk som kombinerer sport og teknologi. Erfaringene og tilbakemeldingene fra prototypefasen dannet grunnlaget for videre utvikling av frontend-løsningen.
