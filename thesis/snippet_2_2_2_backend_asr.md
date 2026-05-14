# Snippet — 2.2.2 Backend (utvidelse om ASR-rensepipeline)

> Limes inn etter den eksisterende Python-teksten i 2.2.2 Backend.
> Beholder den eksisterende Python-introduksjonen som er.

---

Backenden i prosjektet består av to samspillende deler. Den ene delen er ASR-rensepipelinen, som tar imot rå transkripsjoner fra Whisper og produserer renset, strukturert tekst klar for indeksering. Den andre delen er søkemotoren basert på Elasticsearch, som tar imot den rensede teksten og gjør den tilgjengelig for spørringer fra frontend.

ASR-rensepipelinen er bygget i Python og bruker en rekke åpne biblioteker for talegjenkjenning, naturlig språkbehandling og maskinlæring. Pipelinen er delt inn i flere selvstendige moduler som hver har et tydelig avgrenset ansvar, slik at det er enkelt å bytte ut, justere eller deaktivere enkeltkomponenter uten at resten av systemet påvirkes. Modulene kobles sammen i en sentral koordinator som styrer rekkefølgen og videreformidler data mellom dem. Designet legger vekt på at alle terskler og parametre skal være konfigurerbare på ett sted, og at hver bug-fiks skal følges av en tilhørende test. Dette er gjort for å gjøre videre utvikling og vedlikehold etter prosjektets slutt enklere for Forzasys.
