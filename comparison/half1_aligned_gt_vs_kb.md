# Content-aligned GT vs KB-Whisper (NEW URL audio) — half 1

Each row pairs a GT segment with the KB segment that has the most overlapping content words (Jaccard on tokens >= 4 chars). "jac" is the Jaccard score; 1.00 = identical word sets, 0.00 = no overlap.

| GT time | GT text | KB time | KB text | jac |
|---|---|---|---|---|
| 0-8s | Nu är det likadant som han på sistone med landslagsamlingen också. Der han inte fått plats. | 7-9s | Att han inte fick plats. | 0.25 |
| 8-13s | Men nu som avskättsmatchen är igång. | 11-14s | Men nu så. Avskedsmatchen är igång! | 0.33 |
| 13-18s | Det här är en halvlurig boll på grund av Fredrik Nissen. | 24-27s | Vad händer med bollen där? Det här är en halvlurig boll. | 0.29 |
| 18-22s | han gør sin fjarde start | 30-33s | Matchen, det är hans fjärde start. | 0.20 |
| 28-33s | Schyberg kliver in framför Flataker. | 40-43s | Holmberg kliver in framför Flathaker. | 0.33 |
| 33-37s | Det är en bra boll. | 1689-1692s | Men även om det är på boll så, ja, det ser vi sen | 0.50 |
| 38-43s | Lundberg kliver in framför flathaker. | 40-43s | Holmberg kliver in framför Flathaker. | 0.60 |
| 49-52s | Mikkjal Thomassen. | — | — | 0.00 |
| 52-55s | De har mycket rykten där också. De har ju kontrakt nästa år också. | 50-55s | Odomichal Tomasen, har mycket rykten där också. | 0.43 |
| 55-59s | Men 7-trendaren på 7 år. | — | — | 0.00 |
| 59-63s | Han har ju haft att vänta som lämna efter säsongen. | 60-66s | ?? vänta tills han lämnar efter säsongen, får se hur det blir med den saken. | 0.33 |
| 63-66s | Vi får se hur det blir med den saken. | 60-66s | ?? vänta tills han lämnar efter säsongen, får se hur det blir med den saken. | 0.29 |
| 66-72s | Någon har ju kopplat ihop med dansken Søren Krog och portugisaren Joao Enriques men dom to skrevs jo av . | 66-71s | Gnaget har ju kopplats ihop med dansken Sören Krog, och Porte Visa och Enrique. | 0.19 |
| 72-76s | Det är lite expressen. | 71-76s | Men de två skrevs ju av, enligt Expressen. | 0.25 |
| 81-88s | Vi kan snacka om Fredrik Wisur Hansen också som kommer att lämna scouting och rekryteringsansvariga. | 86-89s | Fredrik Wiesoransson också, som kommer att lämna. Scouterna och rekryteringsansvariga. | 0.45 |
| 89-92s | Hansen ska bli ersätta där. | 90-93s | Vem ska ersätta där? | 0.50 |
| 92-96s | Det kan bli ett annat AIK nästa år. | 93-97s | Det kan bli ett annat AIK nästa år, men... | 1.00 |
| 100-106s | Här och nu är det laget som lagar fjärde platsen. | 30-33s | Matchen, det är hans fjärde start. | 0.14 |
| 106-110s | Nissen som nickar upp till Nordfeldt. | 108-110s | En isare som nickar hem till Nordfeldt. | 0.60 |
| 111-118s | Suschor med en halv snåla bollen. | 116-120s | Ett sus går med den halvsnåla bollen. | 0.20 |
| 128-132s | Flataker i bakken, får frispark. | 131-133s | Backen får frispark. | 0.25 |
| 141-145s | Saletros. | 883-890s | Här då! Saletros. AH:HV fanns utvändigt för Saletros. | 0.25 |
| 160-164s | Besirovic. | 560-563s | 65 stycken för Besirovic. 411 stycken | 0.50 |
| 170-180s | Mads Thychosen får ta några kliv till inkastpositionen där ute på Halmstads planhalva. | 180-192s | Rastikusen får ta några kliv ut till inkastpositionen där ute på Almstads plan Alvar. | 0.33 |
| 180-184s | Erlandsson. | 356-357s | Erlandsson. | 1.00 |
| 184-189s | Kåren som sitter på utgående kontrakt. | 205-208s | Kåren som sitter på utgående kontrakt. | 1.00 |
| 253-257s | Thychosen ser etter Saletros. | 644-647s | Men gör det där elegant mot Thychosen. | 0.25 |
| 276-279s | Thychosen, til Celina | 644-647s | Men gör det där elegant mot Thychosen. | 0.33 |
| 280-283s | Maenpaa en tå före Ali  | 283-285s | Per. 2 före Walid. | 0.33 |
| 285-287s | En frispark för Granath. | 287-289s | Frispark för Granath. | 1.00 |
| 293-298s | Det är ett AIK som knappast varit det bästa laget. | 295-300s | Det är ett AIK som knappast varit det bästa. | 0.75 |
| 298-301s | Snarare tvärtom vad gäller dom här parametrarna, | 300-303s | Snarare tvärtom vad gäller de här parametrarna. | 1.00 |
| 301-304s | passningsprocent, tid där bollen är i spel. | 303-307s | Passningsprocent. Tid där bollen är i spel. | 1.00 |
| 304-308s | Där är de  aller sist med dom här långa inkastor och så vidare. | 307-310s | Där är de allra sist med de här långa inkastarna. | 0.29 |
| 308-313s | När det gäller klara avslut, där det bara är en spelare som ställs mot en målvakt. | 310-315s | Vad gäller klara avslut där det bara är en spelare som ställs mot en målvakt. | 1.00 |
| 313-316s | Kontringsavslut, då avslut via högpress. | 315-319s | Omkringsavslut - då avslut med hög press. | 0.20 |
| 316-319s | Tittar man till där de har varit vassa. | 319-322s | Tittar man till där de varit vassa. | 1.00 |
| 319-322s | Avslut emot, det är ju mycket defensiva. | 322-324s | Avslut emot. Det är mycket defensiva. | 1.00 |
| 322-325s | Klara avslut emot, mål på fasta situationer framåt då. | 324-325s | Klara avslut emot. | 0.50 |
| 325-328s | Insläppta mål på fasta situationer. | 328-330s | Insläppta mål på fasta situationer. | 1.00 |
| 329-332s | Antal lyckade inlägg. | 330-333s | lyckade inlägg. | 0.67 |
| 332-337s | Ni får en bild av vad det är för AIK vi har sett. | 333-340s | Ni får en bild av vad det är för AIK vi har sett den här säsongen. | 0.67 |
| 337-340s | Den här säsongen. | 333-340s | Ni får en bild av vad det är för AIK vi har sett den här säsongen. | 0.33 |
| 344-348s | för er som inte har hengt med hela vägen. | 345-348s | Det är någon som inte har hängt med hela vägen. | 0.50 |
| 351-356s | Kurtulus, inte Schyberg, Erlandsson . | 356-357s | Erlandsson. | 0.25 |
| 357-360s | De på mittfältet, spindeln i nätet . | 360-362s | Spindeln i nätet. | 0.67 |
| 362-363s | För HBK. | — | — | 0.00 |
| 363-366s | Den bollen är intressant för Granath mot Csongvai. | 366-377s | Att den bollen är intressant för Granath. | 0.75 |
| 366-369s | Tunna Csongvai, Granat igenom. | — | — | 0.00 |
| 381-383s | Vilken möjlighet för Halmstad. | 382-389s | Vilken möjlighet för Halmstad. Boman hittar den här passningen genom skäraren. | 0.38 |
| 383-385s | Boman som hittar den här passningen. | 382-389s | Vilken möjlighet för Halmstad. Boman hittar den här passningen genom skäraren. | 0.38 |
| 385-387s | Genomskärrar den. | — | — | 0.00 |
| 387-391s | Löpningen från Granath som gör allt rätt egentligen. | 390-396s | från Granath, som gör allt rätt förutom avslutet. | 0.50 |
| 391-393s | Förutom avslutet. | 390-396s | från Granath, som gör allt rätt förutom avslutet. | 0.33 |
| 406-407s | Kaib. | 408-417s | Kaib. Skrämselläge för Halmstad. | 0.33 |
| 410-415s | Skrämselläget för Halmstad. | 1170-1173s | 1-0 Halmstad! | 0.50 |
| 415-417s | Alldeles nyss. | 417-419s | Alldeles nyss. | 1.00 |
| 431-433s | Illary. | — | — | 0.00 |
| 435-437s | Som går upp. | — | — | 0.00 |
| 437-440s | Vänder ut till Kaib och tar en ny löpning. | 766-770s | Får loss den till Khaid. Kommer med en löpning från Jadoha. | 0.22 |
| 440-442s | Får tillbaka bollen. | 116-120s | Ett sus går med den halvsnåla bollen. | 0.33 |
| 442-444s | Mot Csongvai. | — | — | 0.00 |
| 444-446s | Tilbake til Kaib som har en bra inläggsfot. | 408-417s | Kaib. Skrämselläge för Halmstad. | 0.20 |
| 446-448s | Isherwood rensar rakt for Saletros. | 434-450s | Tillbaka till Khaib som har en bra inläggsfot. Ichuoud rensar rakt på Salétros. | 0.20 |
| 448-450s | Undan till slut. | 450-452s | Går undan till slut. | 1.00 |
| 459-461s | Allansson. | 647-653s | Kommer till lill-digg också! Yrsjö mot bort. Rätt ut till Manpapp och Allansson! | 0.11 |
| 461-463s | Mäenpaa. | — | — | 0.00 |
| 469-471s | Kurtulus. | 1650-1654s | En rejäl satsning från Kurtulus – som sen springer in! | 0.20 |
| 472-474s | Schyberg. | — | — | 0.00 |
| 474-477s | Granath skakar av seg ALi. | 287-289s | Frispark för Granath. | 0.33 |
| 477-479s | Snäl ball sen ut mot Boman. | 2510-2513s | Chauvey med diagonalare som Boman får lite problem med. | 0.14 |
| 479-482s | Som ändå löser det där på något sätt. | 480-484s | sen ut mot Boban, som ändå löser det där på något sätt. | 0.80 |
| 501-503s | Allansson. | 647-653s | Kommer till lill-digg också! Yrsjö mot bort. Rätt ut till Manpapp och Allansson! | 0.11 |
| 525-526s | Schyberg. | — | — | 0.00 |
| 526-528s | Pressad her av Saletros. | 527-530s | Pressad här av Salétros. | 0.33 |
| 531-534s | Schyberg kan ta ett problemtäck ut i innspärk. | — | — | 0.00 |
| 534-536s | Halmstad. | 1170-1173s | 1-0 Halmstad! | 1.00 |
| 543-547s | Här ser vi då dom som är bekräftad att han lämnar Aik. | 60-66s | ?? vänta tills han lämnar efter säsongen, får se hur det blir med den saken. | 0.12 |
| 547-549s | Vi kastar in Dino Besirovic också. | 257-259s | Som är också med oss. | 0.25 |
| 549-551s | Inte bekräftat där. | 2160-2162s | Som inte har... | 0.50 |
| 551-556s | Men vi kastade in honom här vad gäller bara antal tävlingsmatcher. | 554-560s | vi kastade in honom här vad gäller antal tävlingsmatcher för AIK | 0.83 |
| 556-558s | För AIK. | — | — | 0.00 |
| 558-560s | 65 stycken från Besirovic. | 560-563s | 65 stycken för Besirovic. 411 stycken | 0.67 |
| 560-563s | 411 som eventuellt då. | 563-565s | som eventuellt då... | 1.00 |
| 563-565s | Frågetecknet kring Besirovic. | 566-568s | frågetecknet kring Besirovic | 1.00 |
| 565-567s | Som kan försvinna. | 568-570s | som kan försvinna. | 1.00 |
| 567-570s | Och det var ju ytterligare. | 673-676s | ytterligare ett sådant skrämselskott. | 0.33 |
| 574-576s | Matcher med Sotirios Papagiannopoulos. | 570-579s | Det var ytterligare matcher med Sotirios Papagionnopoulos. | 0.40 |
| 576-579s | Han kom upp i 589 matcher. | 579-582s | Han kom upp i 589 matcher. | 1.00 |
| 579-581s | Men så förlängde han. | 582-584s | Men så förlängde han! | 1.00 |
| 581-583s | Så sent som igår. | 584-586s | Så sent som igår. | 1.00 |
| 583-588s | Till många AIK-supporters glädje. | 586-590s | Till många AIK-supportrars glädje. | 0.60 |
| 591-595s | Man får inleda den här matchen från bäcken. | 594-597s | Han får inleda den här matchen på bänken. | 0.40 |
| 599-602s | Någonstans i bakgrunden. | 600-603s | Någonstans i bakklubben. | 0.33 |
| 605-608s | Vi undrar också om han kommer att få någon speltid | 609-610s | Undrar om han kommer få någon speltid ... | 0.80 |
| 608-610s | Vi hörde Mikkjal Thomassens tankar | — | — | 0.00 |
| 610-614s | Om Gudettis chanser att få minuter här. | 613-615s | och Guidettis chanser att få minuter här. | 0.50 |
| 620-622s | Hove. | 810-840s | Rättvänd Hove i en härlig ficka för AIK:s del. Han kommer loss. | 0.14 |
| 626-628s | HBK som spelar loss. | 615-622s | KBK som spelar loss alla spel | 0.50 |
| 628-631s | Allansson med yta. Boman sticker. | 630-633s | Kommer i yta, bommen sticker. | 0.20 |
| 638-639s | Intressant. | 719-720s | Om någon är intressant. | 0.50 |
| 639-642s | Men Kaib lyckas inte suga ner den. | 640-644s | Intressant, men Kaji lyckas inte suga ner den. | 0.50 |
| 642-644s | Men gör det där elegant mot Thychosen. | 644-647s | Men gör det där elegant mot Thychosen. | 1.00 |
| 644-646s | Kommer till inlägg också. | 900-902s | Kommer ett inlägg här. | 0.50 |
| 646-647s | Isherwood bort. | 938-941s | Den satsas bort med skallen. | 0.25 |
| 647-650s | Rett ut til Maenpaa och Allansson. | 647-653s | Kommer till lill-digg också! Yrsjö mot bort. Rätt ut till Manpapp och Allansson! | 0.09 |
| 671-672s | Ytterligare ett sånt. | 673-676s | ytterligare ett sådant skrämselskott. | 0.25 |
| 672-674s | Strömselskott. | — | — | 0.00 |
| 678-681s | Lite småoroligt. | 681-686s | Lite småoroligt på läktarna. | 0.67 |
| 681-683s | Läktarna. | 681-686s | Lite småoroligt på läktarna. | 0.33 |
| 683-686s | Lite nervöst i alla fall med den här. | 686-689s | Lite nervöst i alla fall med den här ... | 1.00 |
| 688-691s | Inte särskilt förtroendeingivande starten. | 690-694s | Inte särskilt förtroendeingivande starten för AIK. | 1.00 |
| 691-692s | Från AIK. | 2497-2500s | Vi får se från Lingby. | 0.50 |
| 692-694s | Eller snarare. | 694-696s | Eller snarare... | 1.00 |
| 694-695s | Man ska inte. | 2160-2162s | Som inte har... | 1.00 |
| 695-698s | Någonstans från Hamstad med två bra lägen. | 696-700s | En önskande start ändå från Halmstad med två bra lägen. | 0.25 |
| 708-711s | Det måste komma mycket. | 1179-1180s | Då måste AIK svara. | 0.25 |
| 711-714s | Det är lite liten. | 716-717s | Väldigt liten. | 0.33 |
| 714-716s | Den bollen är intressant. | 366-377s | Att den bollen är intressant för Granath. | 0.67 |
| 716-718s | Allansson lyckas inte få tag i den. | 720-722s | Alansson lyckas inte få tag i den. | 0.50 |
| 718-719s | Istället Hove. | 722-726s | Istället HV, ser löpningen från Besirovic på andra kanten. | 0.14 |
| 719-722s | Ser löpningen från Besirovic på andra kanten. | 722-726s | Istället HV, ser löpningen från Besirovic på andra kanten. | 0.83 |
| 722-725s | Bollen kommer bakom Besirovic. | 726-728s | Bollen kommer bakom Besirovic. | 1.00 |
| 725-728s | Han får hämta in den där. | 730-732s | Som får hämta in den där. | 1.00 |
| 728-730s | HBK samlat. | 733-734s | Robert K samlat. | 0.50 |
| 737-738s | Nissen. | 1140-1142s | Med Nissen. | 1.00 |
| 738-740s | Vidare. | 2878-2879s | och så vidare. | 1.00 |
| 740-742s | Från Celina. | 2497-2500s | Vi får se från Lingby. | 0.33 |
| 743-745s | Morgen. | — | — | 0.00 |
| 761-764s | Illary mot två AIK-spelare får löst den till Kaib. | 120-123s | till Ahmederli och Ali. | 0.17 |
| 764-769s | Kommer en löpning här från Yeboah. | 766-770s | Får loss den till Khaid. Kommer med en löpning från Jadoha. | 0.38 |
| 832-839s | Här da Hove rätt vänd i en härlig ficka för AIKs del. Kommer loss. Söker Flataker. | 810-840s | Rättvänd Hove i en härlig ficka för AIK:s del. Han kommer loss. | 0.55 |
| 844-846s | Saletros där och trycker. | 846-848s | Ser ledtråd där och trycker. | 0.33 |
| 847-850s | Kommer bollen mot Yeboah. Ensam ber uppe mot Isherwood. | 726-728s | Bollen kommer bakom Besirovic. | 0.25 |
| 862-865s | Thychosen | 644-647s | Men gör det där elegant mot Thychosen. | 0.50 |
| 878-878s | Csongvai | — | — | 0.00 |
| 879-881s | Pascal Gregor upp på den. | 881-883s | Skall Gregor upp på den? | 0.33 |
| 881-882s | Här har Saletros. | 883-890s | Här då! Saletros. AH:HV fanns utvändigt för Saletros. | 0.25 |
| 883-888s | Hove fanns utvändigt för Saletros. | 883-890s | Här då! Saletros. AH:HV fanns utvändigt för Saletros. | 0.60 |
| 889-892s | Ställde ett boll här mot lagkaptenen. | 891-894s | Han ställer ett boll här, mot lagkaptenen. | 0.50 |
| 892-894s | Som kan sleppa ut som kan slå inn själv | 897-900s | Han kan släppa ut. Han kan slå in själv. | 0.33 |
| 894-896s | Kommer ett inlägg här. | 900-902s | Kommer ett inlägg här. | 1.00 |
| 896-898s | Ut av Schyberg | — | — | 0.00 |
| 898-900s | Ali up In i gröten igen. | 904-906s | In i gröten igen! | 1.00 |
| 900-902s | Allansson dit. | 647-653s | Kommer till lill-digg också! Yrsjö mot bort. Rätt ut till Manpapp och Allansson! | 0.11 |
| 910-912s | Retning Sirius | — | — | 0.00 |
| 916-918s | Tar emot tabelljumbon Värnamo. | 324-325s | Klara avslut emot. | 0.20 |
| 918-920s | som kommer att spela Superettan . | 900-902s | Kommer ett inlägg här. | 0.25 |
| 920-922s | Fotboll nästa år. | 93-97s | Det kan bli ett annat AIK nästa år, men... | 0.33 |
| 926-928s | Celina. | — | — | 0.00 |
| 928-930s | Den bollen är intressant mot Flataker. | 366-377s | Att den bollen är intressant för Granath. | 0.50 |
| 932-934s |  Bort. | 938-941s | Den satsas bort med skallen. | 0.33 |
| 934-936s | I skallen. | 938-941s | Den satsas bort med skallen. | 0.33 |
| 938-940s | Besirovic da. | 560-563s | 65 stycken för Besirovic. 411 stycken | 0.50 |
| 942-944s | Celina. Tre svartklädda där inne. | 947-951s | Selina, tre svartklädda där inne. Ali löper in. | 0.40 |
| 944-946s | Ali loper inn. | — | — | 0.00 |
| 946-948s | Her kommer bollen mot Thychosen. | 955-959s | Där kommer bollen mot Thychosen, som tagit plats vid straffpunkten. | 0.50 |
| 948-950s | Som har tagit plats där på straffpunkten | 955-959s | Där kommer bollen mot Thychosen, som tagit plats vid straffpunkten. | 0.50 |
| 954-956s | Lite lösare boll där Maenpaa. | 960-966s | Lite lösare boll där, men hade det varit 2-1-läge... | 0.43 |
| 956-958s | Det hade varit två mot ett läge. | 960-966s | Lite lösare boll där, men hade det varit 2-1-läge... | 0.29 |
| 958-960s | Han har spelat med små marginaler. | 966-968s | Då hade han spelat med små marginaler. | 0.67 |
| 960-962s | Aik även där. | 968-971s | AIK även där. | 1.00 |
| 964-966s | Thychosen. Vad hittar han då? | 644-647s | Men gör det där elegant mot Thychosen. | 0.33 |
| 970-972s | Det är en liten boll. | 716-717s | Väldigt liten. | 0.33 |
| 978-980s | Hove Bakken. | 810-840s | Rättvänd Hove i en härlig ficka för AIK:s del. Han kommer loss. | 0.12 |
| 992-994s |  Bättre Aik-period nu. | 990-996s | Bättre AIK-period nu! | 1.00 |
| 996-998s | Senaste fem minutane. | 999-1001s | De senaste fem minuterna. | 0.33 |
| 1026-1028s | innkast. | — | — | 0.00 |
| 1064-1079s | Han har en kvarten in, ganska tillknäppt på flera redor. | 1077-1080s | Kvarten in. Ganska tillknäppt på flera arenor | 0.67 |
| 1079-1089s | Sirius som har tagit ledningen. Och Hammarby mot Elfsborg. | 1080-1081s | Hammarby mot Elfsborg. | 0.40 |
| 1090-1093s | Och AiK kan bryta dödläget här, Celina. | 1084-1095s | AIK kan bryta dödläget här, se linan! | 0.50 |
| 1095-1097s | En bra löpning bakom honom. | 1097-1100s | Vi har en bra löpning bakom honom ifrån attacker. | 0.60 |
| 1098-1101s | Och den här löpningen är intressant från Granath. | 1100-1103s | Den här löpningen är intressant, från Granath. | 1.00 |
| 1101-1105s | En rejäl kamp med Nissen. Granath kommer loss. Granath är ren. | 1103-1105s | En rejäl kamp med Nissen! | 0.50 |
| 1105-1109s | Granath avslutar i mål. Det är ledning, Helsingborg eller Halmstad. | 1106-1110s | Granath är ren! Granath avslutar i mål! Det är ledning Helsingborg! | 0.67 |
| 1111-1114s | HBK upp i ledning. | 1114-1117s | HBK - upp i ledning! | 1.00 |
| 1114-1118s | HBK upp i ledning. Villiam Granath. | 1106-1110s | Granath är ren! Granath avslutar i mål! Det är ledning Helsingborg! | 0.40 |
| 1129-1134s | Hans fjärde mål för säsongen. Han chockar Nationalarenan. | 1131-1137s | Hans fjärde mål för säsongen. Han chockar nationalarenan. | 1.00 |
| 1135-1139s | Efter en löpduell, brottnings match med nissen. | 1137-1140s | Efter en löpduell brottningsmatch... | 0.33 |
| 1144-1148s | Han har ingen fel på avslutet den här gången. | 1153-1155s | Inget fel på avslutet. | 0.25 |
| 1148-1152s | Det kändes som om Nissen hade koll på läget där någonstans. | 1155-1160s | Det kändes som att Nissen hade koll på läget. Där någonstans! touchen från Granath. | 0.67 |
| 1152-1155s | Så kommer den där touchen från Granath. | 1105-1106s | Granath kommer loss! | 0.40 |
| 1156-1161s | Skakar av sig Nissen. Och är ensam mot Nordfeldt. | 1161-1164s | Han skakar av sig Nissen. | 0.50 |
| 1164-1167s | Bara raka in. 1-0 Halmstad. | 1170-1173s | 1-0 Halmstad! | 0.33 |
| 1167-1172s | Nu måste AIK svara. Kanske genom en fast situation. | 1179-1180s | Då måste AIK svara. | 0.33 |
| 1172-1175s | Matchens första hörna. | 1182-1185s | Kanske ger de en fast situation, matchens första hörna. | 0.50 |
| 1197-1200s | Det är en bra match. | 1464-1466s | per match. | 1.00 |
| 1212-1215s | Flataker nære at skalla vidare | 2878-2879s | och så vidare. | 0.25 |
| 1217-1220s | Inlägget från Csongvai. | 2497-2500s | Vi får se från Lingby. | 0.25 |
| 1220-1224s | Men han är alldeles för lång och hårt slagen. | 1217-1224s | Läge för Chant Buyen att lyfta in men är alldeles för lång och hårt slagen. | 0.50 |
| 1227-1231s | Just nu Göteborg på den fjärde platsen. | 2797-2799s | just nu Göteborg har ... | 0.50 |
| 1248-1251s | Jublats där. | — | — | 0.00 |
| 1251-1254s | På Gamle Ullevi, vil jag lova. | 1254-1256s | På Gamla Ullevi, vill jag lova. | 0.40 |
| 1317-1319s | Rätt upp i luften. | 1320-1325s | Rätt upp i luften. Zildovic vidare till Selina. | 0.33 |
| 1320-1322s | Besirovic vidare till Celina. | 741-743s | Vidare till Sina. | 0.40 |
| 1332-1335s | Kommer till inlägg. Flatakern nikker inåt. | 1335-1338s | Kommer till inlägg. Flatthacken nickar inåt! | 0.50 |
| 1336-1338s | Skottläge för Thychosen drar till på volley. | 1338-1342s | Skottläge för Thychosen, drar till på volleyn. | 0.67 |
| 1343-1345s | Tänker ut till ett inkast. | 1346-1348s | Tom Groth, till ett inkast. | 0.50 |
| 1381-1384s | Det blir redan små irriterat. | 1802-1804s | Blir det farligt här då, kanske. | 0.20 |
| 1387-1393s | Lite halvknacket passningsspel från hemma laget och inte minst underläge. | 1393-1395s | och inte minst underläge! | 0.33 |
| 1394-1401s | Efter Granaths ledningsmål från HBK. | 1396-1403s | Efter Granaths ledningsmål. Frobeca. | 0.60 |
| 1429-1434s | Och medans Isherwood behover plastras om. | — | — | 0.00 |
| 1440-1446s | Så kan vi väl titta till de här siffrorna. Granatn är ju där. Topp tre. | 1443-1445s | titta på de här siffrorna. | 0.40 |
| 1446-1449s | Vad gäller antal höghastighetslöpningar? Det är ju två löpmaskiner där. | 1448-1451s | höghastighetslöpningar. Det är ju två löpmaskiner här. | 0.50 |
| 1449-1452s | Niklas Röjkjaer som är etta där. Han finns ju inte ens kvar i allsvenskan. | 1451-1453s | Niklas Röjker som är etta där han finns ju | 0.38 |
| 1452-1459s | Så han skulle vi kunna stryka och dra upp Saletros på första plats och William Granatn på andra plats. | 1456-1458s | skulle vi kunna stryka och dra upp Saletros | 0.44 |
| 1459-1464s | I snittet alltså 47 de två per match. | 1464-1466s | per match. | 0.33 |
| 1466-1472s | Det tänker man kanske inte med Saletros men det är många höghastighetslöpningar det. | 1467-1470s | Där tänker man kanske inte med Saletros | 0.67 |
| 1482-1490s | Då ser vi Lilla Stå som gör premiär idag. | 1488-1491s | Lilla stå, som gör premiär idag. | 1.00 |
| 1491-1493s | Ståplatsläktare. | 1493-1495s | Ståplats/läktare... | 1.00 |
| 1494-1498s | För de mindre. | 1496-1499s | för de mindre. | 1.00 |
| 1506-1511s | Premiär för det. Här är den sista omgången. | 1511-1516s | Här är den sista omgången. | 0.67 |
| 1512-1522s | Flaggor och lite konfetti och så vidare. | 1516-1530s | Det är fartgård och lite konfetti och så vidare. | 0.60 |
| 1543-1550s |   bra från Besirovic som sen sättar upp den här omställningsmöjligheten. | 722-726s | Istället HV, ser löpningen från Besirovic på andra kanten. | 0.25 |
| 1550-1555s | Men Celina kör fast i Pascal Gregor. | 1530-1556s | ...angrytningen är bra för Lecidovitj som iscensätter den här omställningsmöjligheten, men Selina kör fast i Pascal Gregor. | 0.33 |
| 1560-1563s | Kurutlus tar kliv. | 1560-1563s | Skönt att lossa och ta kliv. | 0.25 |
| 1564-1569s | Till Kaib rätt i ryggen på Thychosen. | 1566-1570s | Kaib i ryggen på Thychosen. | 0.60 |
| 1582-1585s | Innläget plukkes av Nordfeldt. Flattakare sticker. | 1581-1589s | Kaib har plockats av Nordfeldt. Flathaker sticker och upptäcker Nordfeldt som skickar upp den där bollen. | 0.18 |
| 1585-1590s | Upptäcker Nordfeldt som skickar upp den där bollen. Det är ingen dum boll. | 1581-1589s | Kaib har plockats av Nordfeldt. Flathaker sticker och upptäcker Nordfeldt som skickar upp den där bollen. | 0.40 |
| 1591-1596s | Landar ändå hos Erlandsson och AIK supportrarna. | 1590-1596s | Landar ändå hos Erlandsson. | 0.75 |
| 1596-1601s | Ändrar att han la vantarna på den utanför straffområdet. | 1599-1603s | Lägger vantarna på den, utanför straffområdet. | 0.60 |
| 1602-1608s | Richard Sundell. | 1609-1611s | Där är Rickard Sundell! | 0.33 |
| 1650-1654s | Kurtulus som sen springer in i Hove. | 1650-1654s | En rejäl satsning från Kurtulus – som sen springer in! | 0.33 |
| 1655-1662s | Det ser nästan ut som att han fick ett knö mot huvudet. Uppe på benen nu. | 1658-1662s | Det ser ut som att han fick en knö mot huvudet. | 0.40 |
| 1671-1677s | Celina lyfter upp mot Johan Hove som är offside. | 1673-1679s | Selina lyfter upp mot Manove, som är offside. | 0.29 |
| 1677-1682s | Ser vi situationen igen. | 1680-1686s | Ser vi situationen igen. | 1.00 |
| 1682-1687s | Först den satsningen. Den är ju brusk. | 1686-1689s | Först den satsningen, den är ju brysk. | 0.50 |
| 1687-1694s | Även om det är på boll så ser vi sen han springer in i Hove. | 1689-1692s | Men även om det är på boll så, ja, det ser vi sen | 0.50 |
| 1700-1706s | Lagets främsta målgörare. Johan Hove med sina åtta fullträffar. | 1702-1707s | Lagets främsta målgörare, Jojarova, med sina åtta fullträffar. | 0.67 |
| 1717-1722s | Titta på Grimstad. | 1443-1445s | titta på de här siffrorna. | 0.33 |
| 1722-1727s | Några kilometer herifrån Dijan Vukojevic. | 1724-1729s | Några kilometer därifrån, Dijan Vukojevic, 1-0 Degerfors. | 0.57 |
| 1796-1803s | Utanför bilden Andreas Johansson uppe och viftar. Ansvarig för de fasta situationerna. Blir det farligt här då kanskje? | 1798-1802s | Andreas Johansson är uppe och viftar, ansvarig för de fasta situationerna. | 0.58 |
| 1803-1808s | Gregor som försökte sätta ner för Mäenpaa. | 1805-1807s | Det var någon som försökte sätta ner bollen. | 0.33 |
| 1826-1831s | Besirovic som ligger kvar. | 2779-2780s | Två mål kvar. | 0.33 |
| 1831-1836s | Celina spela vidare. | 2878-2879s | och så vidare. | 0.33 |
| 1836-1841s | Inkast for Halmstad . | 1170-1173s | 1-0 Halmstad! | 0.50 |
| 1923-1925s | Landar lite olyckligt där Filip Schyberg. | 1927-1943s | han hamnar lite olyckligt där.. | 0.33 |
| 1953-1961s | Det är Fredrik Söderberg som kan konstatera, AIK-toppen då, som kan konstatera att | 1957-1970s | som kan konstatera AIK-toppen att positionen är utanför topp 4. | 0.29 |
| 1962-1967s | Just nu er utanför topp fyra. | 1957-1970s | som kan konstatera AIK-toppen att positionen är utanför topp 4. | 0.29 |
| 1969-1972s | Fredrik Wisur Hansen också ska alltid vara rekryteringschefen där. | 1971-1975s | Jag tror det blir Soro Hansen också, Scouten och rekryteringschefen där. | 0.27 |
| 1977-1979s | Som ska lämna | 1979-1981s | Som ska lämna. | 1.00 |
| 1980-1984s | och bara spelare. | 1983-1986s | Några spelare | 0.33 |
| 1985-1989s | Det är totalt flera hundra matcher tillsammans. | 1987-1990s | Vi har gått på flera hundra matcher tillsammans. | 0.67 |
| 1994-1999s | De har att jobba på under vintern. AIK. | 1997-2001s | Vad dem har att jobba på under vintern, AIK. | 1.00 |
| 2010-2014s | Celina letar alternativen. Har han med sig då? Ja, tvingas hem. Men inte ens där. | 2015-2017s | Han tvingas hem, men inte ens där! | 0.40 |
| 2015-2018s | Finns någon spelare . | 2017-2023s | finns någon spelare, som får ta på hörluren, så får de börja om. | 0.60 |
| 2019-2021s | Så får de börja om. | 2017-2023s | finns någon spelare, som får ta på hörluren, så får de börja om. | 0.20 |
| 2032-2034s | Flataker klackar vidare. | 2878-2879s | och så vidare. | 0.33 |
| 2054-2057s | Litt For lang boll sen for Thychosen . | 644-647s | Men gör det där elegant mot Thychosen. | 0.20 |
| 2059-2066s | Spelet med boll klickade inte riktigt för AIK på för sig i plan halva. | 2047-2068s | Bombens framför bollen, men spelet med boll klickar inte riktigt för AIK på offensiv planhalva. | 0.31 |
| 2069-2072s | Knapp ens på egen plan halva. | 2091-2095s | Svampens på egen planhalla. | 0.17 |
| 2084-2086s | Allansson | 647-653s | Kommer till lill-digg också! Yrsjö mot bort. Rätt ut till Manpapp och Allansson! | 0.11 |
| 2090-2093s | Det är inte ens på egen plan halva. | 2160-2162s | Som inte har... | 0.25 |
| 2093-2095s | Alltså. | 2857-2866s | han har alltså ledning 1-0. | 0.50 |
| 2108-2110s | Frispark Halmstad | 1170-1173s | 1-0 Halmstad! | 0.50 |
| 2141-2147s | 0,02 i XG för AIK. | — | — | 0.00 |
| 2147-2153s | Efter 35 minuter hemma mot Halmstad. | 2150-2154s | Efter 35 minuter hemma mot... | 0.75 |
| 2155-2162s | På den nedre halvan som inte har nånting att spela för egentligen. | 2158-2160s | På den nedre halvan. | 0.33 |
| 2164-2166s | Såklart. | 2167-2168s | Såklart. | 1.00 |
| 2168-2170s | Inte godkänt. | 2171-2172s | Inte godkänt. | 1.00 |
| 2177-2180s | Men här kanske de kan hitta på något. | 2184-2186s | Men här kanske de kan hitta på något. | 1.00 |
| 2182-2186s | För de är vassa på fasta situationer. 12 mål har de gjort där. | 2188-2192s | För de är vassa vid fasta situationer, tolv mål har de gjort där! | 0.80 |
| 2194-2199s | Halmstad. Det är laget som gjort andra först. | 2197-2204s | Halmstad. Det laget som gjort det allra färst... | 0.43 |
| 2201-2204s | Saletros vid bollen. | 116-120s | Ett sus går med den halvsnåla bollen. | 0.33 |
| 2205-2209s | Saletros vid bollen. Eller ska det bli Csongvai? | 2207-2212s | Salétros vid bollen, eller ska det bli Chongwai? | 0.33 |
| 2209-2212s | Ja, det blir Saletros i muren. | 2212-2215s | Det blir Salétros i muren... | 0.50 |
| 2212-2217s | Isherwood holder den i spel med Besirovic med läget. Och det är ett jätteläge dessutom. | 2216-2221s | Ishwood håller den i spel – Besirovic med läget! Ett jätteläge! | 0.44 |
| 2220-2223s | Men bollen fastnar någonstans på vägen mot mål. | 2222-2226s | Men bollen fastnar på vägen mot mål. | 0.75 |
| 2234-2238s | Så ska det vara. | — | — | 0.00 |
| 2324-2325s | Tillbaka också. | 257-259s | Som är också med oss. | 0.50 |
| 2326-2330s | Kan ha lite avslut. Släpper till Yeboah istället som testar och bågar den där bollen mot mål. | 2327-2330s | Han har lite avslut, släpper till Gueboa istället | 0.50 |
| 2342-2343s | Ser vi frisparken igen. | 2343-2345s | Sen ser vi frisparken igen! | 1.00 |
| 2345-2348s | Och läget för Besirovic inte minst. | 2347-2350s | Och läget för Besirovic inte minst. | 1.00 |
| 2355-2358s | Jag kan inte associerar med London. | 2160-2162s | Som inte har... | 0.33 |
| 2372-2375s | Han djupslagsens. | — | — | 0.00 |
| 2378-2378s | Vidare. | 2878-2879s | och så vidare. | 1.00 |
| 2383-2387s | Det räcker inte för att vi inte har fått den här bollen. | 2385-2388s | Men det räcker inte för att hinna upp den där bollen. | 0.60 |
| 2451-2455s | Gregor som kommer in i tufft och åker på matchens första gulan. | 2467-2478s | Det är Bonos som kommer in tufft och åker på matchens första gula. | 0.56 |
| 2455-2473s | Det är ju guld som kommer in i tufft och åker på matchens första gulan. | 2467-2478s | Det är Bonos som kommer in tufft och åker på matchens första gula. | 0.56 |
| 2511-2513s | som man får lite problem med. | 2510-2513s | Chauvey med diagonalare som Boman får lite problem med. | 0.40 |
| 2513-2516s | Han nikkade ut till AIKs andra hörn i matchen. | 2515-2517s | Läcker ut till AIK:s andra hörna i matchen. | 0.50 |
| 2517-2520s | Som sagt, AIK-fassa situationer. | 2519-2521s | Som sagt, AIK-fasta situationer. | 0.50 |
| 2521-2523s | Där är de farliga. | 2523-2525s | Där är de farliga. | 1.00 |
| 2523-2525s | Och allra främst alltså, | 2857-2866s | han har alltså ledning 1-0. | 0.25 |
| 2525-2527s | i Allsvenskan den her säsongen och | 333-340s | Ni får en bild av vad det är för AIK vi har sett den här säsongen. | 0.25 |
| 2527-2529s | inte minst, | 1393-1395s | och inte minst underläge! | 0.67 |
| 2529-2531s | tack vare Csonvais . | 2530-2534s | tack vare Chongwais högerfot. | 0.40 |
| 2531-2533s | Högerfot. | 2530-2534s | tack vare Chongwais högerfot. | 0.25 |
| 2536-2538s | Som nu ska skicka in den här. | 2538-2540s | Som nu ska skicka in den här. | 1.00 |
| 2538-2540s | Vem är siktan på då? | — | — | 0.00 |
| 2541-2543s | Det blir Hove, på den bortre ytan. | 2543-2545s | Chewhowe på den bortre ytan. | 0.40 |
| 2543-2545s | Som han huvudet på den. | 1658-1662s | Det ser ut som att han fick en knö mot huvudet. | 0.50 |
| 2546-2548s | Kurtulus tar sig dit och | 1650-1654s | En rejäl satsning från Kurtulus – som sen springer in! | 0.20 |
| 2548-2550s | får Csongvai springa över. | 2550-2553s | Chongwai springer över. | 0.20 |
| 2553-2555s | Och den här | — | — | 0.00 |
| 2555-2557s | Den hörn också, AIK tredje i matchen. | 2557-2559s | Hörnan också. Oikpos tredje i matchen. | 0.50 |
| 2571-2573s | Här kommer då Hörnan. | 2579-2581s | Där kommer då hörnan. Han gör på av. | 1.00 |
| 2573-2575s | Yeboah. | — | — | 0.00 |
| 2585-2587s | Och kolla på ytan här då. | 2587-2602s | kolla på ytan, 3 mot 2-läge! | 0.67 |
| 2587-2589s | Tre mot två läge. | 1217-1224s | Läge för Chant Buyen att lyfta in men är alldeles för lång och hårt slagen. | 0.12 |
| 2589-2591s | För att ha en bra match. | 1464-1466s | per match. | 1.00 |
| 2591-2593s | Och det är en bra match. | 1464-1466s | per match. | 1.00 |
| 2599-2601s | Och den här då, tre mot två läge. | 1217-1224s | Läge för Chant Buyen att lyfta in men är alldeles för lång och hårt slagen. | 0.12 |
| 2601-2603s | För Ali hinner inte upp med Maenpaa. | 2160-2162s | Som inte har... | 0.33 |
| 2603-2605s | Och det finns yta för Yeboah. | 1451-1453s | Niklas Röjker som är etta där han finns ju | 0.20 |
| 2605-2607s | Och inte minst Granath. | 1393-1395s | och inte minst underläge! | 0.50 |
| 2607-2609s | Kan han komma till avslut? | 2609-2614s | Kan han komma till avslut? Det stannar upp lite, men från Nilari istället! | 0.38 |
| 2609-2611s | Den stallar upp lite. | 681-686s | Lite småoroligt på läktarna. | 0.25 |
| 2611-2613s | Men från Illary i stället... | 2497-2500s | Vi får se från Lingby. | 0.25 |
| 2613-2615s | ...kommer ett skott. | 2615-2617s | Kommer ett skott, seglar över ribban. | 0.40 |
| 2615-2617s | Som seller det över ribban. | 2615-2617s | Kommer ett skott, seglar över ribban. | 0.33 |
| 2659-2665s | Och kolla, Öster tar ledningen mot Djurgården. | 2662-2667s | Kolla. Öster tar ledningen mot Djurgården. | 1.00 |
| 2667-2672s | Det innebär att vi har tre lag där nere. | 2669-2674s | Det innebär att vi har tre lag där nere. | 1.00 |
| 2674-2676s | Rinsträcket på 30 poäng. | 2703-2704s | Ja, 29 poäng. Förlåt! | 0.33 |
| 2689-2699s | Det betyder också att AIK håller Djurgården bakom sig lite högre upp i tabellen. | 2698-2702s | håller Djurgården bakom sig högre i tabellen. | 0.62 |
| 2699-2702s | 29 poäng, förlåt. | 2703-2704s | Ja, 29 poäng. Förlåt! | 1.00 |
| 2702-2709s | Ett Göteborgs mål hade ju ändrat på saker och ting. | 2707-2711s | Ett Göteborgsmål hade ju ändrat på saker och ting. | 0.67 |
| 2719-2735s | Da hade väl Öster och Norrköping hamnat på samma målskydnad. | 966-968s | Då hade han spelat med små marginaler. | 0.12 |
| 2735-2738s | Om jag räknar rätt, minus 16. | 2737-2739s | Räknar rätt. Minus 16. | 1.00 |
| 2738-2742s | De får ju mest gjorda mål och där är Norrköping bättre. | 2740-2743s | De får flest gjorda mål och där är Norrköping bättre. | 0.60 |
| 2742-2746s | Eller förlåt, ja men visst, Norrköping bättre med 55 gjorda. | 2744-2748s | Eller ja, Norrköping bättre med 55 gjorda. | 0.67 |
| 2746-2748s | Nu ska vi se här då. | — | — | 0.00 |
| 2751-2755s | Det är mycket att hålla koll på i slutomgången. | 2753-2756s | Det är mycket att hålla koll på så här i slutomgången... | 1.00 |
| 2757-2761s | En utlingas på flera håll. | 2759-2762s | När det plingas på flera håll. | 0.50 |
