# Search Quality: RAW Whisper vs CLEANED pipeline output

Both indexed in the same Elasticsearch instance with identical
BM25 + fuzzy multi_match + phrase-match (boost 5) hybrid query.
Tests whether the cleaning pipeline adds practical value when
the search backend already does fuzzy matching.

- RAW index: `match_id = chelsea-liverpool-2016-RAW` (1528 segments → 510 windows)
- CLEANED index: `match_id = chelsea-liverpool-2016` (1524 segments → 509 windows)
- Same audio, same Whisper model, only difference is the ASR-cleaning pipeline.

Each query asks: does the top-3 search result contain a canonical
entity name we expected (`hit?` column)? Score is the joint BM25
+ phrase boost — higher means stronger lexical match.

---

### A1: `"Sturridge goal"`  (target entities: ['Sturridge'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 11.96 @56min | `in his second goal Daniel Sturridge applauding the Liverpool fans and not the Chelsea majo` | ✓ Sturridge || 11.60 @43min | `Both dressing rooms. Which is a flick on. Was never threatening to Mignolet's goal. Mignol` | ✓ Sturridge |
| 2 | 11.60 @43min | `was never threatening to Mignolet's goal Mignolet wanted to throw it out quickly and Danie` | ✓ Sturridge || 10.21 @56min | `He played a part in that Chelsea equaliser at Swansea, didn't he? By Trivian, the ball loo` | ✓ Sturridge |
| 3 | 10.20 @24min | `Sturridge could have gone in off anyone shooting Daniel Sturridge but again the switch off` | ✓ Sturridge || 10.20 @24min | `Sturridge. Could have gone in off anyone. Shooting. Daniel Sturridge, but again. The switc` | ✓ Sturridge |

### A2: `"Diego Costa shot"`  (target entities: ['Diego Costa', 'Costa'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 16.11 @70min | `Spinecretta Azza Diego Costa not quite on the money covering by Klein` | ✓ Diego Costa || 19.28 @51min | `Willian. Diego Costa coming in. Diego Costa coming in with reach. By Mignolet. Picked hims` | ✓ Diego Costa |
| 2 | 15.53 @51min | `chance of leaving it away Willian Diego Costa coming in good reach by Mignolet` | ✓ Diego Costa || 16.74 @70min | `Ivanovic. Spinecretta. Azza. Diego Costa. Not quite on the money.` | ✓ Diego Costa |
| 3 | 15.45 @45min | `here's Matic William looks like Oscars gone further forward closer to Diego Costa can't a ` | ✓ Diego Costa || 15.45 @45min | `Here's Matic. Willian looks like Oscar gone further forward. Closer to Diego Costa? Can't ` | ✓ Diego Costa |

### A3: `"Hazard cross"`  (target entities: ['Hazard'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 17.18 @6min | `Costa might have another bite at it here that's a good cross from Oscar Oqueta Hazard` | ✓ Hazard || 17.18 @6min | `Costa. Might have another bite at it here. That's a good cross. From Oscar Oqueta. Hazard.` | ✓ Hazard |
| 2 | 12.32 @81min | `Chelsea have three lined up to come on but they make a Mourinho like situation we've taken` | ✓ Hazard || 9.72 @55min | `His cross in. Actually. It's better. They can feed it in. To feet.` | ✗ |
| 3 | 10.63 @69min | `and throw Matic Hazard Hazard having set up the goal down that side` | ✓ Hazard || 9.54 @77min | `Back from Lovren. Spots: Matip. Henderson. For Clyne. He's put in a great cross.` | ✗ |

### A4: `"Mignolet save"`  (target entities: ['Mignolet'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 17.85 @43min | `could really change the half time team talks both dressing rooms which is a flick on was n` | ✓ Mignolet || 17.85 @43min | `Both dressing rooms. Which is a flick on. Was never threatening to Mignolet's goal. Mignol` | ✓ Mignolet |
| 2 | 17.85 @43min | `was never threatening to Mignolet's goal Mignolet wanted to throw it out quickly and Danie` | ✓ Mignolet || 11.81 @51min | `Willian. Diego Costa coming in. Diego Costa coming in with reach. By Mignolet. Picked hims` | ✓ Mignolet |
| 3 | 12.23 @51min | `chance of leaving it away Willian Diego Costa coming in good reach by Mignolet` | ✓ Mignolet || 11.60 @51min | `By Mignolet. Picked himself up. And he's back in position. The defenders weren't sure abou` | ✓ Mignolet |

### A5: `"Klopp tactics"`  (target entities: ['Klopp', 'Jürgen'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 8.79 @76min | `they know that but others kick every ball and Jurgen Klopp is in that category` | ✓ Klopp || 9.61 @62min | `Say it have to get through the first 15 minutes, the second half, if you were. Jurgen Klop` | ✓ Klopp |
| 2 | 8.64 @62min | `that's a big test for Liverpool now Jurgen Klopp sat in his dugout fearing what might` | ✓ Klopp || 8.95 @62min | `One touch away. From equalising. That's a big test. For Liverpool now. Jurgen Klopp.` | ✓ Klopp |
| 3 | 8.49 @39min | `Theatric 3-1, remember they won here last season in October, shortly after Jurgen Klopp ha` | ✓ Klopp || 8.95 @62min | `For Liverpool now. Jurgen Klopp. Sat in his dugout. Fearing what might. Happen here.` | ✓ Klopp |

### B1: `"Aspilicueta header"`  (target entities: ['Azpilicueta', 'Aspilicueta'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 11.78 @30min | `side Matic Interesting had Mane been able to control that it's a real test for him Swapped` | ✓ Aspilicueta || 11.51 @29min | `Through Liverpool. Lovren's header. Counter. One good attack they had. Is when they did ge` | ✗ |
| 2 | 11.51 @29min | `through Liverpool Lovren's header counter one good attack they had is when they did get it` | ✗ || 10.97 @1min | `Azpilicueta, I think knew that Daniel Sturridge. Would cut back on his left foot. Didn't r` | ✓ Azpilicueta |
| 3 | 10.97 @1min | `Haspilicueta I think knew that Daniel Sturridge would cut back on his left foot didn't rea` | ✓ Aspilicueta || 9.94 @41min | `All about players and Chelsea. Than Chelsea. I'm certainly told. Azpilicueta. Henderson in` | ✓ Azpilicueta |

### B2: `"Davi Luiz pass"`  (target entities: ['David Luiz', 'Davi Luiz'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 19.64 @50min | `and Sturridge over on the far side and a runner on his knee missed the pass contact David ` | ✓ David Luiz || 21.81 @50min | `And a runner on his knee. Missed the pass. Contact. David Luiz. Remember.` | ✓ David Luiz |
| 2 | 13.22 @55min | `nuisance David Luiz taking on himself gives it to Ivanovic` | ✓ David Luiz || 14.20 @52min | `Real combinations happening. Little on that. The next pass isn't there, he's got, he's on ` | ✓ David Luiz |
| 3 | 12.96 @50min | `contact David Luiz remember his first debut was a home defeat` | ✓ David Luiz || 13.83 @53min | `Running it on for Mane trying to take. David Luiz on. On the half turn. Hasn't changed his` | ✓ David Luiz |

### B3: `"Diogo Costa Chelsea"`  (target entities: ['Diego Costa', 'Diogo Costa'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 16.07 @32min | `here's Diego Costa getting better Chelsea there's no doubt it's last five minutes they cou` | ✓ Diego Costa || 17.12 @51min | `Willian. Diego Costa coming in. Diego Costa coming in with reach. By Mignolet. Picked hims` | ✓ Diego Costa |
| 2 | 14.90 @42min | `gets it back again from Oscar and goes on for Chelsea, Diego Costa that's hard coming in f` | ✓ Diego Costa || 15.25 @42min | `And goes on for Chelsea, Diego Costa. That's hard coming in from the opposite flank. Great` | ✓ Diego Costa |
| 3 | 14.84 @42min | `on the snare side, looking so comfortable Willian has done well gets it back again from Os` | ✓ Diego Costa || 15.07 @32min | `With a bit more work to do in midfield. And even deeper than that. Here's Diego Costa. Get` | ✓ Diego Costa |

### B4: `"Havanovic defending"`  (target entities: ['Ivanovic', 'Havanovic'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 16.62 @54min | `wrong foot's Milner Lovren is secure by the near post good defending Ivanovic and they nea` | ✓ Ivanovic || 15.82 @54min | `Lovren is secure by the near post. Good defending. Ivanovic. And they nearly pinched that.` | ✓ Ivanovic |
| 2 | 11.03 @61min | `to stop, not go with that run even Adam Milano he really is poor defending they've caused ` | ✗ || 11.60 @61min | `Adam, Adam Lallana. He really is. Poor defending. They've caused their own problems. It's ` | ✗ |
| 3 | 8.11 @61min | `poor defending they've caused their own problems it's good for Matic Matip doesn't do enou` | ✗ || 9.90 @54min | `Just as it seemed to fit. What's here? Big ball from Nemanja Matić to Willian. Wrong foot'` | ✗ |

### B5: `"Marcus Alonso run"`  (target entities: ['Marcos Alonso', 'Marcus Alonso'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 23.97 @58min | `I'm not sure here's Oscar he's going for goal I suppose Marcus Alonso might come into that` | ✓ Marcus Alonso || 20.36 @58min | `Here's Oscar. He's going for goal. I suppose Marcos Alonso might come. Into that category ` | ✓ Marcos Alonso |
| 2 | 21.54 @58min | `I suppose Marcus Alonso might come into that category as a win back maybe but there's such` | ✓ Marcus Alonso || 8.62 @79min | `Origi. Lallana. Wanting to run it to the left. But there wasn't one. Certainly controlling` | ✗ |
| 3 | 8.77 @79min | `both sides waiting to make a change Origi Lallana wanting to run it to the left` | ✗ || 8.20 @61min | `So poor for Liverpool. Jordan Henderson there. To stop, not go with that run. Adam, Adam L` | ✗ |

### C1: `"Coutinho Lallana goal"`  (target entities: ['Coutinho', 'Lallana'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 15.66 @50min | `James Milner Lallana Henderson lovely touch from Mane and Coutinho` | ✓ Coutinho || 15.36 @50min | `Lallana. Henderson. Lovely touch from Mane. And Coutinho. It's the thing.` | ✓ Coutinho |
| 2 | 14.40 @37min | `but it's a wonderful goal and an own goal off Adam Lallana I think he's saying he's being ` | ✓ Lallana || 14.99 @37min | `And. And own goal off Adam Lallana. Off Adam Lallana. I think he's saying he's being nudge` | ✓ Lallana |
| 3 | 14.02 @80min | `Lallana Milner have a pass on and Martin Atkinson a bit of a problem for Coutinho` | ✓ Coutinho || 14.40 @37min | `The defensive midfield. Player of Liverpool. But it's a wonderful goal. And. And own goal ` | ✓ Lallana |

### C2: `"Henderson midfield"`  (target entities: ['Henderson'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 15.31 @42min | `Henderson Milano infield is Klein taking it on the burst and taking Chelsea on` | ✓ Henderson || 19.01 @42min | `Henderson again. Mane. Henderson. Milano. Infield is Klein.` | ✓ Henderson |
| 2 | 11.15 @37min | `you could have would be Gary Cale's clearance not clearing the defensive midfield` | ✗ || 11.15 @37min | `Would be Gary Cahill's. Clearance. Not clearing. The defensive midfield. Player of Liverpo` | ✗ |
| 3 | 10.94 @37min | `not clearing the defensive midfield player of Liverpool but it's a wonderful goal and` | ✗ || 10.20 @37min | `The defensive midfield. Player of Liverpool. But it's a wonderful goal. And. And own goal ` | ✗ |

### C3: `"first goal Liverpool"`  (target entities: ['goal', 'Liverpool'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 13.35 @37min | `over and over again in the first goal the second goal let's have to say it was wonderful` | ✓ goal || 14.13 @37min | `I'm sure. Gary. Over and over again. In the first goal. The second goal.` | ✓ goal |
| 2 | 12.34 @37min | `you can pick holes I'm sure Gary over and over again in the first goal` | ✓ goal || 13.35 @37min | `In the first goal. The second goal. Let's have to say: It was wonderful. Well, the only cr` | ✓ goal |
| 3 | 11.21 @81min | `happy to be hooked defensive change Lucas's last goal for Liverpool was six years ago this` | ✓ goal || 12.57 @65min | `That's the Liverpool first game of season 4-3. The Manchester derby, and now this one. All` | ✓ goal |

### C4: `"free kick wall"`  (target entities: ['free kick', 'wall'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 21.78 @88min | `he's won the free kick Lucas booked well well Liverpool have had so much supremacy` | ✓ free kick || 21.78 @87min | `He does that. He's won the free kick. Lucas booked. Well, well. Liverpool have had.` | ✓ free kick |
| 2 | 17.46 @52min | `one of those areas where 2-0 up away from home he might have been happy to have the free k` | ✓ free kick || 17.17 @16min | `Giving away free kicks in dangerous areas as well. Variation on the theme from Liverpool. ` | ✓ free kick |
| 3 | 17.17 @16min | `giving away free kicks in dangerous areas as well variation on the theme from Liverpool go` | ✓ free kick || 16.95 @8min | `He's lost focus. Wandering back in now. To pick up Lovren. It's one of those runs as a def` | ✓ free kick |

### D1: `"Conte signing"`  (target entities: ['Conte'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 11.22 @54min | `It's just a bizarre signing really because as I said earlier on in the game he's played wi` | ✗ || 10.52 @54min | `A bizarre signing really. Because, as I said earlier, On in the game. He's played with Cah` | ✗ |
| 2 | 9.76 @54min | `to sign and I'm not sure how near the top of it he was the other targets presumably were o` | ✗ || 10.05 @54min | `How near the top of it he was. The other targets presumably were out of reach. It's just. ` | ✗ |
| 3 | 9.38 @29min | `through Liverpool Lovren's header counter one good attack they had is when they did get it` | ✗ || 9.38 @29min | `Through Liverpool. Lovren's header. Counter. One good attack they had. Is when they did ge` | ✗ |

### D2: `"Willian winger"`  (target entities: ['Willian'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 14.78 @58min | `wing backs here that I can see at Chelsea you might ask William but is he really` | ✗ || 15.86 @58min | `That I can see. At Chelsea. You might ask Willian. But is he really? A wing back.` | ✓ Willian |
| 2 | 14.78 @58min | `you might ask William but is he really a wing back I'm not sure here's Oscar` | ✗ || 8.86 @29min | `There's been a threat awaiting on the left. Willian on the right. Can he get to him? Couti` | ✓ Willian |
| 3 | 10.43 @58min | `where are the wing backs there's an obvious energetic up and down wing backs here that I c` | ✗ || 8.86 @42min | `Not to the ground. There's a little bit of arrogance in Liverpool's play. On the snare sid` | ✓ Willian |

### D3: `"Origi striker"`  (target entities: ['Origi'])

| # | RAW Whisper score | RAW top hit (truncated) | hit? || CLEANED score | CLEANED top hit | hit? |
|---|---|---|---|---|---|---|---|
| 1 | 15.04 @67min | `David Luiz, Gattinio nudging it on for Milner, so many chances to get forward in the secon` | ✓ Origi || 18.79 @67min | `A run for Wijnaldum, Henderson, Gattinio from a standing start, Good strike from Origi. Vo` | ✓ Origi |
| 2 | 12.94 @56min | `majority who used to cheer him and Divock Origi comes on in place being the striker down t` | ✓ Origi || 13.83 @67min | `This holds back into the game. This next five minutes, big five minutes. David Luiz. David` | ✓ Origi |
| 3 | 12.02 @51min | `might metaphorically strike twice in the same place for the fellow on the ball here` | ✗ || 12.02 @51min | `Lightning around. The south east of England. Might metaphorically - Strike twice. In the s` | ✗ |

---

## Top-1 hit-rate per category

| Category | Queries | RAW top-1 hits | CLEANED top-1 hits |
|---|---|---|---|
| A — canonical spelling | 5 | 5/5 | 5/5 |
| B — Whisper-style misspelling | 5 | 5/5 | 4/5 |
| C — multi-entity / semantic | 4 | 4/4 | 4/4 |
| D — tricky (validation cases) | 3 | 1/3 | 2/3 |
| **Total** | **17** | **15/17 (88%)** | **15/17 (88%)** |

## Average top-1 BM25 score (fuzziness=AUTO)

- RAW: **15.71**
- CLEANED: **16.33**
- Δ: **+0.62** (+3.9% rel)

---

## Strict mode: fuzziness=0 (no fuzzy match)

This shows what happens when the search backend does NOT have ES's
`fuzziness: AUTO` safety net. Many production search systems
(SQL LIKE, plain Lucene, Solr without fuzzy) operate in this mode.

| Category | Queries | RAW strict | CLEANED strict |
|---|---|---|---|
| A — canonical spelling | 5 | 4/5 | 5/5 |
| B — Whisper-style misspelling | 5 | 5/5 | 3/5 |
| C — multi-entity / semantic | 4 | 3/4 | 3/4 |
| D — tricky (validation cases) | 3 | 1/3 | 2/3 |
| **Total** | **17** | **13/17 (76%)** | **13/17 (76%)** |

### Strict-mode top-1 BM25 score

- RAW: **14.43**
- CLEANED: **14.07**
- Δ: **-0.36** (-2.5% rel)

## Interpretation guide

- **Category A (canonical)**: Both RAW and CLEANED *should* find the entity, because the user types the right name. If RAW does well here, the cleaning didn't help on this axis — but it shouldn't hurt either.
- **Category B (misspelling)**: This is where cleaning is supposed to win. The user types `Aspilicueta`, RAW only contains canonical `Aspilicueta` (because that's what Whisper said). CLEANED contains `Azpilicueta`. A user who types the canonical form should NOT find any matches in the misspelled-only RAW corpus. ES fuzzy `AUTO` may compensate within edit distance ≤2; cleaning extends the reach beyond that.
- **Category C (multi-entity/semantic)**: BM25 alone may struggle when only 1 of 2 entities is exact. k-NN re-ranking via embeddings should compensate; cleaning's contribution is marginal here.
- **Category D (tricky)**: Cases the cleaning pipeline introduced corrections for. If the cleaned corpus *removed* good matches, this is where it shows.
