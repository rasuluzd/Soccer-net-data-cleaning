# Entity Correction Analysis — Stage E rejection breakdown

Match: **2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool**

## 1. Accepted Stage E corrections (1)

| Half | Seg | Original → Corrected | Method | Score |
|---|---|---|---|---|
| 1 | 316 | `Shooting Daniel Sturridge` → `Daniel Sturridge` | tfidf_shortcut | 92.3 |

**By method:** {'tfidf_shortcut': 1}

## 2. Rejection breakdown (would-be candidates)

These tokens *passed* NER detection but were filtered before any correction was applied. Each row shows the top gazetteer canonical the TF-IDF retrieve picked, with the gate that rejected the pair.

### shortcut_reject_low_cosine — 217 cases

| Token | Top canonical | cosine | fuzz | extra |
|---|---|---|---|---|
| `starry` | `John Terry` | 0.348 | ? |  |
| `fair` | `Cesc Fabregas` | 0.215 | ? |  |
| `London` | `Jordan Henderson` | 0.227 | ? |  |
| `area` | `a` | 0.322 | ? |  |
| `heart` | `Kevin Stewart` | 0.287 | ? |  |
| `knew` | `Kevin Stewart` | 0.193 | ? |  |
| `love` | `Dejan Lovren` | 0.353 | ? |  |
| `Arsenal` | `a` | 0.136 | ? |  |
| `Premier League` | `Pedro Rodriguez` | 0.16 | ? |  |
| `Manchester City's` | `Sadio Mane` | 0.218 | ? |  |
| `stats` | `S` | 0.265 | ? |  |
| `view` | `Victor Moses` | 0.211 | ? |  |
| `Louise` | `Loris Karius` | 0.201 | ? |  |
| `match` | `Joel Matip` | 0.372 | ? |  |
| `Leicester` | `Lucas Leiva` | 0.308 | ? |  |
| `winner` | `James Milner` | 0.317 | ? |  |
| `Mike` | `James Milner` | 0.175 | ? |  |
| `Chennava Alana` | `Adam Lallana` | 0.399 | ? |  |
| `Sam` | `Sadio Mane` | 0.233 | ? |  |
| `proven` | `Dejan Lovren` | 0.231 | ? |  |
| `pair` | `Asmir Begovic` | 0.241 | ? |  |
| `Cale` | `Gary Cahill` | 0.207 | ? |  |
| `jolt` | `Joel Matip` | 0.234 | ? |  |
| `country` | `Thibaut Courtois` | 0.239 | ? |  |
| `Belgium` | `Georginio Wijnaldum` | 0.185 | ? |  |

*(showing 25 of 217)*

### frequency_heuristic — 78 cases

| Token | Top canonical | cosine | fuzz | extra |
|---|---|---|---|---|
| `many` | `Sadio Mane` | ? | ? | 6 |
| `that` | `t` | ? | ? | 99 |
| `last` | `Adam Lallana` | ? | ? | 31 |
| `another` | `James Milner` | ? | ? | 8 |
| `contact` | `Antonio Conte` | ? | ? | 6 |
| `Sturridge` | `Daniel Sturridge` | ? | ? | 17 |
| `maybe` | `Sadio Mane` | ? | ? | 13 |
| `across` | `a` | ? | ? | 9 |
| `Wijnaldum` | `Georginio Wijnaldum` | ? | ? | 12 |
| `good` | `d` | ? | ? | 40 |
| `been` | `Asmir Begovic` | ? | ? | 41 |
| `William` | `Willian` | ? | ? | 13 |
| `goes` | `g` | ? | ? | 8 |
| `Mane` | `Sadio Mane` | ? | ? | 17 |
| `Henderson` | `Jordan Henderson` | ? | ? | 21 |
| `start` | `Kevin Stewart` | ? | ? | 7 |
| `game` | `James Milner` | ? | ? | 50 |
| `even` | `e` | ? | ? | 7 |
| `they` | `t` | ? | ? | 69 |
| `started` | `Kevin Stewart` | ? | ? | 9 |
| `brilliant` | `Willian` | ? | ? | 7 |
| `lost` | `Diego Costa` | ? | ? | 5 |
| `Lallana` | `Adam Lallana` | ? | ? | 15 |
| `goal` | `g` | ? | ? | 39 |
| `Milner` | `James Milner` | ? | ? | 15 |

*(showing 25 of 78)*

### mcq_pregate_short_low_fuzz — 13 cases

| Token | Top canonical | cosine | fuzz | extra |
|---|---|---|---|---|
| `Davi` | `David Luiz` | 0.528 | 57.14285714285714 |  |
| `iffy` | `i` | 0.418 | 40.0 |  |
| `ante` | `Antonio Conte` | 0.526 | 35.29411764705882 |  |
| `Mati` | `Joel Matip` | 0.5 | 57.14285714285714 |  |
| `Mané` | `Sadio Mane` | 0.413 | 42.85714285714286 |  |
| `hard` | `Eden Hazard` | 0.537 | 40.0 |  |
| `cost` | `Diego Costa` | 0.505 | 40.0 |  |
| `EFL` | `e` | 0.43 | 0.0 |  |
| `Lana` | `Adam Lallana` | 0.586 | 50.0 |  |
| `sign` | `Simon Mignolet` | 0.409 | 33.333333333333336 |  |
| `live` | `Liverpool` | 0.547 | 46.15384615384615 |  |
| `rigi` | `Divock Origi` | 0.49 | 50.0 |  |
| `firm` | `Roberto Firmino` | 0.469 | 31.57894736842105 |  |

### mcq_pregate_low_fuzz — 34 cases

| Token | Top canonical | cosine | fuzz | extra |
|---|---|---|---|---|
| `stamford bridge` | `Daniel Sturridge` | 0.444 | 45.16129032258065 |  |
| `Coutinho Alana` | `Philippe Coutinho` | 0.583 | 51.61290322580645 |  |
| `Sam Allardyce` | `Adam Lallana` | 0.419 | 48.0 |  |
| `Oscar Oqueta` | `Oscar` | 0.79 | 58.82352941176471 |  |
| `Novanovich` | `Branislav Ivanovic` | 0.469 | 50.0 |  |
| `Clyne` | `Nathaniel Clyne` | 0.609 | 50.0 |  |
| `along` | `Marcos Alonso` | 0.466 | 44.44444444444444 |  |
| `dramatic` | `Nemanja Matic` | 0.5 | 47.61904761904761 |  |
| `marking` | `Mark Clattenburg` | 0.418 | 43.47826086956522 |  |
| `cheap` | `Chelsea` | 0.529 | 50.0 |  |
| `Kevin` | `Kevin Stewart` | 0.647 | 55.55555555555556 |  |
| `Kante` | `N'Golo Kante` | 0.649 | 58.82352941176471 |  |
| `Joachim Klopp` | `Jurgen Klopp` | 0.614 | 56.00000000000001 |  |
| `Brazilian` | `Willian` | 0.468 | 62.5 |  |
| `control` | `Antonio Conte` | 0.478 | 30.000000000000004 |  |
| `Filiqueta` | `Cesar Azpilicueta` | 0.429 | 53.84615384615385 |  |
| `Gary Kane` | `Gary Cahill` | 0.47 | 60.0 |  |
| `Oscar Acosta` | `Oscar` | 0.758 | 58.82352941176471 |  |
| `defenders` | `Jordan Henderson` | 0.476 | 56.00000000000001 |  |
| `Cruyff` | `f` | 0.521 | 28.57142857142857 |  |
| `James` | `James Milner` | 0.682 | 58.82352941176471 |  |
| `Swansea` | `Chelsea` | 0.423 | 42.85714285714286 |  |
| `cheer` | `Chelsea` | 0.435 | 50.0 |  |
| `Luis Cahill` | `Gary Cahill` | 0.638 | 63.63636363636363 |  |
| `Kosta` | `Diego Costa` | 0.533 | 50.0 |  |

*(showing 25 of 34)*

### validation_c2_length_tolerance — 3 cases

| Token | Top canonical | cosine | fuzz | extra |
|---|---|---|---|---|
| `Mignolet` | `Simon Mignolet` | 0.784 | 72.72727272727273 | c2_len_tol |
| `Terry` | `John Terry` | 0.745 | 66.66666666666667 | c2_len_tol |
| `Rodriguez` | `Pedro Rodriguez` | 0.825 | 75.0 | c2_len_tol |

## 3. Summary counts

| Bucket | Count |
|---|---|
| Accepted | 1 |
| shortcut_reject_low_cosine | 217 |
| frequency_heuristic | 78 |
| mcq_pregate_short_low_fuzz | 13 |
| mcq_pregate_low_fuzz | 34 |
| validation_dict_veto | 0 |
| validation_c1_fuzzy_floor | 0 |
| validation_c2_length_tolerance | 3 |

## 4. OLD `learned_corrections.json` not present in repo
