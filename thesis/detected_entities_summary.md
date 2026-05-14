# Detected Entities — Verification Report

Match: **2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool**
Source variant: `_v3_nbest`
Gazetteer canonicals: **53**

## Counts

| Metric | Value |
|---|---|
| Total entity occurrences | **766** |
| Unique entity tokens (case-folded) | **220** |
| Already-canonical occurrences | 289 (38%) |
| Non-canonical (potential corrections) | 477 (62%) |

**Note**: "occurrences" counts every detection — a player name said 20 times produces 20 rows. This is by design: Stage E's per-match cache short-circuits repeats so the MCQ judge only runs once per unique (entity, top-3-cands) tuple.

## By detection source

| Source | Count |
|---|---|
| spacy | 632 |
| heuristic_capitalized_non_function | 94 |
| heuristic_short_segment | 38 |
| heuristic_gazetteer_fuzz | 2 |

## Top 50 most-frequent detected entity tokens

| Token | Count | In gazetteer? |
|---|---|---|
| `Chelsea` | 108 | ✓ |
| `Liverpool` | 78 | ✓ |
| `Coutinho` | 29 | ✗ |
| `David Luiz` | 22 | ✓ |
| `Ivanovic` | 19 | ✗ |
| `Henderson` | 15 | ✗ |
| `Hazard` | 14 | ✗ |
| `William` | 13 | ✗ |
| `Lallana` | 13 | ✗ |
| `Conte` | 13 | ✗ |
| `Wijnaldum` | 12 | ✗ |
| `Lovren` | 12 | ✗ |
| `Sturridge` | 11 | ✗ |
| `Oscar` | 11 | ✓ |
| `Diego Costa` | 10 | ✓ |
| `Willian` | 10 | ✓ |
| `Mane` | 9 | ✗ |
| `Klein` | 9 | ✗ |
| `Costa` | 9 | ✗ |
| `Lucas` | 8 | ✗ |
| `Milner` | 7 | ✗ |
| `Jurgen Klopp` | 7 | ✓ |
| `Matic` | 7 | ✗ |
| `Antonio Conte` | 7 | ✓ |
| `Fabregas` | 7 | ✗ |
| `Jordan Henderson` | 6 | ✓ |
| `Courtois` | 6 | ✗ |
| `Daniel Sturridge` | 5 | ✓ |
| `Cahill` | 5 | ✗ |
| `Premier` | 5 | ✗ |
| `League` | 5 | ✗ |
| `the Premier League` | 5 | ✗ |
| `Aspilicueta` | 5 | ✗ |
| `Origi` | 5 | ✗ |
| `David Lewis` | 4 | ✗ |
| `Mané` | 4 | ✗ |
| `Martin Atkinson` | 4 | ✗ |
| `Moses` | 4 | ✗ |
| `Sadio Mane` | 3 | ✓ |
| `Arsenal` | 3 | ✗ |
| `Luiz` | 3 | ✗ |
| `Matip` | 3 | ✗ |
| `Matty` | 3 | ✗ |
| `Watford` | 3 | ✗ |
| `Gary Cale` | 3 | ✗ |
| `Kante` | 3 | ✗ |
| `Mignolet` | 3 | ✗ |
| `England` | 3 | ✗ |
| `Gary` | 3 | ✗ |
| `Sunderland` | 3 | ✗ |

## Full data

All 766 occurrences are in `thesis\detected_entities.csv` (open in Excel / VS Code). Filter `is_dup_in_match=True` to see only first occurrence of each unique entity.