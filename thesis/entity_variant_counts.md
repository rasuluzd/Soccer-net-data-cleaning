# Entity Surface-Form Variant Counts: RAW vs CLEANED

Match: **2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool**

For each canonical lineup name, we count how many distinct *surface
forms* of that name appear in the transcript (fuzz.ratio ≥ 70 to
the canonical). A downstream event-aggregation or knowledge-graph
system buckets events by exact surface form — so 5 surface variants
of the same player means 5 broken event-buckets per match.

**This is the cleaning-pipeline value that retrieval and LLM-RAG
benchmarks miss.** ES fuzzy AUTO bridges variants at query time;
an analytics pipeline aggregating by exact match cannot.

| Canonical | RAW variants | RAW total mentions | CLEANED variants | CLEANED total | Δ variants |
|---|---|---|---|---|---|
| `Adam Lallana` | **1** | 2 | **5** | 8 | +4 ⚠ expanded |
| `Antonio Conte` | **2** | 8 | **3** | 9 | +1 ⚠ expanded |
| `Cesar Azpilicueta` | **0** | 0 | **5** | 5 | +5 ⚠ expanded |
| `Cesc Fabregas` | **1** | 2 | **2** | 3 | +1 ⚠ expanded |
| `Chelsea` | **2** | 108 | **2** | 118 | +0  |
| `Daniel Sturridge` | **2** | 6 | **5** | 9 | +3 ⚠ expanded |
| `David Luiz` | **3** | 27 | **3** | 30 | +0  |
| `Dejan Lovren` | **2** | 2 | **2** | 2 | +0  |
| `Diego Costa` | **3** | 14 | **1** | 17 | -2 ✓ collapsed |
| `Divock Origi` | **1** | 2 | **1** | 2 | +0  |
| `Eden Hazard` | **1** | 1 | **1** | 2 | +0  |
| `Gary Cahill` | **3** | 5 | **2** | 5 | -1 ✓ collapsed |
| `James Milner` | **4** | 4 | **6** | 8 | +2 ⚠ expanded |
| `John Terry` | **2** | 2 | **3** | 3 | +1 ⚠ expanded |
| `Jordan Henderson` | **5** | 11 | **7** | 14 | +2 ⚠ expanded |
| `Jurgen Klopp` | **2** | 9 | **1** | 10 | -1 ✓ collapsed |
| `Kevin Stewart` | **1** | 2 | **1** | 2 | +0  |
| `Liverpool` | **2** | 79 | **2** | 84 | +0  |
| `Lucas Leiva` | **4** | 5 | **4** | 5 | +0  |
| `Marcos Alonso` | **1** | 1 | **1** | 1 | +0  |
| `Michy Batshuayi` | **1** | 1 | **1** | 1 | +0  |
| `N'Golo Kante` | **1** | 1 | **1** | 1 | +0  |
| `Nathaniel Clyne` | **1** | 2 | **1** | 2 | +0  |
| `Nemanja Matic` | **1** | 1 | **2** | 3 | +1 ⚠ expanded |
| `Oscar` | **3** | 15 | **2** | 19 | -1 ✓ collapsed |
| `Philippe Coutinho` | **1** | 1 | **2** | 2 | +1 ⚠ expanded |
| `Sadio Mane` | **2** | 4 | **4** | 7 | +2 ⚠ expanded |
| `Simon Mignolet` | **0** | 0 | **1** | 1 | +1 ⚠ expanded |
| `Thibaut Courtois` | **0** | 0 | **1** | 1 | +1 ⚠ expanded |
| `Victor Moses` | **1** | 2 | **1** | 2 | +0  |
| `Willian` | **5** | 26 | **5** | 32 | +0  |

## Summary

- Total surface variants across all canonicals: **RAW 58** vs **CLEANED 78**
  (Δ = **+20** variants)
- Canonicals where cleaning *collapsed* variants: **4**

## Detailed variant breakdown (top 15 most-affected canonicals)

### `Diego Costa` (raw 3 variants → cleaned 1 variants)

| RAW only | CLEANED only | Both |
|---|---|---|
| `lega costa` (1) |  | `diego costa` (12→17) |
| `diogo costa` (1) |  |  |

### `Gary Cahill` (raw 3 variants → cleaned 2 variants)

| RAW only | CLEANED only | Both |
|---|---|---|
| `gary cale` (3) | `gary cahills` (1) | `gary cahill` (1→4) |
| `gary cales` (1) |  |  |

### `Jurgen Klopp` (raw 2 variants → cleaned 1 variants)

| RAW only | CLEANED only | Both |
|---|---|---|
| `jürgen klopp` (2) |  | `jurgen klopp` (7→10) |

### `Oscar` (raw 3 variants → cleaned 2 variants)

| RAW only | CLEANED only | Both |
|---|---|---|
| `oskar` (1) |  | `oscar` (12→17) |
|  |  | `oscars` (2→2) |

### `David Luiz` (raw 3 variants → cleaned 3 variants)

| RAW only | CLEANED only | Both |
|---|---|---|
| `davi luiz` (1) | `david luis` (1) | `david luiz` (22→25) |
|  |  | `david lewis` (4→4) |

### `Marcos Alonso` (raw 1 variants → cleaned 1 variants)

| RAW only | CLEANED only | Both |
|---|---|---|
| `marcus alonso` (1) | `marcos alonso` (1) |  |

## Why this matters for event aggregation / knowledge-graph systems

Take a use-case: "Show me the top scorers in this match." The
downstream event-extraction system will:

1. NER over each segment → list of PERSON mentions
2. Group mentions by exact surface string (or by canonical ID
   if it has one)
3. Count goal-events per group

With **N surface variants per player**, the same player splits
into N rows in the top-scorers list — `Daniel Sturridge: 1 goal`,
`Sturridge: 1 goal`, `Starridge: 0 goals`, `Daniel Klain: 0 goals`
— and the analytics is broken.

Cleaning collapses variants → **1 row per player → correct counts.**
ES fuzzy doesn't help because the aggregation step happens AFTER
retrieval, on the raw text. LLM RAG doesn't help because the LLM
answers single questions, it doesn't run the analytics pipeline.