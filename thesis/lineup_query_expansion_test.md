# Lineup-Aware Query Expansion Test

Match: **2016-09-16 - 22-00 Chelsea 1 - 2 Liverpool**

Lineup index: 41 entries (players, coaches, teams, referees)
Surface variants harvested from raw Whisper text: 113 total

## Per-query top-1 hit?

| ID | Query | RAW+std | RAW+expand | CLEANED+std | CLEANED+expand |
|---|---|---|---|---|---|
| A1 | `Sturridge goal` | ✓ | ✓ | ✓ | ✓ |
| A2 | `Diego Costa shot` | ✓ | ✓ | ✓ | ✓ |
| A3 | `Hazard cross` | ✓ | ✓ | ✓ | ✓ |
| A4 | `Mignolet save` | ✓ | ✓ | ✓ | ✓ |
| A5 | `Klopp tactics` | ✓ | ✓ | ✓ | ✓ |
| B1 | `Aspilicueta header` | ✓ | ✓ | ✗ | ✓ |
| B2 | `Davi Luiz pass` | ✓ | ✓ | ✓ | ✓ |
| B3 | `Diogo Costa Chelsea` | ✓ | ✓ | ✓ | ✓ |
| B4 | `Havanovic defending` | ✓ | ✓ | ✓ | ✓ |
| B5 | `Marcus Alonso run` | ✓ | ✓ | ✓ | ✓ |
| C1 | `Coutinho Lallana goal` | ✓ | ✓ | ✓ | ✓ |
| C2 | `Henderson midfield` | ✓ | ✓ | ✓ | ✓ |
| C3 | `first goal Liverpool` | ✓ | ✓ | ✓ | ✓ |
| C4 | `free kick wall` | ✓ | ✓ | ✓ | ✓ |
| D1 | `Conte signing` | ✗ | ✗ | ✗ | ✗ |
| D2 | `Willian winger` | ✗ | ✗ | ✓ | ✓ |
| D3 | `Origi striker` | ✓ | ✓ | ✓ | ✓ |

## Top-1 hit-rate (out of 17)

| Configuration | Hit rate |
|---|---|
| RAW + standard query | **15/17 (88 %)** |
| **RAW + lineup-expanded query** | **15/17 (88 %)** |
| CLEANED + standard query | **15/17 (88 %)** |
| CLEANED + lineup-expanded query | **16/17 (94 %)** |

## Example query expansions

**Original:** `Sturridge goal`

Matched entities:
- `Sturridge` (metaphone: `STRJ`) → canonical `Daniel Sturridge` → expansions: `Daniel Sturridge`, `Mane Sturridge`, `Sturridge`

**Original:** `Diego Costa shot`

Matched entities:
- `Diego` (metaphone: `TK`) → canonical `Diego Costa` → expansions: `Acosta`, `Costa`, `Diego Costa`, `Diego costa`, `Diogo Costa`, `Kosta`, `Lega Costa`
- `Costa` (metaphone: `KST`) → canonical `Diego Costa` → expansions: `Acosta`, `Costa`, `Diego Costa`, `Diego costa`, `Diogo Costa`, `Kosta`, `Lega Costa`

**Original:** `Hazard cross`

Matched entities:
- `Hazard` (metaphone: `HSRT`) → canonical `Eden Hazard` → expansions: `Azhar`, `Eden Hazard`, `Hazard`

**Original:** `Mignolet save`

Matched entities:
- `Mignolet` (metaphone: `MKNLT`) → canonical `Simon Mignolet` → expansions: `Mignolet`, `Simon Mignolet`

**Original:** `Klopp tactics`

Matched entities:
- `Klopp` (metaphone: `KLP`) → canonical `Jurgen Klopp` → expansions: `Jurgen Klopp`, `Jürgen Klopp`, `Klopp`

**Original:** `Aspilicueta header`

Matched entities:
- `Aspilicueta` (metaphone: `ASPLKT`) → canonical `Cesar Azpilicueta` → expansions: `Aspilicueta`, `Azpilicueta`, `Cesar Azpilicueta`, `Filiqueta`, `Haspilicueta`

## Interpretation

**If RAW+expand ≥ CLEANED+std:** The 32-minute Step L + Step P
cleaning pipeline can be replaced by lineup-aware query expansion
at search time. Cleaning's only remaining justification is for
non-search consumers (LLM RAG answer fluency, NER-based event
aggregation).

**If RAW+expand << CLEANED+std:** Cleaning still earns its place
at indexing time. (We don't expect this — fuzzy AUTO is already
known to bridge edit-distance ≤ 2 errors, and our expansion
explicitly targets the larger errors fuzzy can't reach.)