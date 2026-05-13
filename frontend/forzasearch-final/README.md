# ┌FORZASEARCH┐

Full-stack sports highlight search platform.

## Quick Start

```bash
npm install
cp .env.example .env
```

Start Elasticsearch:
```bash
docker run -d --name elasticsearch -p 9200:9200 -e "discovery.type=single-node" -e "xpack.security.enabled=false" -e "xpack.security.http.ssl.enabled=false" docker.elastic.co/elasticsearch/elasticsearch:8.17.0
```

Optional — start Ollama for AI answers:
```bash
ollama serve
ollama pull mistral
```

Ingest + run:
```bash
npm run ingest
npm run dev
```

Open http://localhost:3000

## Seeded Admin

| Email | Password |
|-------|----------|
| admin@forzasearch.com | admin123 |

## Adding Matches

### Option A — from the Python cleaning pipeline (recommended)

The Python pipeline at the repo root (`pipeline/orchestrator.py`) writes a
ready-to-index `es_chunks.json` per match under
`cleaned_data/.../commentary_data/`. To pull every cleaned match into the
frontend in one shot:

```bash
npm run ingest:pipeline                    # incremental
npm run ingest:pipeline -- --force         # recreate the ES index first
npm run ingest:pipeline -- --match "AIK"   # restrict to a substring
```

This auto-creates an entry in `matches/registry.json` for each match (title
derived from the cleaned-data folder name, league/season from the path). To
enable video playback, hand-edit the entry's `video` field with the HLS
playlist URL and, for the second half, `part2_offset_sec`.

Override the data location with `PIPELINE_DATA_DIR=/path/to/cleaned_data`.

### Option B — hand-curated kamp.json

1. Create `matches/your-match-id/kamp.json` (commentary only — no label.json needed)
2. Add entry to `matches/registry.json` with video URL
3. Run `npm run ingest -- --force`

## Customizing

- Logo: replace placeholder in `src/app/page.tsx` hero
- Team photos: update `src/components/TeamCarousel.tsx`
- Promo video: place in `public/video/promo.mp4`, set src in landing page
- Colors: edit `tailwind.config.ts`
