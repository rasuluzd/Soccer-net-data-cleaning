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

1. Create `matches/your-match-id/kamp.json` (commentary only — no label.json needed)
2. Add entry to `matches/registry.json` with video URL
3. Run `npm run ingest -- --force`

## Customizing

- Logo: replace placeholder in `src/app/page.tsx` hero
- Team photos: update `src/components/TeamCarousel.tsx`
- Promo video: place in `public/video/promo.mp4`, set src in landing page
- Colors: edit `tailwind.config.ts`
