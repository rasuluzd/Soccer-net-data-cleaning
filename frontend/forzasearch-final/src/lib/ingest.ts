import fs from "fs";
import path from "path";
import { Client } from "@elastic/elasticsearch";
import { embedMany, EMBEDDING_DIMS } from "./embeddings";

const INDEX = "forzasearch-windows";
const WINDOW_SIZE = 5;
const OVERLAP = 2;
const MATCHES_DIR = path.join(process.cwd(), "matches");
const ES_URL = process.env.ELASTICSEARCH_URL || "http://localhost:9200";

function parseCommentary(filePath: string) {
  let raw = fs.readFileSync(filePath, "utf-8");

  // Clean common issues
  raw = raw.replace(/\r\n/g, "\n");
  raw = raw.replace(/(\d+)\.\s+(\d+)/g, "$1.$2"); // fix "140. 24"

  const part2Pos = raw.indexOf("Part 2");
  const hasTwoParts = part2Pos !== -1;

  // Robust regex that skips malformed entries
  const pattern = /"(\d+)"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*"((?:[^"\\]|\\.)*)"\s*\]/gs;
  const segments: { part: number; start: number; end: number; text: string }[] = [];
  let match;
  let skipped = 0;

  while ((match = pattern.exec(raw)) !== null) {
    const text = match[4].replace(/\\n/g, " ").replace(/\n/g, " ").replace(/\\t/g, " ").trim();

    // Skip empty or garbage segments
    if (!text || text.length < 3) { skipped++; continue; }

    const start = parseFloat(match[2]);
    const end = parseFloat(match[3]);
    if (isNaN(start) || isNaN(end) || end <= start) { skipped++; continue; }

    segments.push({
      part: hasTwoParts && match.index >= part2Pos ? 2 : 1,
      start,
      end,
      text,
    });
  }

  if (skipped > 0) console.log(`    (skipped ${skipped} malformed segments)`);
  return segments;
}

function buildWindows(segments: ReturnType<typeof parseCommentary>) {
  const windows: { part: number; start_sec: number; end_sec: number; match_minute: number; text: string }[] = [];
  for (const partNum of [1, 2]) {
    const partSegs = segments.filter((s) => s.part === partNum).sort((a, b) => a.start - b.start);
    let i = 0;
    while (i < partSegs.length) {
      const chunk = partSegs.slice(i, i + WINDOW_SIZE);
      if (!chunk.length) break;
      windows.push({
        part: partNum,
        start_sec: Math.round(chunk[0].start * 10) / 10,
        end_sec: Math.round(chunk[chunk.length - 1].end * 10) / 10,
        match_minute: (partNum === 2 ? 45 : 0) + Math.floor(chunk[0].start / 60),
        text: chunk.map((c) => c.text).join(" "),
      });
      i += WINDOW_SIZE - OVERLAP;
    }
  }
  return windows;
}

async function main() {
  const force = process.argv.includes("--force");
  const elastic = new Client({ node: ES_URL });

  console.log("\n⚽ ForzaSearch — Ingestion\n");

  const registryPath = path.join(MATCHES_DIR, "registry.json");
  if (!fs.existsSync(registryPath)) { console.error("Missing matches/registry.json"); process.exit(1); }
  const registry = JSON.parse(fs.readFileSync(registryPath, "utf-8"));

  const exists = await elastic.indices.exists({ index: INDEX });
  if (force && exists) { await elastic.indices.delete({ index: INDEX }); console.log("  Deleted existing index."); }
  if (force || !exists) {
    await elastic.indices.create({
      index: INDEX,
      body: {
        settings: { number_of_shards: 1, number_of_replicas: 0, analysis: {
          analyzer: {
            swedish_custom: { type: "custom", tokenizer: "standard", filter: ["lowercase", "swedish_stop", "swedish_stemmer"] },
            general: { type: "custom", tokenizer: "standard", filter: ["lowercase", "asciifolding"] },
          },
          filter: { swedish_stop: { type: "stop", stopwords: "_swedish_" }, swedish_stemmer: { type: "stemmer", language: "swedish" } },
        }},
        mappings: { properties: {
          text: { type: "text", analyzer: "swedish_custom", fields: { general: { type: "text", analyzer: "general" } } },
          part: { type: "integer" }, start_sec: { type: "float" }, end_sec: { type: "float" },
          match_minute: { type: "integer" }, match_id: { type: "keyword" },
          embedding: { type: "dense_vector", dims: EMBEDDING_DIMS, index: true, similarity: "cosine" },
        }},
      },
    });
    console.log("  Created index.");
  }

  let total = 0;
  for (const entry of registry.matches) {
    const kampPath = path.join(MATCHES_DIR, entry.folder, "kamp.json");
    if (!fs.existsSync(kampPath)) { console.log(`  ⚠ Skipping ${entry.id} — no kamp.json`); continue; }
    console.log(`  ${entry.title} (${entry.id})`);

    const segments = parseCommentary(kampPath);
    console.log(`    ${segments.length} segments parsed`);

    const windows = buildWindows(segments);
    console.log(`    ${windows.length} windows built`);

    console.log(`    computing embeddings...`);
    const embeddings = await embedMany(windows.map((w) => w.text));

    const body = windows.flatMap((w, i) => [
      { index: { _index: INDEX } },
      { ...w, match_id: entry.id, embedding: embeddings[i] },
    ]);
    await elastic.bulk({ body, refresh: true });
    total += windows.length;
    console.log(`    ✓ Indexed`);
  }

  console.log(`\n  Total: ${total} windows. Done.\n`);
}

main().catch((e) => { console.error(e); process.exit(1); });
