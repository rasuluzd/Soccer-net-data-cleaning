import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { searchWindows } from "@/lib/elastic";
import { generateAnswer, translateQueryToSwedish } from "@/lib/llm";

const MATCHES_DIR = path.join(process.cwd(), "matches");
const MIN_SCORE = Number(process.env.SEARCH_MIN_SCORE ?? "5");
const SEARCH_SIZE = Number(process.env.SEARCH_SIZE ?? "5");

type Hit = {
  _score?: number;
  _source: { match_id: string; part: number; start_sec: number; end_sec: number; match_minute: number; text: string };
};

function loadRegistry() {
  const p = path.join(MATCHES_DIR, "registry.json");
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

function noMatchResponse(query: string) {
  return NextResponse.json({
    answer: `No moment in the commentary matches "${query}". Try rephrasing — for example a player name, a specific event (goal, save, card), or a match minute.`,
    clips: [],
    esHits: 0,
  });
}

export async function POST(req: Request) {
  try {
    const { query, matchId } = await req.json();
    if (!query?.trim()) return NextResponse.json({ error: "Query is required" }, { status: 400 });
    if (!matchId?.trim()) return NextResponse.json({ error: "matchId is required" }, { status: 400 });

    const originalQuery = query.trim();
    const searchQuery = await translateQueryToSwedish(originalQuery);
    if (searchQuery !== originalQuery) {
      console.log(`[search] translated query "${originalQuery}" -> "${searchQuery}"`);
    }

    const rawHits = (await searchWindows(searchQuery, matchId, SEARCH_SIZE)) as Hit[];

    if (!rawHits.length) return noMatchResponse(originalQuery);

    // Keep only confidently-scored hits. If nothing clears the bar, treat as "no hit".
    const hits = rawHits.filter((h) => (h._score ?? 0) >= MIN_SCORE);
    console.log(
      `[search] query="${originalQuery}" rawHits=${rawHits.length} topScore=${rawHits[0]._score?.toFixed(2) ?? "n/a"} keptAfterThreshold=${hits.length}`,
    );
    if (!hits.length) return noMatchResponse(originalQuery);

    const registry = loadRegistry();
    const topMatchId = hits[0]._source.match_id;
    const matchEntry = registry.matches.find((m: { id: string }) => m.id === topMatchId);

    // Pass the original (English) query to the LLM so its 2-3 sentence answer
    // echoes the user's wording. The Swedish translation was only needed to
    // hit the Swedish index.
    const { answer, refused, relevantIndices } = await generateAnswer(
      originalQuery,
      hits as Parameters<typeof generateAnswer>[1],
      matchEntry?.title || topMatchId,
    );

    // If the LLM says none of the retrieved context actually describes the query,
    // return no clips — they would contradict the answer.
    if (refused) {
      return NextResponse.json({ answer, clips: [], esHits: 0 });
    }

    // Keep only hits the LLM flagged as actually relevant. This is the "semantic"
    // step on top of lexical BM25 — drops windows that happen to contain the query
    // words but describe a different event.
    const relevantHits = relevantIndices.map((i) => hits[i]).filter(Boolean);
    console.log(
      `[search] llm kept ${relevantHits.length}/${hits.length} hits as relevant (indices: ${relevantIndices.join(",")})`,
    );

    // Merge overlapping windows. Windows overlap by design (5 segs, step 3), so
    // two adjacent windows cover nearly the same moment and would show as
    // duplicate clips. Group by (match_id, part), sort by start, and fuse any
    // pair whose time ranges overlap or are within 2s of each other.
    type Merged = { match_id: string; part: number; start_sec: number; end_sec: number; match_minute: number; texts: string[] };
    const groups = new Map<string, Merged[]>();
    for (const h of relevantHits) {
      const s = h._source;
      const k = `${s.match_id}|${s.part}`;
      if (!groups.has(k)) groups.set(k, []);
      groups.get(k)!.push({
        match_id: s.match_id, part: s.part,
        start_sec: s.start_sec, end_sec: s.end_sec,
        match_minute: s.match_minute, texts: [s.text],
      });
    }
    const merged: Merged[] = [];
    for (const arr of groups.values()) {
      arr.sort((a, b) => a.start_sec - b.start_sec);
      for (const cur of arr) {
        const prev = merged[merged.length - 1];
        if (
          prev &&
          prev.match_id === cur.match_id &&
          prev.part === cur.part &&
          cur.start_sec <= prev.end_sec + 2
        ) {
          prev.end_sec = Math.max(prev.end_sec, cur.end_sec);
          // Only add the new text if it adds information (windows share segments).
          for (const t of cur.texts) if (!prev.texts.includes(t)) prev.texts.push(t);
        } else {
          merged.push({ ...cur, texts: [...cur.texts] });
        }
      }
    }

    const clips: Record<string, unknown>[] = merged.map((m) => {
      const entry = registry.matches.find((e: { id: string }) => e.id === m.match_id);
      // When windows merge, the shared overlap text would appear twice in a naive
      // concat. We dedupe by unique window text, which is already good enough —
      // the human-visible commentary stays readable.
      return {
        matchId: m.match_id,
        matchTitle: entry?.title || m.match_id,
        part: m.part,
        start: m.start_sec,
        end: m.end_sec,
        matchMinute: m.match_minute,
        title: `~${m.match_minute}' — ${m.part === 1 ? "1st half" : "2nd half"}`,
        commentary: m.texts.join(" "),
        video: entry?.video || {},
      };
    });

    return NextResponse.json({ answer, clips, esHits: clips.length });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Search failed";
    return NextResponse.json({ error: "Search failed", detail: message }, { status: 500 });
  }
}
