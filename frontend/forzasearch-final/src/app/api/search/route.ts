import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { searchWindows } from "@/lib/elastic";
import { generateAnswer } from "@/lib/llm";

const MATCHES_DIR = path.join(process.cwd(), "matches");

function loadRegistry() {
  const p = path.join(MATCHES_DIR, "registry.json");
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

export async function POST(req: Request) {
  try {
    const { query, matchId } = await req.json();
    if (!query?.trim()) return NextResponse.json({ error: "Query is required" }, { status: 400 });

    const hits = await searchWindows(query.trim(), matchId, 1);

    if (!hits.length) {
      return NextResponse.json({
        answer: "No match moments found. Try a player name, event (goal, corner, card), or minute.",
        clips: [], esHits: 0,
      });
    }

    const topMatchId = (hits[0] as { _source: { match_id: string } })._source.match_id;
    const registry = loadRegistry();
    const matchEntry = registry.matches.find((m: { id: string }) => m.id === topMatchId);

    const answer = await generateAnswer(
      query.trim(),
      hits as Parameters<typeof generateAnswer>[1],
      matchEntry?.title || topMatchId
    );

    // Build clips — include commentary text for the user.
    // Return all unique matching clips found in the search results.
    const seen = new Set<string>();
    const clips: Record<string, unknown>[] = [];
    for (const hit of hits as { _source: { match_id: string; part: number; start_sec: number; end_sec: number; match_minute: number; text: string } }[]) {
      const s = hit._source;
      const key = `${s.match_id}-${s.part}-${s.start_sec}-${s.end_sec}`;
      if (seen.has(key)) continue;
      seen.add(key);

      const entry = registry.matches.find((m: { id: string }) => m.id === s.match_id);
      clips.push({
        matchId: s.match_id,
        matchTitle: entry?.title || s.match_id,
        part: s.part,
        start: s.start_sec,
        end: s.end_sec,
        matchMinute: s.match_minute,
        title: `~${s.match_minute}' — ${s.part === 1 ? "1st half" : "2nd half"}`,
        commentary: s.text,
        video: entry?.video || {},
      });
    }

    return NextResponse.json({ answer, clips, esHits: hits.length });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Search failed";
    return NextResponse.json({ error: "Search failed", detail: message }, { status: 500 });
  }
}
