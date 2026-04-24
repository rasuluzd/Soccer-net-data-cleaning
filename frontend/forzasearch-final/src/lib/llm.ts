const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "mistral";

interface EsHit {
  _source: { text: string; part: number; start_sec: number; end_sec: number; match_minute: number; match_id: string };
}

// Quick heuristic: if the query contains any Swedish-specific characters (å/ä/ö)
// or any token the index actually uses in Swedish, assume it's Swedish and skip
// translation. Otherwise translate.
const SWEDISH_FOOTBALL_TERMS = [
  "frispark", "hörna", "straff", "mål", "rödakort", "rött", "kort",
  "räddning", "målvakt", "assist", "anfall", "försvar", "avslut",
  "halvlek", "matchen", "spelare", "domare", "gul",
];

function looksSwedish(query: string): boolean {
  if (/[åäöÅÄÖ]/.test(query)) return true;
  const lower = query.toLowerCase();
  return SWEDISH_FOOTBALL_TERMS.some((w) => lower.includes(w));
}

/**
 * Translate an English football query to Swedish so it can hit the Swedish-analyzed
 * index. Falls back to the original query on any failure — better to run a weak
 * search than to fail the request outright.
 */
export async function translateQueryToSwedish(query: string): Promise<string> {
  if (looksSwedish(query)) return query;

  const prompt = [
    `Translate this English football search query to Swedish, using the terminology a Swedish football commentator would use.`,
    `- Keep player names exactly as written.`,
    `- Translate football-specific terms: "goal" → "mål", "free kick" → "frispark", "corner" → "hörna", "penalty" → "straff", "save" → "räddning", "red card" → "rött kort", "yellow card" → "gult kort", "header" → "nickmål" or "nick".`,
    `- Output ONLY the translated query, nothing else. No quotes, no explanation.`,
    ``,
    `English: ${query}`,
    `Swedish:`,
  ].join("\n");

  try {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt,
        stream: false,
        options: { temperature: 0, num_predict: 40 },
      }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) return query;
    const data = await res.json();
    const text: string = (data.response || "").trim();
    // Strip common artifacts: leading/trailing quotes, trailing periods, extra lines.
    const cleaned = text.split("\n")[0].replace(/^["'`]|["'`.]$/g, "").trim();
    return cleaned || query;
  } catch {
    return query;
  }
}

export interface AnswerResult {
  answer: string;
  refused: boolean;
  /** 0-based indices into the `hits` array that the LLM considers actually about the query. */
  relevantIndices: number[];
}

const REFUSAL_MESSAGE = "No moment in the commentary matches that query. Try rephrasing, or search for a player name or specific event.";

function looksLikeRefusal(text: string): boolean {
  if (/\bNO_MATCH\b/.test(text)) return true;
  const lower = text.toLowerCase();
  return (
    lower.includes("not described in the commentary") ||
    lower.includes("not mentioned in the commentary") ||
    lower.includes("no mention of") ||
    lower.includes("cannot find") ||
    lower.includes("does not describe")
  );
}

function parseLlmResponse(text: string, hitCount: number): { answer: string; relevantIndices: number[] } {
  // Expected shape:
  //   RELEVANT: 1, 3
  //   ANSWER: <prose>
  // Small models are inconsistent, so parse defensively.
  const relevantMatch = text.match(/RELEVANT\s*:\s*([^\n]+)/i);
  const answerMatch = text.match(/ANSWER\s*:\s*([\s\S]+)/i);

  let relevantIndices: number[] = [];
  if (relevantMatch) {
    const raw = relevantMatch[1].trim().toLowerCase();
    if (raw !== "none" && raw !== "") {
      const nums = Array.from(raw.matchAll(/\d+/g)).map((m) => Number(m[0]));
      // Prompt uses 1-based indices; convert to 0-based and drop out-of-range / duplicates.
      relevantIndices = Array.from(new Set(nums))
        .map((n) => n - 1)
        .filter((i) => i >= 0 && i < hitCount);
    }
  }

  const answer = (answerMatch ? answerMatch[1] : text).trim();
  return { answer, relevantIndices };
}

export async function generateAnswer(query: string, hits: EsHit[], matchTitle: string): Promise<AnswerResult> {
  const top = hits.slice(0, 8);
  const context = top
    .map((h, i) => {
      const s = h._source;
      const sm = Math.floor(s.start_sec / 60);
      const ss = Math.floor(s.start_sec % 60);
      return `[${i + 1}] Part ${s.part} | ${sm}:${String(ss).padStart(2, "0")} | ~${s.match_minute}'\n${s.text}`;
    })
    .join("\n\n");

  const prompt = [
    `You are analyzing Swedish football commentary for a specific match.`,
    ``,
    `Match: "${matchTitle}"`,
    ``,
    `Commentary segments (Swedish, numbered). Each segment header shows:`,
    `[N] Part <half> | <video timestamp mm:ss> | ~<approximate match minute>'`,
    ``,
    context,
    ``,
    `Query: "${query}"`,
    ``,
    `Task:`,
    `1. Identify which numbered segments actually describe what the user is asking about, IN THIS MATCH. A segment counts as relevant only if the event it describes happened during this match — not in a previous round, not in the spring, not in a career background.`,
    `2. CRITICAL: Swedish football commentators frequently reference OTHER matches as background. These references are NOT events in the current match and are NOT relevant to the user's query. Specifically, ignore a segment (treat it as not relevant) if its main content is about:`,
    `   - A previous round or earlier match: phrases like "i den förra omgången", "förförra omgången", "senaste matchen", "borta mot X", "i våras", "tidigare i säsongen", "den här säsongen".`,
    `   - A career summary or history: phrases like "har gjort X mål för säsongen", "första allsvenska match", "sedan återkomsten".`,
    `   - Another match happening at the same time (other Allsvenskan games being followed): phrases like "Norrköping har 0-0", "Göteborg tar ledningen", "Djurgården i Växjö".`,
    `   Example: a segment saying "Första målet kom här alltså i den förförra omgången mot öster för Barsoum" is about Barsoum's previous-match goal, NOT about the first goal of this match. It is NOT relevant to a query about "the first goal".`,
    `3. Understand natural equivalences when judging relevance (only applied to events in THIS match):`,
    `   - A query about "the Nth goal" or "the N-N goal" matches segments describing the scoreline becoming that number in this match ("2-0 Degerfors", "är reducerat", "i mål!", "i nätet").`,
    `   - A query about a goal matches the segments describing the ball going in during this match, regardless of which Swedish words are used.`,
    `   - A query about a save matches segments describing a keeper stopping a shot in this match ("räddar", "klarar").`,
    `   - A query about a card matches segments mentioning "gult kort" or "rött kort" being issued in this match.`,
    `4. Using ONLY the relevant segments, write a 2-3 sentence answer in English, grounded exclusively in what those segments say.`,
    `5. Do not use outside knowledge about players, teams, or the match. Do not invent any detail (names, minute numbers, scorelines, assists, card counts, etc.) that is not explicitly present in the Swedish text of the relevant segments or in their headers.`,
    `6. SPECIFIC RULE for match minutes: only state a minute number if (a) the Swedish text itself states a minute — e.g. "I den trettionde minuten" (30th) — or (b) you derive it from the "~N'" value in a relevant segment's header. If neither is available, do NOT state a minute number. Never guess.`,
    `7. If no segment is clearly about the query (after applying rules 1-3), output NO_MATCH as the answer and list no relevant segments.`,
    ``,
    `Respond in EXACTLY this format, with nothing before or after:`,
    `RELEVANT: <comma-separated segment numbers, or the word "none">`,
    `ANSWER: <your 2-3 sentence answer, or NO_MATCH>`,
  ].join("\n");

  try {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt,
        stream: false,
        options: { temperature: 0.2, num_predict: 250 },
      }),
      signal: AbortSignal.timeout(60000),
    });
    if (res.ok) {
      const data = await res.json();
      const text = data.response?.trim();
      if (text) {
        console.log(`[llm] raw response:\n${text}\n---`);
        const { answer, relevantIndices } = parseLlmResponse(text, top.length);
        if (looksLikeRefusal(answer) || relevantIndices.length === 0) {
          return { answer: REFUSAL_MESSAGE, refused: true, relevantIndices: [] };
        }
        return { answer, refused: false, relevantIndices };
      }
    }
  } catch (err) {
    console.error("[Ollama] error:", err);
  }

  // Template fallback (Ollama unreachable). Keep the top hit only — safer than blending.
  const s = hits[0]._source;
  const half = s.part === 1 ? "first half" : "second half";
  return {
    answer: `Match moment found at approximately ${s.match_minute}' in the ${half} of ${matchTitle}. The commentary around this timestamp mentions content related to your search "${query}".`,
    refused: false,
    relevantIndices: [0],
  };
}
