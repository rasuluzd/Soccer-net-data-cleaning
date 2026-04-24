const OLLAMA_URL = process.env.OLLAMA_URL || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || "mistral";

interface EsHit {
  _source: { text: string; part: number; start_sec: number; end_sec: number; match_minute: number; match_id: string };
}

export async function generateAnswer(query: string, hits: EsHit[], matchTitle: string): Promise<string> {
  const top = hits.slice(0, 8);
  const context = top
    .map((h, i) => {
      const s = h._source;
      const sm = Math.floor(s.start_sec / 60);
      const ss = Math.floor(s.start_sec % 60);
      return `[${i + 1}] Part ${s.part} | ${sm}:${String(ss).padStart(2, "0")} | ~${s.match_minute}'\n${s.text}`;
    })
    .join("\n\n");

  // Try Ollama
  try {
    const res = await fetch(`${OLLAMA_URL}/api/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: OLLAMA_MODEL,
        prompt: `You are a football match analyst. Given these Swedish commentary segments from "${matchTitle}", answer the user's query in English in 2-3 sentences.\n\nCommentary:\n${context}\n\nQuery: "${query}"\n\nAnswer:`,
        stream: false,
        options: { temperature: 0.3, num_predict: 200 },
      }),
      signal: AbortSignal.timeout(60000),
    });
    if (res.ok) {
      const data = await res.json();
      if (data.response?.trim()) return data.response.trim();
    }
  } catch (err) {
    console.error("[Ollama] error:", err);
  }

  // Template fallback
  const s = hits[0]._source;
  const half = s.part === 1 ? "first half" : "second half";
  return `Match moment found at approximately ${s.match_minute}' in the ${half} of ${matchTitle}. The commentary around this timestamp mentions content related to your search "${query}".`;
}
