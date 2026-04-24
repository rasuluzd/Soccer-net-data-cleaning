"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/context/AuthContext";
import VideoPlayer from "@/components/VideoPlayer";
import Logo from "@/components/Logo";

interface Clip {
  matchId: string; matchTitle: string; part: number; start: number; end: number;
  matchMinute: number; title: string; commentary: string;
  video: { type: string; parts: Record<string, string>; part2_offset_sec?: number };
}
interface SearchResult { answer: string; clips: Clip[]; esHits: number; }
interface HistoryItem { id: string; query: string; answer: string; clips: Clip[]; time: string; }
interface MatchOption { id: string; title: string; subtitle?: string; date?: string }

const fmtTime = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

export default function AppPage() {
  const { user, loading: authLoading, logout, token } = useAuth();
  const router = useRouter();

  const [query, setQuery] = useState("");
  const [searching, setSearching] = useState(false);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [sidebar, setSidebar] = useState(true);
  const [matches, setMatches] = useState<MatchOption[]>([]);
  const [matchId, setMatchId] = useState<string>("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!user) return;
    fetch("/api/matches")
      .then((r) => (r.ok ? r.json() : []))
      .then((data: MatchOption[]) => {
        setMatches(data);
        if (data.length && !matchId) setMatchId(data[0].id);
      })
      .catch(() => setMatches([]));
  }, [user?.id]);

  useEffect(() => {
    if (!authLoading && !user) router.push("/login");
  }, [authLoading, user, router]);

  const historyKey = user ? `forzasearch-history-${user.id}` : "forzasearch-history";

  useEffect(() => {
    if (!user) {
      setHistory([]);
      return;
    }
    try {
      const saved = localStorage.getItem(historyKey);
      if (saved) setHistory(JSON.parse(saved));
      else setHistory([]);
    } catch {
      setHistory([]);
    }
  }, [user?.id]);

  const saveHistory = (items: HistoryItem[]) => {
    setHistory(items);
    if (user) {
      localStorage.setItem(historyKey, JSON.stringify(items.slice(0, 50)));
    }
  };

  const deleteHistoryItem = (id: string) => {
    saveHistory(history.filter((h) => h.id !== id));
  };

  const clearHistory = () => {
    if (user) localStorage.removeItem(historyKey);
    saveHistory([]);
    setResult(null);
    setQuery("");
  };

  const handleSearch = useCallback(async () => {
    const q = query.trim();
    if (!q || searching) return;
    if (!matchId) {
      setResult({ answer: "Pick a match first.", clips: [], esHits: 0 });
      return;
    }
    setSearching(true);
    setResult(null);

    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
        body: JSON.stringify({ query: q, matchId }),
      });
      const data = await res.json();
      if (data.error) {
        setResult({ answer: `Error: ${data.detail || data.error}`, clips: [], esHits: 0 });
      } else {
        setResult(data);
        const item: HistoryItem = { id: Date.now().toString(), query: q, answer: data.answer, clips: data.clips || [], time: new Date().toISOString() };
        saveHistory([item, ...history]);
      }
    } catch {
      setResult({ answer: "Could not reach the server. Is it running?", clips: [], esHits: 0 });
    } finally {
      setSearching(false);
    }
  }, [query, searching, token, history, matchId]);

  const loadHistory = (item: HistoryItem) => {
    setQuery(item.query);
    setResult({ answer: item.answer, clips: item.clips, esHits: 0 });
  };

  if (authLoading || !user) {
    return <div className="min-h-screen bg-brand-navy flex items-center justify-center text-white/50">Loading...</div>;
  }

  return (
    <div className="flex h-screen font-body overflow-hidden bg-brand-navy">

      {/* ═══ SIDEBAR ═══ */}
      <div className={`${sidebar ? "w-72" : "w-0"} min-w-0 bg-brand-navy border-r border-white/5 flex flex-col transition-all duration-300 overflow-hidden`}>
        <div className="p-5 border-b border-white/5">
          <Link href="/" className="hover:opacity-70 transition-opacity block">
            <Logo className="h-6 w-auto" />
          </Link>
        </div>

        <div className="p-3">
          <button onClick={() => { setQuery(""); setResult(null); inputRef.current?.focus(); }}
            className="w-full flex items-center gap-2 px-4 py-2.5 rounded-lg border border-white/10 text-white/80 text-sm hover:bg-white/5 transition-colors">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path d="M12 4v16m8-8H4"/></svg>
            New search
          </button>
        </div>

        {/* History */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-2">
          <div className="px-2 py-2 flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-widest text-white/30 font-semibold">Recent</span>
            {history.length > 0 && (
              <button onClick={clearHistory} className="text-[10px] text-white/30 hover:text-red-400 transition-colors">Clear all</button>
            )}
          </div>
          {history.map((item) => (
            <div key={item.id} className="group flex items-center gap-1 mb-0.5">
              <button onClick={() => loadHistory(item)}
                className="flex-1 text-left px-3 py-2 rounded-lg text-white/50 text-sm hover:bg-white/5 hover:text-white/80 transition-colors truncate">
                {item.query}
              </button>
              <button onClick={() => deleteHistoryItem(item.id)}
                className="opacity-0 group-hover:opacity-100 p-1 text-white/20 hover:text-red-400 transition-all flex-shrink-0">
                <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path d="M6 18L18 6M6 6l12 12"/></svg>
              </button>
            </div>
          ))}
          {history.length === 0 && <p className="text-white/20 text-xs text-center py-6">Search history appears here</p>}
        </div>

        {/* User + logout */}
        <div className="p-3 border-t border-white/5">
          <button onClick={() => { logout(); router.push("/"); }}
            className="text-white/30 hover:text-red-400 text-xs transition-colors mb-2 px-3">
            Log out
          </button>
          <div className="flex items-center gap-3 px-3 py-2">
            <div className="w-8 h-8 rounded-full bg-brand-gold flex items-center justify-center text-white text-xs font-bold">
              {user.firstName[0]}{user.lastName[0]}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-white text-sm font-medium truncate">{user.firstName} {user.lastName}</p>
              <p className="text-white/40 text-xs truncate">{user.email}</p>
            </div>
          </div>
        </div>
      </div>

      {/* ═══ MAIN ═══ */}
      <div className="flex-1 flex flex-col bg-gradient-to-br from-brand-gold to-brand-gold-dark relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_30%_20%,rgba(255,255,255,0.04)_0%,transparent_60%)] pointer-events-none" />

        {/* Topbar */}
        <div className="h-14 flex items-center px-4 relative z-10">
          <button onClick={() => setSidebar(!sidebar)} className="w-9 h-9 rounded-lg bg-black/20 flex items-center justify-center text-white/80 hover:text-white">
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path d="M3 12h18M3 6h18M3 18h18"/></svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex flex-col items-center overflow-y-auto scrollbar-thin px-6 pb-6 relative z-10"
          style={{ justifyContent: result ? "flex-start" : "center" }}>

          {!result && !searching && (
            <div className="text-center mb-8">
              <h1 className="font-display text-4xl md:text-5xl font-bold text-white leading-tight mb-3">
                Find your sports highlight<br />immediately.
              </h1>
              <p className="text-white/45 text-sm">Search match commentary · Get AI answers · Watch the video clip</p>
            </div>
          )}

          {/* Match picker — queries are scoped to one match */}
          <div className="w-full max-w-2xl mb-3 flex-shrink-0">
            <label className="block text-white/70 text-xs mb-1.5">Match</label>
            <select
              value={matchId}
              onChange={(e) => setMatchId(e.target.value)}
              disabled={!matches.length}
              className="w-full bg-black/30 border border-white/15 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-white/40 disabled:opacity-50"
            >
              {matches.length === 0 && <option value="">No matches available</option>}
              {matches.map((m) => (
                <option key={m.id} value={m.id} className="text-black">
                  {m.title}{m.date ? ` — ${m.date}` : ""}
                </option>
              ))}
            </select>
          </div>

          {/* Search bar — no + icon */}
          <div className="w-full max-w-2xl mb-6 flex-shrink-0">
            <div className="flex items-center bg-white/90 rounded-full px-5 py-1 shadow-lg">
              <input ref={inputRef} value={query} onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="Search for a match moment..."
                className="flex-1 bg-transparent border-none outline-none text-gray-800 text-base py-3 font-body placeholder:text-gray-400" />
              <button onClick={handleSearch} disabled={searching || !query.trim()}
                className="w-10 h-10 rounded-full flex items-center justify-center hover:bg-gray-100 transition-colors disabled:opacity-30">
                {searching ? (
                  <div className="w-5 h-5 border-2 border-gray-300 border-t-gray-600 rounded-full animate-spin" />
                ) : (
                  <svg className="w-5 h-5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
                )}
              </button>
            </div>
          </div>

          {searching && (
            <div className="text-center text-white/60 text-sm">
              <p className="text-[10px] uppercase tracking-widest text-white/30 mb-1">Searching commentary...</p>
              Querying Elasticsearch + generating AI answer
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="w-full max-w-3xl space-y-4">
              {/* Answer */}
              <div className="bg-black/30 backdrop-blur-xl border border-white/8 rounded-2xl p-6">
                <p className="text-white text-[15px] leading-relaxed whitespace-pre-wrap">{result.answer}</p>
                {result.esHits > 0 && <p className="text-white/20 text-[11px] mt-3 font-mono">{result.esHits} Elasticsearch hits</p>}
              </div>

              {/* Clips — show timestamp info + commentary, user controls video */}
              {result.clips?.map((clip, i) => (
                <div key={i} className="bg-black/30 backdrop-blur-xl border border-white/8 rounded-2xl overflow-hidden">
                  <div className="px-5 py-3">
                    <div className="flex justify-between items-center mb-2">
                      <p className="text-white text-sm font-semibold">{clip.matchTitle || clip.matchId}</p>
                      <span className="text-white/40 text-xs font-mono">~{clip.matchMinute}&apos; match minute</span>
                    </div>
                    <p className="text-white/50 text-xs mb-1">
                      Part {clip.part} · Video timestamp {fmtTime(clip.start)} → {fmtTime(clip.end)}
                    </p>
                    {clip.commentary && (
                      <p className="text-white/35 text-xs italic mt-2 leading-relaxed line-clamp-3">{clip.commentary}</p>
                    )}
                  </div>
                  <div className="px-3 pb-3">
                    <VideoPlayer
                      src={clip.video?.parts?.["1"] || ""}
                      type={(clip.video?.type as "hls" | "direct") || "hls"}
                    />
                  </div>
                </div>
              ))}

              {(!result.clips || result.clips.length === 0) && (
                <div className="bg-black/30 backdrop-blur-xl border border-white/8 rounded-2xl p-5 text-center text-white/40 text-sm">
                  No video clips could be linked to this moment.
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
