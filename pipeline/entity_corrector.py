"""Stage E: TF-IDF retrieval + Qwen MCQ judge for entity correction.

Per detected entity:
  1. Validated cross-match cache hit -> apply.
  2. Per-match cache hit -> reuse.
  3. TF-IDF char-bigram retrieve top-K canonical candidates.
  4. Auto-accept on high cosine + clear winner. Auto-reject on low cosine.
  5. Uncertain band -> Qwen MCQ judge (A/B/C/D=keep/E=unsure).
  6. xlm-roberta MLM veto on the MCQ pick.
  7. Final validation gates (dict veto, fuzz floor, length tolerance).
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.config import (
    FREQUENCY_HEURISTIC_THRESHOLD,
    MCQ_MIN_TOKEN_LEN,
    MCQ_MIN_FUZZ_TO_INVOKE,
    MCQ_SHORT_TOKEN_MIN_FUZZ,
    MCQ_SELF_CONSISTENCY_SAMPLES,
    MLM_VETO_ON_MCQ_ENABLED,
    VALIDATED_CACHE_ENABLED,
    VALIDATED_CACHE_MIN_CONSENSUS,
    VALIDATED_CACHE_MIN_FUZZY,
    VALIDATED_CACHE_PATH,
)
from pipeline.loader import Segment


# ─── Thresholds ───────────────────────────────────────────────────────

SHORTCUT_ACCEPT_TFIDF = 0.90    # auto-accept when cosine clears this with a clear gap
SHORTCUT_ACCEPT_GAP   = 0.10
SHORTCUT_REJECT_TFIDF = 0.40    # below this, skip the entity entirely
TOP_K_CANDIDATES      = 5
MCQ_OPTIONS_SHOWN     = 3       # A/B/C in the MCQ prompt (+ D=keep + E=unsure)
TFIDF_NGRAM_RANGE     = (2, 4)  # char_wb bigrams to 4-grams


# ─── Telemetry ────────────────────────────────────────────────────────

@dataclass
class _Telemetry:
    total_entities: int = 0
    auto_accept_shortcut: int = 0
    auto_reject_low: int = 0
    auto_reject_dict_veto: int = 0
    auto_reject_freq_heuristic: int = 0
    mcq_invoked: int = 0
    mcq_chose_keep: int = 0
    mcq_chose_unsure: int = 0
    mcq_chose_candidate: int = 0
    mcq_call_failed: int = 0
    mlm_vetoed_mcq: int = 0
    cache_hit_per_match: int = 0
    cache_hit_validated: int = 0
    cache_promoted_this_run: int = 0
    accepted: int = 0
    examples: dict = field(default_factory=lambda: {
        "shortcut": [], "mcq_picked": [], "mcq_kept": [], "mlm_vetoed": [],
    })


_LAST_TELEMETRY: dict = {}


def get_last_telemetry() -> dict:
    return dict(_LAST_TELEMETRY)


# ─── Per-match decision cache (cleared at start of each match) ───────

_PER_MATCH_CACHE: dict[tuple, Optional[str]] = {}


def _cache_clear() -> None:
    _PER_MATCH_CACHE.clear()


def _cache_lookup(key: tuple) -> tuple[bool, Optional[str]]:
    if key in _PER_MATCH_CACHE:
        return True, _PER_MATCH_CACHE[key]
    return False, None


def _cache_store(key: tuple, decision: Optional[str]) -> None:
    _PER_MATCH_CACHE[key] = decision


# ─── Cross-match validated cache (consensus-based) ───────────────────

def _load_validated_cache() -> dict:
    """Load the on-disk cross-match cache. Returns {} on any error."""
    p = Path(VALIDATED_CACHE_PATH)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_validated_cache(cache: dict) -> None:
    p = Path(VALIDATED_CACHE_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _validated_cache_lookup(entity_lower: str, cache: dict) -> Optional[str]:
    """Cached canonical when consensus has been reached, else None."""
    entry = cache.get(entity_lower)
    if not entry:
        return None
    if len(entry.get("matches_seen", [])) >= VALIDATED_CACHE_MIN_CONSENSUS:
        return entry.get("correct")
    return None


def _validated_cache_record(
    cache: dict, entity_lower: str, correction: str,
    match_id: str, fuzz_score: float,
) -> bool:
    """Record one MCQ-accepted correction. True iff the entry just hit consensus."""
    if fuzz_score < VALIDATED_CACHE_MIN_FUZZY:
        return False
    entry = cache.setdefault(entity_lower, {
        "correct": correction,
        "matches_seen": [],
        "fuzzy_avg": 0.0,
    })
    # Don't let a different correction overwrite an established mapping.
    if entry["correct"] != correction:
        return False
    if match_id in entry["matches_seen"]:
        return False
    was_validated = len(entry["matches_seen"]) >= VALIDATED_CACHE_MIN_CONSENSUS
    entry["matches_seen"].append(match_id)
    n = len(entry["matches_seen"])
    entry["fuzzy_avg"] = (entry["fuzzy_avg"] * (n - 1) + fuzz_score) / n
    entry["last_updated"] = date.today().isoformat()
    is_validated = n >= VALIDATED_CACHE_MIN_CONSENSUS
    return is_validated and not was_validated


# ─── TF-IDF retrieval over gazetteer ─────────────────────────────────

class _GazetteerIndex:
    """Char n-gram TF-IDF index over gazetteer canonicals. Built once per match."""

    def __init__(self, gazetteer: dict[str, str]):
        self.canonicals: list[str] = sorted(set(gazetteer.values()))
        if not self.canonicals:
            self.vectorizer = None
            self.matrix = None
            return
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=TFIDF_NGRAM_RANGE,
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(self.canonicals)

    def retrieve(self, query: str, top_k: int = TOP_K_CANDIDATES) -> list[tuple[str, float]]:
        if not self.canonicals or self.vectorizer is None or not query.strip():
            return []
        q_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.matrix)[0]
        q_lower = query.lower()
        combined = np.array(tfidf_scores, dtype=float, copy=True)

        # Rescue high-fuzz typos whose cosine fell below the reject floor.
        # Clamp them below shortcut so they still go through MCQ + gates.
        for i, canon in enumerate(self.canonicals):
            words = canon.split() or [canon]
            best_word_fuzz = max(
                fuzz.ratio(q_lower, w.lower()) / 100.0 for w in words
            )
            if best_word_fuzz >= MCQ_MIN_FUZZ_TO_INVOKE / 100.0:
                combined[i] = max(
                    float(tfidf_scores[i]),
                    min(best_word_fuzz, SHORTCUT_ACCEPT_TFIDF - 1e-6),
                )

        top_idx = np.argsort(-combined)[:top_k]
        return [
            (self.canonicals[i], float(combined[i]))
            for i in top_idx if combined[i] > 0
        ]


# ─── Validation gates (live in fuzzy_corrector for now) ─────────────

def _validation_gate(original: str, corrected: str, language: str = "en") -> bool:
    from pipeline.fuzzy_corrector import passes_conservative_gates
    return passes_conservative_gates(original, corrected, language=language)


def _reduce_to_best_word(entity: str, canonical: str) -> str:
    """If entity is one word and canonical is multi-word, pick the closest
    canonical word (usually the surname). ASR rarely produces full names so
    expanding to "Daniel Sturridge" would trip the fuzz floor anyway."""
    if not entity or not canonical:
        return canonical
    if " " in entity or " " not in canonical:
        return canonical
    canon_parts = canonical.split()
    best_word, best_score = canonical, -1.0
    for w in canon_parts:
        s = fuzz.ratio(entity.lower(), w.lower())
        if s > best_score:
            best_score = s
            best_word = w
    return best_word


def _entity_core(text: str) -> str:
    from pipeline.fuzzy_corrector import extract_entity_core
    return extract_entity_core(text)


def _rebuild(original_text: str, canonical: str, language: str) -> str:
    from pipeline.fuzzy_corrector import extract_and_rebuild_entity
    return extract_and_rebuild_entity(original_text, canonical, language=language)


# ─── MCQ Judge prompt (Qwen via llm_corrector handle) ────────────────

# Qwen 1.5B has a strong "always pick A" bias on short ambiguous tokens.
# The rules + negative examples push it toward D when the original is itself
# a real entity (other-team player, city, day of week).
_MCQ_SYSTEM_TEMPLATE = (
    "You correct football commentary transcribed by Whisper. "
    "Pick the candidate ONLY when the original token is clearly an ASR "
    "mishearing of a player from the listed lineup. Otherwise keep it.\n"
    "{context}\n"
    "Rules:\n"
    "- Reply with EXACTLY ONE LETTER (A, B, C, D, or E).\n"
    "- A/B/C are candidate corrections from the lineup above.\n"
    "- D = keep the original token unchanged.\n"
    "- E = unsure (treated as keep).\n"
    "- USE D when the original is itself a real name (different player, "
    "city, common word, day of week) even if a candidate looks similar.\n"
    "- USE D when the surrounding context is generic and any candidate "
    "would equally fit — picking randomly hurts more than helps.\n"
    "- Only pick A/B/C when there is a STRONG signal: the original "
    "sounds like the candidate AND the context names that specific player.\n"
    "\n"
    "Examples:\n"
    'Original: "Kane" | A) Mane B) Cahill C) Henderson | Context: "shot from Kane outside the box"\n'
    "Answer: D  (Harry Kane is a real player not in this lineup)\n"
    "\n"
    'Original: "Northampton" | A) Southampton B) Tottenham | Context: "Northampton in the FA Cup"\n'
    "Answer: D  (Northampton is a real club, not a mishearing)\n"
    "\n"
    'Original: "Saturday" | A) Sturridge B) Mane | Context: "starry Saturday night at Stamford"\n'
    "Answer: D  (calendar day, not an ASR error)\n"
    "\n"
    'Original: "Starridge" | A) Sturridge B) Cahill | Context: "Starridge through on goal"\n'
    "Answer: A  (Sturridge in the lineup, clear ASR mishearing, context fits)\n"
    "\n"
    'Original: "Hendrik" | A) Henderson B) Hazard | Context: "Hendrik passes to Lallana"\n'
    "Answer: A  (Henderson — clearly mis-transcribed, Liverpool midfielder fits)"
)


def _build_mcq_user_msg(
    original: str,
    candidates: list[str],
    prev_text: str,
    next_text: str,
    segment_text: str,
) -> str:
    options: list[str] = []
    letters = ["A", "B", "C"]
    for i, cand in enumerate(candidates[:MCQ_OPTIONS_SHOWN]):
        options.append(f"{letters[i]}) {cand}")
    while len(options) < MCQ_OPTIONS_SHOWN:
        options.append(f"{letters[len(options)]}) (no candidate)")
    options.append("D) keep original")
    options.append("E) unsure")
    return (
        f'Segment: "{segment_text.strip()}"\n'
        f'Previous: "{prev_text.strip()}"\n'
        f'Next: "{next_text.strip()}"\n'
        f'Original token: "{original}"\n'
        f"Choices:\n  " + "\n  ".join(options) + "\n\n"
        f"Answer (single letter):"
    )


def _build_match_context_block(
    gazetteer: dict[str, str],
    entity_types: dict[str, str],
    match_name: str,
    half: int,
) -> str:
    by_type: dict[str, list[str]] = {}
    for canon, etype in (entity_types or {}).items():
        by_type.setdefault(etype, []).append(canon)
    players = sorted(by_type.get("player", []))[:30]
    teams = sorted(by_type.get("team", []))[:8]
    referees = sorted(by_type.get("referee", []))[:3]
    lines = [f"Match: {match_name} | Half: {half}"]
    if teams:
        lines.append("Teams: " + ", ".join(teams))
    if players:
        lines.append("Players: " + ", ".join(players))
    if referees:
        lines.append("Referee: " + ", ".join(referees))
    return "\n".join(lines)


def _single_mcq_sample(
    original: str, candidates: list[str], prev_text: str, next_text: str,
    segment_text: str, context_block: str, temperature: float = 0.0,
) -> Optional[str]:
    """One Qwen MCQ call. Returns A-E or None on failure."""
    from pipeline import llm_corrector as lc
    if not lc._ensure_llm():
        return None
    system = _MCQ_SYSTEM_TEMPLATE.format(context=context_block)
    user = _build_mcq_user_msg(original, candidates, prev_text, next_text, segment_text)
    try:
        resp = lc._LLM.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=8,
            stop=["\n", ".", " "],
        )
        content = resp["choices"][0]["message"]["content"].strip().upper()
        m = re.search(r"\b([A-E])\b", content)
        return m.group(1) if m else None
    except Exception as e:
        print(f"  [entity_corrector] MCQ call failed: {e}", file=sys.stderr)
        return None


def _mcq_call(
    original: str,
    candidates: list[str],
    prev_text: str,
    next_text: str,
    segment_text: str,
    context_block: str,
) -> Optional[str]:
    """MCQ with majority-vote self-consistency. First sample at temp 0,
    rest at temp 0.3. Default samples=1 since Qwen 1.5B is deterministic
    on single-letter output. No clear majority -> 'D' (keep)."""
    samples: list[str] = []
    for i in range(MCQ_SELF_CONSISTENCY_SAMPLES):
        temp = 0.0 if i == 0 else 0.3
        result = _single_mcq_sample(
            original, candidates, prev_text, next_text,
            segment_text, context_block, temperature=temp,
        )
        if result:
            samples.append(result)
    if not samples:
        return None
    from collections import Counter
    counts = Counter(samples)
    top_letter, top_count = counts.most_common(1)[0]
    if top_count > len(samples) / 2:
        return top_letter
    return "D"


# ─── MLM veto on MCQ picks (reuses xlm-roberta from llm_corrector) ──

def _mlm_veto_mcq_pick(
    original: str, picked: str, segment_text: str,
) -> bool:
    """True iff xlm-roberta prefers the original over the MCQ pick by
    MLM_VETO_RATIO. Reuses the MLM handle from Step L."""
    if not MLM_VETO_ON_MCQ_ENABLED:
        return False
    try:
        from pipeline import llm_corrector as lc
        from pipeline.config import MLM_VETO_RATIO
        tokens = segment_text.split()
        idx = -1
        ent_clean = original.strip(" .,!?;:'\"()-").lower()
        for i, tok in enumerate(tokens):
            if tok.strip(" .,!?;:'\"()-").lower() == ent_clean:
                idx = i
                break
        if idx < 0:
            return False
        lp_orig = lc._mlm_pseudo_logprob(tokens, idx, original)
        lp_pick = lc._mlm_pseudo_logprob(tokens, idx, picked)
        if lp_orig is None or lp_pick is None:
            return False
        return (lp_orig - lp_pick) >= math.log(MLM_VETO_RATIO)
    except Exception:
        return False


# ─── Public API ──────────────────────────────────────────────────────

def correct_match(
    segments: list[Segment],
    gazetteer: dict[str, str],
    entity_types: dict[str, str],
    segment_entities_map: dict[tuple[int, str], list],
    match_id: str,
    match_name: str = "",
    language: str = "en",
) -> tuple[list[Segment], list[dict]]:
    """Run Stage E across a match. segment_entities_map keys are (half, segment_id)."""
    global _LAST_TELEMETRY
    telem = _Telemetry()

    if not gazetteer or not segments:
        _LAST_TELEMETRY = _telem_to_dict(telem)
        return list(segments), []

    _cache_clear()
    validated_cache = _load_validated_cache() if VALIDATED_CACHE_ENABLED else {}

    index = _GazetteerIndex(gazetteer)

    # Token frequency over the whole match (for the freq heuristic).
    token_freq: Counter = Counter()
    for s in segments:
        for tok in s.text.split():
            token_freq[tok.strip(" .,!?;:'\"()-").lower()] += 1

    # Build segment_id → idx map for O(1) prev/next lookup
    seg_idx_by_uid = {(s.half, s.segment_id): i for i, s in enumerate(segments)}

    out_segments: list[Segment] = []
    out_corrections: list[dict] = []

    for seg in segments:
        entities = segment_entities_map.get((seg.half, seg.segment_id), [])
        if not entities:
            out_segments.append(seg)
            continue

        text = seg.text
        # Word indices we corrected, so Step L can freeze them.
        # Computed on the FINAL text after all per-segment edits land.
        corrected_canonicals: list[str] = []
        # Process right-to-left so character offsets stay valid.
        sorted_ents = sorted(entities, key=lambda e: e.start_char, reverse=True)

        for entity in sorted_ents:
            telem.total_entities += 1
            entity_text = _entity_core(entity.text)
            if not entity_text or len(entity_text.strip()) <= 2:
                continue
            ent_lower = entity_text.lower()

            # ── 1. Validated cross-match cache ──────────────────────
            if VALIDATED_CACHE_ENABLED:
                cached = _validated_cache_lookup(ent_lower, validated_cache)
                if cached and cached.lower() != ent_lower:
                    if _validation_gate(entity_text, cached, language):
                        text = _splice(text, entity, cached, language)
                        out_corrections.append(_correction_dict(
                            entity_text, cached, seg, "validated_cache", 100.0,
                        ))
                        corrected_canonicals.append(cached)
                        telem.cache_hit_validated += 1
                        telem.accepted += 1
                        continue

            # ── 2. TF-IDF retrieval ─────────────────────────────────
            candidates = index.retrieve(entity_text)
            if not candidates:
                telem.auto_reject_low += 1
                continue

            best_canon, best_score = candidates[0]
            second_score = candidates[1][1] if len(candidates) > 1 else 0.0

            # Skip self-match (entity already canonical)
            if best_canon.lower() == ent_lower:
                continue

            # ── 3. Per-match cache lookup ───────────────────────────
            cache_key = (ent_lower, tuple(c[0] for c in candidates[:3]))
            cache_hit, cached_decision = _cache_lookup(cache_key)
            if cache_hit:
                telem.cache_hit_per_match += 1
                if cached_decision is not None:
                    text = _splice(text, entity, cached_decision, language)
                    out_corrections.append(_correction_dict(
                        entity_text, cached_decision, seg, "per_match_cache", 100.0,
                    ))
                    corrected_canonicals.append(cached_decision)
                    telem.accepted += 1
                continue

            # ── 4. Frequency heuristic ──────────────────────────────
            if token_freq.get(ent_lower, 0) >= FREQUENCY_HEURISTIC_THRESHOLD:
                # Probably a common word (e.g. Swedish "kommer" appearing 17x).
                telem.auto_reject_freq_heuristic += 1
                _cache_store(cache_key, None)
                continue

            # ── 5. Shortcut-reject ──────────────────────────────────
            if best_score < SHORTCUT_REJECT_TFIDF:
                telem.auto_reject_low += 1
                _cache_store(cache_key, None)
                continue

            # ── 6. Shortcut-accept (high cosine + clear winner + gates) ──
            if (
                best_score >= SHORTCUT_ACCEPT_TFIDF
                and (best_score - second_score) >= SHORTCUT_ACCEPT_GAP
            ):
                # For one-word entities, swap to the matching canonical word
                # ("Sturridge" not "Daniel Sturridge").
                applied_canon = _reduce_to_best_word(entity_text, best_canon)
                if _validation_gate(entity_text, applied_canon, language):
                    text = _splice(text, entity, applied_canon, language)
                    out_corrections.append(_correction_dict(
                        entity_text, applied_canon, seg, "tfidf_shortcut",
                        best_score * 100,
                    ))
                    corrected_canonicals.append(applied_canon)
                    telem.auto_accept_shortcut += 1
                    telem.accepted += 1
                    if len(telem.examples["shortcut"]) < 5:
                        telem.examples["shortcut"].append(
                            f"{entity_text} → {applied_canon} ({best_score:.2f})"
                        )
                    if VALIDATED_CACHE_ENABLED:
                        fuzz_s = fuzz.ratio(entity_text.lower(), applied_canon.lower())
                        if _validated_cache_record(
                            validated_cache, ent_lower, applied_canon, match_id, fuzz_s,
                        ):
                            telem.cache_promoted_this_run += 1
                    _cache_store(cache_key, applied_canon)
                    continue
                else:
                    telem.auto_reject_dict_veto += 1
                    _cache_store(cache_key, None)
                    continue

            # ── 7. Uncertain band → MCQ eligibility check ───────────
            cand_names = [c[0] for c in candidates[:MCQ_OPTIONS_SHOWN]]

            # Top candidate's word-level fuzz to the original. Filters TF-IDF
            # noise hits where cosine is high but the actual tokens are unrelated.
            top_reduced = _reduce_to_best_word(entity_text, cand_names[0])
            top_fuzz = fuzz.ratio(entity_text.lower(), top_reduced.lower())

            # Block short tokens unless fuzz is very high. Catches Sako->Sakho
            # while keeping Kane->Mane (~75) blocked.
            ent_core = entity_text.strip()
            if (
                len(ent_core) < MCQ_MIN_TOKEN_LEN
                and top_fuzz < MCQ_SHORT_TOKEN_MIN_FUZZ
            ):
                telem.auto_reject_low += 1
                _cache_store(cache_key, None)
                continue

            if top_fuzz < MCQ_MIN_FUZZ_TO_INVOKE:
                telem.auto_reject_low += 1
                _cache_store(cache_key, None)
                continue

            # ── 8. MCQ judge (3-sample self-consistency) ────────────
            telem.mcq_invoked += 1
            seg_idx = seg_idx_by_uid.get((seg.half, seg.segment_id), -1)
            prev_text = segments[seg_idx - 1].text if seg_idx > 0 else ""
            next_text = (
                segments[seg_idx + 1].text
                if 0 <= seg_idx < len(segments) - 1 else ""
            )
            ctx_block = _build_match_context_block(
                gazetteer, entity_types, match_name, seg.half,
            )

            choice = _mcq_call(
                entity_text, cand_names, prev_text, next_text, seg.text, ctx_block,
            )

            if choice is None:
                telem.mcq_call_failed += 1
                _cache_store(cache_key, None)
                continue
            if choice == "E":
                telem.mcq_chose_unsure += 1
                _cache_store(cache_key, None)
                continue
            if choice == "D":
                telem.mcq_chose_keep += 1
                if len(telem.examples["mcq_kept"]) < 5:
                    telem.examples["mcq_kept"].append(
                        f"{entity_text} (candidates: {cand_names})"
                    )
                _cache_store(cache_key, None)
                continue

            letter_idx = {"A": 0, "B": 1, "C": 2}.get(choice, -1)
            if letter_idx < 0 or letter_idx >= len(cand_names):
                telem.mcq_chose_unsure += 1
                _cache_store(cache_key, None)
                continue

            picked_canon = cand_names[letter_idx]
            picked = _reduce_to_best_word(entity_text, picked_canon)
            if not _validation_gate(entity_text, picked, language):
                telem.auto_reject_dict_veto += 1
                _cache_store(cache_key, None)
                continue

            # ── 9. MLM veto on MCQ pick (xlm-roberta second opinion) ──
            if _mlm_veto_mcq_pick(entity_text, picked, seg.text):
                telem.mlm_vetoed_mcq += 1
                if len(telem.examples["mlm_vetoed"]) < 5:
                    telem.examples["mlm_vetoed"].append(
                        f"{entity_text} → {picked} (MLM kept original)"
                    )
                _cache_store(cache_key, None)
                continue

            text = _splice(text, entity, picked, language)
            out_corrections.append(_correction_dict(
                entity_text, picked, seg, "mcq_judge",
                best_score * 100, judge_choice=choice,
            ))
            corrected_canonicals.append(picked)
            telem.mcq_chose_candidate += 1
            telem.accepted += 1
            if len(telem.examples["mcq_picked"]) < 5:
                telem.examples["mcq_picked"].append(
                    f"{entity_text} → {picked} ({choice})"
                )
            if VALIDATED_CACHE_ENABLED:
                fuzz_s = fuzz.ratio(entity_text.lower(), picked.lower())
                if _validated_cache_record(
                    validated_cache, ent_lower, picked, match_id, fuzz_s,
                ):
                    telem.cache_promoted_this_run += 1
            _cache_store(cache_key, picked)

        # Frozen word indices on the FINAL text — Step L treats these as non-editable.
        frozen: list[int] = []
        if corrected_canonicals:
            words_lower = [w.strip(" .,!?;:'\"()-").lower() for w in text.split()]
            for canon in corrected_canonicals:
                for canon_word in canon.split():
                    cw_lower = canon_word.strip(" .,!?;:'\"()-").lower()
                    for i, w in enumerate(words_lower):
                        if w == cw_lower and i not in frozen:
                            frozen.append(i)
                            break

        out_segments.append(Segment(
            segment_id=seg.segment_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=text,
            half=seg.half,
            global_id=seg.global_id,
            words=seg.words,
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
            frozen_word_indices=frozen if frozen else None,
        ))

    if VALIDATED_CACHE_ENABLED:
        _save_validated_cache(validated_cache)

    _LAST_TELEMETRY = _telem_to_dict(telem)
    print(
        f"  [entity_corrector] entities={telem.total_entities} "
        f"shortcut={telem.auto_accept_shortcut} "
        f"mcq={telem.mcq_invoked} "
        f"(picked={telem.mcq_chose_candidate} keep={telem.mcq_chose_keep} "
        f"unsure={telem.mcq_chose_unsure} fail={telem.mcq_call_failed} "
        f"mlm_veto={telem.mlm_vetoed_mcq}) "
        f"cache_hit={telem.cache_hit_per_match}+{telem.cache_hit_validated}val "
        f"promoted={telem.cache_promoted_this_run} "
        f"accepted={telem.accepted}",
        file=sys.stderr,
    )
    return out_segments, out_corrections


# ─── Helpers ─────────────────────────────────────────────────────────

def _splice(text: str, entity, canonical: str, language: str) -> str:
    rebuilt = _rebuild(entity.text, canonical, language)
    return text[:entity.start_char] + rebuilt + text[entity.end_char:]


def _correction_dict(
    original: str, corrected: str, seg, method: str, score: float,
    judge_choice: Optional[str] = None,
) -> dict:
    d = {
        "segment_id": seg.segment_id,
        "half": seg.half,
        "original": original,
        "corrected": corrected,
        "score": round(float(score), 1),
        "method": method,
        "stage": "E",
    }
    if judge_choice:
        d["judge_choice"] = judge_choice
    return d


def _telem_to_dict(t: _Telemetry) -> dict:
    return {
        "total_entities": t.total_entities,
        "accepted": t.accepted,
        "auto_accept_shortcut": t.auto_accept_shortcut,
        "auto_reject_low": t.auto_reject_low,
        "auto_reject_dict_veto": t.auto_reject_dict_veto,
        "auto_reject_freq_heuristic": t.auto_reject_freq_heuristic,
        "mcq_invoked": t.mcq_invoked,
        "mcq_chose_candidate": t.mcq_chose_candidate,
        "mcq_chose_keep": t.mcq_chose_keep,
        "mcq_chose_unsure": t.mcq_chose_unsure,
        "mcq_call_failed": t.mcq_call_failed,
        "mlm_vetoed_mcq": t.mlm_vetoed_mcq,
        "cache_hit_per_match": t.cache_hit_per_match,
        "cache_hit_validated": t.cache_hit_validated,
        "cache_promoted_this_run": t.cache_promoted_this_run,
        "examples": dict(t.examples),
    }
