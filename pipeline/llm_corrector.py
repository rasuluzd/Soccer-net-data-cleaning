"""
pipeline/llm_corrector.py — confidence-gated Generative Error Correction.

Step L of the SOTA refactor (see plans/check-all-files-and-peppy-lemur.md).
This module replaces the legacy Tier 2 / Tier 3 / Stage 4 / Stage 4B /
learned_dictionary cascade with one coherent decision-maker:

  1. **Confidence gating.** Every segment's words come with an avg
     ``probability`` from faster-whisper (``Segment.words``). Tokens with
     ``log(prob) > LLM_LOGPROB_GATE`` are deemed acoustically confident
     and stay verbatim; only the low-confidence tokens are wrapped
     ``<token>`` in the LLM prompt. The LLM is instructed to leave
     unwrapped tokens untouched. This is the "Confidence-Guided Error
     Correction" pattern (arxiv:2509.25048) — its key insight is that
     LLMs over-correct already-correct text, and constraining the LLM
     to only edit low-confidence regions eliminates that failure mode.

  2. **Match-context prompt.** Each prompt carries the gazetteer typed
     by entity (Players / Teams / Referee / Venue / Coaches) plus
     ``LLM_CTX_PREVIOUS_SEGMENTS`` previous segments and
     ``LLM_CTX_NEXT_SEGMENTS`` next segments for local context. This is
     the "domain-aware GER" pattern (Whispering-LLaMA EMNLP 2023, GER-
     LoRA ACL Findings 2025).

  3. **MLM veto.** After the LLM proposes an edit, we mask the proposed
     token in context and ask the existing xlm-roberta-base handle from
     ``pipeline.error_detector`` whether the *original* token has a
     higher pseudo-log-likelihood than the proposed one. If it does by a
     factor ``MLM_VETO_RATIO`` or more, we reject the LLM edit. This
     catches the residual over-correction failure mode and keeps the
     pipeline language-agnostic (no static word lists).

Backend
-------
Qwen2.5-0.5B-Instruct (q4_k GGUF) via ``llama-cpp-python``. CPU-friendly:
~500-800 ms per segment on a single thread; gated to ~30-50% of segments
by the confidence filter so a 600-segment match takes ~5 minutes of LLM
time.

Graceful degradation
--------------------
If the GGUF model file is not present or ``llama-cpp-python`` is not
installed, the module is a no-op (prints a warning, returns segments
unchanged). The orchestrator continues with the rest of the pipeline.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pipeline.config import (
    LLM_CORRECTION_ENABLED,
    LLM_CTX_NEXT_SEGMENTS,
    LLM_CTX_PREVIOUS_SEGMENTS,
    LLM_CTX_WINDOW,
    LLM_LOGPROB_GATE,
    LLM_MAX_NEW_TOKENS,
    LLM_MIN_TOKENS_TO_INVOKE,
    LLM_MODEL_FILENAME,
    LLM_MODEL_PATH,
    LLM_MODEL_REPO,
    LLM_NUM_THREADS,
    LLM_TEMPERATURE,
    MLM_VETO_ENABLED,
    MLM_VETO_RATIO,
    get_rejected_pos_tags,
)
from pipeline.loader import Segment


# ─── Singleton model handles ─────────────────────────────────────────

_LLM = None
_LLM_LOAD_ERROR: Optional[str] = None
_MLM_HANDLE = None


def _ensure_llm() -> bool:
    """Lazy-load Qwen via llama-cpp-python. Returns True if available."""
    global _LLM, _LLM_LOAD_ERROR
    if _LLM is not None:
        return True
    if _LLM_LOAD_ERROR is not None:
        return False

    model_path = Path(LLM_MODEL_PATH)
    if not model_path.exists():
        # Try to auto-download from HF if possible.
        try:
            from huggingface_hub import hf_hub_download
            model_path.parent.mkdir(parents=True, exist_ok=True)
            print(
                f"  [llm_corrector] downloading {LLM_MODEL_REPO} / "
                f"{LLM_MODEL_FILENAME} (~400 MB, one-time) ...",
                file=sys.stderr,
            )
            downloaded = hf_hub_download(
                repo_id=LLM_MODEL_REPO,
                filename=LLM_MODEL_FILENAME,
                local_dir=str(model_path.parent),
            )
            # Move/rename to canonical path if hf placed it elsewhere
            if Path(downloaded).resolve() != model_path.resolve():
                try:
                    Path(downloaded).rename(model_path)
                except Exception:
                    model_path = Path(downloaded)
        except Exception as e:
            _LLM_LOAD_ERROR = (
                f"GGUF not at {model_path} and auto-download failed ({e}); "
                "manually download Qwen2.5-0.5B-Instruct-GGUF q4_k_m and place at this path."
            )
            print(f"  [llm_corrector] DISABLED — {_LLM_LOAD_ERROR}", file=sys.stderr)
            return False

    try:
        from llama_cpp import Llama
    except ImportError as e:
        _LLM_LOAD_ERROR = f"llama-cpp-python not installed: {e}"
        print(f"  [llm_corrector] DISABLED — {_LLM_LOAD_ERROR}", file=sys.stderr)
        return False

    try:
        print(
            f"  [llm_corrector] loading {model_path.name} via llama-cpp-python ...",
            file=sys.stderr,
        )
        _LLM = Llama(
            model_path=str(model_path),
            n_ctx=LLM_CTX_WINDOW,
            n_threads=(LLM_NUM_THREADS or os.cpu_count() or 4),
            verbose=False,
            chat_format="qwen",
        )
        return True
    except Exception as e:
        _LLM_LOAD_ERROR = f"failed to initialise llama.cpp model: {e}"
        print(f"  [llm_corrector] DISABLED — {_LLM_LOAD_ERROR}", file=sys.stderr)
        return False


def _ensure_mlm():
    """Lazy-load xlm-roberta MLM handle (reuses error_detector's loader)."""
    global _MLM_HANDLE
    if _MLM_HANDLE is not None:
        return _MLM_HANDLE
    try:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        from pipeline.config import MLM_VETO_MODEL
        tokenizer = AutoTokenizer.from_pretrained(MLM_VETO_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(MLM_VETO_MODEL)
        model.eval()
        _MLM_HANDLE = (tokenizer, model)
        print(
            f"  [llm_corrector] loaded MLM veto model {MLM_VETO_MODEL}",
            file=sys.stderr,
        )
        return _MLM_HANDLE
    except Exception as e:
        print(
            f"  [llm_corrector] MLM veto unavailable ({e}); proceeding without veto",
            file=sys.stderr,
        )
        return None


# ─── Confidence gating ───────────────────────────────────────────────

_CAPITALIZED = re.compile(
    r"^[A-ZÅÄÖÆØ][a-zà-ÿåäöæøüß]+(?:['-][A-Za-zÅÄÖÆØà-ÿåäöæøüß]+)*[.,!?;:]?$"
)


def _wrap_low_confidence_tokens(
    segment: Segment,
    gazetteer: dict[str, str] | None = None,
    language: str = "en",
) -> tuple[str, int, list[bool]]:
    """Return (wrapped_text, n_wrapped, per_word_editable_mask).

    The mask aligns 1:1 with the words of ``segment.text`` (after
    whitespace split): True at indices of low-confidence tokens that the
    LLM is allowed to edit, False everywhere else. This mask is used to
    HARD-CONSTRAIN the LLM output later — any change at a False position
    is treated as drift and the entire correction is rejected.

    Two paths:
      * Schema-2 (``segment.words`` present): use per-word
        ``avg_logprob`` and gate by ``LLM_LOGPROB_GATE``.
      * Schema-1 (``segment.words`` is None): fall back to capitalised
        tokens, but POS-filter sentence-start common words (``And``,
        ``It's``, ``He``, ``The``, ...). This stops the LLM from being
        invited to "improve" segments where every sentence-start cap
        is grammar, not an entity. Mid-sentence capitalised tokens stay
        editable. Tokens whose stripped form is in the gazetteer also
        stay editable regardless of POS (so known canonical names like
        ``David Luiz`` still pass even if a small POS model labels them
        oddly).
    """
    seg_words = segment.text.split()
    # Frozen positions (set by entity_corrector) that Step L must NEVER edit.
    # Even if the word would normally be wrapped, we force mask[i]=False here
    # so the editable-drift guard rejects any LLM proposal that touches a
    # canonical name we just placed.
    frozen = set(segment.frozen_word_indices or [])
    if segment.words:
        wrapped: list[str] = []
        n_wrapped = 0
        mask: list[bool] = []
        # Align segment.words to whitespace tokens. faster-whisper's word
        # tokens often include leading whitespace and may not 1:1 with
        # the .split() tokens. We use the .split() tokens as the truth
        # and look up each one's average prob from the words list.
        word_iter = iter(segment.words)
        cur_word = next(word_iter, None)
        for i, tok in enumerate(seg_words):
            prob = None
            while cur_word is not None:
                cw = (cur_word.get("word") or "").strip(" .,!?;:")
                if cw and (tok.startswith(cw) or cw.startswith(tok)):
                    prob = cur_word.get("prob", 0.0)
                    cur_word = next(word_iter, None)
                    break
                cur_word = next(word_iter, None)
            if prob is None:
                logp = -10.0
            elif prob > 0:
                logp = math.log(prob)
            else:
                logp = -10.0
            if i in frozen:
                wrapped.append(tok)
                mask.append(False)
            elif logp < LLM_LOGPROB_GATE:
                wrapped.append(f"<{tok}>")
                mask.append(True)
                n_wrapped += 1
            else:
                wrapped.append(tok)
                mask.append(False)
        return " ".join(wrapped).strip(), n_wrapped, mask

    # ── Schema-1 fallback: POS-filtered capitalised-token wrapping ──
    pos_tags = _pos_tags_for(segment.text, language)
    rejected = get_rejected_pos_tags(language)
    gaz_lc = {k.lower() for k in (gazetteer or {})}
    wrapped: list[str] = []
    mask: list[bool] = []
    n_wrapped = 0
    for i, tok in enumerate(seg_words):
        if i in frozen:
            wrapped.append(tok)
            mask.append(False)
            continue
        if not _CAPITALIZED.match(tok):
            wrapped.append(tok)
            mask.append(False)
            continue
        stripped_lc = tok.strip(" .,!?;:'\"-—–…").lower()
        if stripped_lc in gaz_lc:
            wrapped.append(f"<{tok}>")
            mask.append(True)
            n_wrapped += 1
            continue
        is_sentence_start = (i == 0)
        if is_sentence_start:
            tag = pos_tags.get(i)
            if tag is not None and tag in rejected:
                wrapped.append(tok)
                mask.append(False)
                continue
        wrapped.append(f"<{tok}>")
        mask.append(True)
        n_wrapped += 1
    return " ".join(wrapped).strip(), n_wrapped, mask


def _pos_tags_for(text: str, language: str) -> dict[int, str]:
    """Return {whitespace_index → POS tag} for ``text``. Empty on failure.

    Lightweight wrapper around ``pipeline.ner_extractor.get_nlp(language)``
    that maps spaCy's character offsets back to whitespace-split indices.
    Returning an empty dict (not raising) keeps the LLM corrector usable
    when spaCy is unavailable — the wrapper degrades to wrapping every
    capitalised token, which was the previous behaviour.
    """
    try:
        from pipeline.ner_extractor import get_nlp
        nlp = get_nlp(language)
    except Exception:
        return {}
    try:
        doc = nlp(text)
    except Exception:
        return {}
    tags: dict[int, str] = {}
    seg_words = text.split()
    if not seg_words:
        return tags
    # Map by walking both sequences in lockstep on stripped surface form.
    spacy_iter = iter([t for t in doc if not t.is_space])
    cur = next(spacy_iter, None)
    for i, tok in enumerate(seg_words):
        clean = tok.strip(" .,!?;:'\"-—–…")
        while cur is not None:
            sp_clean = cur.text.strip(" .,!?;:'\"-—–…")
            if sp_clean and (clean.startswith(sp_clean) or sp_clean.startswith(clean)):
                tags[i] = cur.pos_
                cur = next(spacy_iter, None)
                break
            cur = next(spacy_iter, None)
    return tags


def _strip_punct(s: str) -> str:
    return s.strip(" .,!?;:'\"-—–…")


def _editable_drift(
    original: str, corrected: str, editable_mask: list[bool]
) -> bool:
    """True if the LLM changed words at positions that were NOT marked editable.

    This is the hard constraint that prevents Qwen-0.5B from "improving"
    the segment by inserting articles, rewording for fluency, etc. Any
    change outside the bracketed-token positions is treated as drift and
    triggers rejection of the whole correction.
    """
    o, n = original.split(), corrected.split()
    if len(o) != len(n):
        return True   # length mismatch — easiest signal of rewriting
    for i, (ow, nw) in enumerate(zip(o, n)):
        if ow == nw:
            continue
        # Tolerate punctuation-only differences at any position
        if _strip_punct(ow).lower() == _strip_punct(nw).lower():
            continue
        if i >= len(editable_mask) or not editable_mask[i]:
            return True  # change at non-editable position
    return False


# ─── Prompt construction ────────────────────────────────────────────

def _build_context_prompt(
    gazetteer: dict[str, str],
    entity_types: dict[str, str],
    prev_segments: list[Segment],
    next_segments: list[Segment],
    match_name: str,
    half: int,
) -> str:
    """Build the per-segment context string injected into the LLM prompt."""
    by_type: dict[str, list[str]] = {}
    for canon, etype in entity_types.items():
        by_type.setdefault(etype, []).append(canon)
    players = sorted(by_type.get("player", []))[:30]
    teams = sorted(by_type.get("team", []))[:8]
    coaches = sorted(by_type.get("coach", []))[:6]
    referees = sorted(by_type.get("referee", []))[:3]
    venues = sorted(by_type.get("venue", []))[:2]

    lines = [f"Match: {match_name} | Half: {half}"]
    if teams:
        lines.append("Teams: " + ", ".join(teams))
    if players:
        lines.append("Players: " + ", ".join(players))
    if coaches:
        lines.append("Coaches: " + ", ".join(coaches))
    if referees:
        lines.append("Referee: " + ", ".join(referees))
    if venues:
        lines.append("Venue: " + ", ".join(venues))
    for s in prev_segments[-LLM_CTX_PREVIOUS_SEGMENTS:]:
        lines.append(f'Prev: "{s.text.strip()}"')
    for s in next_segments[:LLM_CTX_NEXT_SEGMENTS]:
        lines.append(f'Next: "{s.text.strip()}"')
    return "\n".join(lines)


_SYSTEM_PROMPT_TEMPLATE = (
    "You are an editor for English football match commentary transcribed by Whisper. "
    "Your only job is to fix obvious misheard player or team names so they match the LINEUP.\n"
    "{context}\n"
    "Rules:\n"
    "- Output ONLY the corrected sentence on a single line.\n"
    "- Do NOT echo any field labels (Match, Half, Players, Teams, Input, Corrected, Output, Players, Teams, Coaches, Referee, Venue, Prev, Next).\n"
    "- Preserve word count and word order. Make the smallest possible edit.\n"
    "- Replace only words that are clearly misheard names; keep all other words verbatim.\n"
    "- If unsure, return the input sentence unchanged.\n\n"
    "Examples:\n"
    'Input: Sturridge passes to Hassard.\n'
    'Output: Sturridge passes to Hazard.\n\n'
    'Input: Free kick has given Chelsea\'s way.\n'
    'Output: Free kick has given Chelsea\'s way.\n\n'
    'Input: Hendrik to the keeper.\n'
    'Output: Henderson to the keeper.'
)


# Tokens that signal the LLM is echoing the prompt instead of correcting.
# If any whole-word match shows up in the LLM output, we treat the call
# as failed and keep the original segment.
_PROMPT_ECHO_MARKERS = {
    "Match:", "Half:", "Teams:", "Players:", "Coaches:", "Referee:",
    "Venue:", "Prev:", "Next:", "Input:", "Output:", "Corrected:",
    "LINEUP", "<|", "|>",
}


def _llm_correct(
    segment_text_wrapped: str,
    context_block: str,
) -> Optional[str]:
    """One LLM call. Returns corrected sentence or None on failure."""
    if not _ensure_llm():
        return None
    system_msg = _SYSTEM_PROMPT_TEMPLATE.format(context=context_block)
    user_msg = f"Input: {segment_text_wrapped}\nOutput:"
    try:
        resp = _LLM.create_chat_completion(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_NEW_TOKENS,
            stop=["\nInput:", "\nOutput:", "\n\n"],
        )
        choice = resp["choices"][0]["message"]["content"]
        return _clean_llm_output(choice)
    except Exception as e:
        print(f"  [llm_corrector] call failed: {e}", file=sys.stderr)
        return None


def _clean_llm_output(text: str) -> str:
    """Strip wrapper artefacts (quotes, leading 'Corrected:', etc.) from LLM output."""
    text = text.strip()
    # Drop any leading "Output:" / "Corrected:" / "Answer:" prefix.
    text = re.sub(r"^(corrected|output|answer)\s*[:\-]\s*", "", text, flags=re.IGNORECASE)
    if text.startswith(('"', "'")) and text.endswith(('"', "'")):
        text = text[1:-1]
    text = text.strip()
    # Take only the first line — the model sometimes continues with extra text.
    text = text.split("\n", 1)[0].strip()
    return text


def _is_prompt_echo(output: str) -> bool:
    """Detect prompt-echo failures (Qwen-0.5B leaks the system context)."""
    return any(marker in output for marker in _PROMPT_ECHO_MARKERS)


def _is_length_anomaly(original: str, output: str) -> bool:
    """Reject outputs that are wildly different in length from the input.

    GER should make the SMALLEST possible edit. An output that's <50% or
    >150% of the input length is almost always a hallucination or a
    rephrase that hurts WER more than it helps.
    """
    o, n = len(original.split()), len(output.split())
    if o == 0:
        return n > 0
    ratio = n / o
    return ratio < 0.5 or ratio > 1.5


# ─── MLM veto ───────────────────────────────────────────────────────

def _mlm_pseudo_logprob(
    sentence_tokens: list[str], idx: int, candidate: str,
) -> Optional[float]:
    """Compute log P(candidate | context) by masking ``sentence_tokens[idx]``."""
    handle = _ensure_mlm()
    if handle is None:
        return None
    tokenizer, model = handle
    masked = sentence_tokens[:idx] + [tokenizer.mask_token] + sentence_tokens[idx + 1 :]
    sentence = " ".join(masked)
    try:
        import torch
        enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits  # (1, T, V)
        # Find the mask position in tokenized output
        mask_id = tokenizer.mask_token_id
        mask_positions = (enc.input_ids[0] == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            return None
        pos = int(mask_positions[0])
        cand_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        if not cand_ids:
            return None
        log_probs = torch.log_softmax(logits[0, pos], dim=-1)
        return float(log_probs[cand_ids[0]].item())
    except Exception:
        return None


def _veto(original_text: str, corrected_text: str) -> str:
    """Apply MLM veto on a per-token-edit basis. Rejects edits where MLM
    prefers the original token by a factor of MLM_VETO_RATIO or more.
    Returns the (possibly rolled-back) corrected text.
    """
    if not MLM_VETO_ENABLED:
        return corrected_text
    orig_toks = original_text.split()
    new_toks = corrected_text.split()
    if len(orig_toks) != len(new_toks):
        return corrected_text  # length mismatch — let the change stand
    out_toks: list[str] = list(new_toks)
    for i, (o, n) in enumerate(zip(orig_toks, new_toks)):
        if o == n:
            continue
        lp_orig = _mlm_pseudo_logprob(out_toks, i, o)
        lp_new = _mlm_pseudo_logprob(out_toks, i, n)
        if lp_orig is None or lp_new is None:
            continue
        # Reject the LLM edit if the original is more plausible by the
        # configured margin (log-ratio ≥ log(MLM_VETO_RATIO)).
        if lp_orig - lp_new >= math.log(MLM_VETO_RATIO):
            out_toks[i] = o
    return " ".join(out_toks)


# ─── Public API ──────────────────────────────────────────────────────

@dataclass
class LlmCorrection:
    """One per-token edit recorded by the LLM corrector."""
    segment_id: str
    half: int
    original: str
    corrected: str
    method: str = "llm_ger"
    score: float = 100.0
    stage: str = "L"


_LAST_TELEMETRY: dict = {}


def get_last_telemetry() -> dict:
    """Return the telemetry dict from the most recent ``correct_match`` call.

    The orchestrator can persist this to ``cleaning_metadata.llm_telemetry``
    so per-rejection counts (drift, prompt echo, MLM veto, etc.) appear
    next to the corrections themselves and downstream eval can ratio
    accepted/proposed edits without re-running the LLM.
    """
    return dict(_LAST_TELEMETRY)


def _format_telemetry(t: dict) -> str:
    return (
        f"segs={t['total_segments']} "
        f"eligible={t['eligible_segments']} "
        f"wrapped_tokens={t['total_wrapped_tokens']} "
        f"skipped<min={t['skipped_below_min']} "
        f"call_failed={t['llm_call_failed']} "
        f"echo={t['rejected_prompt_echo']} "
        f"length={t['rejected_length_anomaly']} "
        f"drift={t['rejected_editable_drift']} "
        f"veto_rolled_back={t['mlm_rolled_back']} "
        f"no_op={t['no_op']} "
        f"accepted={t['accepted']}"
    )


def correct_match(
    segments: list[Segment],
    gazetteer: dict[str, str],
    entity_types: dict[str, str],
    match_name: str = "",
    language: str = "en",
) -> tuple[list[Segment], list[dict]]:
    """Run confidence-gated GER over a full match. Honours config flags.

    Side effect: populates ``_LAST_TELEMETRY`` with per-stage counts so
    the orchestrator can persist them. Reset on every call.
    """
    global _LAST_TELEMETRY
    telem = {
        "total_segments": len(segments),
        "eligible_segments": 0,
        "total_wrapped_tokens": 0,
        "skipped_below_min": 0,
        "llm_call_failed": 0,
        "rejected_prompt_echo": 0,
        "rejected_length_anomaly": 0,
        "rejected_editable_drift": 0,
        "mlm_rolled_back": 0,
        "no_op": 0,
        "accepted": 0,
        "examples": {
            "prompt_echo": [],
            "length_anomaly": [],
            "editable_drift": [],
        },
    }

    if not LLM_CORRECTION_ENABLED:
        telem["disabled_reason"] = "LLM_CORRECTION_ENABLED=False"
        _LAST_TELEMETRY = telem
        return segments, []
    if not _ensure_llm():
        telem["disabled_reason"] = _LLM_LOAD_ERROR or "llm not available"
        _LAST_TELEMETRY = telem
        return segments, []

    out: list[Segment] = []
    corrections: list[dict] = []
    # We do per-segment processing; build prev/next windows on the fly.
    for i, seg in enumerate(segments):
        wrapped_text, n_wrapped, editable_mask = _wrap_low_confidence_tokens(
            seg, gazetteer=gazetteer, language=language,
        )
        telem["total_wrapped_tokens"] += n_wrapped
        if n_wrapped < LLM_MIN_TOKENS_TO_INVOKE:
            telem["skipped_below_min"] += 1
            out.append(seg)
            continue
        telem["eligible_segments"] += 1
        prev = segments[max(0, i - LLM_CTX_PREVIOUS_SEGMENTS) : i]
        nxt = segments[i + 1 : i + 1 + LLM_CTX_NEXT_SEGMENTS]
        ctx = _build_context_prompt(
            gazetteer, entity_types, prev, nxt, match_name, seg.half,
        )
        proposal = _llm_correct(wrapped_text, ctx)
        if not proposal:
            telem["llm_call_failed"] += 1
            out.append(seg)
            continue
        proposal = re.sub(r"<([^<>]+)>", r"\1", proposal).strip()
        if not proposal or proposal == seg.text:
            telem["no_op"] += 1
            out.append(seg)
            continue
        if _is_prompt_echo(proposal):
            telem["rejected_prompt_echo"] += 1
            if len(telem["examples"]["prompt_echo"]) < 5:
                telem["examples"]["prompt_echo"].append(
                    {"original": seg.text, "proposal": proposal[:200]}
                )
            out.append(seg)
            continue
        if _is_length_anomaly(seg.text, proposal):
            telem["rejected_length_anomaly"] += 1
            if len(telem["examples"]["length_anomaly"]) < 5:
                telem["examples"]["length_anomaly"].append(
                    {"original": seg.text, "proposal": proposal[:200]}
                )
            out.append(seg)
            continue
        if _editable_drift(seg.text, proposal, editable_mask):
            telem["rejected_editable_drift"] += 1
            if len(telem["examples"]["editable_drift"]) < 5:
                telem["examples"]["editable_drift"].append(
                    {"original": seg.text, "proposal": proposal[:200]}
                )
            out.append(seg)
            continue
        # MLM veto pass — count when it rolled back AT LEAST one token
        final_text = _veto(seg.text, proposal)
        if final_text == seg.text:
            telem["no_op"] += 1
            out.append(seg)
            continue
        if final_text != proposal:
            telem["mlm_rolled_back"] += 1
        telem["accepted"] += 1
        corrections.append({
            "segment_id": seg.segment_id,
            "half": seg.half,
            "original": seg.text,
            "corrected": final_text,
            "score": 100.0,
            "method": "llm_ger",
            "stage": "L",
        })
        out.append(Segment(
            segment_id=seg.segment_id,
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=final_text,
            half=seg.half,
            global_id=seg.global_id,
            words=seg.words,
            avg_logprob=seg.avg_logprob,
            no_speech_prob=seg.no_speech_prob,
            nbest=seg.nbest,
            speaker_id=seg.speaker_id,
        ))
    print(f"  [llm_corrector] telemetry: {_format_telemetry(telem)}", file=sys.stderr)
    _LAST_TELEMETRY = telem
    return out, corrections
