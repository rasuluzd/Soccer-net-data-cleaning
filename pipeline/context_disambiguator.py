"""
Contextual Disambiguation using Sentence-Transformers.

When Tier 2 (fuzzy + phonetic) can't confidently match an entity,
this module uses semantic context to disambiguate. It encodes the
surrounding sentence with a sentence-transformer and compares against
candidate player descriptions using cosine similarity.

This handles the hardest ASR errors where string/phonetic similarity
fails, e.g. "Zurich" → "Djuricic" (completely different spelling
and phonetics, but context says "fantastic ball from Zurich" implies
a player, not a city).
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from pipeline.config import (
    CONTEXT_MODEL_NAME,
    CONTEXT_SIMILARITY_THRESHOLD,
    CONTEXT_WINDOW_SIZE,
)


# ─── Singleton model loading ────────────────────────────────────────
# The model is loaded once on first use and cached in memory.
# This avoids loading the ~80MB model on every call.
_model = None
_model_available = None  # None = not checked, True/False = result


def _check_model_available() -> bool:
    """Check if sentence-transformers is installed."""
    global _model_available
    if _model_available is not None:
        return _model_available
    try:
        from sentence_transformers import SentenceTransformer
        _model_available = True
    except ImportError:
        _model_available = False
        print("  ⚠ sentence-transformers not installed — Tier 3 contextual disambiguation disabled")
    return _model_available


def load_model():
    """Load the sentence-transformer model (singleton, loaded once)."""
    global _model
    if _model is not None:
        return _model

    if not _check_model_available():
        return None

    from sentence_transformers import SentenceTransformer
    print(f"  Loading context model: {CONTEXT_MODEL_NAME}...")
    _model = SentenceTransformer(CONTEXT_MODEL_NAME)
    print(f"  Context model loaded ({CONTEXT_MODEL_NAME})")
    return _model


# ─── Data structures ────────────────────────────────────────────────

@dataclass
class DisambiguationResult:
    """Result of a contextual disambiguation attempt."""
    entity_text: str       # Original entity text
    corrected: str         # Best candidate name
    similarity: float      # Cosine similarity score
    segment_id: str        # Which segment this came from


# ─── Core logic ─────────────────────────────────────────────────────

def build_candidate_descriptions(
    gazetteer: dict[str, str],
    labels: Optional[dict] = None,
    entity_types: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """
    Build descriptive strings for each canonical name in the gazetteer.

    These descriptions provide context for the sentence-transformer to
    match against. Format: "{name} {position} {team} football player"

    Args:
        gazetteer: variant → canonical mapping
        labels: Labels-caption.json dict (for team/position info)

    Returns:
        Dict mapping canonical name → description string
    """
    # Collect unique canonical names, filtering to only player/coach types
    # if entity_types is available. Teams and venues should never be
    # disambiguation candidates.
    all_canonicals = set(gazetteer.values())
    if entity_types:
        canonical_names = {
            name for name in all_canonicals
            if entity_types.get(name, "player") in ("player", "coach", "referee")
        }
    else:
        canonical_names = all_canonicals
    descriptions: dict[str, str] = {}

    # Build a player → team mapping from labels
    player_teams: dict[str, str] = {}
    if labels:
        home_team = labels.get("gameHomeTeam", "")
        away_team = labels.get("gameAwayTeam", "")

        for side, team_name in [("home", home_team), ("away", away_team)]:
            lineup = labels.get("lineup", {}).get(side, {})
            for player in lineup.get("players", []):
                long_name = player.get("long_name", "")
                if long_name:
                    player_teams[long_name] = team_name

            for coach in lineup.get("coach", []):
                long_name = coach.get("long_name", "")
                if long_name:
                    player_teams[long_name] = f"{team_name} manager"

    for name in canonical_names:
        team_info = player_teams.get(name, "")
        if team_info:
            descriptions[name] = f"{name} {team_info} football"
        else:
            descriptions[name] = f"{name} football player"

    return descriptions


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def disambiguate_entity(
    entity_text: str,
    context: str,
    candidate_descriptions: dict[str, str],
    candidate_embeddings: dict[str, np.ndarray],
    context_embedding: np.ndarray,
    entity_embedding: np.ndarray,
) -> Optional[DisambiguationResult]:
    """
    Use semantic similarity to pick the best candidate for an entity.

    Uses a MARGIN-BASED approach to reject ambiguous matches:
    - The best candidate must beat the 2nd-best by at least MIN_GAP
    - This eliminates false positives where ALL candidates score similarly
      (e.g., "Palace" context → all players score 0.28–0.43)
    - Genuine matches like "Ivanjevic"→Ivanovic have clear gaps (0.09+)

    Also combines entity-level and context-level similarity:
    - Context similarity: does the sentence context match this player?
    - Entity similarity: does the entity text itself resemble a player name?
    - Combined = 0.6 * context + 0.4 * entity (entity matters for name matching)

    Args:
        entity_text: the unresolved entity
        context: the full segment text for context
        candidate_descriptions: name → description mapping
        candidate_embeddings: name → pre-computed embedding
        context_embedding: pre-computed embedding for the context
        entity_embedding: pre-computed embedding for the entity text alone

    Returns:
        DisambiguationResult if a confident match is found, else None
    """
    MIN_GAP = 0.05  # Minimum gap between 1st and 2nd best

    # Compute combined similarity (context + entity) for each candidate
    scored: list[tuple[float, str]] = []
    for name, embedding in candidate_embeddings.items():
        ctx_sim = _cosine_similarity(context_embedding, embedding)
        ent_sim = _cosine_similarity(entity_embedding, embedding)
        # Weighted combination: context provides setting, entity provides name signal
        combined = 0.6 * ctx_sim + 0.4 * ent_sim
        scored.append((combined, name))

    # Sort descending by score
    scored.sort(reverse=True)

    if len(scored) < 2:
        return None

    best_sim, best_name = scored[0]
    second_sim, _ = scored[1]
    gap = best_sim - second_sim

    # Require BOTH: minimum threshold AND clear margin over 2nd place
    if best_sim >= CONTEXT_SIMILARITY_THRESHOLD and gap >= MIN_GAP:
        return DisambiguationResult(
            entity_text=entity_text,
            corrected=best_name,
            similarity=best_sim,
            segment_id="",
        )

    return None


def batch_disambiguate(
    unresolved_entities: list[dict],
    all_segments: list,
    gazetteer: dict[str, str],
    labels: Optional[dict] = None,
    entity_types: Optional[dict[str, str]] = None,
) -> list[DisambiguationResult]:
    """
    Disambiguate a batch of unresolved entities using sentence-transformers.

    This is the main entry point called by the orchestrator after Tier 2.

    Args:
        unresolved_entities: list of dicts with keys:
            - "text": entity text
            - "segment_id": segment identifier
            - "segment_idx": index into all_segments
        all_segments: list of all segments (for building context windows)
        gazetteer: the current gazetteer
        labels: Labels-caption.json dict

    Returns:
        List of DisambiguationResult for successfully disambiguated entities
    """
    model = load_model()
    if model is None:
        return []

    if not unresolved_entities:
        return []

    # Step 1: Build candidate descriptions and pre-encode them
    descriptions = build_candidate_descriptions(
        gazetteer, labels, entity_types=entity_types,
    )

    if not descriptions:
        return []

    # Batch-encode all candidate descriptions at once (efficiency)
    candidate_names = list(descriptions.keys())
    candidate_texts = list(descriptions.values())
    candidate_emb_matrix = model.encode(candidate_texts, show_progress_bar=False)
    candidate_embeddings = {
        name: candidate_emb_matrix[i]
        for i, name in enumerate(candidate_names)
    }

    # Step 2: Build context strings and entity texts for each unresolved entity
    context_texts = []
    entity_texts = []
    for ue in unresolved_entities:
        seg_idx = ue["segment_idx"]

        # Build context window: current segment + neighbors
        window_start = max(0, seg_idx - CONTEXT_WINDOW_SIZE)
        window_end = min(len(all_segments), seg_idx + CONTEXT_WINDOW_SIZE + 1)

        context_parts = []
        for i in range(window_start, window_end):
            context_parts.append(all_segments[i].text)

        context_texts.append(" ".join(context_parts))
        entity_texts.append(ue["text"])

    # Batch-encode all context strings AND entity texts (efficiency)
    context_embeddings = model.encode(context_texts, show_progress_bar=False)
    entity_embeddings = model.encode(entity_texts, show_progress_bar=False)

    # Step 3: Disambiguate each entity
    results = []
    for i, ue in enumerate(unresolved_entities):
        result = disambiguate_entity(
            entity_text=ue["text"],
            context=context_texts[i],
            candidate_descriptions=descriptions,
            candidate_embeddings=candidate_embeddings,
            context_embedding=context_embeddings[i],
            entity_embedding=entity_embeddings[i],
        )
        if result:
            result.segment_id = ue["segment_id"]

            # Apply single-word logic: if entity is 1 word, use surname
            entity_words = ue["text"].split()
            if len(entity_words) == 1 and " " in result.corrected:
                result.corrected = result.corrected.split()[-1]

            results.append(result)

    return results
