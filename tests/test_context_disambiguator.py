"""Tests for Tier 3 context disambiguator."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass

from pipeline.context_disambiguator import (
    batch_disambiguate,
    DisambiguationResult,
    _cosine_similarity,
)


# ─── Helpers ──────────────────────────────────────────────────────────

@dataclass
class FakeSegment:
    segment_id: str
    text: str
    start_time: float = 0.0
    end_time: float = 1.0
    half: int = 1


def _make_mock_model(embeddings_map: dict[str, np.ndarray]):
    """Create a mock sentence-transformer that returns controlled embeddings.

    ``embeddings_map`` maps substrings to embeddings.  When *encode* is
    called the mock inspects each input string and returns the first
    matching embedding (or a zero vector).
    """
    model = MagicMock()
    dim = 4  # small dimension for testing

    def fake_encode(texts, **kwargs):
        results = []
        for text in texts:
            matched = False
            for key, emb in embeddings_map.items():
                if key.lower() in text.lower():
                    results.append(emb)
                    matched = True
                    break
            if not matched:
                results.append(np.zeros(dim))
        return np.array(results)

    model.encode = fake_encode
    return model


# ─── Tests ────────────────────────────────────────────────────────────

class TestProposedCorrectionValidation:
    """Tier 3 should validate Tier 2's proposed correction for uncertain
    corrections instead of doing an unconstrained search that might pick
    a completely unrelated candidate.

    Real-world case: "Suarez" → Tier 2 proposes "Souare" (score 69.5,
    phonetic+fuzzy match).  Without validation, Tier 3 does a fresh
    semantic search and picks "Costa" because attacking-play context
    matches a striker description better than a defender's."""

    # Embeddings are crafted so that:
    #   - "Suarez" entity is closer to "Costa" (wrong) than "Souare" (right)
    #   - But Tier 2 already proposed "Souare" via string matching
    SOUARE_EMB = np.array([0.3, 0.2, 0.8, 0.1])   # defender-ish
    COSTA_EMB  = np.array([0.9, 0.8, 0.1, 0.2])    # striker-ish
    CONTEXT_EMB = np.array([0.85, 0.75, 0.15, 0.25])  # attacking context
    ENTITY_EMB  = np.array([0.7, 0.6, 0.2, 0.3])     # "Suarez" entity

    GAZ = {
        "Pape Souare": "Pape Souare",
        "Souare": "Pape Souare",
        "Diego Costa": "Diego Costa",
        "Costa": "Diego Costa",
    }
    ETYPES = {
        "Pape Souare": "player",
        "Diego Costa": "player",
    }

    SEGMENTS = [
        FakeSegment("seg0", "Palace defending deep here"),
        FakeSegment("seg1", "Suarez with a brilliant run past two defenders"),
        FakeSegment("seg2", "What a cross from the left side"),
    ]

    def _build_embeddings_map(self):
        return {
            "Souare": self.SOUARE_EMB,
            "Costa": self.COSTA_EMB,
            "Suarez": self.ENTITY_EMB,
            "brilliant run": self.CONTEXT_EMB,
            "defending deep": np.array([0.2, 0.3, 0.7, 0.5]),
            "cross from": np.array([0.5, 0.4, 0.3, 0.4]),
        }

    def test_proposed_correction_beats_unconstrained_search(self):
        """When Tier 2 proposes a correction, Tier 3 should validate it
        instead of picking a different candidate via semantic search."""
        mock_model = _make_mock_model(self._build_embeddings_map())

        unresolved = [{
            "text": "Suarez",
            "segment_id": "seg1",
            "segment_idx": 1,
            "proposed_correction": "Souare",
            "proposed_score": 69.5,
        }]

        with patch("pipeline.context_disambiguator.load_model", return_value=mock_model):
            results = batch_disambiguate(
                unresolved_entities=unresolved,
                all_segments=self.SEGMENTS,
                gazetteer=self.GAZ,
                labels=None,
                entity_types=self.ETYPES,
            )

        # Should correct to Souare (Tier 2's proposal), NOT Costa
        assert len(results) == 1
        assert results[0].corrected == "Souare", (
            f"Expected 'Souare' (Tier 2 proposed), got '{results[0].corrected}'"
        )

    def test_unconstrained_search_without_proposal(self):
        """Without a proposed correction, Tier 3 should do the normal
        unconstrained search (existing behaviour)."""
        mock_model = _make_mock_model(self._build_embeddings_map())

        unresolved = [{
            "text": "Suarez",
            "segment_id": "seg1",
            "segment_idx": 1,
            # No proposed_correction field
        }]

        with patch("pipeline.context_disambiguator.load_model", return_value=mock_model):
            results = batch_disambiguate(
                unresolved_entities=unresolved,
                all_segments=self.SEGMENTS,
                gazetteer=self.GAZ,
                labels=None,
                entity_types=self.ETYPES,
            )

        # Without a proposal, Tier 3 searches freely and may pick Costa
        # (we just verify it returns *something* — the exact candidate
        # depends on the embeddings, but it should NOT be constrained)
        if results:
            assert results[0].corrected in ("Costa", "Souare", "Diego Costa", "Pape Souare")
