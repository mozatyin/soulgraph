"""DualSoul — Deep Soul + Surface Soul architecture.

Inspired by Kahneman's Thinking Fast and Slow:
- Surface Soul (Think Fast): live extraction, captures current state
- Deep Soul (Think Slow): compressed long-term personality, periodic consolidation
"""
from __future__ import annotations

import json
import math
import time

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge

_EMB_MODEL: SentenceTransformer | None = None

def _get_emb_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB_MODEL


class DualSoul:
    """Two-graph architecture: Surface (live) + Deep (compressed)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        deep_cycle: int = 100,
        max_surface_nodes: int = 200,
        carry_forward_k: int = 10,
    ):
        self._api_key = api_key
        self._model = model
        self.deep_cycle = deep_cycle
        self.max_surface_nodes = max_surface_nodes
        self.carry_forward_k = carry_forward_k

        self._detector = Detector(api_key=api_key, model=model)
        self._messages: list[Message] = []
        self.total_utterances: int = 0
        self._consolidation_count: int = 0
        self._deep = SoulGraph(owner_id="unknown")

        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)

    @property
    def surface(self) -> SoulGraph:
        return self._detector.detected_graph

    @property
    def deep(self) -> SoulGraph:
        return self._deep

    @property
    def stats(self) -> dict:
        return {
            "total_utterances": self.total_utterances,
            "consolidation_count": self._consolidation_count,
            "surface_items": len(self.surface.items),
            "surface_edges": len(self.surface.edges),
            "deep_items": len(self._deep.items),
            "deep_edges": len(self._deep.edges),
        }

    def ingest(self, text: str) -> None:
        """Add utterance, extract to Surface. Auto-consolidates when needed."""
        self._messages.append(Message(role="speaker", content=text))
        self.total_utterances += 1
        self._detector.listen_and_detect(self._messages)

        # Auto-consolidate triggers
        if (self.total_utterances % self.deep_cycle == 0
                or len(self.surface.items) > self.max_surface_nodes):
            self.consolidate()

    def _adaptive_merge_threshold(self) -> float:
        """Merge threshold decreases as Deep grows. Smooth log curve."""
        n = max(len(self._deep.items), 1)
        return max(0.40, 0.82 - 0.06 * math.log(n))

    def consolidate(self) -> dict:
        """Stub — will be implemented in Task 4."""
        return {"merged": 0, "added": 0, "decayed": 0}
