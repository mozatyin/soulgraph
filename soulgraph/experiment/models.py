"""Experiment data models."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from soulgraph.comparator.models import GraphSimilarity
from soulgraph.graph.models import SoulGraph


class Message(BaseModel):
    role: str  # "speaker" or "detector"
    content: str


class ExperimentResult(BaseModel):
    conversation: list[Message]
    ground_truth: SoulGraph
    detected_graph: SoulGraph
    similarity: GraphSimilarity
    turns: int
    embedding_scores: dict[str, Any] | None = None
    ranking_scores: dict[str, Any] | None = None


class MultiSessionResult(BaseModel):
    session_scores: list[dict[str, Any]]
    rank_improvement: float
    final_scores: dict[str, Any]
    num_sessions: int
    turns_per_session: int
