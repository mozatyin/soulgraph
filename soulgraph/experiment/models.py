"""Experiment data models."""
from __future__ import annotations

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
