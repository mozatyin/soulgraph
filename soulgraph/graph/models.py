from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SoulItem(BaseModel):
    """A single node in the soul graph."""

    id: str
    text: str
    domains: list[str]
    confidence: float = 0.5
    specificity: float = 0.5
    source_turn: int = 0
    source_session: str = ""
    created_at: datetime | None = None
    last_referenced: datetime | None = None
    mention_count: int = 0

    @field_validator("confidence", "specificity", mode="before")
    @classmethod
    def clamp_floats(cls, v: float) -> float:
        return _clamp(v)

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        now = _utcnow()
        if self.created_at is None:
            self.created_at = now
        if self.last_referenced is None:
            self.last_referenced = self.created_at
        return self


class SoulEdge(BaseModel):
    """A directed edge between two soul items."""

    from_id: str
    to_id: str
    relation: str
    strength: float = 0.5
    confidence: float = 0.5
    created_at: datetime | None = None

    @field_validator("strength", "confidence", mode="before")
    @classmethod
    def clamp_floats(cls, v: float) -> float:
        return _clamp(v)

    @model_validator(mode="after")
    def set_defaults(self) -> Self:
        if self.created_at is None:
            self.created_at = _utcnow()
        return self


class SoulGraph(BaseModel):
    """Append-only graph of soul items. No delete operations."""

    owner_id: str
    items: list[SoulItem] = []
    edges: list[SoulEdge] = []

    def add_item(self, item: SoulItem) -> None:
        if any(existing.id == item.id for existing in self.items):
            return
        self.items.append(item)

    def add_edge(self, edge: SoulEdge) -> None:
        self.edges.append(edge)

    def strengthen(self, item_id: str, delta: float) -> None:
        for item in self.items:
            if item.id == item_id:
                item.confidence = _clamp(item.confidence + delta)
                item.mention_count += 1
                item.last_referenced = _utcnow()
                return

    def strengthen_edge(self, from_id: str, to_id: str, delta: float) -> None:
        for edge in self.edges:
            if edge.from_id == from_id and edge.to_id == to_id:
                edge.strength = _clamp(edge.strength + delta)
                return

    def get_hubs(self, top_k: int = 5) -> list[SoulItem]:
        edge_count: dict[str, int] = {}
        for edge in self.edges:
            edge_count[edge.to_id] = edge_count.get(edge.to_id, 0) + 1
            edge_count[edge.from_id] = edge_count.get(edge.from_id, 0) + 1
        scored = [
            (item, item.mention_count + edge_count.get(item.id, 0))
            for item in self.items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:top_k]]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> SoulGraph:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)
