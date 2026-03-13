from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from typing import Self

from pydantic import BaseModel, field_validator, model_validator


class ItemType(str, Enum):
    """Type of soul item — determines how it appears in the star map."""
    COGNITIVE = "cognitive"    # Self-understanding intention (认知型)
    ACTION = "action"          # Real-world action intention (行动型)
    BACKGROUND = "background"  # Background fact, value, or context


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SoulItem(BaseModel):
    """A single node in the soul graph."""

    id: str
    text: str
    domains: list[str]
    item_type: ItemType = ItemType.BACKGROUND
    confidence: float = 0.5
    specificity: float = 0.5
    source_turn: int = 0
    source_session: str = ""
    created_at: datetime | None = None
    last_referenced: datetime | None = None
    mention_count: int = 0
    tags: list[str] = []

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

    def merge_items(self, keep_id: str, remove_id: str) -> None:
        """Merge remove_id into keep_id: rewire edges and drop the duplicate."""
        keep = next((i for i in self.items if i.id == keep_id), None)
        remove = next((i for i in self.items if i.id == remove_id), None)
        if not keep or not remove:
            return
        # Boost kept item
        keep.confidence = _clamp(max(keep.confidence, remove.confidence))
        keep.mention_count += remove.mention_count
        # Merge domains
        for d in remove.domains:
            if d not in keep.domains:
                keep.domains.append(d)
        # Rewire edges
        for edge in self.edges:
            if edge.from_id == remove_id:
                edge.from_id = keep_id
            if edge.to_id == remove_id:
                edge.to_id = keep_id
        # Remove self-loops created by merge
        self.edges = [e for e in self.edges if e.from_id != e.to_id]
        # Remove duplicate edges (same from/to/relation)
        seen: set[tuple[str, str, str]] = set()
        deduped: list[SoulEdge] = []
        for e in self.edges:
            key = (e.from_id, e.to_id, e.relation)
            if key not in seen:
                seen.add(key)
                deduped.append(e)
        self.edges = deduped
        # Remove the merged item
        self.items = [i for i in self.items if i.id != remove_id]

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
