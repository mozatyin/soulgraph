from __future__ import annotations

from datetime import datetime, timezone
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
