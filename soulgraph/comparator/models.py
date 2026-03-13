"""Comparator data models for graph similarity measurement."""
from __future__ import annotations
from pydantic import BaseModel, computed_field


class HubRecall(BaseModel):
    ground_truth_hubs: list[str]
    detected_hubs: list[str]
    recall: float
    semantic_threshold: float = 0.8


class LocalStructureSimilarity(BaseModel):
    hub_id: str
    neighbor_recall: float
    edge_type_accuracy: float
    combined_score: float


class GraphSimilarity(BaseModel):
    hub_recall: HubRecall
    local_similarities: list[LocalStructureSimilarity]

    @computed_field
    @property
    def overall_score(self) -> float:
        if not self.local_similarities:
            return self.hub_recall.recall * 0.4
        avg_local = sum(ls.combined_score for ls in self.local_similarities) / len(self.local_similarities)
        return self.hub_recall.recall * 0.4 + avg_local * 0.6
