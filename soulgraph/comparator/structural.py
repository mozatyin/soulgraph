"""Structural graph comparison: hub recall + local structure similarity."""
from __future__ import annotations

from soulgraph.comparator.models import GraphSimilarity, HubRecall, LocalStructureSimilarity
from soulgraph.comparator.semantic import SemanticMatcher
from soulgraph.graph.models import SoulGraph


class GraphComparator:
    def __init__(self, matcher: SemanticMatcher):
        self._matcher = matcher

    def compare(
        self, ground_truth: SoulGraph, detected: SoulGraph, hub_top_k: int = 5
    ) -> GraphSimilarity:
        gt_hubs = ground_truth.get_hubs(top_k=hub_top_k)
        if not gt_hubs:
            return GraphSimilarity(
                hub_recall=HubRecall(
                    ground_truth_hubs=[], detected_hubs=[], recall=0.0
                ),
                local_similarities=[],
            )

        item_mapping = self._matcher.match_items(ground_truth.items, detected.items)

        gt_hub_ids = [h.id for h in gt_hubs]
        detected_hub_ids = [
            item_mapping[h_id] for h_id in gt_hub_ids if h_id in item_mapping
        ]

        hub_recall = HubRecall(
            ground_truth_hubs=gt_hub_ids,
            detected_hubs=detected_hub_ids,
            recall=len(detected_hub_ids) / len(gt_hub_ids) if gt_hub_ids else 0.0,
        )

        local_sims: list[LocalStructureSimilarity] = []
        for gt_hub_id in gt_hub_ids:
            if gt_hub_id not in item_mapping:
                continue
            det_hub_id = item_mapping[gt_hub_id]

            gt_neighbors = self._get_neighbors(ground_truth, gt_hub_id)
            det_neighbors = self._get_neighbors(detected, det_hub_id)

            matched_neighbors = sum(
                1
                for n_id in gt_neighbors
                if n_id in item_mapping and item_mapping[n_id] in det_neighbors
            )
            neighbor_recall = (
                matched_neighbors / len(gt_neighbors) if gt_neighbors else 1.0
            )

            gt_edge_types = self._get_edge_types(ground_truth, gt_hub_id)
            det_edge_types = self._get_edge_types(detected, det_hub_id)

            type_matches = 0
            type_total = 0
            for gt_n_id, gt_rel in gt_edge_types.items():
                if gt_n_id in item_mapping:
                    det_n_id = item_mapping[gt_n_id]
                    if det_n_id in det_edge_types:
                        type_total += 1
                        if det_edge_types[det_n_id] == gt_rel:
                            type_matches += 1

            edge_accuracy = type_matches / type_total if type_total else 0.0

            combined = neighbor_recall * 0.6 + edge_accuracy * 0.4
            local_sims.append(
                LocalStructureSimilarity(
                    hub_id=gt_hub_id,
                    neighbor_recall=neighbor_recall,
                    edge_type_accuracy=edge_accuracy,
                    combined_score=combined,
                )
            )

        return GraphSimilarity(hub_recall=hub_recall, local_similarities=local_sims)

    @staticmethod
    def _get_neighbors(graph: SoulGraph, node_id: str) -> set[str]:
        neighbors: set[str] = set()
        for edge in graph.edges:
            if edge.from_id == node_id:
                neighbors.add(edge.to_id)
            elif edge.to_id == node_id:
                neighbors.add(edge.from_id)
        return neighbors

    @staticmethod
    def _get_edge_types(graph: SoulGraph, node_id: str) -> dict[str, str]:
        types: dict[str, str] = {}
        for edge in graph.edges:
            if edge.from_id == node_id:
                types[edge.to_id] = edge.relation
            elif edge.to_id == node_id:
                types[edge.from_id] = edge.relation
        return types
