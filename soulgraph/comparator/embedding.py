"""Embedding-based semantic matching — replaces LLM-as-judge for deterministic comparison."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer

from soulgraph.graph.models import SoulEdge, SoulGraph, SoulItem


class EmbeddingMatcher:
    """Deterministic semantic matcher using sentence embeddings + Hungarian algorithm."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", node_threshold: float = 0.55):
        self._model = SentenceTransformer(model_name)
        self._node_threshold = node_threshold

    def match_items(
        self, ground_truth: list[SoulItem], detected: list[SoulItem]
    ) -> dict[str, str]:
        """Optimal node matching using embeddings + Hungarian algorithm. Returns {gt_id: det_id}."""
        if not ground_truth or not detected:
            return {}

        gt_texts = [item.text for item in ground_truth]
        det_texts = [item.text for item in detected]

        gt_embs = self._model.encode(gt_texts, normalize_embeddings=True)
        det_embs = self._model.encode(det_texts, normalize_embeddings=True)

        # Cosine similarity matrix (already normalized, so dot product = cosine)
        sim_matrix = gt_embs @ det_embs.T  # shape: (n_gt, n_det)

        # Convert to cost matrix for Hungarian algorithm (minimize cost = maximize similarity)
        cost_matrix = 1.0 - sim_matrix

        # Optimal assignment
        gt_indices, det_indices = linear_sum_assignment(cost_matrix)

        # Build mapping, filtering by threshold
        mapping: dict[str, str] = {}
        for gi, di in zip(gt_indices, det_indices):
            if sim_matrix[gi, di] >= self._node_threshold:
                mapping[ground_truth[gi].id] = detected[di].id

        return mapping

    def compute_similarity(
        self, ground_truth: SoulGraph, detected: SoulGraph, hub_top_k: int = 5
    ) -> dict:
        """Full graph comparison: node recall, triple soft matching, overall score."""
        # 1. Node matching
        item_mapping = self.match_items(ground_truth.items, detected.items)
        node_recall = len(item_mapping) / len(ground_truth.items) if ground_truth.items else 0.0

        # Precision: how many detected items have a GT match
        reverse_mapping = {v: k for k, v in item_mapping.items()}
        node_precision = len(reverse_mapping) / len(detected.items) if detected.items else 0.0

        # 2. Hub recall
        gt_hubs = ground_truth.get_hubs(top_k=hub_top_k)
        gt_hub_ids = [h.id for h in gt_hubs]
        matched_hubs = [hid for hid in gt_hub_ids if hid in item_mapping]
        hub_recall = len(matched_hubs) / len(gt_hub_ids) if gt_hub_ids else 0.0

        # 3. Triple-level soft matching
        gt_triples = self._build_triples(ground_truth)
        det_triples = self._build_triples(detected)
        triple_recall, triple_precision, triple_f1 = self._soft_triple_match(
            gt_triples, det_triples, item_mapping
        )

        # 4. Overall score (weighted)
        # Hub recall 0.3 + node recall 0.2 + triple F1 0.5
        overall = hub_recall * 0.3 + node_recall * 0.2 + triple_f1 * 0.5

        return {
            "node_recall": round(node_recall, 3),
            "node_precision": round(node_precision, 3),
            "hub_recall": round(hub_recall, 3),
            "triple_recall": round(triple_recall, 3),
            "triple_precision": round(triple_precision, 3),
            "triple_f1": round(triple_f1, 3),
            "overall": round(overall, 3),
            "matched_nodes": len(item_mapping),
            "gt_nodes": len(ground_truth.items),
            "det_nodes": len(detected.items),
            "gt_edges": len(ground_truth.edges),
            "det_edges": len(detected.edges),
        }

    def _build_triples(self, graph: SoulGraph) -> list[str]:
        """Linearize edges as natural language triples."""
        item_map = {item.id: item.text for item in graph.items}
        triples = []
        for edge in graph.edges:
            subj = item_map.get(edge.from_id, edge.from_id)
            obj = item_map.get(edge.to_id, edge.to_id)
            triples.append(f"{subj} {edge.relation} {obj}")
        return triples

    def _soft_triple_match(
        self,
        gt_triples: list[str],
        det_triples: list[str],
        item_mapping: dict[str, str],
        threshold: float = 0.50,
    ) -> tuple[float, float, float]:
        """Soft matching between linearized triples using embeddings."""
        if not gt_triples or not det_triples:
            return (0.0, 0.0, 0.0)

        gt_embs = self._model.encode(gt_triples, normalize_embeddings=True)
        det_embs = self._model.encode(det_triples, normalize_embeddings=True)

        sim_matrix = gt_embs @ det_embs.T

        # Greedy matching: for each GT triple, find best detected match
        used_det: set[int] = set()
        matches = 0
        for gi in range(len(gt_triples)):
            best_di = -1
            best_sim = threshold
            for di in range(len(det_triples)):
                if di not in used_det and sim_matrix[gi, di] > best_sim:
                    best_sim = sim_matrix[gi, di]
                    best_di = di
            if best_di >= 0:
                matches += 1
                used_det.add(best_di)

        recall = matches / len(gt_triples) if gt_triples else 0.0
        precision = matches / len(det_triples) if det_triples else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return (recall, precision, f1)
