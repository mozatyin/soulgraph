"""Ranking-based graph comparison — V4 evaluation metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from soulgraph.comparator.embedding import EmbeddingMatcher
from soulgraph.graph.models import SoulGraph


class RankingComparator:
    """Compare graphs by PageRank correlation and domain NDCG."""

    def __init__(self, node_threshold: float = 0.55):
        self._matcher = EmbeddingMatcher(node_threshold=node_threshold)

    def compare(self, ground_truth: SoulGraph, detected: SoulGraph) -> dict:
        """Full V4 comparison: rank correlation, domain NDCG, absorption, intention recall."""
        if not ground_truth.items or not detected.items:
            return {
                "rank_correlation": 0.0,
                "domain_ndcg": 0.0,
                "absorption_rate": 0.0,
                "intention_recall": 0.0,
                "overall": 0.0,
            }

        # 1. Node matching (reuse embedding matcher)
        item_mapping = self._matcher.match_items(ground_truth.items, detected.items)

        # 2. Absorption rate
        absorption_rate = len(item_mapping) / len(ground_truth.items)

        # 3. Intention recall
        gt_intentions = [i for i in ground_truth.items if "intention" in i.tags]
        if gt_intentions:
            matched_intentions = sum(1 for i in gt_intentions if i.id in item_mapping)
            intention_recall = matched_intentions / len(gt_intentions)
        else:
            intention_recall = absorption_rate  # fallback if no tags

        # 4. Rank correlation (Spearman)
        rank_correlation = self._compute_rank_correlation(
            ground_truth, detected, item_mapping
        )

        # 5. Domain NDCG
        domain_ndcg = self._compute_domain_ndcg(
            ground_truth, detected, item_mapping
        )

        # 6. Overall
        overall = (
            rank_correlation * 0.3
            + domain_ndcg * 0.3
            + absorption_rate * 0.2
            + intention_recall * 0.2
        )
        overall = max(0.0, min(1.0, overall))

        return {
            "rank_correlation": round(rank_correlation, 3),
            "domain_ndcg": round(domain_ndcg, 3),
            "absorption_rate": round(absorption_rate, 3),
            "intention_recall": round(intention_recall, 3),
            "overall": round(overall, 3),
            "matched_items": len(item_mapping),
            "gt_items": len(ground_truth.items),
            "det_items": len(detected.items),
        }

    def _compute_rank_correlation(
        self,
        gt: SoulGraph,
        det: SoulGraph,
        mapping: dict[str, str],
    ) -> float:
        """Spearman rank correlation between GT and detected PageRank."""
        if len(mapping) < 3:
            return 0.0

        gt_ranks = gt.pagerank()
        det_ranks = det.pagerank()

        gt_scores = []
        det_scores = []
        for gt_id, det_id in mapping.items():
            if gt_id in gt_ranks and det_id in det_ranks:
                gt_scores.append(gt_ranks[gt_id])
                det_scores.append(det_ranks[det_id])

        if len(gt_scores) < 3:
            return 0.0

        corr, _ = spearmanr(gt_scores, det_scores)
        if np.isnan(corr):
            return 0.0
        # Normalize to [0, 1] range
        return (corr + 1.0) / 2.0

    def _compute_domain_ndcg(
        self,
        gt: SoulGraph,
        det: SoulGraph,
        mapping: dict[str, str],
        top_k: int = 5,
    ) -> float:
        """Average NDCG@K across all domains."""
        if len(mapping) < 2:
            return 0.0

        all_domains: set[str] = set()
        for item in gt.items:
            all_domains.update(item.domains)

        if not all_domains:
            return 0.0

        ndcg_scores = []
        for domain in all_domains:
            score = self._domain_ndcg_single(gt, det, mapping, domain, top_k)
            if score is not None:
                ndcg_scores.append(score)

        if not ndcg_scores:
            return 0.0
        return float(np.mean(ndcg_scores))

    def _domain_ndcg_single(
        self,
        gt: SoulGraph,
        det: SoulGraph,
        mapping: dict[str, str],
        domain: str,
        top_k: int,
    ) -> float | None:
        """NDCG@K for a single domain."""
        gt_domain_ranks = gt.domain_pagerank(domain)
        det_domain_ranks = det.domain_pagerank(domain)

        gt_sorted = sorted(gt_domain_ranks.items(), key=lambda x: x[1], reverse=True)[:top_k]
        if not gt_sorted:
            return None

        true_relevance = []
        predicted_scores = []
        for gt_id, gt_score in gt_sorted:
            true_relevance.append(gt_score)
            det_id = mapping.get(gt_id)
            if det_id and det_id in det_domain_ranks:
                predicted_scores.append(det_domain_ranks[det_id])
            else:
                predicted_scores.append(0.0)

        true_rel = np.array([true_relevance])
        pred_scores = np.array([predicted_scores])

        if true_rel.sum() == 0:
            return None

        try:
            return float(ndcg_score(true_rel, pred_scores, k=top_k))
        except ValueError:
            return None
