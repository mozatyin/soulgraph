from unittest.mock import MagicMock

from soulgraph.comparator.models import HubRecall, LocalStructureSimilarity, GraphSimilarity
from soulgraph.comparator.semantic import SemanticMatcher
from soulgraph.comparator.structural import GraphComparator
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge


class TestComparatorModels:
    def test_hub_recall(self):
        hr = HubRecall(ground_truth_hubs=["si_001", "si_002", "si_003"],
                       detected_hubs=["si_001", "si_003"], recall=2/3)
        assert hr.recall == 2/3

    def test_local_structure_similarity(self):
        lss = LocalStructureSimilarity(hub_id="si_001", neighbor_recall=0.8,
                                        edge_type_accuracy=0.6, combined_score=0.7)
        assert lss.combined_score == 0.7

    def test_graph_similarity_score(self):
        hr = HubRecall(ground_truth_hubs=["a", "b"], detected_hubs=["a"], recall=0.5)
        lss = LocalStructureSimilarity(hub_id="a", neighbor_recall=0.8,
                                        edge_type_accuracy=0.6, combined_score=0.7)
        gs = GraphSimilarity(hub_recall=hr, local_similarities=[lss])
        # 0.5 * 0.4 + 0.7 * 0.6 = 0.62
        assert abs(gs.overall_score - 0.62) < 0.01

    def test_graph_similarity_no_local(self):
        hr = HubRecall(ground_truth_hubs=["a"], detected_hubs=[], recall=0.0)
        gs = GraphSimilarity(hub_recall=hr, local_similarities=[])
        assert gs.overall_score == 0.0


class TestSemanticMatcher:
    def test_match_identical_text(self):
        matcher = SemanticMatcher(api_key="fake")
        assert matcher.is_match("重视家庭", "重视家庭") is True

    def test_match_uses_llm(self):
        matcher = SemanticMatcher(api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"is_match": true, "similarity": 0.9}')]
        matcher._client = MagicMock()
        matcher._client.messages.create.return_value = mock_response
        result = matcher.is_match("重视家庭", "家庭很重要")
        assert result is True
        matcher._client.messages.create.assert_called_once()

    def test_match_items_returns_mapping(self):
        matcher = SemanticMatcher(api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"matches": [{"gt_id": "gt_1", "det_id": "det_1", "similarity": 0.9}]}')]
        matcher._client = MagicMock()
        matcher._client.messages.create.return_value = mock_response

        gt_items = [
            SoulItem(id="gt_1", text="loves family", domains=["family"]),
            SoulItem(id="gt_2", text="wants SUV", domains=["purchase"]),
        ]
        det_items = [
            SoulItem(id="det_1", text="loves family", domains=["family"]),
            SoulItem(id="det_3", text="likes sports", domains=["health"]),
        ]
        mapping = matcher.match_items(gt_items, det_items)
        assert mapping == {"gt_1": "det_1"}

    def test_batch_match_items(self):
        matcher = SemanticMatcher(api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"matches": [{"gt_id": "gt_1", "det_id": "det_1", "similarity": 0.9}, {"gt_id": "gt_2", "det_id": "det_2", "similarity": 0.7}]}')]
        matcher._client = MagicMock()
        matcher._client.messages.create.return_value = mock_response

        gt = [SoulItem(id="gt_1", text="loves family", domains=["family"]),
              SoulItem(id="gt_2", text="wants SUV", domains=["purchase"])]
        det = [SoulItem(id="det_1", text="family important", domains=["family"]),
               SoulItem(id="det_2", text="buying SUV", domains=["purchase"])]

        mapping = matcher.match_items(gt, det)
        assert mapping == {"gt_1": "det_1", "gt_2": "det_2"}


class TestGraphComparator:
    def _make_gt_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="gt")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family"], mention_count=5))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase"], mention_count=3))
        g.add_item(SoulItem(id="si_003", text="预算有限", domains=["finance"], mention_count=1))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        g.add_edge(SoulEdge(from_id="si_003", to_id="si_002", relation="constrains"))
        return g

    def _make_detected_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="detected")
        g.add_item(SoulItem(id="d_001", text="重视家庭", domains=["family"]))
        g.add_item(SoulItem(id="d_002", text="想买SUV", domains=["purchase"]))
        g.add_edge(SoulEdge(from_id="d_001", to_id="d_002", relation="drives"))
        return g

    def test_compare_returns_graph_similarity(self):
        gt = self._make_gt_graph()
        det = self._make_detected_graph()
        matcher = MagicMock()
        matcher.match_items.return_value = {"si_001": "d_001", "si_002": "d_002"}
        comparator = GraphComparator(matcher=matcher)
        result = comparator.compare(gt, det, hub_top_k=2)
        assert result.hub_recall.recall > 0
        assert len(result.local_similarities) > 0
        assert 0.0 <= result.overall_score <= 1.0

    def test_compare_perfect_match(self):
        gt = self._make_gt_graph()
        matcher = MagicMock()
        matcher.match_items.return_value = {"si_001": "si_001", "si_002": "si_002", "si_003": "si_003"}
        comparator = GraphComparator(matcher=matcher)
        result = comparator.compare(gt, gt, hub_top_k=2)
        assert result.hub_recall.recall == 1.0
        assert result.overall_score > 0.8

    def test_compare_no_match(self):
        gt = self._make_gt_graph()
        det = SoulGraph(owner_id="empty")
        matcher = MagicMock()
        matcher.match_items.return_value = {}
        comparator = GraphComparator(matcher=matcher)
        result = comparator.compare(gt, det, hub_top_k=2)
        assert result.hub_recall.recall == 0.0
        assert result.overall_score == 0.0
