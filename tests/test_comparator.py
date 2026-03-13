from unittest.mock import MagicMock

from soulgraph.comparator.models import HubRecall, LocalStructureSimilarity, GraphSimilarity
from soulgraph.comparator.semantic import SemanticMatcher
from soulgraph.graph.models import SoulItem


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
        matcher.is_match = lambda a, b: a == b
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
