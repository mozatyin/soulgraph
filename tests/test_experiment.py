from unittest.mock import MagicMock, patch

from soulgraph.experiment.models import Message, ExperimentResult
from soulgraph.experiment.runner import ExperimentRunner
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge
from soulgraph.comparator.models import GraphSimilarity, HubRecall


class TestExperimentModels:
    def test_message(self):
        msg = Message(role="speaker", content="我最近在考虑买车")
        assert msg.role == "speaker"

    def test_experiment_result(self):
        result = ExperimentResult(
            conversation=[
                Message(role="detector", content="你最近在想什么？"),
                Message(role="speaker", content="想买车"),
            ],
            ground_truth=SoulGraph(owner_id="gt"),
            detected_graph=SoulGraph(owner_id="det"),
            similarity=GraphSimilarity(
                hub_recall=HubRecall(ground_truth_hubs=[], detected_hubs=[], recall=0.0),
                local_similarities=[],
            ),
            turns=1,
        )
        assert result.turns == 1
        assert len(result.conversation) == 2


class TestExperimentRunner:
    def _make_gt_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="gt")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family"], mention_count=3))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase"]))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        return g

    @patch("soulgraph.experiment.runner.GraphComparator")
    @patch("soulgraph.experiment.runner.SemanticMatcher")
    @patch("soulgraph.experiment.runner.EmbeddingMatcher")
    @patch("soulgraph.experiment.runner.Speaker")
    @patch("soulgraph.experiment.runner.Detector")
    def test_run_completes(self, MockDetector, MockSpeaker, MockEmbMatcher, MockSemMatcher, MockGraphComp):
        mock_speaker = MockSpeaker.return_value
        mock_speaker.respond.return_value = "我最近在想买车"

        mock_detector = MockDetector.return_value
        mock_detector.ask_next_question.return_value = "你最近在想什么？"
        mock_detector.listen_and_detect.return_value = SoulGraph(owner_id="det")
        mock_detector.detected_graph = SoulGraph(owner_id="det")

        # Mock embedding matcher
        MockEmbMatcher.return_value.compute_similarity.return_value = {
            "node_recall": 0.5, "node_precision": 0.5, "hub_recall": 0.5,
            "triple_recall": 0.0, "triple_precision": 0.0, "triple_f1": 0.0,
            "overall": 0.25, "matched_nodes": 1, "gt_nodes": 2, "det_nodes": 0,
            "gt_edges": 1, "det_edges": 0,
        }

        # Mock legacy comparator
        MockGraphComp.return_value.compare.return_value = GraphSimilarity(
            hub_recall=HubRecall(ground_truth_hubs=[], detected_hubs=[], recall=0.0),
            local_similarities=[],
        )

        runner = ExperimentRunner(api_key="fake")
        result = runner.run(self._make_gt_graph(), max_turns=3, verbose=False)

        assert result.turns == 3
        assert len(result.conversation) == 6  # 3 turns x 2 messages
        assert mock_speaker.respond.call_count == 3
        assert mock_detector.listen_and_detect.call_count == 3
