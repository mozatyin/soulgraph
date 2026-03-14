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
    @patch("soulgraph.experiment.runner.RankingComparator")
    @patch("soulgraph.experiment.runner.EmbeddingMatcher")
    @patch("soulgraph.experiment.runner.Speaker")
    @patch("soulgraph.experiment.runner.Detector")
    def test_run_completes(self, MockDetector, MockSpeaker, MockEmbMatcher, MockRankComp, MockSemMatcher, MockGraphComp):
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

        # Mock ranking comparator
        MockRankComp.return_value.compare.return_value = {
            "rank_correlation": 0.5, "domain_ndcg": 0.5,
            "absorption_rate": 0.5, "intention_recall": 0.5,
            "overall": 0.5, "matched_items": 1, "gt_items": 2, "det_items": 0,
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


class TestMultiSessionResult:
    def test_multi_session_result(self):
        from soulgraph.experiment.models import MultiSessionResult
        result = MultiSessionResult(
            session_scores=[
                {"rank_correlation": 0.4, "absorption_rate": 0.3},
                {"rank_correlation": 0.5, "absorption_rate": 0.6},
                {"rank_correlation": 0.7, "absorption_rate": 0.9},
            ],
            rank_improvement=0.3,
            final_scores={"rank_correlation": 0.7, "absorption_rate": 0.9, "overall": 0.8},
            num_sessions=3,
            turns_per_session=10,
        )
        assert result.num_sessions == 3
        assert result.rank_improvement == 0.3
        assert len(result.session_scores) == 3


class TestMultiSessionRunner:
    def _make_gt_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="gt")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family"], tags=["intention"]))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase"]))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        return g

    @patch("soulgraph.experiment.runner.RankingComparator")
    @patch("soulgraph.experiment.runner.Speaker")
    @patch("soulgraph.experiment.runner.Detector")
    def test_run_multi_session(self, MockDetector, MockSpeaker, MockRankComp):
        mock_speaker = MockSpeaker.return_value
        mock_speaker.respond.return_value = "我最近在想买车"

        mock_detector = MockDetector.return_value
        mock_detector.ask_next_question.return_value = "你在想什么？"
        mock_detector.listen_and_detect.return_value = SoulGraph(owner_id="det")
        mock_detector.detected_graph = SoulGraph(owner_id="det")
        mock_detector.session_number = 0

        MockRankComp.return_value.compare.return_value = {
            "rank_correlation": 0.5, "domain_ndcg": 0.5,
            "absorption_rate": 0.5, "intention_recall": 0.5,
            "overall": 0.5, "matched_items": 1, "gt_items": 2, "det_items": 0,
        }

        runner = ExperimentRunner(api_key="fake")
        session_configs = [
            {"turns": 3, "topic_hints": ["family"]},
            {"turns": 3, "topic_hints": ["career"]},
        ]
        result = runner.run_multi_session(
            self._make_gt_graph(), session_configs=session_configs, verbose=False
        )
        assert result["num_sessions"] == 2
        assert len(result["session_scores"]) == 2
        assert "rank_improvement" in result


class TestQueryEvalPhase:
    def _make_gt_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="gt")
        g.add_item(SoulItem(id="si_001", text="想创业", domains=["career"], tags=["intention"]))
        g.add_item(SoulItem(id="si_002", text="房贷压力", domains=["finance"]))
        g.add_edge(SoulEdge(from_id="si_002", to_id="si_001", relation="constrains"))
        return g

    @patch("soulgraph.experiment.runner.RetrievalEvaluator")
    @patch("soulgraph.experiment.runner.RankingComparator")
    @patch("soulgraph.experiment.runner.Speaker")
    @patch("soulgraph.experiment.runner.Detector")
    def test_multi_session_with_queries(self, MockDetector, MockSpeaker, MockRankComp, MockRetEval):
        mock_speaker = MockSpeaker.return_value
        mock_speaker.respond.return_value = "我想创业"

        mock_detector = MockDetector.return_value
        mock_detector.ask_next_question.return_value = "你在想什么？"
        mock_detector.listen_and_detect.return_value = SoulGraph(owner_id="det")
        mock_detector.detected_graph = SoulGraph(owner_id="det")
        mock_detector.detected_graph.add_item(SoulItem(id="d1", text="创业", domains=["career"]))
        mock_detector.session_number = 0

        MockRankComp.return_value.compare.return_value = {
            "rank_correlation": 0.5, "domain_ndcg": 0.5,
            "absorption_rate": 0.5, "intention_recall": 0.5,
            "overall": 0.5, "matched_items": 1, "gt_items": 2, "det_items": 1,
        }

        MockRetEval.return_value.evaluate.return_value = {
            "node_count": 1, "edge_count": 0, "cross_domain_coverage": 1,
            "is_connected": True, "density": 0.0,
            "faithfulness": 0.8, "comprehensiveness": 0.7, "diversity": 0.5,
            "structural_score": 0.5, "retrieval_score": 0.65,
        }

        runner = ExperimentRunner(api_key="fake")
        queries = [{"query": "创业", "expected_domains": ["career"]}]
        result = runner.run_multi_session(
            self._make_gt_graph(),
            session_configs=[{"turns": 2, "topic_hints": ["career"]}],
            queries=queries,
            verbose=False,
        )
        assert "query_scores" in result
        assert len(result["query_scores"]) == 1
        assert "mean_retrieval_score" in result


class TestDetectorSession:
    def test_detector_session_number(self):
        from soulgraph.experiment.detector import Detector
        det = Detector(api_key="fake", session_number=2)
        assert det.session_number == 2

    def test_detector_default_session(self):
        from soulgraph.experiment.detector import Detector
        det = Detector(api_key="fake")
        assert det.session_number == 0
