from unittest.mock import patch, MagicMock

from soulgraph.comparator.retrieval import RetrievalEvaluator
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge


class TestRetrievalStructural:
    def _make_subgraph(self) -> SoulGraph:
        g = SoulGraph(owner_id="sub")
        g.add_item(SoulItem(id="s1", text="创业想法", domains=["career", "identity"]))
        g.add_item(SoulItem(id="s2", text="房贷压力", domains=["finance"]))
        g.add_item(SoulItem(id="s3", text="害怕失败", domains=["career", "family"]))
        g.add_edge(SoulEdge(from_id="s2", to_id="s3", relation="drives", strength=0.8))
        g.add_edge(SoulEdge(from_id="s3", to_id="s1", relation="conflicts_with", strength=0.9))
        return g

    def test_cross_domain_coverage(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        # career, identity, finance, family = 4
        assert metrics["cross_domain_coverage"] >= 3

    def test_connectivity(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert metrics["is_connected"] is True

    def test_disconnected_graph(self):
        evaluator = RetrievalEvaluator()
        g = SoulGraph(owner_id="disc")
        g.add_item(SoulItem(id="a", text="节点A", domains=["x"]))
        g.add_item(SoulItem(id="b", text="节点B", domains=["y"]))
        metrics = evaluator.structural_metrics(g)
        assert metrics["is_connected"] is False

    def test_density(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert 0.0 <= metrics["density"] <= 1.0

    def test_node_edge_counts(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert metrics["node_count"] == 3
        assert metrics["edge_count"] == 2

    def test_empty_subgraph(self):
        evaluator = RetrievalEvaluator()
        g = SoulGraph(owner_id="empty")
        metrics = evaluator.structural_metrics(g)
        assert metrics["node_count"] == 0
        assert metrics["cross_domain_coverage"] == 0


class TestRetrievalLLMJudge:
    def _make_subgraph(self) -> SoulGraph:
        g = SoulGraph(owner_id="sub")
        g.add_item(SoulItem(id="s1", text="想创业做AI工具", domains=["career"]))
        g.add_item(SoulItem(id="s2", text="房贷压力大", domains=["finance"]))
        g.add_edge(SoulEdge(from_id="s2", to_id="s1", relation="constrains", strength=0.8))
        return g

    @patch("soulgraph.comparator.retrieval.anthropic")
    def test_evaluate_returns_all_metrics(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"score": 0.8, "reasoning": "good"}')]
        )

        evaluator = RetrievalEvaluator(api_key="fake")
        sub = self._make_subgraph()
        result = evaluator.evaluate(
            full_graph=sub,
            subgraph=sub,
            query="创业",
            conversation_transcript="我想创业做AI工具，但是房贷压力大",
        )
        assert "faithfulness" in result
        assert "comprehensiveness" in result
        assert "diversity" in result
        assert "retrieval_score" in result
        assert 0.0 <= result["retrieval_score"] <= 1.0

    @patch("soulgraph.comparator.retrieval.anthropic")
    def test_evaluate_includes_structural(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"score": 0.7, "reasoning": "ok"}')]
        )

        evaluator = RetrievalEvaluator(api_key="fake")
        sub = self._make_subgraph()
        result = evaluator.evaluate(
            full_graph=sub,
            subgraph=sub,
            query="创业",
            conversation_transcript="我想创业",
        )
        assert "node_count" in result
        assert "cross_domain_coverage" in result
        assert "is_connected" in result

    def test_no_api_key_returns_fallback(self):
        evaluator = RetrievalEvaluator(api_key="")
        sub = self._make_subgraph()
        result = evaluator.evaluate(
            full_graph=sub,
            subgraph=sub,
            query="创业",
            conversation_transcript="test",
        )
        # Without API key, LLM scores fall back to 0.5
        assert result["faithfulness"] == 0.5
        assert result["comprehensiveness"] == 0.5
        assert result["diversity"] == 0.5
