from soulgraph.comparator.ranking import RankingComparator
from soulgraph.graph.models import SoulEdge, SoulGraph, SoulItem


def _make_gt() -> SoulGraph:
    """Ground truth graph: 6 items, clear hub structure."""
    g = SoulGraph(owner_id="gt")
    g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family", "values"]))
    g.add_item(SoulItem(id="si_002", text="想创业做AI工具", domains=["career"],
                        tags=["intention"]))
    g.add_item(SoulItem(id="si_003", text="预算有限", domains=["finance"]))
    g.add_item(SoulItem(id="si_004", text="妻子支持创业", domains=["family"]))
    g.add_item(SoulItem(id="si_005", text="害怕失败", domains=["emotion"],
                        tags=["intention"]))
    g.add_item(SoulItem(id="si_006", text="骑自行车", domains=["transport", "hobby"]))
    g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
    g.add_edge(SoulEdge(from_id="si_003", to_id="si_002", relation="constrains"))
    g.add_edge(SoulEdge(from_id="si_004", to_id="si_002", relation="enables"))
    g.add_edge(SoulEdge(from_id="si_005", to_id="si_002", relation="conflicts_with"))
    g.add_edge(SoulEdge(from_id="si_006", to_id="si_001", relation="enables"))
    return g


def _make_detected_good() -> SoulGraph:
    """Detected graph that closely matches GT structure."""
    g = SoulGraph(owner_id="det")
    g.add_item(SoulItem(id="d_001", text="很看重家庭", domains=["family"]))
    g.add_item(SoulItem(id="d_002", text="想做AI创业", domains=["career"],
                        tags=["intention"]))
    g.add_item(SoulItem(id="d_003", text="经济压力大", domains=["finance"]))
    g.add_item(SoulItem(id="d_004", text="妻子很支持", domains=["family"]))
    g.add_item(SoulItem(id="d_005", text="担心创业失败", domains=["emotion"],
                        tags=["intention"]))
    g.add_item(SoulItem(id="d_006", text="喜欢骑车", domains=["transport"]))
    g.add_item(SoulItem(id="d_007", text="IT行业收入稳定", domains=["career"]))  # extra
    g.add_edge(SoulEdge(from_id="d_001", to_id="d_002", relation="drives"))
    g.add_edge(SoulEdge(from_id="d_003", to_id="d_002", relation="constrains"))
    g.add_edge(SoulEdge(from_id="d_004", to_id="d_002", relation="enables"))
    g.add_edge(SoulEdge(from_id="d_005", to_id="d_002", relation="conflicts_with"))
    g.add_edge(SoulEdge(from_id="d_006", to_id="d_001", relation="enables"))
    g.add_edge(SoulEdge(from_id="d_007", to_id="d_002", relation="enables"))
    return g


class TestRankingComparator:
    def test_compare_returns_all_metrics(self):
        comp = RankingComparator()
        scores = comp.compare(_make_gt(), _make_detected_good())
        assert "rank_correlation" in scores
        assert "domain_ndcg" in scores
        assert "absorption_rate" in scores
        assert "intention_recall" in scores
        assert "overall" in scores

    def test_good_detection_has_positive_rank_correlation(self):
        comp = RankingComparator()
        scores = comp.compare(_make_gt(), _make_detected_good())
        assert scores["rank_correlation"] > 0.3

    def test_absorption_rate(self):
        comp = RankingComparator()
        scores = comp.compare(_make_gt(), _make_detected_good())
        assert scores["absorption_rate"] >= 0.8

    def test_intention_recall(self):
        comp = RankingComparator()
        scores = comp.compare(_make_gt(), _make_detected_good())
        assert scores["intention_recall"] >= 0.5

    def test_overall_bounded(self):
        comp = RankingComparator()
        scores = comp.compare(_make_gt(), _make_detected_good())
        assert 0.0 <= scores["overall"] <= 1.0

    def test_empty_detected_graph(self):
        comp = RankingComparator()
        scores = comp.compare(_make_gt(), SoulGraph(owner_id="empty"))
        assert scores["absorption_rate"] == 0.0
        assert scores["overall"] == 0.0
