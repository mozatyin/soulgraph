import tempfile
from datetime import datetime, timezone
from pathlib import Path

from soulgraph.graph.models import SoulItem, SoulEdge, SoulGraph, _clamp
from soulgraph.graph.filter import filter_by_domain, filter_by_time


class TestSoulItem:
    def test_create_soul_item(self):
        item = SoulItem(
            id="si_001",
            text="重视家庭",
            domains=["family", "values"],
            confidence=0.8,
            specificity=0.3,
            source_turn=1,
            source_session="session_001",
            created_at=datetime(2026, 3, 13, tzinfo=timezone.utc),
            last_referenced=datetime(2026, 3, 13, tzinfo=timezone.utc),
            mention_count=1,
        )
        assert item.id == "si_001"
        assert item.text == "重视家庭"
        assert "family" in item.domains
        assert item.mention_count == 1

    def test_soul_item_defaults(self):
        now = datetime.now(timezone.utc)
        item = SoulItem(id="si_002", text="喜欢速度", domains=["lifestyle"])
        assert item.confidence == 0.5
        assert item.specificity == 0.5
        assert item.source_turn == 0
        assert item.source_session == ""
        assert item.mention_count == 0
        assert item.created_at >= now

    def test_confidence_clamped(self):
        item = SoulItem(id="si_003", text="test", domains=["x"], confidence=1.5)
        assert item.confidence == 1.0
        item2 = SoulItem(id="si_004", text="test", domains=["x"], confidence=-0.3)
        assert item2.confidence == 0.0


class TestSoulEdge:
    def test_create_soul_edge(self):
        edge = SoulEdge(
            from_id="si_001",
            to_id="si_002",
            relation="drives",
            strength=0.7,
            confidence=0.8,
        )
        assert edge.from_id == "si_001"
        assert edge.relation == "drives"

    def test_edge_defaults(self):
        edge = SoulEdge(from_id="si_001", to_id="si_002", relation="causes")
        assert edge.strength == 0.5
        assert edge.confidence == 0.5

    def test_edge_strength_clamped(self):
        edge = SoulEdge(
            from_id="si_001", to_id="si_002", relation="causes", strength=2.0
        )
        assert edge.strength == 1.0


class TestSoulGraph:
    def _make_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="user_001")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family", "values"]))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase", "family"]))
        g.add_item(SoulItem(id="si_003", text="预算有限", domains=["finance"]))
        g.add_edge(
            SoulEdge(from_id="si_001", to_id="si_002", relation="drives", strength=0.8)
        )
        g.add_edge(
            SoulEdge(
                from_id="si_003", to_id="si_002", relation="constrains", strength=0.6
            )
        )
        return g

    def test_add_item(self):
        g = SoulGraph(owner_id="user_001")
        g.add_item(SoulItem(id="si_001", text="test", domains=["x"]))
        assert len(g.items) == 1

    def test_add_duplicate_item_ignored(self):
        g = SoulGraph(owner_id="user_001")
        g.add_item(SoulItem(id="si_001", text="test", domains=["x"]))
        g.add_item(SoulItem(id="si_001", text="different", domains=["y"]))
        assert len(g.items) == 1

    def test_add_edge(self):
        g = self._make_graph()
        assert len(g.edges) == 2

    def test_no_delete_method(self):
        g = SoulGraph(owner_id="user_001")
        assert not hasattr(g, "delete_item")
        assert not hasattr(g, "remove_item")
        assert not hasattr(g, "delete_edge")
        assert not hasattr(g, "remove_edge")

    def test_strengthen_item(self):
        g = self._make_graph()
        old_conf = g.items[0].confidence
        g.strengthen("si_001", 0.2)
        assert g.items[0].confidence == _clamp(old_conf + 0.2)
        assert g.items[0].mention_count == 1

    def test_strengthen_edge(self):
        g = self._make_graph()
        old_strength = g.edges[0].strength
        g.strengthen_edge("si_001", "si_002", 0.1)
        assert g.edges[0].strength == _clamp(old_strength + 0.1)

    def test_get_hubs(self):
        g = self._make_graph()
        hubs = g.get_hubs(top_k=2)
        assert hubs[0].id == "si_002"  # 2 incoming edges

    def test_save_and_load_json(self):
        g = self._make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "soul.json"
            g.save(path)
            loaded = SoulGraph.load(path)
            assert loaded.owner_id == "user_001"
            assert len(loaded.items) == 3
            assert len(loaded.edges) == 2


class TestFilters:
    def _make_multi_domain_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="user_001")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family", "values"]))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase", "family"]))
        g.add_item(SoulItem(id="si_003", text="预算有限", domains=["finance"]))
        g.add_item(SoulItem(id="si_004", text="喜欢跑步", domains=["health"]))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        g.add_edge(SoulEdge(from_id="si_003", to_id="si_002", relation="constrains"))
        g.add_edge(SoulEdge(from_id="si_004", to_id="si_001", relation="enables"))
        return g

    def test_filter_by_domain_nodes(self):
        g = self._make_multi_domain_graph()
        sub = filter_by_domain(g, "family")
        ids = {item.id for item in sub.items}
        assert ids == {"si_001", "si_002"}

    def test_filter_by_domain_edges(self):
        g = self._make_multi_domain_graph()
        sub = filter_by_domain(g, "family")
        assert len(sub.edges) == 1
        assert sub.edges[0].from_id == "si_001"
        assert sub.edges[0].to_id == "si_002"

    def test_filter_by_domain_empty(self):
        g = self._make_multi_domain_graph()
        sub = filter_by_domain(g, "career")
        assert len(sub.items) == 0
        assert len(sub.edges) == 0

    def test_filter_by_time(self):
        g = SoulGraph(owner_id="user_001")
        g.add_item(SoulItem(id="si_001", text="old", domains=["x"],
                            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)))
        g.add_item(SoulItem(id="si_002", text="new", domains=["x"],
                            created_at=datetime(2026, 3, 1, tzinfo=timezone.utc)))
        sub = filter_by_time(g, start=datetime(2026, 2, 1, tzinfo=timezone.utc),
                             end=datetime(2026, 4, 1, tzinfo=timezone.utc))
        assert len(sub.items) == 1
        assert sub.items[0].id == "si_002"


class TestSoulItemTags:
    def test_tags_default_empty(self):
        item = SoulItem(id="si_001", text="test", domains=["x"])
        assert item.tags == []

    def test_tags_assigned(self):
        item = SoulItem(id="si_001", text="想创业", domains=["career"],
                        tags=["intention", "goal"])
        assert "intention" in item.tags
        assert "goal" in item.tags

    def test_tags_survive_json_roundtrip(self):
        import tempfile
        from pathlib import Path
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="test", domains=["x"],
                            tags=["fact", "preference"]))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.json"
            g.save(path)
            loaded = SoulGraph.load(path)
            assert loaded.items[0].tags == ["fact", "preference"]


class TestPageRank:
    def _make_ranked_graph(self) -> SoulGraph:
        """Graph where si_002 is a clear hub (many incoming edges)."""
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family", "values"]))
        g.add_item(SoulItem(id="si_002", text="想创业", domains=["career"],
                            tags=["intention"]))
        g.add_item(SoulItem(id="si_003", text="预算有限", domains=["finance"]))
        g.add_item(SoulItem(id="si_004", text="妻子支持", domains=["family"]))
        g.add_item(SoulItem(id="si_005", text="骑自行车", domains=["transport", "hobby"]))
        # si_002 has 3 incoming edges — should rank highest
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives", strength=0.8))
        g.add_edge(SoulEdge(from_id="si_003", to_id="si_002", relation="constrains", strength=0.7))
        g.add_edge(SoulEdge(from_id="si_004", to_id="si_002", relation="enables", strength=0.6))
        # si_005 has 1 edge — should rank lowest
        g.add_edge(SoulEdge(from_id="si_005", to_id="si_001", relation="enables", strength=0.3))
        return g

    def test_pagerank_returns_all_items(self):
        g = self._make_ranked_graph()
        ranks = g.pagerank()
        assert len(ranks) == 5
        assert all(isinstance(v, float) for v in ranks.values())

    def test_pagerank_hub_ranks_highest(self):
        g = self._make_ranked_graph()
        ranks = g.pagerank()
        sorted_ids = sorted(ranks, key=ranks.get, reverse=True)
        assert sorted_ids[0] == "si_002"

    def test_pagerank_leaf_ranks_lowest(self):
        g = self._make_ranked_graph()
        ranks = g.pagerank()
        sorted_ids = sorted(ranks, key=ranks.get, reverse=True)
        assert sorted_ids[-1] == "si_005"

    def test_domain_pagerank_shifts_importance(self):
        g = self._make_ranked_graph()
        career_ranks = g.domain_pagerank("career")
        transport_ranks = g.domain_pagerank("transport")
        career_rank_005 = career_ranks.get("si_005", 0)
        transport_rank_005 = transport_ranks.get("si_005", 0)
        assert transport_rank_005 > career_rank_005

    def test_pagerank_empty_graph(self):
        g = SoulGraph(owner_id="test")
        ranks = g.pagerank()
        assert ranks == {}

    def test_pagerank_no_edges(self):
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="a", domains=["x"]))
        g.add_item(SoulItem(id="si_002", text="b", domains=["x"]))
        ranks = g.pagerank()
        assert len(ranks) == 2
        assert abs(ranks["si_001"] - ranks["si_002"]) < 0.01


class TestFixtures:
    def test_load_car_buyer(self):
        path = Path(__file__).parent.parent / "fixtures" / "car_buyer.json"
        g = SoulGraph.load(path)
        assert g.owner_id == "car_buyer_001"
        assert len(g.items) == 12
        assert len(g.edges) == 13
        hubs = g.get_hubs(top_k=3)
        hub_ids = {h.id for h in hubs}
        assert "si_002" in hub_ids or "si_001" in hub_ids

    def test_load_zhang_wei(self):
        path = Path(__file__).parent.parent / "fixtures" / "zhang_wei.json"
        g = SoulGraph.load(path)
        assert g.owner_id == "zhang_wei"
        assert len(g.items) >= 50
        assert len(g.edges) >= 70
        # Verify PageRank works on this graph
        ranks = g.pagerank()
        assert len(ranks) == len(g.items)
        # Top-ranked item should be an intention
        top_id = max(ranks, key=ranks.get)
        top_item = next(i for i in g.items if i.id == top_id)
        assert "intention" in top_item.tags
        # Verify domain PageRank works
        career_ranks = g.domain_pagerank("career")
        assert len(career_ranks) == len(g.items)
