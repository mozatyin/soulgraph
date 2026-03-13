import tempfile
from datetime import datetime, timezone
from pathlib import Path

from soulgraph.graph.models import SoulItem, SoulEdge, SoulGraph, _clamp


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
