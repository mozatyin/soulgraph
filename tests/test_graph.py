from datetime import datetime, timezone

from soulgraph.graph.models import SoulItem, SoulEdge, _clamp


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
