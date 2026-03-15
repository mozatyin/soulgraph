"""Tests for DualSoul — Dual Soul architecture."""
import json
import math
from unittest.mock import MagicMock, patch

from soulgraph.dual_soul import DualSoul
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge


class TestDualSoulInit:
    @patch("soulgraph.dual_soul.Detector")
    def test_creates_two_graphs(self, MockDetector):
        MockDetector.return_value.detected_graph = SoulGraph(owner_id="unknown")
        ds = DualSoul(api_key="fake")
        assert isinstance(ds.surface, SoulGraph)
        assert isinstance(ds.deep, SoulGraph)

    @patch("soulgraph.dual_soul.Detector")
    def test_default_config(self, MockDetector):
        ds = DualSoul(api_key="fake")
        assert ds.deep_cycle == 100
        assert ds.max_surface_nodes == 200
        assert ds.carry_forward_k == 10
        assert ds.total_utterances == 0

    @patch("soulgraph.dual_soul.Detector")
    def test_custom_config(self, MockDetector):
        ds = DualSoul(api_key="fake", deep_cycle=50, max_surface_nodes=100)
        assert ds.deep_cycle == 50
        assert ds.max_surface_nodes == 100


class TestDualSoulIngest:
    @patch("soulgraph.dual_soul.Detector")
    def test_ingest_calls_detector(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="unknown")
        mock_det.listen_and_detect.return_value = mock_det.detected_graph

        ds = DualSoul(api_key="fake")
        ds.ingest("I love hiking")

        assert mock_det.listen_and_detect.call_count == 1
        assert ds.total_utterances == 1

    @patch("soulgraph.dual_soul.Detector")
    def test_ingest_increments_counter(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="unknown")
        mock_det.listen_and_detect.return_value = mock_det.detected_graph

        ds = DualSoul(api_key="fake")
        ds.ingest("First")
        ds.ingest("Second")
        ds.ingest("Third")

        assert ds.total_utterances == 3

    @patch("soulgraph.dual_soul.Detector")
    def test_surface_is_detector_graph(self, MockDetector):
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="hiking", domains=["hobby"]))
        mock_det = MockDetector.return_value
        mock_det.detected_graph = g
        mock_det.listen_and_detect.return_value = g

        ds = DualSoul(api_key="fake")
        ds.ingest("I love hiking")

        assert len(ds.surface.items) == 1


class TestAdaptiveThreshold:
    @patch("soulgraph.dual_soul.Detector")
    def test_threshold_empty_deep(self, MockDetector):
        ds = DualSoul(api_key="fake")
        # n=1 (min) → 0.82
        assert ds._adaptive_merge_threshold() >= 0.81

    @patch("soulgraph.dual_soul.Detector")
    def test_threshold_decreases_with_size(self, MockDetector):
        ds = DualSoul(api_key="fake")
        for i in range(100):
            ds._deep.add_item(SoulItem(
                id=f"si_{i:03d}", text=f"item {i}", domains=["test"]
            ))
        t100 = ds._adaptive_merge_threshold()
        assert t100 < 0.60

    @patch("soulgraph.dual_soul.Detector")
    def test_threshold_has_floor(self, MockDetector):
        ds = DualSoul(api_key="fake")
        for i in range(10000):
            ds._deep.add_item(SoulItem(
                id=f"si_{i:05d}", text=f"item {i}", domains=["test"]
            ))
        assert ds._adaptive_merge_threshold() >= 0.40


class TestConsolidation:
    @patch("soulgraph.dual_soul.Detector")
    def test_consolidate_empty_surface(self, MockDetector):
        """Consolidation with empty Surface should be a no-op."""
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="test")
        ds = DualSoul(api_key="fake")
        result = ds.consolidate()
        assert result["merged"] == 0
        assert result["added"] == 0

    @patch("soulgraph.dual_soul.Detector")
    def test_consolidate_adds_new_to_deep(self, MockDetector):
        """Surface nodes with no Deep match should be added as new."""
        mock_det = MockDetector.return_value
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="loves hiking", domains=["hobby"], confidence=0.8))
        g.add_item(SoulItem(id="si_002", text="fears poverty", domains=["emotion"], confidence=0.9))
        mock_det.detected_graph = g

        ds = DualSoul(api_key="fake")
        result = ds.consolidate()

        assert result["added"] == 2
        assert len(ds.deep.items) == 2


class TestBatchMerge:
    @patch("soulgraph.dual_soul.Detector")
    def test_batch_merge_returns_texts(self, MockDetector):
        ds = DualSoul(api_key="fake")
        # Mock the LLM client
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"deep_id": "di_001", "merged_text": "merged concept A and B"}]')]
        ds._client = MagicMock()
        ds._client.messages.create.return_value = mock_response

        s_item = SoulItem(id="si_001", text="concept A", domains=["test"])
        d_item = SoulItem(id="di_001", text="concept B", domains=["test"])
        pairs = [(s_item, d_item)]

        result = ds._batch_merge(pairs)
        assert len(result) == 1
        assert "merged" in result[0].lower()

    @patch("soulgraph.dual_soul.Detector")
    def test_batch_merge_empty(self, MockDetector):
        ds = DualSoul(api_key="fake")
        result = ds._batch_merge([])
        assert result == []


class TestEdgeMigration:
    @patch("soulgraph.dual_soul.Detector")
    def test_edges_remapped_to_deep(self, MockDetector):
        ds = DualSoul(api_key="fake")
        remap = {"si_001": "di_001", "si_002": "di_002"}
        ds._detector.detected_graph = SoulGraph(owner_id="test")
        ds.surface.add_item(SoulItem(id="si_001", text="a", domains=["x"]))
        ds.surface.add_item(SoulItem(id="si_002", text="b", domains=["x"]))
        ds.surface.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        ds._deep.add_item(SoulItem(id="di_001", text="a", domains=["x"]))
        ds._deep.add_item(SoulItem(id="di_002", text="b", domains=["x"]))

        ds._migrate_edges(remap)

        assert len(ds._deep.edges) >= 1
        e = ds._deep.edges[0]
        assert e.from_id == "di_001"
        assert e.to_id == "di_002"


class TestDecay:
    @patch("soulgraph.dual_soul.Detector")
    def test_unreinforced_nodes_decay(self, MockDetector):
        ds = DualSoul(api_key="fake")
        ds._consolidation_count = 5
        ds._deep.add_item(SoulItem(
            id="di_001", text="old item", domains=["x"],
            confidence=0.8, last_reinforced_cycle=1
        ))
        ds._deep.add_item(SoulItem(
            id="di_002", text="recent item", domains=["x"],
            confidence=0.8, last_reinforced_cycle=5
        ))

        decayed = ds._apply_decay()

        old = next(i for i in ds._deep.items if i.id == "di_001")
        recent = next(i for i in ds._deep.items if i.id == "di_002")
        assert old.confidence < 0.8
        assert recent.confidence == 0.8
        assert decayed >= 1


class TestCarryForward:
    @patch("soulgraph.dual_soul.Detector")
    def test_carry_forward_keeps_top_k(self, MockDetector):
        mock_det = MockDetector.return_value
        g = SoulGraph(owner_id="test")
        for i in range(20):
            g.add_item(SoulItem(id=f"si_{i:03d}", text=f"item {i}", domains=["x"]))
        for i in range(1, 15):
            g.add_edge(SoulEdge(from_id=f"si_{i:03d}", to_id="si_000", relation="drives"))
        mock_det.detected_graph = g

        ds = DualSoul(api_key="fake", carry_forward_k=5)
        ds._carry_forward_and_reset()

        assert len(ds.surface.items) <= 5
        carried_ids = {i.id for i in ds.surface.items}
        assert "si_000" in carried_ids


class TestQueryRouting:
    @patch("soulgraph.dual_soul.Detector")
    def test_recency_keywords(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("What is she thinking right now?")
        assert sw > dw

    @patch("soulgraph.dual_soul.Detector")
    def test_enduring_keywords(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("What kind of person is she generally?")
        assert dw > sw

    @patch("soulgraph.dual_soul.Detector")
    def test_default_routing(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("Tell me about her")
        assert sw == dw == 0.5

    @patch("soulgraph.dual_soul.Detector")
    def test_chinese_recency(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("她现在在想什么？")
        assert sw > dw

    @patch("soulgraph.dual_soul.Detector")
    def test_chinese_enduring(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("她的性格是怎样的？")
        assert dw > sw


class TestDualQuery:
    @patch("soulgraph.dual_soul.Detector")
    def test_query_empty_graphs(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="test")
        ds = DualSoul(api_key="fake")
        result = ds.query("What drives this person?")
        assert "empty" in result.lower()

    @patch("soulgraph.dual_soul.Detector")
    def test_query_with_items(self, MockDetector):
        mock_det = MockDetector.return_value
        surface = SoulGraph(owner_id="test")
        surface.add_item(SoulItem(id="si_001", text="thinking about dinner", domains=["daily"]))
        mock_det.detected_graph = surface

        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(id="di_001", text="core belief in family", domains=["values"]))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="She is currently thinking about dinner but deeply values family.")]
        ds._client = MagicMock()
        ds._client.messages.create.return_value = mock_response

        result = ds.query("What is she thinking?")
        assert len(result) > 0


class TestDualSoulPersistence:
    @patch("soulgraph.dual_soul.Detector")
    def test_save_and_load(self, MockDetector, tmp_path):
        mock_det = MockDetector.return_value
        surface = SoulGraph(owner_id="test")
        surface.add_item(SoulItem(id="si_001", text="surface item", domains=["x"]))
        mock_det.detected_graph = surface

        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(id="di_001", text="deep item", domains=["y"]))
        ds.total_utterances = 50
        ds._consolidation_count = 2

        path = str(tmp_path / "dual_soul.json")
        ds.save(path)

        # Load into new instance
        ds2 = DualSoul(api_key="fake")
        ds2.load(path)

        assert len(ds2.deep.items) == 1
        assert ds2.deep.items[0].text == "deep item"
        assert len(ds2.surface.items) == 1
        assert ds2.total_utterances == 50
        assert ds2._consolidation_count == 2


class TestMetaConsolidation:
    @patch("soulgraph.dual_soul.Detector")
    def test_meta_consolidate_empty_deep_is_noop(self, MockDetector):
        ds = DualSoul(api_key="fake")
        result = ds.meta_consolidate()
        assert result["roots_created"] == 0
        assert result["roots_updated"] == 0
        assert result["edges_created"] == 0

    @patch("soulgraph.dual_soul.Detector")
    def test_meta_consolidate_creates_root_with_multi_framework_tags(self, MockDetector):
        ds = DualSoul(api_key="fake")
        for i, text in enumerate([
            "wants a pretty dress to impress",
            "needs to marry a wealthy man",
            "will fight to keep Tara land",
            "runs lumber mill for money",
            "fears going hungry again",
        ]):
            ds._deep.add_item(SoulItem(
                id=f"di_{i:04d}", text=text, domains=["survival"],
                confidence=0.8, mention_count=3,
            ))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {
                "root_text": "Survival and material security — must never be vulnerable",
                "motivation_tags": {
                    "maslow": "safety",
                    "sdt": "autonomy",
                    "reiss": "tranquility",
                },
                "concrete_ids": ["di_0000", "di_0001", "di_0002", "di_0003", "di_0004"],
                "confidence": 0.95,
            }
        ]))]
        ds._client = MagicMock()
        ds._client.messages.create.return_value = mock_response

        result = ds.meta_consolidate()

        assert result["roots_created"] >= 1
        assert result["edges_created"] >= 5
        roots = [i for i in ds._deep.items if i.abstraction_level == 1]
        assert len(roots) >= 1
        assert roots[0].motivation_tags["maslow"] == "safety"
        assert roots[0].motivation_tags["sdt"] == "autonomy"
        manifest_edges = [e for e in ds._deep.edges if e.relation == "manifests-as"]
        assert len(manifest_edges) >= 5

    @patch("soulgraph.dual_soul.Detector")
    def test_meta_consolidate_updates_existing_roots(self, MockDetector):
        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(
            id="ri_0001", text="survival security need", domains=["survival"],
            abstraction_level=1, motivation_tags={"maslow": "safety"},
            confidence=0.8, mention_count=3,
        ))
        ds._deep.add_item(SoulItem(
            id="di_0010", text="must earn money through lumber business",
            domains=["business"], confidence=0.7, mention_count=2,
        ))
        ds._deep.add_item(SoulItem(
            id="di_0011", text="fears going hungry again",
            domains=["survival"], confidence=0.8, mention_count=3,
        ))
        ds._deep.add_item(SoulItem(
            id="di_0012", text="will fight to protect the land",
            domains=["survival"], confidence=0.7, mention_count=2,
        ))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {
                "root_id": "ri_0001",
                "root_text": "Survival and material security — must never be vulnerable",
                "motivation_tags": {"maslow": "safety", "sdt": "autonomy"},
                "concrete_ids": ["di_0010"],
                "confidence": 0.9,
            }
        ]))]
        ds._client = MagicMock()
        ds._client.messages.create.return_value = mock_response

        result = ds.meta_consolidate()

        assert result["roots_updated"] >= 1
        root = next(i for i in ds._deep.items if i.id == "ri_0001")
        assert root.mention_count > 3
        assert root.motivation_tags["sdt"] == "autonomy"
        manifest_edges = [e for e in ds._deep.edges
                          if e.relation == "manifests-as" and e.to_id == "di_0010"]
        assert len(manifest_edges) == 1

    @patch("soulgraph.dual_soul.Detector")
    def test_meta_consolidate_discovers_multiple_roots(self, MockDetector):
        ds = DualSoul(api_key="fake")
        for i, (text, domain) in enumerate([
            ("wants Ashley's love", "romance"),
            ("needs to be most admired", "esteem"),
            ("fears hunger and poverty", "survival"),
            ("will fight to keep Tara", "survival"),
            ("wants Rhett's attention", "romance"),
        ]):
            ds._deep.add_item(SoulItem(
                id=f"di_{i:04d}", text=text, domains=[domain],
                confidence=0.8, mention_count=2,
            ))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps([
            {
                "root_text": "Deep need for love and emotional connection",
                "motivation_tags": {"maslow": "love", "sdt": "relatedness", "attachment": "anxious"},
                "concrete_ids": ["di_0000", "di_0004"],
                "confidence": 0.85,
            },
            {
                "root_text": "Survival and material security",
                "motivation_tags": {"maslow": "safety", "reiss": "tranquility"},
                "concrete_ids": ["di_0002", "di_0003"],
                "confidence": 0.9,
            },
            {
                "root_text": "Need for recognition and social status",
                "motivation_tags": {"maslow": "esteem", "reiss": "status", "sdt": "competence"},
                "concrete_ids": ["di_0001"],
                "confidence": 0.75,
            },
        ]))]
        ds._client = MagicMock()
        ds._client.messages.create.return_value = mock_response

        result = ds.meta_consolidate()

        roots = [i for i in ds._deep.items if i.abstraction_level == 1]
        assert len(roots) == 3
        maslow_set = {r.motivation_tags.get("maslow") for r in roots}
        assert maslow_set == {"love", "safety", "esteem"}


class TestQueryWithRoots:
    @patch("soulgraph.dual_soul.Detector")
    def test_query_system_prompt_includes_roots(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="test")

        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(
            id="ri_0001", text="survival security need", domains=["survival"],
            abstraction_level=1, motivation_tags={"maslow": "safety"},
            confidence=0.9, mention_count=10,
        ))
        ds._deep.add_item(SoulItem(
            id="di_0001", text="fears hunger", domains=["survival"],
            confidence=0.8, mention_count=5,
        ))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Driven by survival need.")]
        ds._client = MagicMock()
        ds._client.messages.create.return_value = mock_response

        ds.query("What drives her?")

        call_args = ds._client.messages.create.call_args
        system_prompt = call_args.kwargs.get("system", "")
        assert "Root Motivation" in system_prompt
        assert "safety" in system_prompt


class TestMetaConsolidationTrigger:
    @patch("soulgraph.dual_soul.Detector")
    def test_meta_triggers_every_n_consolidations(self, MockDetector):
        mock_det = MockDetector.return_value
        g = SoulGraph(owner_id="test")
        for i in range(5):
            g.add_item(SoulItem(id=f"si_{i:03d}", text=f"item {i}", domains=["x"]))
        mock_det.detected_graph = g
        mock_det.listen_and_detect.return_value = g

        ds = DualSoul(api_key="fake", deep_cycle=1, meta_cycle=2)
        ds._client = MagicMock()

        meta_calls = []
        def mock_meta(**kwargs):
            meta_calls.append(1)
            return {"roots_created": 0, "roots_updated": 0, "edges_created": 0}
        ds.meta_consolidate = mock_meta

        for i in range(4):
            ds.ingest(f"utterance {i}")

        assert len(meta_calls) == 2

    @patch("soulgraph.dual_soul.Detector")
    def test_meta_cycle_persisted_in_save_load(self, MockDetector, tmp_path):
        MockDetector.return_value.detected_graph = SoulGraph(owner_id="test")
        ds = DualSoul(api_key="fake", meta_cycle=5)
        ds._consolidation_count = 3
        path = str(tmp_path / "test.json")
        ds.save(path)

        ds2 = DualSoul(api_key="fake")
        ds2.load(path)
        assert ds2.meta_cycle == 5

    @patch("soulgraph.dual_soul.Detector")
    def test_root_items_survive_save_load(self, MockDetector, tmp_path):
        MockDetector.return_value.detected_graph = SoulGraph(owner_id="test")
        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(
            id="ri_0001", text="survival need", domains=["survival"],
            abstraction_level=1, motivation_tags={"maslow": "safety", "sdt": "autonomy"},
            confidence=0.9, mention_count=10,
        ))
        path = str(tmp_path / "test.json")
        ds.save(path)

        ds2 = DualSoul(api_key="fake")
        ds2.load(path)
        root = next(i for i in ds2._deep.items if i.id == "ri_0001")
        assert root.abstraction_level == 1
        assert root.motivation_tags["maslow"] == "safety"
        assert root.motivation_tags["sdt"] == "autonomy"
