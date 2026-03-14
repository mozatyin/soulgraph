"""Tests for DualSoul — Dual Soul architecture."""
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
