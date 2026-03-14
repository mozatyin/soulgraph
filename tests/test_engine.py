"""Tests for SoulEngine — the public SDK interface."""
from unittest.mock import MagicMock, patch

from soulgraph.engine import SoulEngine
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge


class TestSoulEngineInit:
    def test_creates_empty_graph(self):
        with patch("soulgraph.engine.Detector"):
            engine = SoulEngine(api_key="fake")
            assert engine.graph is not None

    def test_graph_property(self):
        with patch("soulgraph.engine.Detector") as MockDet:
            mock_graph = SoulGraph(owner_id="test")
            MockDet.return_value.detected_graph = mock_graph
            engine = SoulEngine(api_key="fake")
            assert engine.graph is mock_graph


class TestSoulEngineIngest:
    @patch("soulgraph.engine.Detector")
    def test_ingest_calls_detector(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="test")
        mock_det.listen_and_detect.return_value = mock_det.detected_graph

        engine = SoulEngine(api_key="fake")
        result = engine.ingest("I love hiking in the mountains")

        assert mock_det.listen_and_detect.call_count == 1
        assert len(engine._messages) == 1
        assert engine._messages[0].role == "speaker"
        assert engine._messages[0].content == "I love hiking in the mountains"

    @patch("soulgraph.engine.Detector")
    def test_ingest_multiple(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="test")
        mock_det.listen_and_detect.return_value = mock_det.detected_graph

        engine = SoulEngine(api_key="fake")
        engine.ingest("First message")
        engine.ingest("Second message")

        assert mock_det.listen_and_detect.call_count == 2
        assert len(engine._messages) == 2


class TestSoulEngineQuery:
    @patch("soulgraph.engine.Detector")
    def test_query_empty_graph(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="test")

        engine = SoulEngine(api_key="fake")
        result = engine.query("What drives this person?")
        assert "empty" in result.lower()

    @patch("soulgraph.engine.anthropic")
    @patch("soulgraph.engine.Detector")
    def test_query_with_items(self, MockDetector, mock_anthropic):
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="loves hiking", domains=["hobbies"]))
        g.add_item(SoulItem(id="si_002", text="values nature", domains=["values"]))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))

        mock_det = MockDetector.return_value
        mock_det.detected_graph = g

        # Mock the anthropic client response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This person loves hiking because they deeply value nature.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        engine = SoulEngine(api_key="fake")
        answer = engine.query("What does this person care about?")

        assert len(answer) > 0
        assert mock_client.messages.create.call_count == 1


class TestSoulEngineSaveLoad:
    @patch("soulgraph.engine.Detector")
    def test_save_and_load(self, MockDetector, tmp_path):
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="test item", domains=["test"]))

        mock_det = MockDetector.return_value
        mock_det.detected_graph = g

        engine = SoulEngine(api_key="fake")
        path = str(tmp_path / "test_graph.json")
        engine.save(path)

        # Load into new engine
        engine2 = SoulEngine(api_key="fake")
        engine2.load(path)
        assert len(engine2.graph.items) == 1
        assert engine2.graph.items[0].text == "test item"
