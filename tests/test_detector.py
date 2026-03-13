"""Tests for Detector."""
from unittest.mock import MagicMock

import json

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message


class TestDetector:
    def _mock_detect_response(self, items, edges):
        return json.dumps(
            {"new_items": items, "new_edges": edges, "strengthen_ids": []}
        )

    def test_listen_and_detect_adds_items(self):
        detector = Detector(api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(
                text=self._mock_detect_response(
                    items=[
                        {
                            "id": "si_001",
                            "text": "重视家庭",
                            "domains": ["family"],
                            "confidence": 0.7,
                            "specificity": 0.3,
                        }
                    ],
                    edges=[],
                )
            )
        ]
        detector._client = MagicMock()
        detector._client.messages.create.return_value = mock_response

        conversation = [Message(role="speaker", content="家庭对我来说很重要")]
        graph = detector.listen_and_detect(conversation)

        assert len(graph.items) == 1
        assert graph.items[0].text == "重视家庭"

    def test_listen_and_detect_incremental(self):
        detector = Detector(api_key="fake")

        mock1 = MagicMock()
        mock1.content = [
            MagicMock(
                text=self._mock_detect_response(
                    items=[
                        {
                            "id": "si_001",
                            "text": "重视家庭",
                            "domains": ["family"],
                            "confidence": 0.7,
                            "specificity": 0.3,
                        }
                    ],
                    edges=[],
                )
            )
        ]
        mock2 = MagicMock()
        mock2.content = [
            MagicMock(
                text=self._mock_detect_response(
                    items=[
                        {
                            "id": "si_002",
                            "text": "想买SUV",
                            "domains": ["purchase"],
                            "confidence": 0.6,
                            "specificity": 0.7,
                        }
                    ],
                    edges=[
                        {
                            "from_id": "si_001",
                            "to_id": "si_002",
                            "relation": "drives",
                            "strength": 0.7,
                            "confidence": 0.6,
                        }
                    ],
                )
            )
        ]

        detector._client = MagicMock()
        detector._client.messages.create.side_effect = [mock1, mock2]

        conv1 = [Message(role="speaker", content="家庭对我来说很重要")]
        detector.listen_and_detect(conv1)
        assert len(detector.detected_graph.items) == 1

        conv2 = conv1 + [
            Message(role="detector", content="那你最近有什么打算吗？"),
            Message(role="speaker", content="在考虑买辆SUV，方便带家人出去"),
        ]
        detector.listen_and_detect(conv2)
        assert len(detector.detected_graph.items) == 2
        assert len(detector.detected_graph.edges) == 1

    def test_ask_next_question_returns_string(self):
        detector = Detector(api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="你最近有什么开心的事情吗？")]
        detector._client = MagicMock()
        detector._client.messages.create.return_value = mock_response

        question = detector.ask_next_question(conversation=[])
        assert isinstance(question, str)
        assert len(question) > 0
