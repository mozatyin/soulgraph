"""Tests for emotional field extraction in Detector."""
import json
import pytest
from unittest.mock import MagicMock, patch
from soulgraph.experiment.detector import Detector, _DETECT_SYSTEM


class TestDetectSystemPrompt:
    def test_prompt_mentions_emotional_valence(self):
        assert "emotional_valence" in _DETECT_SYSTEM

    def test_prompt_mentions_authenticity_hint(self):
        assert "authenticity_hint" in _DETECT_SYSTEM


class TestAddItemEmotionalFields:
    def test_add_item_with_emotional_fields(self):
        det = Detector.__new__(Detector)
        from soulgraph.graph.models import SoulGraph
        det.detected_graph = SoulGraph(owner_id="test")
        det.session_number = 0
        det._add_item({
            "text": "I never want to see him again",
            "domains": ["relationship"],
            "emotional_valence": "extreme",
            "authenticity_hint": "amplified",
        }, "si_001")
        item = det.detected_graph.items[0]
        assert item.emotional_valence == "extreme"
        assert item.authenticity_hint == "amplified"

    def test_add_item_defaults_when_missing(self):
        det = Detector.__new__(Detector)
        from soulgraph.graph.models import SoulGraph
        det.detected_graph = SoulGraph(owner_id="test")
        det.session_number = 0
        det._add_item({
            "text": "I like coffee",
            "domains": ["preference"],
        }, "si_001")
        item = det.detected_graph.items[0]
        assert item.emotional_valence == "neutral"
        assert item.authenticity_hint == "unknown"
