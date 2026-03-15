"""Tests for JSON parse fallback in experiment script."""
import json
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.scarlett_intention_experiment import compare_phase


class TestJsonParseFallback:
    def test_valid_json_parses_normally(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"phase": 1, "overall_understanding": 0.85, "commentary": "good"}')]
        mock_client.messages.create.return_value = mock_response
        result = compare_phase(1, {"name": "Test", "intentions": []}, [], [], mock_client, "test-model")
        assert result["overall_understanding"] == 0.85

    def test_malformed_json_uses_regex_fallback(self):
        mock_client = MagicMock()
        # Malformed JSON — missing comma
        bad_json = '{"phase": 1 "overall_understanding": 0.90, "commentary": "broken"}'
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=bad_json)]
        mock_client.messages.create.return_value = mock_response
        result = compare_phase(1, {"name": "Test", "intentions": []}, [], [], mock_client, "test-model")
        assert result["overall_understanding"] == 0.90

    def test_total_failure_returns_negative_one(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="I cannot produce valid output")]
        mock_client.messages.create.return_value = mock_response
        result = compare_phase(1, {"name": "Test", "intentions": []}, [], [], mock_client, "test-model")
        assert result["overall_understanding"] == -1
