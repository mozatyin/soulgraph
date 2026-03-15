"""Tests for SoulItem model extensions."""

import json

from soulgraph.graph.models import SoulItem


class TestSoulItemAbstraction:
    def test_default_abstraction_level_is_zero(self):
        item = SoulItem(id="si_001", text="wants food", domains=["survival"])
        assert item.abstraction_level == 0

    def test_root_intention_has_level_one(self):
        item = SoulItem(id="ri_001", text="survival need", domains=["survival"], abstraction_level=1)
        assert item.abstraction_level == 1

    def test_motivation_tags_defaults_to_empty_dict(self):
        item = SoulItem(id="si_001", text="wants food", domains=["survival"])
        assert item.motivation_tags == {}

    def test_motivation_tags_multi_framework(self):
        item = SoulItem(
            id="ri_001", text="survival need", domains=["survival"],
            abstraction_level=1,
            motivation_tags={
                "maslow": "safety",
                "sdt": "autonomy",
                "reiss": "tranquility",
                "attachment": "anxious",
            },
        )
        assert item.motivation_tags["maslow"] == "safety"
        assert item.motivation_tags["sdt"] == "autonomy"
        assert len(item.motivation_tags) == 4

    def test_serialization_round_trip(self):
        item = SoulItem(
            id="ri_001", text="root", domains=["x"],
            abstraction_level=1,
            motivation_tags={"maslow": "safety", "reiss": "tranquility"},
        )
        data = item.model_dump()
        assert data["abstraction_level"] == 1
        assert data["motivation_tags"]["maslow"] == "safety"
        json_str = item.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["motivation_tags"]["reiss"] == "tranquility"
