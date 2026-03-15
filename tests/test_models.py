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


class TestSoulItemEmotionalFields:
    def test_default_emotional_valence_is_neutral(self):
        item = SoulItem(id="t1", text="test", domains=["d"])
        assert item.emotional_valence == "neutral"

    def test_default_authenticity_hint_is_unknown(self):
        item = SoulItem(id="t1", text="test", domains=["d"])
        assert item.authenticity_hint == "unknown"

    def test_emotional_fields_roundtrip_json(self):
        item = SoulItem(
            id="t1", text="test", domains=["d"],
            emotional_valence="extreme",
            authenticity_hint="slip",
        )
        data = json.loads(item.model_dump_json())
        assert data["emotional_valence"] == "extreme"
        assert data["authenticity_hint"] == "slip"

    def test_emotional_valence_normalizes_invalid(self):
        item = SoulItem(id="t1", text="test", domains=["d"], emotional_valence="EXTREME")
        assert item.emotional_valence == "extreme"

    def test_emotional_valence_rejects_unknown(self):
        item = SoulItem(id="t1", text="test", domains=["d"], emotional_valence="very_angry")
        assert item.emotional_valence == "neutral"

    def test_authenticity_hint_normalizes_case(self):
        item = SoulItem(id="t1", text="test", domains=["d"], authenticity_hint="SLIP")
        assert item.authenticity_hint == "slip"

    def test_authenticity_hint_rejects_unknown_value(self):
        item = SoulItem(id="t1", text="test", domains=["d"], authenticity_hint="maybe")
        assert item.authenticity_hint == "unknown"

    def test_emotional_fields_in_model_validate(self):
        data = {
            "id": "t1", "text": "test", "domains": ["d"],
            "emotional_valence": "aroused",
            "authenticity_hint": "amplified",
        }
        item = SoulItem.model_validate(data)
        assert item.emotional_valence == "aroused"
        assert item.authenticity_hint == "amplified"
