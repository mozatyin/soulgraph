"""Semantic matching between soul items using LLM-as-judge."""
from __future__ import annotations

import json

import anthropic

from soulgraph.graph.models import SoulItem

_MATCH_PROMPT = """\
Compare these two soul descriptions and determine if they refer to the same \
underlying concept, intention, or characteristic of a person.

Text A: {text_a}
Text B: {text_b}

Return JSON: {{"is_match": true/false, "similarity": 0.0-1.0}}
"""


class SemanticMatcher:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        threshold: float = 0.8,
    ):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
        self._threshold = threshold

    def is_match(self, text_a: str, text_b: str) -> bool:
        if text_a == text_b:
            return True
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": _MATCH_PROMPT.format(text_a=text_a, text_b=text_b),
                }
            ],
        )
        raw = response.content[0].text
        try:
            data = json.loads(raw)
            return data.get("similarity", 0.0) >= self._threshold
        except (json.JSONDecodeError, KeyError):
            return False

    def match_items(
        self, ground_truth: list[SoulItem], detected: list[SoulItem]
    ) -> dict[str, str]:
        mapping: dict[str, str] = {}
        used_det: set[str] = set()
        for gt in ground_truth:
            for det in detected:
                if det.id in used_det:
                    continue
                if self.is_match(gt.text, det.text):
                    mapping[gt.id] = det.id
                    used_det.add(det.id)
                    break
        return mapping
