"""Semantic matching between soul items using LLM-as-judge (batch)."""
from __future__ import annotations

import json

import anthropic

from soulgraph.graph.models import SoulItem

_BATCH_MATCH_PROMPT = """\
You are comparing two sets of soul descriptions to find semantic matches.

## Ground Truth Items
{gt_items_json}

## Detected Items
{det_items_json}

For each ground truth item, find the best matching detected item (if any).
Two items match if they describe the same underlying concept, intention, or characteristic.

Return JSON:
{{
  "matches": [
    {{"gt_id": "<ground truth id>", "det_id": "<detected id>", "similarity": 0.0-1.0}},
    ...
  ]
}}

Rules:
- Only include matches with similarity >= 0.6
- Each detected item can only match ONE ground truth item
- If no good match exists for a ground truth item, omit it
- Be generous with matching — if the core meaning is the same, it's a match even if wording differs
"""


class SemanticMatcher:
    def __init__(
        self,
        api_key: str,
        model: str = "claude-haiku-4-5-20251001",
        threshold: float = 0.6,
    ):
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
        self._threshold = threshold

    def is_match(self, text_a: str, text_b: str) -> bool:
        """Single pair match — kept for backward compat but prefer match_items."""
        if text_a == text_b:
            return True
        response = self._client.messages.create(
            model=self._model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Do these describe the same concept?\n"
                        f"A: {text_a}\nB: {text_b}\n"
                        f'Return JSON: {{"is_match": true/false, "similarity": 0.0-1.0}}'
                    ),
                }
            ],
        )
        raw = response.content[0].text
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            return data.get("similarity", 0.0) >= self._threshold
        except (json.JSONDecodeError, KeyError):
            return False

    def match_items(
        self, ground_truth: list[SoulItem], detected: list[SoulItem]
    ) -> dict[str, str]:
        """Batch match using single LLM call. Returns {gt_id: det_id}."""
        if not ground_truth or not detected:
            return {}

        gt_json = json.dumps(
            [{"id": i.id, "text": i.text} for i in ground_truth],
            ensure_ascii=False,
            indent=2,
        )
        det_json = json.dumps(
            [{"id": i.id, "text": i.text} for i in detected],
            ensure_ascii=False,
            indent=2,
        )

        prompt = _BATCH_MATCH_PROMPT.format(
            gt_items_json=gt_json, det_items_json=det_json
        )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text

        # Parse response
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            last = raw.rfind("}")
            if start == -1 or last == -1:
                return {}
            try:
                data = json.loads(raw[start : last + 1])
            except json.JSONDecodeError:
                return {}

        # Build mapping from matches
        mapping: dict[str, str] = {}
        used_det: set[str] = set()
        # Sort by similarity descending to get best matches first
        matches = sorted(
            data.get("matches", []),
            key=lambda m: m.get("similarity", 0),
            reverse=True,
        )
        for match in matches:
            gt_id = match.get("gt_id", "")
            det_id = match.get("det_id", "")
            sim = match.get("similarity", 0.0)
            if (
                sim >= self._threshold
                and gt_id not in mapping
                and det_id not in used_det
            ):
                mapping[gt_id] = det_id
                used_det.add(det_id)

        return mapping
