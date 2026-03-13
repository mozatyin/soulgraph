"""Detector: incrementally reconstructs a soul graph from conversation."""
from __future__ import annotations

import json

import anthropic

from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulEdge, SoulGraph, SoulItem

_DETECT_SYSTEM = """\
You are a soul graph detector. You listen to a conversation and extract the speaker's \
inner world as a graph of soul items (intentions, facts, personality traits, experiences, \
values, emotions) and their relationships.

## Current Detected Graph
{current_graph_json}

## Rules
1. Extract NEW soul items not already in the graph.
2. Extract relationships between items (causes, enables, compensates, manifests_as, \
drives, conflicts_with, decomposes_to, next_step).
3. If existing items are reinforced by new evidence, list their IDs in strengthen_ids.
4. Each item needs: id (continue from highest existing), text, domains, confidence (0-1), specificity (0-1).
5. Be conservative — only extract what is clearly supported by the conversation.

Return JSON:
{{
  "new_items": [{{"id": "si_NNN", "text": "...", "domains": [...], "confidence": 0.0-1.0, "specificity": 0.0-1.0}}],
  "new_edges": [{{"from_id": "...", "to_id": "...", "relation": "...", "strength": 0.0-1.0, "confidence": 0.0-1.0}}],
  "strengthen_ids": ["si_001", ...]
}}
"""

_QUESTION_SYSTEM = """\
You are a skilled listener trying to understand someone's inner world. Based on what \
you've learned so far, ask the single most valuable next question to uncover more about \
their soul — their deep motivations, hidden experiences, values, or intentions.

## What You Know So Far
{current_graph_json}

## Rules
1. Ask naturally — like a curious friend, not an interviewer.
2. Target sparse areas of the graph (domains with few items, or items with no connections).
3. One question only. Be specific, not generic.
4. Respond in the same language as the conversation.

Return ONLY the question text, nothing else.
"""


class Detector:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.detected_graph = SoulGraph(owner_id="unknown")
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def listen_and_detect(self, conversation: list[Message]) -> SoulGraph:
        current_json = self.detected_graph.model_dump_json(indent=2)
        system = _DETECT_SYSTEM.format(current_graph_json=current_json)
        conv_text = "\n".join(f"[{m.role}]: {m.content}" for m in conversation)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": f"Conversation so far:\n{conv_text}"}],
        )
        raw = response.content[0].text
        self._apply_detection(raw)
        return self.detected_graph

    def ask_next_question(self, conversation: list[Message]) -> str:
        current_json = self.detected_graph.model_dump_json(indent=2)
        system = _QUESTION_SYSTEM.format(current_graph_json=current_json)
        if conversation:
            conv_text = "\n".join(f"[{m.role}]: {m.content}" for m in conversation)
            user_msg = f"Conversation so far:\n{conv_text}\n\nWhat should I ask next?"
        else:
            user_msg = "This is the start of the conversation. What opening question should I ask?"
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = response.content[0].text.strip()
        # Strip markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return text

    def _apply_detection(self, raw: str) -> None:
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            last = raw.rfind("}")
            if start == -1 or last == -1:
                return
            try:
                data = json.loads(raw[start : last + 1])
            except json.JSONDecodeError:
                return

        next_id = len(self.detected_graph.items) + 1
        for item_data in data.get("new_items", []):
            item_id = item_data.get("id", f"si_{next_id:03d}")
            self.detected_graph.add_item(
                SoulItem(
                    id=item_id,
                    text=item_data["text"],
                    domains=item_data.get("domains", ["general"]),
                    confidence=item_data.get("confidence", 0.5),
                    specificity=item_data.get("specificity", 0.5),
                )
            )
            next_id += 1
        for edge_data in data.get("new_edges", []):
            self.detected_graph.add_edge(
                SoulEdge(
                    from_id=edge_data["from_id"],
                    to_id=edge_data["to_id"],
                    relation=edge_data.get("relation", "relates_to"),
                    strength=edge_data.get("strength", 0.5),
                    confidence=edge_data.get("confidence", 0.5),
                )
            )
        for sid in data.get("strengthen_ids", []):
            self.detected_graph.strengthen(sid, 0.1)
