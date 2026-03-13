"""Detector: incrementally reconstructs a soul graph from conversation."""
from __future__ import annotations

import json

import anthropic

from soulgraph.experiment.models import Message
from soulgraph.graph.models import ItemType, SoulEdge, SoulGraph, SoulItem

_CONSOLIDATE_PROMPT = """\
You are deduplicating a list of soul items extracted from conversation. \
Find items that describe EXACTLY the same concept and should be merged.

## Items
{items_json}

Return JSON:
{{"merges": [{{"keep_id": "<id to keep>", "remove_id": "<id to merge into keep>"}}]}}

Rules:
- ONLY merge items that say the SAME thing in different words (true duplicates)
- Do NOT merge items that are merely RELATED or CONNECTED — those are different nodes
- Example of a merge: "想买SUV" and "考虑换一辆SUV" → same concept
- Example of NOT merging: "想买SUV" and "家庭需要大空间" → related but distinct
- Keep the item with higher confidence or more specific text
- When in doubt, do NOT merge. Prefer keeping items separate.
- If no true duplicates exist, return {{"merges": []}}
"""

_DETECT_SYSTEM = """\
You are a soul graph detector for a self-discovery app. You analyze conversation to extract \
the speaker's inner world as a graph of soul items — intentions, emotions, values, fears, \
and background facts — and their relationships.

## Current Detected Graph
{current_graph_json}

## Item Types
Each soul item has an item_type:
- "cognitive": Self-understanding intention — insights about oneself (认知型)
  Example: "说不清自己到底在焦虑什么" "害怕被别人看出自己不行"
- "action": Real-world action intention — something the person wants to DO (行动型)
  Example: "想试试冥想" "想找心理咨询师" "想多运动"
- "background": Context, facts, experiences that shape the person
  Example: "工作每天加班到很晚" "小时候父母要求很高"

## Rules
1. Extract ONLY items clearly evidenced by the conversation. Do NOT infer or guess.
2. Classify each item as cognitive, action, or background.
3. Prefer fewer, higher-confidence items over many low-confidence ones.
4. Each item should be distinct — do NOT create near-duplicates of existing items.
5. Before adding a new item, check if an existing item already covers the same concept. \
If so, add its ID to strengthen_ids instead.
6. Relationships MUST be one of these exact types:
   - drives (A motivates/causes/leads to B)
   - enables (A makes B possible/supports B)
   - constrains (A limits/restricts/blocks B)
   - conflicts_with (A contradicts/opposes B, internal tension)
   - manifests_as (A expresses itself as B)
   - decomposes_to (A breaks down into B, part-whole)
   - compensates (A balances/offsets B)
   - next_step (A leads to B sequentially)
   Do NOT invent other relationship types.
7. Confidence: 0.9 = stated directly, 0.5 = implied, 0.3 = weak inference.

Return JSON:
{{
  "new_items": [{{"id": "si_NNN", "text": "...", "domains": [...], "item_type": "cognitive|action|background", "confidence": 0.0-1.0, "specificity": 0.0-1.0}}],
  "new_edges": [{{"from_id": "...", "to_id": "...", "relation": "...", "strength": 0.0-1.0, "confidence": 0.0-1.0}}],
  "strengthen_ids": ["si_001", ...]
}}

IMPORTANT: Only return items with confidence >= 0.4. Max 3-5 new items per extraction cycle.
"""

_QUESTION_SYSTEM = """\
You are a skilled listener building a deep understanding of someone through conversation. \
You use motivational interviewing techniques: reflect back understanding, elicit discrepancy, \
and explore connections between what someone values and how they act.

## What You Know So Far
{current_graph_json}

## Strategy (choose based on graph state)
1. **Empty/sparse graph (0-3 items)**: Ask a broad, warm opening question. Build rapport.
2. **Growing graph (4-7 items)**: Use REFLECTIVE SUMMARY + QUESTION. Summarize 2-3 things \
you've noticed about them, then ask about the connection between two items. \
Example: "听起来你一方面...另一方面...这两者之间是什么样的关系？"
3. **Rich graph (8+ items)**: Look for MISSING CONNECTIONS. If two items seem related but \
have no edge, ask directly about how they connect. Also look for UNEXPLORED DOMAINS \
(items mentioned once with no connections).
4. **Key technique — ELICIT DISCREPANCY**: When you see potential conflicts (e.g., wanting X \
but fearing Y), reflect both sides back and ask the person to elaborate on the tension.

## Rules
- One question only. Open-ended (not yes/no).
- When graph has 4+ items, ALWAYS start with a brief reflection of what you understand \
before asking the question. This validates the speaker and often triggers deeper disclosure.
- Ask about CONNECTIONS between items, not just new topics.
- Respond in the same language as the conversation.

Return ONLY the question text (with optional reflection prefix), nothing else.
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
        # Consolidate if graph is getting large (conservative — only true dupes)
        if len(self.detected_graph.items) >= 12:
            self._consolidate()
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
            confidence = item_data.get("confidence", 0.5)
            if confidence < 0.4:
                continue  # Skip low-confidence items
            item_id = item_data.get("id", f"si_{next_id:03d}")
            item_type_str = item_data.get("item_type", "background")
            try:
                item_type = ItemType(item_type_str)
            except ValueError:
                item_type = ItemType.BACKGROUND
            self.detected_graph.add_item(
                SoulItem(
                    id=item_id,
                    text=item_data["text"],
                    domains=item_data.get("domains", ["general"]),
                    item_type=item_type,
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

    def _consolidate(self) -> None:
        """Merge near-duplicate items using LLM."""
        if len(self.detected_graph.items) < 6:
            return
        items_json = json.dumps(
            [{"id": i.id, "text": i.text, "confidence": i.confidence} for i in self.detected_graph.items],
            ensure_ascii=False,
            indent=2,
        )
        prompt = _CONSOLIDATE_PROMPT.format(items_json=items_json)
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
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
        for merge in data.get("merges", []):
            keep_id = merge.get("keep_id", "")
            remove_id = merge.get("remove_id", "")
            if keep_id and remove_id and keep_id != remove_id:
                self.detected_graph.merge_items(keep_id, remove_id)
