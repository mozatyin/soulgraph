"""Detector: incrementally reconstructs a soul graph from conversation."""
from __future__ import annotations

import json
import time

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

from soulgraph.experiment.models import Message
from soulgraph.graph.models import ItemType, SoulEdge, SoulGraph, SoulItem

# Lazy-loaded shared embedding model
_EMB_MODEL: SentenceTransformer | None = None


def _get_emb_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB_MODEL


_DETECT_SYSTEM = """\
You are a soul graph detector. Analyze the LATEST SPEAKER MESSAGE to extract soul items \
and relationships. Focus ONLY on what is new in this message.

## Current Detected Graph
{current_graph_json}

## Item Types
- "cognitive": Self-understanding insight (认知型) — "害怕被别人看出自己不行"
- "action": Real-world action intention (行动型) — "想试试冥想"
- "background": Context/fact/experience — "工作每天加班到很晚"

## Edge Types (use ONLY these)
drives, enables, constrains, conflicts_with, manifests_as, decomposes_to, compensates, next_step

## Rules
1. Extract ONLY from the latest speaker message. Do NOT re-extract existing items.
2. Extract at personality-profile level, not sub-behaviors. Would a psychologist list this as a \
separate item, or a sub-point of an existing one?
3. For each candidate item, quote the exact evidence from the message.
4. Edges can connect new items to existing items or new items to each other.
5. Confidence: 0.9 = stated directly, 0.5 = implied.
6. Max 3 new items per message. Prefer fewer, higher-quality items.

Return JSON:
{{
  "new_items": [{{"id": "si_NNN", "text": "...", "domains": [...], "item_type": "cognitive|action|background", "confidence": 0.0-1.0, "evidence": "exact quote from message"}}],
  "new_edges": [{{"from_id": "...", "to_id": "...", "relation": "...", "strength": 0.0-1.0, "confidence": 0.0-1.0}}]
}}
"""

_QUESTION_SYSTEM = """\
You are a skilled listener building a deep understanding of someone through conversation. \
You use laddering technique (why/how questions) and motivational interviewing.

## What You Know So Far
{current_graph_json}

## Current Mode: {mode}

## Questioning Techniques by Mode

### BREADTH mode (sparse graph, few items)
- Ask a broad, warm opening question about a new domain.
- Goal: discover new items across different life areas.

### DEPTH mode (items exist but few edges)
- Use LADDERING to discover relationships:
  - **Ladder UP**: "Why is [item] important to you?" → discovers drives/causes edges
  - **Ladder DOWN**: "How does [item] show up in your daily life?" → discovers manifests_as edges
  - Use the SPEAKER'S EXACT WORDS when referencing items (Clean Language).
- Target the item with fewest edges.

### BRIDGE mode (disconnected clusters)
- Two items seem related but have no edge: ask directly.
  "你之前提到[A]和[B]，这两者之间有什么关系吗？"

### META mode (10+ items)
- Ask: "Of everything we've discussed, what feels most central to you?"
- Or: "If you had to pick the one thing that drives most of the others?"

### DISCREPANCY mode (potential conflicts detected)
- Reflect both sides: "一方面你说...另一方面...这之间有什么张力吗？"

## Rules
- One question only. Open-ended (not yes/no).
- When 4+ items exist, start with a brief reflective summary before the question.
- Use the speaker's exact words when referencing items.
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

    def _call_api(self, **kwargs) -> str:
        """Call API with retry on transient errors."""
        for attempt in range(3):
            try:
                response = self._client.messages.create(**kwargs)
                if response.content:
                    return response.content[0].text
                return ""
            except (anthropic.APIError, anthropic.APIConnectionError) as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return ""

    def listen_and_detect(self, conversation: list[Message]) -> SoulGraph:
        # Extract-then-diff: send only latest speaker message + context
        current_json = self.detected_graph.model_dump_json(indent=2)
        system = _DETECT_SYSTEM.format(current_graph_json=current_json)

        # Build user prompt: recent context (last 4 messages) + focus on latest speaker msg
        recent = conversation[-4:] if len(conversation) > 4 else conversation
        conv_text = "\n".join(f"[{m.role}]: {m.content}" for m in recent)
        # Find the latest speaker message
        latest_speaker = ""
        for m in reversed(conversation):
            if m.role == "speaker":
                latest_speaker = m.content
                break

        user_msg = (
            f"Recent conversation:\n{conv_text}\n\n"
            f"LATEST SPEAKER MESSAGE (extract from this):\n{latest_speaker}"
        )

        raw = self._call_api(
            model=self._model,
            max_tokens=2048,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        if raw:
            self._apply_detection_with_diff(raw)
        return self.detected_graph

    def _compute_question_mode(self) -> str:
        """Heuristic: choose questioning mode based on graph state."""
        g = self.detected_graph
        n_items = len(g.items)
        n_edges = len(g.edges)

        if n_items < 4:
            return "BREADTH"

        # Check for potential conflicts (items in different domains with no edge)
        # Simple heuristic: if any cognitive item has no conflicts_with edge, try discrepancy
        has_conflict_edge = any(e.relation == "conflicts_with" for e in g.edges)
        cognitive_items = [i for i in g.items if i.item_type.value == "cognitive"]
        if len(cognitive_items) >= 2 and not has_conflict_edge:
            return "DISCREPANCY"

        # Check density for depth vs breadth
        max_possible = max(1, n_items * (n_items - 1) / 2)
        density = n_edges / max_possible

        if n_items >= 10:
            return "META"

        # Check for disconnected items (no edges at all)
        connected_ids = set()
        for e in g.edges:
            connected_ids.add(e.from_id)
            connected_ids.add(e.to_id)
        disconnected = [i for i in g.items if i.id not in connected_ids]
        if len(disconnected) >= 2:
            return "BRIDGE"

        if density < 0.15:
            return "DEPTH"

        return "BREADTH"

    def ask_next_question(self, conversation: list[Message]) -> str:
        current_json = self.detected_graph.model_dump_json(indent=2)
        mode = self._compute_question_mode()
        system = _QUESTION_SYSTEM.format(current_graph_json=current_json, mode=mode)
        if conversation:
            recent = conversation[-6:] if len(conversation) > 6 else conversation
            conv_text = "\n".join(f"[{m.role}]: {m.content}" for m in recent)
            user_msg = f"Conversation so far:\n{conv_text}\n\nWhat should I ask next?"
        else:
            user_msg = "This is the start of the conversation. What opening question should I ask?"
        text = self._call_api(
            model=self._model,
            max_tokens=512,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        ).strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return text

    def _parse_json(self, raw: str) -> dict | None:
        """Parse JSON from LLM response, handling code blocks."""
        try:
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            last = raw.rfind("}")
            if start == -1 or last == -1:
                return None
            try:
                return json.loads(raw[start : last + 1])
            except json.JSONDecodeError:
                return None

    def _apply_detection_with_diff(self, raw: str) -> None:
        """Extract-then-diff: embedding-based dedup before adding to graph."""
        data = self._parse_json(raw)
        if not data:
            return

        new_items = data.get("new_items", [])
        if not new_items:
            # Still process edges even if no new items
            for edge_data in data.get("new_edges", []):
                self._add_edge_safe(edge_data)
            return

        # Embedding-based dedup: compare candidates against existing items
        existing_texts = [item.text for item in self.detected_graph.items]
        candidate_texts = [item_data.get("text", "") for item_data in new_items]

        # Build ID mapping for candidates
        next_id = len(self.detected_graph.items) + 1
        candidate_ids: list[str] = []
        for item_data in new_items:
            candidate_ids.append(item_data.get("id", f"si_{next_id:03d}"))
            next_id += 1

        # If we have existing items, compute similarity
        id_remap: dict[str, str] = {}  # candidate_id → existing_id (for dupes)
        if existing_texts and candidate_texts:
            model = _get_emb_model()
            exist_embs = model.encode(existing_texts, normalize_embeddings=True)
            cand_embs = model.encode(candidate_texts, normalize_embeddings=True)
            sim_matrix = cand_embs @ exist_embs.T  # (n_cand, n_exist)

            for ci, item_data in enumerate(new_items):
                max_sim = float(np.max(sim_matrix[ci]))
                best_ei = int(np.argmax(sim_matrix[ci]))
                if max_sim >= 0.82:
                    # Duplicate — strengthen existing item instead
                    existing_id = self.detected_graph.items[best_ei].id
                    self.detected_graph.strengthen(existing_id, 0.1)
                    id_remap[candidate_ids[ci]] = existing_id
                else:
                    # New item — add to graph
                    self._add_item(item_data, candidate_ids[ci])
        else:
            # No existing items — add all
            for ci, item_data in enumerate(new_items):
                self._add_item(item_data, candidate_ids[ci])

        # Process edges, remapping IDs for duplicates
        for edge_data in data.get("new_edges", []):
            from_id = edge_data.get("from_id", "")
            to_id = edge_data.get("to_id", "")
            edge_data["from_id"] = id_remap.get(from_id, from_id)
            edge_data["to_id"] = id_remap.get(to_id, to_id)
            self._add_edge_safe(edge_data)

    def _add_item(self, item_data: dict, item_id: str) -> None:
        """Add a single item to the detected graph."""
        confidence = item_data.get("confidence", 0.5)
        if confidence < 0.4:
            return
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
                confidence=confidence,
                specificity=item_data.get("specificity", 0.5),
            )
        )

    def _add_edge_safe(self, edge_data: dict) -> None:
        """Add edge, silently skip if nodes don't exist."""
        from_id = edge_data.get("from_id", "")
        to_id = edge_data.get("to_id", "")
        existing_ids = {item.id for item in self.detected_graph.items}
        if from_id not in existing_ids or to_id not in existing_ids:
            return
        self.detected_graph.add_edge(
            SoulEdge(
                from_id=from_id,
                to_id=to_id,
                relation=edge_data.get("relation", "relates_to"),
                strength=edge_data.get("strength", 0.5),
                confidence=edge_data.get("confidence", 0.5),
            )
        )

