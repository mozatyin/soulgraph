"""Speaker: simulates a person whose conversation is driven by their soul graph."""
from __future__ import annotations

import json

import anthropic

from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulGraph

_SPEAKER_SYSTEM = """\
You are role-playing as a person whose inner world is described by the soul graph below. \
You are having a natural conversation with someone. You do NOT know they are trying to \
understand your soul — you are just chatting naturally.

## Your Soul Graph

### Items (your inner world)
{items_json}

### Relationships
{edges_json}

### Already Disclosed (don't repeat these directly)
{disclosed_json}

## Rules
1. Chat naturally — through stories, emotions, reactions. NEVER list your traits directly.
2. Each response, tend to reveal 1-2 soul items that are NEIGHBORS of recently discussed items \
(follow the graph edges for natural flow).
3. Be human — hesitate, contradict yourself sometimes, go on tangents.
4. Respond in the same language the other person uses.

Return JSON:
{{"response": "<your natural response>", "disclosed_ids": ["<ids of items you revealed in this response>"]}}
"""


class Speaker:
    def __init__(
        self,
        soul_graph: SoulGraph,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.soul = soul_graph
        self.disclosed: set[str] = set()
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model

    def respond(self, question: str, conversation: list[Message]) -> str:
        items_json = json.dumps(
            [{"id": i.id, "text": i.text, "domains": i.domains} for i in self.soul.items],
            indent=2,
            ensure_ascii=False,
        )
        edges_json = json.dumps(
            [{"from": e.from_id, "to": e.to_id, "relation": e.relation} for e in self.soul.edges],
            indent=2,
            ensure_ascii=False,
        )
        disclosed_json = json.dumps(list(self.disclosed), ensure_ascii=False)
        system = _SPEAKER_SYSTEM.format(
            items_json=items_json,
            edges_json=edges_json,
            disclosed_json=disclosed_json,
        )
        messages = [
            {"role": "assistant" if m.role == "speaker" else "user", "content": m.content}
            for m in conversation
        ]
        messages.append({"role": "user", "content": question})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        raw = response.content[0].text
        # Strip markdown code blocks if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            data = json.loads(raw)
            text = data.get("response", raw)
            for sid in data.get("disclosed_ids", []):
                self.disclosed.add(sid)
            return text
        except json.JSONDecodeError:
            # Try to find JSON object in the text
            start = raw.find("{")
            last = raw.rfind("}")
            if start != -1 and last != -1:
                try:
                    data = json.loads(raw[start:last + 1])
                    text = data.get("response", raw)
                    for sid in data.get("disclosed_ids", []):
                        self.disclosed.add(sid)
                    return text
                except json.JSONDecodeError:
                    pass
            return raw
