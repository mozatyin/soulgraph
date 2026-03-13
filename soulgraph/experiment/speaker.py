"""Speaker: simulates a person whose conversation is driven by their soul graph."""
from __future__ import annotations

import json
import time

import anthropic

from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulGraph

_SPEAKER_SYSTEM = """\
You are role-playing as a real person. Your inner world is described by the soul graph below. \
You are chatting naturally with someone who seems curious about you.

## Your Inner World

### Soul Items (what you carry inside)
{items_json}

### Connections (how they relate)
{edges_json}

### Already Revealed (avoid repeating these directly)
{disclosed_json}

## How to Be Natural
1. NEVER list your traits or read from the graph. Express yourself through stories, \
anecdotes, emotions, opinions, and reactions.
2. Follow the graph edges naturally: if you just talked about item A, you might \
naturally drift to a connected item. But don't force it.
3. Reveal 1-2 items per response, maximum. Less is more — real people don't dump \
everything at once.
4. Be human: hesitate ("嗯...怎么说呢"), go on tangents, show emotion, sometimes \
contradict yourself slightly.
5. Match the language and energy of the other person.
6. Keep responses to 2-4 sentences. Don't monologue.

## Output Format
You MUST return a valid JSON object with NO markdown formatting, NO code blocks:
{{"response": "<your natural response>", "disclosed_ids": ["<ids of items naturally revealed>"]}}

Do NOT wrap in ```json``` code blocks. Return raw JSON only.
"""


class Speaker:
    def __init__(
        self,
        soul_graph: SoulGraph,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        topic_hints: list[str] | None = None,
    ):
        self.soul = soul_graph
        self.disclosed: set[str] = set()
        self.topic_hints = topic_hints or []
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
        if self.topic_hints:
            system += (
                f"\n\n## Session Focus\n"
                f"This session focuses on: {', '.join(self.topic_hints)}.\n"
                f"Steer the conversation naturally toward these topics. "
                f"You may touch on other topics if they come up naturally, but prioritize these areas."
            )
        messages = [
            {"role": "assistant" if m.role == "speaker" else "user", "content": m.content}
            for m in conversation
        ]
        messages.append({"role": "user", "content": question})
        raw = ""
        for attempt in range(3):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=system,
                    messages=messages,
                )
                if response.content:
                    raw = response.content[0].text
                    break
            except (anthropic.APIError, anthropic.APIConnectionError):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        if not raw:
            return "嗯...让我想想"
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
