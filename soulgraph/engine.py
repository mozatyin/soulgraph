"""SoulEngine — the public SDK interface for SoulGraph.

Usage:
    engine = SoulEngine(api_key="...")
    engine.ingest("I've been thinking about buying a car")
    engine.ingest("Work pressure is really getting to me lately")
    answer = engine.query("What's really driving this person?")
"""
from __future__ import annotations

import json
import time

import anthropic

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulGraph


_QUERY_SYSTEM = """\
You are answering a question about a person based on their soul graph — a structured \
representation of their inner world built from conversations.

## Retrieved Subgraph (most relevant to the query)

### Nodes
{nodes}

### Connections
{edges}

## Full Graph Stats
{stats}

## Rules
1. Answer based ONLY on the graph data above. Be specific — reference actual items.
2. Surface cross-domain connections: how do seemingly unrelated aspects connect?
3. If the graph doesn't contain enough info, say so honestly.
4. Be concise: 2-4 sentences for simple queries, up to a paragraph for complex ones.
5. Respond in the same language as the query.
"""


class SoulEngine:
    """Ingest conversations, build a soul graph, query it for insights."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ):
        self._api_key = api_key
        self._model = model
        self._detector = Detector(api_key=api_key, model=model)
        self._messages: list[Message] = []
        self._turn = 0

        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)

    @property
    def graph(self) -> SoulGraph:
        return self._detector.detected_graph

    def ingest(self, text: str) -> SoulGraph:
        """Ingest a message and extract soul items/edges.

        Returns the current graph state after extraction.
        """
        self._turn += 1
        self._messages.append(Message(role="speaker", content=text))
        self._detector.listen_and_detect(self._messages)
        return self.graph

    def query(self, question: str, top_k: int = 10) -> str:
        """Query the soul graph and get a personalized answer.

        Uses PPR to retrieve the most relevant subgraph, then asks an LLM
        to synthesize an answer grounded in the graph data.
        """
        if not self.graph.items:
            return "Graph is empty — ingest some conversation first."

        subgraph = self.graph.query_subgraph(question, top_k=top_k)

        nodes_text = "\n".join(
            f"- {item.text} (domains: {', '.join(item.domains)}, "
            f"type: {item.item_type.value}, confidence: {item.confidence:.1f})"
            for item in subgraph.items
        )
        edges_text = "\n".join(
            f"- [{e.from_id}] --{e.relation}--> [{e.to_id}]"
            for e in subgraph.edges
        ) or "(no edges in subgraph)"

        # Find text for edge endpoints for readability
        id_to_text = {item.id: item.text for item in subgraph.items}
        edges_readable = "\n".join(
            f"- \"{id_to_text.get(e.from_id, e.from_id)}\" --{e.relation}--> "
            f"\"{id_to_text.get(e.to_id, e.to_id)}\""
            for e in subgraph.edges
        ) or "(no connections in subgraph)"

        stats = (
            f"Total: {len(self.graph.items)} items, {len(self.graph.edges)} edges | "
            f"Subgraph: {len(subgraph.items)} items, {len(subgraph.edges)} edges"
        )

        system = _QUERY_SYSTEM.format(
            nodes=nodes_text,
            edges=edges_readable,
            stats=stats,
        )

        for attempt in range(3):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=1024,
                    system=system,
                    messages=[{"role": "user", "content": question}],
                )
                if response.content:
                    return response.content[0].text
                return ""
            except (anthropic.APIError, anthropic.APIConnectionError):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return ""

    def save(self, path: str) -> None:
        """Save the current graph to a JSON file."""
        from pathlib import Path
        Path(path).write_text(
            self.graph.model_dump_json(indent=2), encoding="utf-8"
        )

    def load(self, path: str) -> None:
        """Load a previously saved graph."""
        from pathlib import Path
        self._detector.detected_graph = SoulGraph.load(Path(path))
