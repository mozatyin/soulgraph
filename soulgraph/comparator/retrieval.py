"""Retrieval-based evaluation — V6 metrics for query-driven subgraph quality."""
from __future__ import annotations

import json
import networkx as nx
import anthropic

from soulgraph.graph.models import SoulGraph


_JUDGE_PROMPT = """You are evaluating the quality of a knowledge subgraph retrieved for a query.

## Query
{query}

## Retrieved Subgraph Nodes
{nodes}

## Conversation Transcript (source)
{transcript}

## Task
Evaluate this dimension: **{dimension}**

{dimension_description}

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""

_DIMENSIONS = {
    "faithfulness": "Is every node in the subgraph directly supported by something said in the conversation? Score 1.0 if all nodes are grounded, 0.0 if none are.",
    "comprehensiveness": "Does this subgraph capture all important aspects of the conversation relevant to the query? Score 1.0 if complete coverage, 0.0 if missing everything relevant.",
    "diversity": "Does this subgraph span multiple different domains or perspectives related to the query? Score 1.0 if highly diverse (3+ domains), 0.5 if moderate (2 domains), 0.2 if single-domain.",
}


class RetrievalEvaluator:
    """Evaluate quality of a retrieved subgraph."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def structural_metrics(self, subgraph: SoulGraph) -> dict:
        """Compute structural quality metrics for a subgraph (no LLM needed)."""
        if not subgraph.items:
            return {
                "node_count": 0,
                "edge_count": 0,
                "cross_domain_coverage": 0,
                "is_connected": False,
                "density": 0.0,
            }

        domains: set[str] = set()
        for item in subgraph.items:
            domains.update(item.domains)

        G = subgraph._to_nx()
        undirected = G.to_undirected()
        is_connected = nx.is_connected(undirected) if len(undirected) > 0 else False
        density = nx.density(G) if len(G) > 1 else 0.0

        return {
            "node_count": len(subgraph.items),
            "edge_count": len(subgraph.edges),
            "cross_domain_coverage": len(domains),
            "is_connected": is_connected,
            "density": round(density, 3),
        }

    def evaluate(
        self,
        full_graph: SoulGraph,
        subgraph: SoulGraph,
        query: str,
        conversation_transcript: str,
    ) -> dict:
        """Full evaluation: structural metrics + LLM-as-Judge."""
        structural = self.structural_metrics(subgraph)

        nodes_text = "\n".join(
            f"- [{item.id}] {item.text} (domains: {', '.join(item.domains)})"
            for item in subgraph.items
        )
        transcript = conversation_transcript[-4000:] if len(conversation_transcript) > 4000 else conversation_transcript

        llm_scores = {}
        for dim, desc in _DIMENSIONS.items():
            llm_scores[dim] = self._judge(query, nodes_text, transcript, dim, desc)

        # Combined structural score
        structural_score = 0.0
        if structural["node_count"] > 0:
            connected_bonus = 0.5 if structural["is_connected"] else 0.0
            domain_bonus = min(structural["cross_domain_coverage"] / 3.0, 1.0) * 0.3
            density_bonus = min(structural["density"] / 0.3, 1.0) * 0.2
            structural_score = connected_bonus + domain_bonus + density_bonus

        retrieval_score = (
            llm_scores.get("faithfulness", 0.0) * 0.3
            + llm_scores.get("comprehensiveness", 0.0) * 0.3
            + llm_scores.get("diversity", 0.0) * 0.2
            + structural_score * 0.2
        )

        return {
            **structural,
            **llm_scores,
            "structural_score": round(structural_score, 3),
            "retrieval_score": round(retrieval_score, 3),
        }

    def _judge(self, query: str, nodes: str, transcript: str, dimension: str, description: str) -> float:
        """Call LLM to judge one dimension. Returns score 0.0-1.0."""
        if not self._api_key:
            return 0.5

        kwargs: dict = {"api_key": self._api_key}
        if self._api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        client = anthropic.Anthropic(**kwargs)

        prompt = _JUDGE_PROMPT.format(
            query=query,
            nodes=nodes,
            transcript=transcript,
            dimension=dimension,
            dimension_description=description,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            data = json.loads(text)
            return max(0.0, min(1.0, float(data["score"])))
        except Exception:
            return 0.5
