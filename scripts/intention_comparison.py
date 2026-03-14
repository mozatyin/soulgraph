"""Compare intention detection: LLM-on-raw-text vs Graph-structure (PageRank)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from soulgraph.graph.models import SoulGraph


def load_scarlett_lines() -> list[str]:
    lines = []
    with open("fixtures/gone_with_wind.jsonl", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["speaker"] == "scarlett":
                lines.append(data["text"])
    return lines


def llm_find_intentions(client: anthropic.Anthropic, lines: list[str]) -> list[dict]:
    """Ask LLM to find top 3 intentions from raw dialogue text."""
    dialogue_text = "\n".join(f"- {line}" for line in lines)

    prompt = f"""Below are all lines spoken by a character in a series of conversations.
Read them carefully and identify this person's **top 3 most important intentions** —
things they deeply want to do, achieve, or become.

Rank by importance: #1 is the most central driving intention.

## Dialogue Lines
{dialogue_text}

## Rules
- An "intention" is a forward-looking desire, goal, or determination (not a fact or emotion)
- Focus on deep/enduring intentions, not momentary wishes
- Each intention should be distinct (not overlapping)

Respond with JSON only:
[
  {{"rank": 1, "intention": "<concise description>", "evidence": "<1-2 quotes from the dialogue>"}},
  {{"rank": 2, "intention": "...", "evidence": "..."}},
  {{"rank": 3, "intention": "...", "evidence": "..."}}
]"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start = raw.find("[")
    end = raw.rfind("]")
    return json.loads(raw[start:end + 1])


def graph_find_intentions(graph: SoulGraph) -> list[dict]:
    """Find top 3 intentions from graph structure using PageRank."""
    # Filter items tagged as intention
    intention_items = [item for item in graph.items if "intention" in item.tags]

    if not intention_items:
        # Fallback: look for action type items
        intention_items = [item for item in graph.items if item.item_type.value == "action"]

    if not intention_items:
        return []

    # Get PageRank scores
    pr = graph.pagerank()

    # Rank intention items by PageRank
    scored = []
    for item in intention_items:
        score = pr.get(item.id, 0.0)
        # Count edges connected to this item
        edge_count = sum(1 for e in graph.edges if e.from_id == item.id or e.to_id == item.id)
        scored.append({
            "id": item.id,
            "intention": item.text,
            "pagerank": round(score, 6),
            "edge_count": edge_count,
            "domains": item.domains,
            "confidence": item.confidence,
        })

    scored.sort(key=lambda x: -x["pagerank"])
    return scored[:3]


def judge_match(client: anthropic.Anthropic, llm_intentions: list[dict], graph_intentions: list[dict]) -> dict:
    """LLM judge: compare the two sets of intentions for semantic overlap."""
    llm_text = "\n".join(
        f"  LLM #{i['rank']}: {i['intention']}"
        for i in llm_intentions
    )
    graph_text = "\n".join(
        f"  Graph #{idx+1}: {i['intention']} (PageRank: {i['pagerank']}, edges: {i['edge_count']})"
        for idx, i in enumerate(graph_intentions)
    )

    prompt = f"""Compare two sets of "top 3 intentions" identified for the same character (Scarlett O'Hara from Gone with the Wind).

## Set A: LLM reading raw dialogue
{llm_text}

## Set B: Graph structure (PageRank ranking)
{graph_text}

## Task
For each intention in Set A, check if it has a semantic match in Set B (same core meaning, even if worded differently).
For each intention in Set B, check if it has a semantic match in Set A.

Respond with JSON:
{{
  "a_to_b_matches": [
    {{"a_rank": 1, "b_match": <1-3 or null>, "match_quality": "exact|partial|none", "reason": "brief"}}
  ],
  "b_to_a_matches": [
    {{"b_rank": 1, "a_match": <1-3 or null>, "match_quality": "exact|partial|none", "reason": "brief"}}
  ],
  "precision_a_in_b": <float, how many A items found in B / 3>,
  "recall_b_in_a": <float, how many B items found in A / 3>,
  "overlap_count": <int, total unique matched pairs>,
  "analysis": "<2-3 sentences: what did the graph find that LLM missed, or vice versa?>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
    start = raw.find("{")
    end = raw.rfind("}")
    return json.loads(raw[start:end + 1])


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    kwargs["timeout"] = 60.0
    client = anthropic.Anthropic(**kwargs)

    # Load data
    lines = load_scarlett_lines()
    graph = SoulGraph.load(Path("results/scarlett_soul.json"))

    print(f"Scarlett: {len(lines)} dialogue lines, graph: {len(graph.items)} items, {len(graph.edges)} edges")

    # Show intention-tagged items in graph
    intention_items = [item for item in graph.items if "intention" in item.tags]
    action_items = [item for item in graph.items if item.item_type.value == "action"]
    print(f"Items tagged 'intention': {len(intention_items)}")
    print(f"Items with type 'action': {len(action_items)}")

    # Step 1: LLM on raw text
    print(f"\n{'='*60}")
    print("  METHOD A: LLM on Raw Dialogue Text")
    print(f"{'='*60}")
    llm_intentions = llm_find_intentions(client, lines)
    for i in llm_intentions:
        print(f"  #{i['rank']}: {i['intention']}")
        print(f"    Evidence: {i['evidence'][:100]}")

    # Step 2: Graph structure
    print(f"\n{'='*60}")
    print("  METHOD B: Graph Structure (PageRank)")
    print(f"{'='*60}")
    graph_intentions = graph_find_intentions(graph)
    for idx, i in enumerate(graph_intentions):
        print(f"  #{idx+1}: {i['intention']}")
        print(f"    PageRank: {i['pagerank']}  Edges: {i['edge_count']}  Domains: {i['domains']}")

    # Step 3: Compare
    print(f"\n{'='*60}")
    print("  COMPARISON (LLM Judge)")
    print(f"{'='*60}")
    comparison = judge_match(client, llm_intentions, graph_intentions)

    print(f"\n  A→B matches (LLM intentions found in Graph):")
    for m in comparison.get("a_to_b_matches", []):
        print(f"    LLM #{m['a_rank']} → Graph #{m.get('b_match', 'none')}: {m['match_quality']} — {m['reason']}")

    print(f"\n  B→A matches (Graph intentions found in LLM):")
    for m in comparison.get("b_to_a_matches", []):
        print(f"    Graph #{m['b_rank']} → LLM #{m.get('a_match', 'none')}: {m['match_quality']} — {m['reason']}")

    print(f"\n  Precision (LLM in Graph): {comparison.get('precision_a_in_b', 0):.2f}")
    print(f"  Recall (Graph in LLM):    {comparison.get('recall_b_in_a', 0):.2f}")
    print(f"  Overlap: {comparison.get('overlap_count', 0)}/3")
    print(f"\n  Analysis: {comparison.get('analysis', '')}")

    # Save
    result = {
        "llm_intentions": llm_intentions,
        "graph_intentions": graph_intentions,
        "comparison": comparison,
    }
    out = Path("results/scarlett_intention_comparison.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
