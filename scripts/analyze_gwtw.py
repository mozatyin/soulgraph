"""Analyze Gone with the Wind dialogue — build separate soul graphs for Scarlett and Rhett."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulGraph


def load_dialogue(path: str) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def build_graph(
    dialogue: list[dict],
    target_speaker: str,
    api_key: str,
) -> tuple[SoulGraph, list[Message]]:
    """Build a soul graph for one speaker from dialogue.

    The Detector sees the full conversation context but only extracts
    from the target speaker's lines.
    """
    detector = Detector(api_key=api_key, model="claude-sonnet-4-20250514")
    conversation: list[Message] = []

    for i, line in enumerate(dialogue):
        role = "speaker" if line["speaker"] == target_speaker else "detector"
        conversation.append(Message(role=role, content=line["text"]))

        # Only extract when the target speaker just spoke
        if line["speaker"] == target_speaker:
            print(f"  [{i+1}/{len(dialogue)}] {target_speaker}: {line['text'][:60]}...")
            try:
                detector.listen_and_detect(conversation)
                g = detector.detected_graph
                print(f"    → {len(g.items)} items, {len(g.edges)} edges")
            except Exception as e:
                print(f"    → ERROR: {e}")

    return detector.detected_graph, conversation


def print_graph_summary(name: str, graph: SoulGraph) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}'s Soul Graph")
    print(f"{'='*60}")
    print(f"  Items: {len(graph.items)}  |  Edges: {len(graph.edges)}")

    # Domains
    domains: dict[str, int] = {}
    for item in graph.items:
        for d in item.domains:
            domains[d] = domains.get(d, 0) + 1
    sorted_domains = sorted(domains.items(), key=lambda x: -x[1])
    print(f"\n  Domains ({len(domains)}):")
    for d, count in sorted_domains[:10]:
        print(f"    {d}: {count} items")

    # Top items by PageRank
    if graph.items:
        pr = graph.pagerank()
        sorted_items = sorted(pr.items(), key=lambda x: -x[1])
        print(f"\n  Top 10 Items (by PageRank):")
        id_to_item = {item.id: item for item in graph.items}
        for item_id, score in sorted_items[:10]:
            item = id_to_item.get(item_id)
            if item:
                print(f"    {score:.4f}  {item.text[:60]}  [{item.item_type.value}] {item.domains}")

    # Edge types
    edge_types: dict[str, int] = {}
    for e in graph.edges:
        edge_types[e.relation] = edge_types.get(e.relation, 0) + 1
    if edge_types:
        print(f"\n  Edge Types:")
        for rel, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            print(f"    {rel}: {count}")


def query_graph(graph: SoulGraph, query: str, api_key: str) -> str:
    """Query a graph using the SoulEngine's query logic."""
    import anthropic
    import time

    if not graph.items:
        return "(empty graph)"

    subgraph = graph.query_subgraph(query, top_k=10)

    nodes_text = "\n".join(
        f"- {item.text} (domains: {', '.join(item.domains)}, "
        f"type: {item.item_type.value})"
        for item in subgraph.items
    )
    id_to_text = {item.id: item.text for item in subgraph.items}
    edges_text = "\n".join(
        f"- \"{id_to_text.get(e.from_id, e.from_id)}\" --{e.relation}--> "
        f"\"{id_to_text.get(e.to_id, e.to_id)}\""
        for e in subgraph.edges
    ) or "(no connections)"

    system = f"""You are analyzing a character's soul graph from Gone with the Wind.

## Retrieved Subgraph

### Nodes
{nodes_text}

### Connections
{edges_text}

## Rules
1. Answer based ONLY on the graph data above.
2. Surface cross-domain connections.
3. Be concise and insightful.
"""

    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    client = anthropic.Anthropic(**kwargs)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": query}],
            )
            return response.content[0].text
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
                continue
            return f"(error: {e})"


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    dialogue_path = Path(__file__).parent.parent / "fixtures" / "gone_with_wind.jsonl"
    dialogue = load_dialogue(str(dialogue_path))
    print(f"Loaded {len(dialogue)} dialogue lines\n")

    # Build Scarlett's graph
    print("=" * 60)
    print("  Building Scarlett's Soul Graph")
    print("=" * 60)
    scarlett_graph, _ = build_graph(dialogue, "scarlett", api_key)

    # Build Rhett's graph
    print("\n" + "=" * 60)
    print("  Building Rhett's Soul Graph")
    print("=" * 60)
    rhett_graph, _ = build_graph(dialogue, "rhett", api_key)

    # Print summaries
    print_graph_summary("Scarlett", scarlett_graph)
    print_graph_summary("Rhett", rhett_graph)

    # Save graphs
    out_dir = Path(__file__).parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    scarlett_path = out_dir / "scarlett_soul.json"
    rhett_path = out_dir / "rhett_soul.json"
    scarlett_path.write_text(scarlett_graph.model_dump_json(indent=2), encoding="utf-8")
    rhett_path.write_text(rhett_graph.model_dump_json(indent=2), encoding="utf-8")
    print(f"\nGraphs saved to {scarlett_path} and {rhett_path}")

    # Query both graphs
    queries = [
        "What is this person's deepest fear?",
        "What drives this person more than anything?",
        "What is this person's relationship with love?",
        "How does this person cope with loss?",
    ]

    print(f"\n{'='*60}")
    print("  QUERIES")
    print(f"{'='*60}")

    for q in queries:
        print(f"\n--- {q} ---")
        print(f"\n  Scarlett:")
        answer = query_graph(scarlett_graph, q, api_key)
        print(f"  {answer}")
        print(f"\n  Rhett:")
        answer = query_graph(rhett_graph, q, api_key)
        print(f"  {answer}")

    # Summary comparison
    print(f"\n{'='*60}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"  Scarlett: {len(scarlett_graph.items)} items, {len(scarlett_graph.edges)} edges")
    print(f"  Rhett:    {len(rhett_graph.items)} items, {len(rhett_graph.edges)} edges")

    s_domains = set()
    for item in scarlett_graph.items:
        s_domains.update(item.domains)
    r_domains = set()
    for item in rhett_graph.items:
        r_domains.update(item.domains)
    shared = s_domains & r_domains
    print(f"\n  Scarlett domains: {sorted(s_domains)}")
    print(f"  Rhett domains:    {sorted(r_domains)}")
    print(f"  Shared domains:   {sorted(shared)}")
    print(f"  Scarlett only:    {sorted(s_domains - r_domains)}")
    print(f"  Rhett only:       {sorted(r_domains - s_domains)}")


if __name__ == "__main__":
    main()
