"""Faithfulness audit: for each dialogue line, check what info exists vs what was extracted."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic


def load_dialogue(path: str) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def load_graph(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


_AUDIT_PROMPT = """You are auditing the faithfulness of a soul graph extraction.

Given a dialogue line and the items extracted from it, evaluate:

1. **Recall**: What real information is in this line? List each distinct piece of info.
2. **Extracted**: Which of those pieces were captured by the extracted items?
3. **Missed**: Which pieces were NOT captured?
4. **Hallucinated**: Which extracted items contain info NOT actually in this line?
5. **Score**: Give a faithfulness score 0.0-1.0 (captured / total real info pieces)

## Context
Speaker: {speaker}
Scene: {scene}
Full conversation context (preceding lines):
{context}

## This Line
"{line}"

## Items Extracted From This Line
{items}

Respond with JSON only:
{{
  "real_info": ["list of distinct real info pieces in this line"],
  "captured": ["list of info pieces correctly captured"],
  "missed": ["list of info pieces missed"],
  "hallucinated": ["list of extracted items that go beyond what the line says"],
  "faithfulness": <float 0.0-1.0>,
  "notes": "<brief observation>"
}}
"""


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    dialogue = load_dialogue("fixtures/gone_with_wind.jsonl")
    graph = load_graph("results/scarlett_soul.json")

    # Build item lookup by approximate creation order
    items = graph["items"]

    # We need to map items to the dialogue line they were extracted from.
    # Items are created in order, so we can use creation timestamps.
    # Group items by created_at timestamp clusters.
    from datetime import datetime

    items_by_time: list[tuple[datetime, dict]] = []
    for item in items:
        ts = datetime.fromisoformat(item["created_at"].replace("Z", "+00:00"))
        items_by_time.append((ts, item))
    items_by_time.sort(key=lambda x: x[0])

    # Find Scarlett's lines
    scarlett_lines = [(i, line) for i, line in enumerate(dialogue) if line["speaker"] == "scarlett"]

    # Cluster items into groups by time gaps (>2s gap = new extraction)
    clusters: list[list[dict]] = []
    current_cluster: list[dict] = []
    prev_ts = None
    for ts, item in items_by_time:
        if prev_ts is not None and (ts - prev_ts).total_seconds() > 2.0:
            if current_cluster:
                clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(item)
        prev_ts = ts
    if current_cluster:
        clusters.append(current_cluster)

    print(f"Scarlett lines: {len(scarlett_lines)}")
    print(f"Item clusters: {len(clusters)}")
    print(f"Total items: {len(items)}")

    # Match clusters to lines (1:1 in order)
    n = min(len(scarlett_lines), len(clusters))

    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    kwargs["timeout"] = 60.0
    client = anthropic.Anthropic(**kwargs)

    all_scores = []

    # Audit a sample: first 10, middle 5, last 5
    sample_indices = list(range(min(10, n)))
    if n > 20:
        mid = n // 2
        sample_indices += list(range(mid - 2, mid + 3))
        sample_indices += list(range(n - 5, n))
    sample_indices = sorted(set(i for i in sample_indices if i < n))

    print(f"\nAuditing {len(sample_indices)} lines out of {n}\n")

    for idx in sample_indices:
        line_idx, line = scarlett_lines[idx]
        cluster = clusters[idx]

        # Build context (preceding lines)
        context_lines = dialogue[max(0, line_idx - 4):line_idx]
        context = "\n".join(f"  [{l['speaker']}]: {l['text']}" for l in context_lines)

        items_text = "\n".join(
            f"  - {item['id']}: {item['text']} (domains: {item['domains']}, tags: {item.get('tags', [])})"
            for item in cluster
        )

        prompt = _AUDIT_PROMPT.format(
            speaker=line["speaker"],
            scene=line["scene"],
            context=context or "(start of conversation)",
            line=line["text"],
            items=items_text or "(no items extracted)",
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            # Parse JSON
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            start = raw.find("{")
            end = raw.rfind("}")
            if start >= 0 and end >= 0:
                data = json.loads(raw[start:end + 1])
            else:
                data = {"faithfulness": 0.0, "notes": "parse error"}
        except Exception as e:
            data = {"faithfulness": 0.0, "notes": f"error: {e}"}

        score = data.get("faithfulness", 0.0)
        all_scores.append(score)

        print(f"[{idx+1}/{n}] Line {line_idx+1}: \"{line['text'][:60]}...\"")
        print(f"  Extracted: {len(cluster)} items")
        print(f"  Real info: {data.get('real_info', [])}")
        print(f"  Missed:    {data.get('missed', [])}")
        print(f"  Halluc:    {data.get('hallucinated', [])}")
        print(f"  Score:     {score:.2f}")
        print(f"  Notes:     {data.get('notes', '')}")
        print(flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"FAITHFULNESS AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"Lines audited: {len(all_scores)}")
    if all_scores:
        mean = sum(all_scores) / len(all_scores)
        print(f"Mean faithfulness: {mean:.3f}")
        print(f"Min: {min(all_scores):.2f}  Max: {max(all_scores):.2f}")
        perfect = sum(1 for s in all_scores if s >= 0.95)
        print(f"Perfect (>=0.95): {perfect}/{len(all_scores)}")
        low = sum(1 for s in all_scores if s < 0.7)
        print(f"Low (<0.70): {low}/{len(all_scores)}")


if __name__ == "__main__":
    main()
