"""Convergence experiment: generate extended Scarlett dialogue with repeated themes,
test node growth with current dedup vs aggressive aggregation."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic
from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message


def generate_extended_dialogue(client: anthropic.Anthropic, original_lines: list[str], target_count: int = 200) -> list[str]:
    """Generate more Scarlett dialogue that repeats and deepens existing themes."""
    themes = """Core themes from the original dialogue:
1. Survival and fear of hunger (Tara, taxes, money)
2. Love for Ashley / rejection of Rhett / eventual realization
3. Social status and proving herself to Atlanta society
4. Self-reliance and leadership ("no one was going to save me")
5. Nostalgia for pre-war life vs pragmatic forward-looking attitude
6. Relationship with Bonnie, Mammy, Melanie
7. Using marriage as a strategic tool
8. Moral justification of her actions
9. Avoidance coping ("I'll think about it tomorrow")
10. Pride and competitive drive"""

    batches = []
    batch_size = 50
    num_batches = (target_count + batch_size - 1) // batch_size

    for batch in range(num_batches):
        remaining = min(batch_size, target_count - len(batches))
        prompt = f"""Generate {remaining} dialogue lines spoken by Scarlett O'Hara from Gone with the Wind.

{themes}

Rules:
- Each line is 1-3 sentences, spoken naturally in conversation
- REPEAT and DEEPEN existing themes — a real person returns to the same topics
- Mix: some lines introduce slight variations, most revisit known themes
- Include emotional repetition: fear of poverty, defensiveness about choices, Ashley obsession
- Vary the phrasing — same meaning, different words (this tests dedup)
- Include some near-duplicates of: "I'll never be hungry again", "I don't love you Rhett", "Tomorrow is another day"
- Batch {batch+1}/{num_batches}

Return a JSON array of strings, each string is one dialogue line.
"""
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
                start = raw.find("[")
                end = raw.rfind("]")
                lines = json.loads(raw[start:end + 1])
                batches.extend(lines[:remaining])
                print(f"  Generated batch {batch+1}: {len(lines)} lines (total: {len(batches)})", flush=True)
                break
            except Exception as e:
                print(f"  Batch {batch+1} error: {e}, retrying...", flush=True)
                time.sleep(2 ** attempt)

    return batches


def run_extraction(lines: list[str], api_key: str, label: str) -> list[dict]:
    """Run extraction and record growth curve."""
    detector = Detector(api_key=api_key, model="claude-sonnet-4-20250514")
    conversation: list[Message] = []
    growth = []

    for i, line in enumerate(lines):
        conversation.append(Message(role="speaker", content=line))
        before = len(detector.detected_graph.items)

        try:
            detector.listen_and_detect(conversation)
        except Exception as e:
            print(f"  [{label}] Line {i+1} ERROR: {e}", flush=True)

        after = len(detector.detected_graph.items)
        edges = len(detector.detected_graph.edges)
        new = after - before

        growth.append({
            "line": i + 1,
            "new_items": new,
            "total_items": after,
            "total_edges": edges,
        })

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{label}] Line {i+1}/{len(lines)}: +{new} items (total: {after} items, {edges} edges)", flush=True)

    return growth


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

    # Load original 57 lines
    original = []
    with open("fixtures/gone_with_wind.jsonl", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["speaker"] == "scarlett":
                original.append(data["text"])

    print(f"Original Scarlett lines: {len(original)}")

    # Generate extended dialogue (200 more lines with repeated themes)
    print("\n=== Generating extended dialogue ===")
    extended = generate_extended_dialogue(client, original, target_count=200)

    # Save generated dialogue
    Path("results").mkdir(exist_ok=True)
    Path("results/scarlett_extended_dialogue.json").write_text(
        json.dumps(extended, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Generated {len(extended)} extended lines")

    # Combine: original 57 + extended 200 = ~257 lines
    all_lines = original + extended
    print(f"\nTotal lines for experiment: {len(all_lines)}")

    # Run extraction on all lines
    print("\n=== Running extraction (current dedup threshold 0.82) ===")
    growth = run_extraction(all_lines, api_key, "current")

    # Save results
    result = {
        "original_lines": len(original),
        "extended_lines": len(extended),
        "total_lines": len(all_lines),
        "growth": growth,
        "final_items": growth[-1]["total_items"] if growth else 0,
        "final_edges": growth[-1]["total_edges"] if growth else 0,
    }

    Path("results/convergence_experiment.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"CONVERGENCE EXPERIMENT RESULTS")
    print(f"{'='*60}")
    print(f"Total lines: {len(all_lines)}")
    print(f"Final nodes: {result['final_items']}")
    print(f"Final edges: {result['final_edges']}")

    # Growth rate at different stages
    stages = [
        ("Lines 1-57 (original)", growth[:57]),
        ("Lines 58-107 (extended 1)", growth[57:107]),
        ("Lines 108-157 (extended 2)", growth[107:157]),
        ("Lines 158-207 (extended 3)", growth[157:207]),
        ("Lines 208-257 (extended 4)", growth[207:]),
    ]
    print(f"\nGrowth by stage:")
    for label, chunk in stages:
        if not chunk:
            continue
        new_total = sum(g["new_items"] for g in chunk)
        avg = new_total / len(chunk)
        print(f"  {label}: +{new_total} items ({avg:.1f}/line), total={chunk[-1]['total_items']}")


if __name__ == "__main__":
    main()
