"""Extract Scarlett's graph and save per-line item mapping for faithfulness audit."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message


def load_dialogue(path: str) -> list[dict]:
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))
    return lines


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    dialogue = load_dialogue("fixtures/gone_with_wind.jsonl")
    target = "scarlett"

    detector = Detector(api_key=api_key, model="claude-sonnet-4-20250514")
    conversation: list[Message] = []

    # mapping: line_index -> list of item IDs added by that line
    mapping: list[dict] = []

    for i, line in enumerate(dialogue):
        role = "speaker" if line["speaker"] == target else "detector"
        conversation.append(Message(role=role, content=line["text"]))

        if line["speaker"] == target:
            before_ids = {item.id for item in detector.detected_graph.items}
            print(f"[{i+1}/{len(dialogue)}] {line['text'][:70]}...", flush=True)

            try:
                detector.listen_and_detect(conversation)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)

            after_ids = {item.id for item in detector.detected_graph.items}
            new_ids = sorted(after_ids - before_ids)

            # Get the actual new items
            id_to_item = {item.id: item for item in detector.detected_graph.items}
            new_items = [
                {"id": nid, "text": id_to_item[nid].text, "domains": id_to_item[nid].domains, "tags": id_to_item[nid].tags}
                for nid in new_ids
            ]

            mapping.append({
                "line_index": i,
                "text": line["text"],
                "scene": line["scene"],
                "new_item_count": len(new_ids),
                "new_items": new_items,
                "total_items": len(detector.detected_graph.items),
                "total_edges": len(detector.detected_graph.edges),
            })

            print(f"  +{len(new_ids)} items (total: {len(detector.detected_graph.items)})", flush=True)
            for ni in new_items:
                print(f"    {ni['id']}: {ni['text'][:60]}", flush=True)

    # Save
    out_path = Path("results/scarlett_extraction_mapping.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nMapping saved to {out_path}")
    print(f"Total lines: {len(mapping)}, Total items: {len(detector.detected_graph.items)}")

    # Quick stats
    zero_lines = sum(1 for m in mapping if m["new_item_count"] == 0)
    print(f"Lines with 0 new items: {zero_lines}")
    avg_items = sum(m["new_item_count"] for m in mapping) / len(mapping)
    print(f"Avg items per line: {avg_items:.1f}")


if __name__ == "__main__":
    main()
