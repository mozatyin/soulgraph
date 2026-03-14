"""Test DualSoul with Gone with the Wind dialogue.

Usage:
    export ANTHROPIC_API_KEY=sk-or-...
    .venv/bin/python scripts/dual_soul_gwtw.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from soulgraph.dual_soul import DualSoul


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Load Scarlett's lines
    fixture = Path(__file__).parent.parent / "fixtures" / "gone_with_wind.jsonl"
    lines = []
    with open(fixture) as f:
        for raw in f:
            data = json.loads(raw.strip())
            if data["speaker"] == "scarlett":
                lines.append(data["text"])

    print(f"Scarlett lines: {len(lines)}")

    ds = DualSoul(api_key=api_key, deep_cycle=30, max_surface_nodes=100)

    for i, line in enumerate(lines):
        try:
            ds.ingest(line)
        except Exception as e:
            print(f"  [warn] Line {i+1} failed: {e}")
            continue
        if (i + 1) % 10 == 0:
            s = ds.stats
            print(f"  Line {i+1}: Surface={s['surface_items']} items, Deep={s['deep_items']} items, Consolidations={s['consolidation_count']}")

    print(f"\nFinal stats: {json.dumps(ds.stats, indent=2)}")

    # Force final consolidation if not triggered
    if ds.surface.items:
        print("\nRunning final consolidation...")
        result = ds.consolidate()
        print(f"  merged={result['merged']}, added={result['added']}, decayed={result['decayed']}")
        print(f"  After: {json.dumps(ds.stats, indent=2)}")

    # Query both souls
    queries = [
        "What is Scarlett thinking about right now?",
        "What kind of person is Scarlett at her core?",
        "What drives Scarlett's decisions?",
        "她最深层的恐惧是什么？",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        try:
            answer = ds.query(q)
            print(f"A: {answer}")
        except Exception as e:
            print(f"A: [error] {e}")

    # Save
    out_path = Path(__file__).parent.parent / "results" / "scarlett_dual_soul.json"
    out_path.parent.mkdir(exist_ok=True)
    ds.save(str(out_path))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
