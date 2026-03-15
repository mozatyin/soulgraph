"""Scarlett Intention Evolution Experiment.

Feeds Scarlett's dialogue phase-by-phase through DualSoul, snapshots
after each phase, then compares detected intentions vs ground truth.

Usage:
    export ANTHROPIC_API_KEY=sk-or-...
    .venv/bin/python scripts/scarlett_intention_experiment.py
"""
import json
import os
import sys
import traceback
from pathlib import Path
from collections import defaultdict

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

sys.path.insert(0, str(Path(__file__).parent.parent))

from soulgraph.dual_soul import DualSoul
from soulgraph.graph.models import SoulGraph, SoulItem

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "scarlett_full.jsonl"
GROUND_TRUTH_PATH = Path(__file__).parent.parent / "fixtures" / "scarlett_intentions.json"
RESULTS_DIR = Path(__file__).parent.parent / "results" / "scarlett_intention_experiment"


def load_dialogue() -> dict[int, list[dict]]:
    """Load dialogue grouped by chapter_phase."""
    phases: dict[int, list[dict]] = defaultdict(list)
    with open(FIXTURE_PATH) as f:
        for line in f:
            data = json.loads(line.strip())
            phase = data.get("chapter_phase", 1)
            phases[phase].append(data)
    return dict(sorted(phases.items()))


def load_ground_truth() -> dict:
    """Load the ground-truth intention graph."""
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def extract_intentions(graph: SoulGraph) -> list[dict]:
    """Extract intention-like items from a SoulGraph."""
    intentions = []
    for item in graph.items:
        # Intention indicators: item_type, tags, or text patterns
        is_intention = (
            item.item_type.value in ("intention", "belief", "preference")
            or any(t in item.tags for t in ["intention", "belief", "desire", "fear", "strategy", "value"])
            or any(kw in item.text.lower() for kw in [
                "want", "need", "must", "will", "desire", "fear",
                "believe", "love", "hate", "refuse", "determined",
                "想要", "必须", "害怕", "相信", "决心",
            ])
        )
        intentions.append({
            "id": item.id,
            "text": item.text,
            "domains": item.domains,
            "confidence": item.confidence,
            "mention_count": item.mention_count,
            "is_intention": is_intention,
            "tags": item.tags,
        })
    return intentions


def compare_phase(
    phase_num: int,
    gt_phase: dict,
    surface_intentions: list[dict],
    deep_intentions: list[dict],
    client,
    model: str,
) -> dict:
    """Use LLM to compare detected vs ground-truth intentions for a phase."""
    gt_items = gt_phase.get("intentions", [])
    gt_text = "\n".join(
        f"- [{i['type']}] {i['text']} (strength: {i['strength']})"
        for i in gt_items
    )

    surface_text = "\n".join(
        f"- {i['text']} (conf: {i['confidence']:.2f}, mentions: {i['mention_count']})"
        for i in surface_intentions[:20]
    ) or "(empty)"

    deep_text = "\n".join(
        f"- {i['text']} (conf: {i['confidence']:.2f}, mentions: {i['mention_count']})"
        for i in deep_intentions[:20]
    ) or "(empty)"

    prompt = f"""Compare the AI's detected soul graph against the ground-truth intentions for Scarlett O'Hara.

## Phase: {gt_phase.get('name', f'Phase {phase_num}')}
{gt_phase.get('description', '')}

## Ground Truth Intentions
{gt_text}

## AI Detected — Surface Soul (recent)
{surface_text}

## AI Detected — Deep Soul (long-term)
{deep_text}

## Evaluation Tasks
1. For each ground-truth intention, did the AI detect it? Score 0-1 for each.
2. Did the AI detect valid intentions NOT in ground truth? List them.
3. Rate the AI's overall understanding of Scarlett at this phase (0-1).
4. What key intentions did the AI miss?
5. What did the AI capture that shows genuine "soul understanding"?

Return JSON:
{{
  "phase": {phase_num},
  "gt_coverage": [
    {{"gt_intention": "...", "detected": true/false, "score": 0.0-1.0, "matched_detected": "..." or null}}
  ],
  "novel_detections": ["..."],
  "overall_understanding": 0.0-1.0,
  "missed": ["..."],
  "soul_insights": ["..."],
  "commentary": "2-3 sentences on how well the AI understands Scarlett at this point"
}}"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= 0:
            return json.loads(raw[start:end + 1])
    except Exception as e:
        print(f"  [warn] LLM comparison failed: {e}")

    return {"phase": phase_num, "overall_understanding": 0.0, "commentary": "comparison failed"}


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    if not FIXTURE_PATH.exists():
        print(f"Error: fixture not found: {FIXTURE_PATH}", file=sys.stderr)
        print("Run the dialogue generation first.", file=sys.stderr)
        sys.exit(1)

    if not GROUND_TRUTH_PATH.exists():
        print(f"Error: ground truth not found: {GROUND_TRUTH_PATH}", file=sys.stderr)
        sys.exit(1)

    import anthropic
    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    client = anthropic.Anthropic(**kwargs)
    model = "claude-sonnet-4-20250514"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    dialogue_by_phase = load_dialogue()
    gt = load_ground_truth()
    gt_phases = {p["phase"]: p for p in gt.get("phases", [])}

    print(f"Loaded {sum(len(v) for v in dialogue_by_phase.values())} dialogue lines across {len(dialogue_by_phase)} phases")
    print(f"Ground truth: {len(gt_phases)} phases")
    print()

    # Initialize DualSoul with smaller cycle for more frequent consolidation
    ds = DualSoul(api_key=api_key, deep_cycle=25, max_surface_nodes=80, carry_forward_k=8)

    phase_results = []

    for phase_num, lines in dialogue_by_phase.items():
        phase_name = gt_phases.get(phase_num, {}).get("name", f"Phase {phase_num}")
        print(f"{'='*60}")
        print(f"Phase {phase_num}: {phase_name} ({len(lines)} lines)")
        print(f"{'='*60}")

        # Ingest all lines in this phase
        for i, line_data in enumerate(lines):
            text = line_data["text"]
            try:
                ds.ingest(text)
                if (i + 1) % 5 == 0 or i == len(lines) - 1:
                    print(f"  ingested {i+1}/{len(lines)}: Surface={len(ds.surface.items)} Deep={len(ds.deep.items)}")
            except Exception as e:
                print(f"  [warn] Line {i+1} failed: {e}")
                traceback.print_exc()
                continue

        # Snapshot
        s = ds.stats
        print(f"  Surface: {s['surface_items']} items, {s['surface_edges']} edges")
        print(f"  Deep:    {s['deep_items']} items, {s['deep_edges']} edges")
        print(f"  Consolidations: {s['consolidation_count']}")

        # Extract intentions from both graphs
        surface_intentions = extract_intentions(ds.surface)
        deep_intentions = extract_intentions(ds.deep)

        surface_intention_count = sum(1 for i in surface_intentions if i["is_intention"])
        deep_intention_count = sum(1 for i in deep_intentions if i["is_intention"])
        print(f"  Surface intentions: {surface_intention_count}/{len(surface_intentions)}")
        print(f"  Deep intentions:    {deep_intention_count}/{len(deep_intentions)}")

        # Compare with ground truth if available
        gt_phase = gt_phases.get(phase_num)
        if gt_phase:
            print(f"  Comparing with ground truth...")
            comparison = compare_phase(
                phase_num, gt_phase,
                surface_intentions, deep_intentions,
                client, model,
            )
            print(f"  Overall understanding: {comparison.get('overall_understanding', '?')}")
            print(f"  {comparison.get('commentary', '')}")
        else:
            comparison = {"phase": phase_num, "commentary": "no ground truth for this phase"}

        # Save phase snapshot
        phase_snapshot = {
            "phase": phase_num,
            "name": phase_name,
            "lines_ingested": len(lines),
            "stats": s,
            "surface_intentions": surface_intentions,
            "deep_intentions": deep_intentions,
            "comparison": comparison,
        }
        phase_results.append(phase_snapshot)

        snapshot_path = RESULTS_DIR / f"phase_{phase_num:02d}_{phase_name.replace(' ', '_')}.json"
        snapshot_path.write_text(json.dumps(phase_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
        print()

    # Save full DualSoul state
    ds.save(str(RESULTS_DIR / "scarlett_dual_soul_final.json"))

    # Generate evolution summary
    print(f"\n{'='*60}")
    print("INTENTION EVOLUTION SUMMARY")
    print(f"{'='*60}")

    evolution = {
        "total_phases": len(phase_results),
        "total_lines": sum(len(v) for v in dialogue_by_phase.values()),
        "final_stats": ds.stats,
        "phase_scores": [],
    }

    for pr in phase_results:
        comp = pr.get("comparison", {})
        score = comp.get("overall_understanding", "N/A")
        name = pr["name"]
        print(f"  Phase {pr['phase']}: {name:30s} | Understanding: {score}")
        evolution["phase_scores"].append({
            "phase": pr["phase"],
            "name": name,
            "understanding": score,
            "surface_items": pr["stats"]["surface_items"],
            "deep_items": pr["stats"]["deep_items"],
        })

    # Final LLM summary of the entire evolution
    print("\nGenerating final soul evolution analysis...")
    phase_summaries = "\n".join(
        f"Phase {pr['phase']} ({pr['name']}): understanding={pr['comparison'].get('overall_understanding', '?')}, "
        f"commentary={pr['comparison'].get('commentary', 'N/A')}"
        for pr in phase_results
    )
    deep_items_text = "\n".join(
        f"- {i.text} (mentions: {i.mention_count}, conf: {i.confidence:.2f})"
        for i in sorted(ds.deep.items, key=lambda x: x.mention_count, reverse=True)[:30]
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            messages=[{"role": "user", "content": f"""Analyze Scarlett O'Hara's soul evolution as detected by an AI system that read her dialogue phase by phase.

## Phase-by-phase scores
{phase_summaries}

## Top 30 Deep Soul items (long-term, compressed)
{deep_items_text}

## Tasks
1. Trace how the AI's understanding of Scarlett evolved across phases
2. Identify which of Scarlett's core transformations the AI captured vs missed
3. Score the overall soul detection quality (0-1)
4. What does the Deep Soul graph reveal about Scarlett that even a casual reader might miss?
5. 用中文写一段总结：这个 AI 系统对斯嘉丽灵魂的理解程度如何？

Return JSON with "evolution_analysis", "transformations_captured", "transformations_missed", "overall_score", "deep_insights", "chinese_summary"."""}],
        )
        raw = response.content[0].text
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end >= 0:
            evolution["final_analysis"] = json.loads(raw[start:end + 1])
            print(f"\n  Overall soul detection score: {evolution['final_analysis'].get('overall_score', '?')}")
            print(f"\n  {evolution['final_analysis'].get('chinese_summary', '')}")
    except Exception as e:
        print(f"  [warn] Final analysis failed: {e}")

    # Save evolution summary
    summary_path = RESULTS_DIR / "evolution_summary.json"
    summary_path.write_text(json.dumps(evolution, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL: {e}")
        traceback.print_exc()
        sys.exit(1)
