"""Faithfulness audit v2: uses the extraction mapping for correct alignment."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import anthropic

_AUDIT_PROMPT = """You are auditing the faithfulness of a soul graph extraction from a single dialogue line.

## Task
For this ONE line of dialogue, evaluate what information it contains and whether the extracted items correctly capture it.

## Speaker: {speaker} (Scene: {scene})

## Conversation context (preceding lines):
{context}

## THIS LINE (evaluate only this):
"{line}"

## Items extracted from THIS LINE:
{items}

## Instructions
1. List every distinct piece of information actually present in this line
2. For each extracted item, mark it as: FAITHFUL (directly supported), INFERRED (reasonable but not explicit), or HALLUCINATED (not supported)
3. Note any information in the line that was MISSED entirely

Respond with JSON only:
{{
  "info_in_line": ["list every distinct piece of info in this line"],
  "item_verdicts": [
    {{"id": "si_NNN", "verdict": "FAITHFUL|INFERRED|HALLUCINATED", "reason": "brief"}}
  ],
  "missed_info": ["info in the line not captured by any item"],
  "faithful_count": <int>,
  "inferred_count": <int>,
  "hallucinated_count": <int>,
  "missed_count": <int>,
  "precision": <float 0-1, faithful / (faithful + hallucinated)>,
  "recall": <float 0-1, captured / total info in line>,
  "notes": "<one sentence>"
}}
"""


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    mapping = json.loads(Path("results/scarlett_extraction_mapping.json").read_text())
    dialogue = []
    with open("fixtures/gone_with_wind.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dialogue.append(json.loads(line.strip()))

    kwargs: dict = {"api_key": api_key}
    if api_key.startswith("sk-or-"):
        kwargs["base_url"] = "https://openrouter.ai/api"
    kwargs["timeout"] = 60.0
    client = anthropic.Anthropic(**kwargs)

    # Sample: every 3rd line (covers ~19 lines evenly distributed)
    sample = list(range(0, len(mapping), 3))
    print(f"Auditing {len(sample)} of {len(mapping)} extraction events\n")

    all_results = []

    for idx in sample:
        entry = mapping[idx]
        line_idx = entry["line_index"]
        line_text = entry["text"]
        new_items = entry["new_items"]

        # Context: preceding 4 lines
        context_lines = dialogue[max(0, line_idx - 4):line_idx]
        context = "\n".join(f"  [{l['speaker']}]: {l['text']}" for l in context_lines) or "(start)"

        items_text = "\n".join(
            f"  - {it['id']}: {it['text']} (domains: {it['domains']}, tags: {it.get('tags', [])})"
            for it in new_items
        ) if new_items else "(no items extracted from this line)"

        prompt = _AUDIT_PROMPT.format(
            speaker="scarlett",
            scene=entry["scene"],
            context=context,
            line=line_text,
            items=items_text,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            start = raw.find("{")
            end = raw.rfind("}")
            data = json.loads(raw[start:end + 1]) if start >= 0 else {}
        except Exception as e:
            data = {"notes": f"error: {e}", "precision": 0, "recall": 0}

        precision = data.get("precision", 0)
        recall = data.get("recall", 0)
        faithful = data.get("faithful_count", 0)
        inferred = data.get("inferred_count", 0)
        halluc = data.get("hallucinated_count", 0)
        missed = data.get("missed_count", 0)

        all_results.append({
            "line_index": line_idx,
            "text": line_text[:60],
            "items_extracted": len(new_items),
            "precision": precision,
            "recall": recall,
            "faithful": faithful,
            "inferred": inferred,
            "hallucinated": halluc,
            "missed": missed,
            **data,
        })

        print(f"[{idx+1}/{len(mapping)}] \"{line_text[:55]}...\"")
        print(f"  Items: {len(new_items)} | Faithful: {faithful} | Inferred: {inferred} | Halluc: {halluc} | Missed: {missed}")
        print(f"  Precision: {precision:.2f} | Recall: {recall:.2f}")
        if data.get("missed_info"):
            print(f"  Missed: {data['missed_info']}")
        print(f"  Notes: {data.get('notes', '')}")
        print(flush=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"FAITHFULNESS AUDIT v2 — SUMMARY")
    print(f"{'='*60}")
    n = len(all_results)
    avg_precision = sum(r["precision"] for r in all_results) / n
    avg_recall = sum(r["recall"] for r in all_results) / n
    total_faithful = sum(r["faithful"] for r in all_results)
    total_inferred = sum(r["inferred"] for r in all_results)
    total_halluc = sum(r["hallucinated"] for r in all_results)
    total_missed = sum(r["missed"] for r in all_results)
    total_items = sum(r["items_extracted"] for r in all_results)

    print(f"Lines audited: {n}")
    print(f"Total items in sample: {total_items}")
    print(f"  Faithful:     {total_faithful} ({total_faithful/max(total_items,1)*100:.0f}%)")
    print(f"  Inferred:     {total_inferred} ({total_inferred/max(total_items,1)*100:.0f}%)")
    print(f"  Hallucinated: {total_halluc} ({total_halluc/max(total_items,1)*100:.0f}%)")
    print(f"  Missed info:  {total_missed}")
    print(f"\nMean Precision: {avg_precision:.3f}")
    print(f"Mean Recall:    {avg_recall:.3f}")

    # Save results
    out_path = Path("results/scarlett_faithfulness_audit.json")
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
