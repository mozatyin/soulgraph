"""CLI for running SoulGraph experiments."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from soulgraph.experiment.runner import ExperimentRunner
from soulgraph.graph.models import SoulGraph


def main() -> None:
    parser = argparse.ArgumentParser(description="SoulGraph — soul detection experiments")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test (3 turns) on car_buyer fixture")
    parser.add_argument("--experiment", type=str, help="Path to a ground truth JSON file to run full experiment")
    parser.add_argument("--turns", type=int, default=20, help="Number of turns (default 20)")
    parser.add_argument("--hubs", type=int, default=5, help="Top-k hubs for comparison")
    parser.add_argument("--output", type=str, help="Path to save result JSON")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for multi-run averaging (default 1)")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    if args.smoke:
        fixture_path = Path(__file__).parent.parent / "fixtures" / "car_buyer.json"
        if not fixture_path.exists():
            print(f"Error: fixture not found at {fixture_path}", file=sys.stderr)
            sys.exit(1)
        gt = SoulGraph.load(fixture_path)
        runner = ExperimentRunner(api_key=api_key)
        print(f"Running smoke test (3 turns) on {fixture_path.name}...")
        result = runner.run(gt, max_turns=3, hub_top_k=3)
        _print_result(result)
        if args.output:
            Path(args.output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
            print(f"\nResult saved to {args.output}")
    elif args.experiment:
        gt_path = Path(args.experiment)
        if not gt_path.exists():
            print(f"Error: file not found: {gt_path}", file=sys.stderr)
            sys.exit(1)
        gt = SoulGraph.load(gt_path)
        runner = ExperimentRunner(api_key=api_key)
        if args.runs > 1:
            print(f"Running multi-run experiment ({args.runs} runs, {args.turns} turns each) on {gt_path.name}...")
            import json
            summary = runner.run_multi(gt, max_turns=args.turns, hub_top_k=args.hubs, num_runs=args.runs)
            if args.output:
                Path(args.output).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"\nSummary saved to {args.output}")
        else:
            print(f"Running experiment ({args.turns} turns) on {gt_path.name}...")
            result = runner.run(gt, max_turns=args.turns, hub_top_k=args.hubs)
            _print_result(result)
            if args.output:
                Path(args.output).write_text(result.model_dump_json(indent=2), encoding="utf-8")
                print(f"\nResult saved to {args.output}")
    else:
        parser.print_help()


def _print_result(result) -> None:
    print(f"\n{'='*60}")
    print(f"Experiment Complete — {result.turns} turns")
    print(f"{'='*60}")
    print(f"\nGround truth: {len(result.ground_truth.items)} items, {len(result.ground_truth.edges)} edges")
    print(f"Detected:     {len(result.detected_graph.items)} items, {len(result.detected_graph.edges)} edges")
    print(f"\n--- Similarity ---")
    print(f"Hub Recall:     {result.similarity.hub_recall.recall:.2f}")
    if result.similarity.local_similarities:
        avg_local = sum(ls.combined_score for ls in result.similarity.local_similarities) / len(result.similarity.local_similarities)
        print(f"Avg Local Sim:  {avg_local:.2f}")
    print(f"Overall Score:  {result.similarity.overall_score:.2f}")
    if result.ranking_scores:
        print(f"\n--- Ranking (V4) ---")
        print(f"Rank Correlation: {result.ranking_scores['rank_correlation']:.3f}")
        print(f"Domain NDCG:      {result.ranking_scores['domain_ndcg']:.3f}")
        print(f"Absorption Rate:  {result.ranking_scores['absorption_rate']:.3f}")
        print(f"Intention Recall: {result.ranking_scores['intention_recall']:.3f}")
        print(f"Overall (V4):     {result.ranking_scores['overall']:.3f}")
    print(f"\n--- Detected Items ---")
    for item in result.detected_graph.items:
        type_tag = f"[{item.item_type.value}]" if hasattr(item, 'item_type') else ""
        print(f"  {item.id}: {item.text[:55]}  {type_tag} conf={item.confidence:.1f}  domains={item.domains}")
    print(f"\n--- Conversation (last 4 messages) ---")
    for msg in result.conversation[-4:]:
        role = "Speaker" if msg.role == "speaker" else "Detector"
        print(f"{role}: {msg.content[:120]}")
