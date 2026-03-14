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
    parser.add_argument("--sessions", type=int, default=0, help="Number of sessions for multi-session experiment (reads session configs from fixture)")
    parser.add_argument("--queries", action="store_true", default=False, help="Run query evaluation phase after multi-session (reads queries from fixture)")
    parser.add_argument("--interact", action="store_true", help="Interactive mode: chat to build a soul graph, then query it")
    parser.add_argument("--dual", action="store_true", help="Use DualSoul (Deep + Surface) architecture in interactive mode")
    parser.add_argument("--load", type=str, help="Load a saved graph before entering interactive mode")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set", file=sys.stderr)
        sys.exit(1)

    if args.interact and args.dual:
        _interactive_dual(api_key, load_path=args.load, save_path=args.output)
        return
    if args.interact:
        _interactive(api_key, load_path=args.load, save_path=args.output)
        return

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
        if args.sessions > 0:
            import json as json_mod
            fixture_data = json_mod.loads(gt_path.read_text(encoding="utf-8"))
            session_configs = fixture_data.get("sessions", [])[:args.sessions]
            if not session_configs:
                print("Error: fixture has no 'sessions' field", file=sys.stderr)
                sys.exit(1)
            for sc in session_configs:
                sc["turns"] = args.turns
            queries = None
            if args.queries:
                queries = fixture_data.get("queries", [])
                if not queries:
                    print("Warning: --queries flag set but fixture has no 'queries' field", file=sys.stderr)
                    queries = None
            print(f"Running multi-session experiment ({args.sessions} sessions, {args.turns} turns each) on {gt_path.name}...")
            summary = runner.run_multi_session(gt, session_configs=session_configs, queries=queries)
            if args.output:
                Path(args.output).write_text(json_mod.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"\nSummary saved to {args.output}")
        elif args.runs > 1:
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


def _interactive(api_key: str, load_path: str | None = None, save_path: str | None = None) -> None:
    """Interactive REPL: chat to build a soul graph, then query it."""
    from soulgraph.engine import SoulEngine

    engine = SoulEngine(api_key=api_key)
    if load_path:
        engine.load(load_path)
        g = engine.graph
        print(f"Loaded graph: {len(g.items)} items, {len(g.edges)} edges")

    print("SoulGraph Interactive Mode")
    print("=" * 40)
    print("Commands:")
    print("  (type text)     → ingest as conversation")
    print("  /query <text>   → query the soul graph")
    print("  /graph          → show current graph summary")
    print("  /items          → list all items")
    print("  /save [path]    → save graph to file")
    print("  /quit           → exit")
    print()

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line == "/quit":
            break

        if line == "/graph":
            g = engine.graph
            print(f"\n  {len(g.items)} items, {len(g.edges)} edges")
            if g.items:
                domains: set[str] = set()
                for item in g.items:
                    domains.update(item.domains)
                print(f"  Domains: {', '.join(sorted(domains))}")
            print()
            continue

        if line == "/items":
            g = engine.graph
            if not g.items:
                print("  (empty graph)")
            else:
                for item in g.items:
                    print(f"  {item.id}: {item.text[:60]}  [{item.item_type.value}] domains={item.domains}")
            print()
            continue

        if line.startswith("/save"):
            parts = line.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else (save_path or "soulgraph_saved.json")
            engine.save(path)
            print(f"  Saved to {path}")
            print()
            continue

        if line.startswith("/query "):
            question = line[7:].strip()
            if not question:
                print("  Usage: /query <your question>")
                continue
            if not engine.graph.items:
                print("  Graph is empty — chat first to build it.")
                continue
            print("  thinking...")
            answer = engine.query(question)
            print(f"\n  {answer}\n")
            continue

        # Default: ingest as conversation
        print("  ingesting...")
        engine.ingest(line)
        g = engine.graph
        print(f"  [{len(g.items)} items, {len(g.edges)} edges]\n")

    # Auto-save on exit if output path specified
    if save_path and engine.graph.items:
        engine.save(save_path)
        print(f"Graph saved to {save_path}")


def _interactive_dual(api_key: str, load_path: str | None = None, save_path: str | None = None) -> None:
    """Interactive REPL using DualSoul architecture."""
    from soulgraph.dual_soul import DualSoul

    ds = DualSoul(api_key=api_key)
    if load_path:
        ds.load(load_path)
        s = ds.stats
        print(f"Loaded: Surface={s['surface_items']} items, Deep={s['deep_items']} items")

    print("SoulGraph DualSoul Interactive Mode")
    print("=" * 40)
    print("Commands:")
    print("  (type text)       → ingest as conversation")
    print("  /query <text>     → query both souls")
    print("  /stats            → show Surface + Deep stats")
    print("  /consolidate      → force consolidation (Surface→Deep)")
    print("  /save [path]      → save to file")
    print("  /quit             → exit")
    print()

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue

        if line == "/quit":
            break

        if line == "/stats":
            s = ds.stats
            print(f"\n  Utterances: {s['total_utterances']}, Consolidations: {s['consolidation_count']}")
            print(f"  Surface: {s['surface_items']} items, {s['surface_edges']} edges")
            print(f"  Deep:    {s['deep_items']} items, {s['deep_edges']} edges")
            print(f"  Merge threshold: {ds._adaptive_merge_threshold():.3f}")
            print()
            continue

        if line == "/consolidate":
            print("  consolidating...")
            result = ds.consolidate()
            print(f"  Done: merged={result['merged']}, added={result['added']}, decayed={result['decayed']}")
            print()
            continue

        if line.startswith("/save"):
            parts = line.split(maxsplit=1)
            path = parts[1] if len(parts) > 1 else (save_path or "dual_soul_saved.json")
            ds.save(path)
            print(f"  Saved to {path}")
            print()
            continue

        if line.startswith("/query "):
            question = line[7:].strip()
            if not question:
                print("  Usage: /query <your question>")
                continue
            if not ds.surface.items and not ds.deep.items:
                print("  Both graphs empty — chat first.")
                continue
            print("  thinking...")
            answer = ds.query(question)
            print(f"\n  {answer}\n")
            continue

        # Default: ingest
        print("  ingesting...")
        ds.ingest(line)
        s = ds.stats
        print(f"  [Surface: {s['surface_items']} items | Deep: {s['deep_items']} items]\n")

    if save_path:
        ds.save(save_path)
        print(f"Saved to {save_path}")


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
