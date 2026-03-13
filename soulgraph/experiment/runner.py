"""Experiment runner: Speaker <-> Detector conversation loop + evaluation."""
from __future__ import annotations

from soulgraph.comparator.embedding import EmbeddingMatcher
from soulgraph.comparator.ranking import RankingComparator
from soulgraph.comparator.semantic import SemanticMatcher
from soulgraph.comparator.structural import GraphComparator
from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import ExperimentResult, Message
from soulgraph.experiment.speaker import Speaker
from soulgraph.graph.models import SoulGraph


class ExperimentRunner:
    def __init__(
        self,
        api_key: str,
        speaker_model: str = "claude-sonnet-4-20250514",
        detector_model: str = "claude-sonnet-4-20250514",
        matcher_model: str = "claude-haiku-4-5-20251001",
    ):
        self._api_key = api_key
        self._speaker_model = speaker_model
        self._detector_model = detector_model
        self._matcher_model = matcher_model

    def run(
        self,
        ground_truth: SoulGraph,
        max_turns: int = 20,
        hub_top_k: int = 5,
        verbose: bool = True,
    ) -> ExperimentResult:
        speaker = Speaker(
            soul_graph=ground_truth,
            api_key=self._api_key,
            model=self._speaker_model,
        )
        detector = Detector(api_key=self._api_key, model=self._detector_model)
        conversation: list[Message] = []

        if verbose:
            print(f"Starting experiment: {len(ground_truth.items)} GT items, {max_turns} turns")

        question = detector.ask_next_question(conversation)

        for turn in range(max_turns):
            if verbose:
                print(f"\n--- Turn {turn + 1}/{max_turns} ---")
                print(f"Detector asks: {question[:80]}...")

            response = speaker.respond(question, conversation)
            conversation.append(Message(role="speaker", content=response))

            if verbose:
                print(f"Speaker says: {response[:80]}...")

            detector.listen_and_detect(conversation)

            if verbose:
                print(f"Detected so far: {len(detector.detected_graph.items)} items, {len(detector.detected_graph.edges)} edges")

            question = detector.ask_next_question(conversation)
            conversation.append(Message(role="detector", content=question))

        if verbose:
            print(f"\nComparing graphs...")

        # V3: Embedding-based comparison (deterministic)
        emb_matcher = EmbeddingMatcher()
        emb_scores = emb_matcher.compute_similarity(
            ground_truth, detector.detected_graph, hub_top_k=hub_top_k
        )

        if verbose:
            print(f"\n--- Embedding Comparison (V3) ---")
            print(f"Node Recall:     {emb_scores['node_recall']:.3f}  (matched {emb_scores['matched_nodes']}/{emb_scores['gt_nodes']})")
            print(f"Node Precision:  {emb_scores['node_precision']:.3f}")
            print(f"Hub Recall:      {emb_scores['hub_recall']:.3f}")
            print(f"Triple Recall:   {emb_scores['triple_recall']:.3f}")
            print(f"Triple Precision:{emb_scores['triple_precision']:.3f}")
            print(f"Triple F1:       {emb_scores['triple_f1']:.3f}")
            print(f"Overall (V3):    {emb_scores['overall']:.3f}")

        # V4: Ranking-based comparison
        ranking_comp = RankingComparator()
        ranking_scores = ranking_comp.compare(ground_truth, detector.detected_graph)

        if verbose:
            print(f"\n--- Ranking Comparison (V4) ---")
            print(f"Rank Correlation: {ranking_scores['rank_correlation']:.3f}")
            print(f"Domain NDCG:      {ranking_scores['domain_ndcg']:.3f}")
            print(f"Absorption Rate:  {ranking_scores['absorption_rate']:.3f}")
            print(f"Intention Recall: {ranking_scores['intention_recall']:.3f}")
            print(f"Overall (V4):     {ranking_scores['overall']:.3f}")

        # Also run legacy comparison for backward compat
        matcher = SemanticMatcher(api_key=self._api_key, model=self._matcher_model)
        comparator = GraphComparator(matcher=matcher)
        similarity = comparator.compare(
            ground_truth, detector.detected_graph, hub_top_k=hub_top_k
        )

        result = ExperimentResult(
            conversation=conversation,
            ground_truth=ground_truth,
            detected_graph=detector.detected_graph,
            similarity=similarity,
            turns=max_turns,
        )
        # Attach V3 scores as extra data
        result.embedding_scores = emb_scores  # type: ignore[attr-defined]
        result.ranking_scores = ranking_scores
        return result

    def run_multi(
        self,
        ground_truth: SoulGraph,
        max_turns: int = 20,
        hub_top_k: int = 5,
        num_runs: int = 3,
        verbose: bool = True,
    ) -> dict:
        """Run multiple experiments and report mean±std for all metrics."""
        import numpy as np

        all_emb_scores: list[dict] = []
        all_rank_scores: list[dict] = []
        for i in range(num_runs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"RUN {i + 1}/{num_runs}")
                print(f"{'='*60}")
            result = self.run(ground_truth, max_turns=max_turns, hub_top_k=hub_top_k, verbose=verbose)
            all_emb_scores.append(result.embedding_scores)  # type: ignore[attr-defined]
            if result.ranking_scores is not None:
                all_rank_scores.append(result.ranking_scores)

        # Aggregate V3 (embedding) metrics
        emb_metrics = ["node_recall", "node_precision", "hub_recall", "triple_recall", "triple_precision", "triple_f1", "overall"]
        summary: dict = {}
        for m in emb_metrics:
            values = [s[m] for s in all_emb_scores]
            summary[m] = {"mean": round(float(np.mean(values)), 3), "std": round(float(np.std(values)), 3)}

        # Aggregate V4 (ranking) metrics
        rank_metrics = ["rank_correlation", "domain_ndcg", "absorption_rate", "intention_recall"]
        if all_rank_scores:
            for m in rank_metrics:
                values = [s[m] for s in all_rank_scores]
                summary[f"v4_{m}"] = {"mean": round(float(np.mean(values)), 3), "std": round(float(np.std(values)), 3)}
            v4_overall = [s["overall"] for s in all_rank_scores]
            summary["v4_overall"] = {"mean": round(float(np.mean(v4_overall)), 3), "std": round(float(np.std(v4_overall)), 3)}

        summary["num_runs"] = num_runs
        summary["raw_scores"] = all_emb_scores
        summary["raw_ranking_scores"] = all_rank_scores

        if verbose:
            print(f"\n{'='*60}")
            print(f"MULTI-RUN SUMMARY ({num_runs} runs)")
            print(f"{'='*60}")
            print(f"  --- V3 (Embedding) ---")
            for m in emb_metrics:
                print(f"  {m:20s}: {summary[m]['mean']:.3f} ± {summary[m]['std']:.3f}")
            if all_rank_scores:
                print(f"  --- V4 (Ranking) ---")
                for m in rank_metrics:
                    key = f"v4_{m}"
                    print(f"  {m:20s}: {summary[key]['mean']:.3f} ± {summary[key]['std']:.3f}")
                print(f"  {'v4_overall':20s}: {summary['v4_overall']['mean']:.3f} ± {summary['v4_overall']['std']:.3f}")

        return summary

    def run_multi_session(
        self,
        ground_truth: SoulGraph,
        session_configs: list[dict],
        hub_top_k: int = 5,
        verbose: bool = True,
    ) -> dict:
        """Run multiple sessions with persistent detector graph."""
        detector = Detector(api_key=self._api_key, model=self._detector_model, session_number=0)
        session_scores: list[dict] = []
        ranking_comp = RankingComparator()

        for si, config in enumerate(session_configs):
            turns = config.get("turns", 10)
            topic_hints = config.get("topic_hints", [])
            detector.session_number = si + 1

            speaker = Speaker(
                soul_graph=ground_truth,
                api_key=self._api_key,
                model=self._speaker_model,
                topic_hints=topic_hints,
            )
            conversation: list[Message] = []

            if verbose:
                print(f"\n{'='*60}")
                print(f"SESSION {si + 1}/{len(session_configs)} — topics: {topic_hints}")
                print(f"{'='*60}")
                print(f"Graph state: {len(detector.detected_graph.items)} items, {len(detector.detected_graph.edges)} edges")

            question = detector.ask_next_question(conversation)

            for turn in range(turns):
                if verbose:
                    print(f"\n--- Session {si + 1}, Turn {turn + 1}/{turns} ---")
                    print(f"Detector asks: {question[:80]}...")

                response = speaker.respond(question, conversation)
                conversation.append(Message(role="speaker", content=response))

                if verbose:
                    print(f"Speaker says: {response[:80]}...")

                detector.listen_and_detect(conversation)

                if verbose:
                    print(f"Detected so far: {len(detector.detected_graph.items)} items, {len(detector.detected_graph.edges)} edges")

                question = detector.ask_next_question(conversation)
                conversation.append(Message(role="detector", content=question))

            # Evaluate after this session
            scores = ranking_comp.compare(ground_truth, detector.detected_graph)
            session_scores.append(scores)

            if verbose:
                print(f"\n--- Session {si + 1} Results ---")
                print(f"Rank Correlation: {scores['rank_correlation']:.3f}")
                print(f"Domain NDCG:      {scores['domain_ndcg']:.3f}")
                print(f"Absorption Rate:  {scores['absorption_rate']:.3f}")
                print(f"Intention Recall: {scores['intention_recall']:.3f}")
                print(f"Overall (V4):     {scores['overall']:.3f}")

        # Cross-session metrics
        rank_improvement = (
            session_scores[-1]["rank_correlation"] - session_scores[0]["rank_correlation"]
            if len(session_scores) >= 2 else 0.0
        )

        result = {
            "session_scores": session_scores,
            "rank_improvement": round(rank_improvement, 3),
            "final_scores": session_scores[-1] if session_scores else {},
            "num_sessions": len(session_configs),
            "turns_per_session": [c.get("turns", 10) for c in session_configs],
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"MULTI-SESSION SUMMARY")
            print(f"{'='*60}")
            for si, scores in enumerate(session_scores):
                print(f"  Session {si + 1}: rank_corr={scores['rank_correlation']:.3f}  absorption={scores['absorption_rate']:.3f}  overall={scores['overall']:.3f}")
            print(f"  Rank Improvement: {rank_improvement:+.3f}")

        return result
