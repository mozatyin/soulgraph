"""Experiment runner: Speaker <-> Detector conversation loop + evaluation."""
from __future__ import annotations

from soulgraph.comparator.embedding import EmbeddingMatcher
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

        all_scores: list[dict] = []
        for i in range(num_runs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"RUN {i + 1}/{num_runs}")
                print(f"{'='*60}")
            result = self.run(ground_truth, max_turns=max_turns, hub_top_k=hub_top_k, verbose=verbose)
            all_scores.append(result.embedding_scores)  # type: ignore[attr-defined]

        # Aggregate
        metrics = ["node_recall", "node_precision", "hub_recall", "triple_recall", "triple_precision", "triple_f1", "overall"]
        summary: dict = {}
        for m in metrics:
            values = [s[m] for s in all_scores]
            summary[m] = {"mean": round(float(np.mean(values)), 3), "std": round(float(np.std(values)), 3)}

        summary["num_runs"] = num_runs
        summary["raw_scores"] = all_scores

        if verbose:
            print(f"\n{'='*60}")
            print(f"MULTI-RUN SUMMARY ({num_runs} runs)")
            print(f"{'='*60}")
            for m in metrics:
                print(f"  {m:20s}: {summary[m]['mean']:.3f} ± {summary[m]['std']:.3f}")

        return summary
