"""Experiment runner: Speaker <-> Detector conversation loop + evaluation."""
from __future__ import annotations

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

        matcher = SemanticMatcher(api_key=self._api_key, model=self._matcher_model)
        comparator = GraphComparator(matcher=matcher)
        similarity = comparator.compare(
            ground_truth, detector.detected_graph, hub_top_k=hub_top_k
        )

        return ExperimentResult(
            conversation=conversation,
            ground_truth=ground_truth,
            detected_graph=detector.detected_graph,
            similarity=similarity,
            turns=max_turns,
        )
