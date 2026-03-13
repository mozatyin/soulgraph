from soulgraph.experiment.models import Message, ExperimentResult
from soulgraph.graph.models import SoulGraph
from soulgraph.comparator.models import GraphSimilarity, HubRecall


class TestExperimentModels:
    def test_message(self):
        msg = Message(role="speaker", content="我最近在考虑买车")
        assert msg.role == "speaker"

    def test_experiment_result(self):
        result = ExperimentResult(
            conversation=[
                Message(role="detector", content="你最近在想什么？"),
                Message(role="speaker", content="想买车"),
            ],
            ground_truth=SoulGraph(owner_id="gt"),
            detected_graph=SoulGraph(owner_id="det"),
            similarity=GraphSimilarity(
                hub_recall=HubRecall(ground_truth_hubs=[], detected_hubs=[], recall=0.0),
                local_similarities=[],
            ),
            turns=1,
        )
        assert result.turns == 1
        assert len(result.conversation) == 2
