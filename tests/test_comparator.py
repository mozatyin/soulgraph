from soulgraph.comparator.models import HubRecall, LocalStructureSimilarity, GraphSimilarity


class TestComparatorModels:
    def test_hub_recall(self):
        hr = HubRecall(ground_truth_hubs=["si_001", "si_002", "si_003"],
                       detected_hubs=["si_001", "si_003"], recall=2/3)
        assert hr.recall == 2/3

    def test_local_structure_similarity(self):
        lss = LocalStructureSimilarity(hub_id="si_001", neighbor_recall=0.8,
                                        edge_type_accuracy=0.6, combined_score=0.7)
        assert lss.combined_score == 0.7

    def test_graph_similarity_score(self):
        hr = HubRecall(ground_truth_hubs=["a", "b"], detected_hubs=["a"], recall=0.5)
        lss = LocalStructureSimilarity(hub_id="a", neighbor_recall=0.8,
                                        edge_type_accuracy=0.6, combined_score=0.7)
        gs = GraphSimilarity(hub_recall=hr, local_similarities=[lss])
        # 0.5 * 0.4 + 0.7 * 0.6 = 0.62
        assert abs(gs.overall_score - 0.62) < 0.01

    def test_graph_similarity_no_local(self):
        hr = HubRecall(ground_truth_hubs=["a"], detected_hubs=[], recall=0.0)
        gs = GraphSimilarity(hub_recall=hr, local_similarities=[])
        assert gs.overall_score == 0.0
