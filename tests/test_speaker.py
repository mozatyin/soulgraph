from unittest.mock import MagicMock

from soulgraph.experiment.speaker import Speaker
from soulgraph.graph.models import SoulEdge, SoulGraph, SoulItem


class TestSpeaker:
    def _make_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family"]))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase"]))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        return g

    def test_respond_returns_string(self):
        graph = self._make_graph()
        speaker = Speaker(soul_graph=graph, api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="最近在想给家里换辆车，想要SUV。")]
        speaker._client = MagicMock()
        speaker._client.messages.create.return_value = mock_response
        result = speaker.respond("你最近在想什么？", conversation=[])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_disclosed_tracking(self):
        graph = self._make_graph()
        speaker = Speaker(soul_graph=graph, api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"response": "想买车", "disclosed_ids": ["si_001"]}')
        ]
        speaker._client = MagicMock()
        speaker._client.messages.create.return_value = mock_response
        speaker.respond("你最近在想什么？", conversation=[])
        assert "si_001" in speaker.disclosed

    def test_system_prompt_contains_soul_graph(self):
        graph = self._make_graph()
        speaker = Speaker(soul_graph=graph, api_key="fake")
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(text='{"response": "test", "disclosed_ids": []}')
        ]
        speaker._client = MagicMock()
        speaker._client.messages.create.return_value = mock_response
        speaker.respond("hello", conversation=[])
        call_kwargs = speaker._client.messages.create.call_args
        system_prompt = call_kwargs.kwargs.get("system", "") or call_kwargs[1].get("system", "")
        assert "重视家庭" in system_prompt
        assert "想买SUV" in system_prompt
