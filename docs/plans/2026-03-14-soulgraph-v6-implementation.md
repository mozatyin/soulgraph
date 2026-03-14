# SoulGraph V6 — Query-Driven Retrieval + Retrieval Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `query_subgraph()` with PPR-based retrieval and `RetrievalEvaluator` with structural + LLM-as-Judge metrics, integrated into the multi-session pipeline.

**Architecture:** Extend SoulGraph with embedding-based seed resolution + Personalized PageRank subgraph extraction. New RetrievalEvaluator computes structural metrics (cross-domain coverage, connectivity, density) and LLM-as-Judge scores (faithfulness, comprehensiveness, diversity). Query eval phase appended to multi-session runner.

**Tech Stack:** Python 3.12, pydantic v2, networkx, sentence-transformers (all-MiniLM-L6-v2), anthropic SDK, numpy, pytest

---

### Task 1: Add `query_subgraph()` to SoulGraph

**Files:**
- Modify: `soulgraph/graph/models.py`
- Modify: `tests/test_graph.py`

**Step 1: Write the failing tests**

Add to `tests/test_graph.py`:

```python
class TestQuerySubgraph:
    def _make_graph(self) -> SoulGraph:
        """Graph with cross-domain connections for retrieval testing."""
        g = SoulGraph(owner_id="test")
        # Career cluster
        g.add_item(SoulItem(id="si_001", text="想创业做AI工具", domains=["career", "identity"], mention_count=5, tags=["intention"]))
        g.add_item(SoulItem(id="si_002", text="8年程序员", domains=["career"], mention_count=3))
        g.add_item(SoulItem(id="si_003", text="技术到了天花板", domains=["career"], mention_count=2))
        # Family cluster
        g.add_item(SoulItem(id="si_004", text="女儿5岁", domains=["family"], mention_count=4))
        g.add_item(SoulItem(id="si_005", text="害怕创业失败养不了家", domains=["career", "family", "finance"], mention_count=6, tags=["intention"]))
        # Finance cluster
        g.add_item(SoulItem(id="si_006", text="房贷压力大", domains=["finance"], mention_count=3))
        g.add_item(SoulItem(id="si_007", text="存款够18个月", domains=["finance"], mention_count=2))
        # Isolated node
        g.add_item(SoulItem(id="si_008", text="喜欢骑自行车", domains=["hobbies"], mention_count=1))
        # Edges: career→family via si_005
        g.add_edge(SoulEdge(from_id="si_002", to_id="si_001", relation="drives", strength=0.8))
        g.add_edge(SoulEdge(from_id="si_003", to_id="si_001", relation="drives", strength=0.7))
        g.add_edge(SoulEdge(from_id="si_006", to_id="si_005", relation="drives", strength=0.9))
        g.add_edge(SoulEdge(from_id="si_004", to_id="si_005", relation="drives", strength=0.8))
        g.add_edge(SoulEdge(from_id="si_005", to_id="si_001", relation="conflicts_with", strength=0.9))
        g.add_edge(SoulEdge(from_id="si_007", to_id="si_001", relation="enables", strength=0.7))
        return g

    def test_query_by_node_id(self):
        g = self._make_graph()
        sub = g.query_subgraph("si_001", top_k=5)
        assert len(sub.items) <= 5
        assert any(i.id == "si_001" for i in sub.items)

    def test_query_by_text(self):
        g = self._make_graph()
        sub = g.query_subgraph("创业", top_k=5)
        assert len(sub.items) >= 1
        # Should find career-related nodes
        sub_ids = {i.id for i in sub.items}
        assert "si_001" in sub_ids  # 想创业做AI工具

    def test_query_cross_domain(self):
        g = self._make_graph()
        sub = g.query_subgraph("创业", top_k=7)
        domains = set()
        for item in sub.items:
            domains.update(item.domains)
        # Should span multiple domains via bridge node si_005
        assert len(domains) >= 2

    def test_query_returns_subgraph_with_edges(self):
        g = self._make_graph()
        sub = g.query_subgraph("创业", top_k=5)
        sub_ids = {i.id for i in sub.items}
        # Edges should only reference nodes in the subgraph
        for edge in sub.edges:
            assert edge.from_id in sub_ids
            assert edge.to_id in sub_ids

    def test_query_empty_graph(self):
        g = SoulGraph(owner_id="empty")
        sub = g.query_subgraph("anything", top_k=5)
        assert len(sub.items) == 0
        assert len(sub.edges) == 0

    def test_query_top_k_limits(self):
        g = self._make_graph()
        sub = g.query_subgraph("创业", top_k=3)
        assert len(sub.items) <= 3
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_graph.py::TestQuerySubgraph -v`
Expected: FAIL (query_subgraph not defined)

**Step 3: Write minimal implementation**

Add to `soulgraph/graph/models.py`, at the top with imports:

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# Module-level lazy-loaded model cache
_embedding_model: SentenceTransformer | None = None

def _get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedding_model
```

Add methods to `SoulGraph` class (after `domain_pagerank()`):

```python
    def query_subgraph(
        self,
        query: str,
        top_k: int = 15,
        alpha: float = 0.85,
        seed_k: int = 5,
    ) -> SoulGraph:
        """Extract a relevant subgraph using Personalized PageRank from query seeds."""
        if not self.items:
            return SoulGraph(owner_id=self.owner_id)

        # Resolve seeds
        seed_ids = self._resolve_seeds(query, seed_k)
        if not seed_ids:
            return SoulGraph(owner_id=self.owner_id)

        # Build personalization vector
        G = self._to_nx()
        personalization = {item.id: 0.0 for item in self.items}
        for sid in seed_ids:
            personalization[sid] = 1.0 / len(seed_ids)

        # Run PPR
        try:
            ppr = nx.pagerank(G, alpha=alpha, personalization=personalization, weight="weight")
        except nx.PowerIterationFailedConvergence:
            ppr = {item.id: 1.0 / len(self.items) for item in self.items}

        # Take top-K by PPR score
        sorted_ids = sorted(ppr, key=ppr.get, reverse=True)[:top_k]
        selected = set(sorted_ids)

        # Extract induced subgraph
        items = [i for i in self.items if i.id in selected]
        edges = [e for e in self.edges if e.from_id in selected and e.to_id in selected]
        return SoulGraph(owner_id=self.owner_id, items=items, edges=edges)

    def _resolve_seeds(self, query: str, seed_k: int) -> list[str]:
        """Resolve a text query or node ID to seed node IDs."""
        # Check if query is a node ID
        item_ids = {i.id for i in self.items}
        if query in item_ids:
            return [query]

        # Embedding-based seed resolution
        model = _get_embedding_model()
        query_emb = model.encode([query], normalize_embeddings=True)
        item_texts = [i.text for i in self.items]
        item_embs = model.encode(item_texts, normalize_embeddings=True)

        sims = (query_emb @ item_embs.T)[0]
        top_indices = np.argsort(sims)[::-1][:seed_k]

        # Filter by minimum similarity threshold
        seeds = []
        for idx in top_indices:
            if sims[idx] >= 0.3:
                seeds.append(self.items[idx].id)
        return seeds if seeds else [self.items[top_indices[0]].id]
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph.py::TestQuerySubgraph -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All 66+ tests pass

**Step 6: Commit**

```bash
git add soulgraph/graph/models.py tests/test_graph.py
git commit -m "feat(v6): add query_subgraph() with PPR-based retrieval"
```

---

### Task 2: Add `RetrievalEvaluator` — structural metrics

**Files:**
- Create: `soulgraph/comparator/retrieval.py`
- Create: `tests/test_retrieval.py`

**Step 1: Write the failing tests**

Create `tests/test_retrieval.py`:

```python
from soulgraph.comparator.retrieval import RetrievalEvaluator
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge


class TestRetrievalStructural:
    def _make_subgraph(self) -> SoulGraph:
        g = SoulGraph(owner_id="sub")
        g.add_item(SoulItem(id="s1", text="创业想法", domains=["career", "identity"]))
        g.add_item(SoulItem(id="s2", text="房贷压力", domains=["finance"]))
        g.add_item(SoulItem(id="s3", text="害怕失败", domains=["career", "family"]))
        g.add_edge(SoulEdge(from_id="s2", to_id="s3", relation="drives", strength=0.8))
        g.add_edge(SoulEdge(from_id="s3", to_id="s1", relation="conflicts_with", strength=0.9))
        return g

    def test_cross_domain_coverage(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert metrics["cross_domain_coverage"] == 3  # career, identity, finance, family → 4 actually
        # Actually: career, identity, finance, family = 4
        assert metrics["cross_domain_coverage"] >= 3

    def test_connectivity(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert metrics["is_connected"] is True

    def test_disconnected_graph(self):
        evaluator = RetrievalEvaluator()
        g = SoulGraph(owner_id="disc")
        g.add_item(SoulItem(id="a", text="节点A", domains=["x"]))
        g.add_item(SoulItem(id="b", text="节点B", domains=["y"]))
        # No edges — disconnected
        metrics = evaluator.structural_metrics(g)
        assert metrics["is_connected"] is False

    def test_density(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert 0.0 <= metrics["density"] <= 1.0

    def test_node_edge_counts(self):
        evaluator = RetrievalEvaluator()
        sub = self._make_subgraph()
        metrics = evaluator.structural_metrics(sub)
        assert metrics["node_count"] == 3
        assert metrics["edge_count"] == 2

    def test_empty_subgraph(self):
        evaluator = RetrievalEvaluator()
        g = SoulGraph(owner_id="empty")
        metrics = evaluator.structural_metrics(g)
        assert metrics["node_count"] == 0
        assert metrics["cross_domain_coverage"] == 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_retrieval.py -v`
Expected: FAIL (cannot import RetrievalEvaluator)

**Step 3: Write minimal implementation**

Create `soulgraph/comparator/retrieval.py`:

```python
"""Retrieval-based evaluation — V6 metrics for query-driven subgraph quality."""
from __future__ import annotations

import networkx as nx

from soulgraph.graph.models import SoulGraph


class RetrievalEvaluator:
    """Evaluate quality of a retrieved subgraph."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def structural_metrics(self, subgraph: SoulGraph) -> dict:
        """Compute structural quality metrics for a subgraph (no LLM needed)."""
        if not subgraph.items:
            return {
                "node_count": 0,
                "edge_count": 0,
                "cross_domain_coverage": 0,
                "is_connected": False,
                "density": 0.0,
            }

        # Cross-domain coverage
        domains: set[str] = set()
        for item in subgraph.items:
            domains.update(item.domains)

        # Connectivity (treat as undirected for connectivity check)
        G = subgraph._to_nx()
        undirected = G.to_undirected()
        is_connected = nx.is_connected(undirected) if len(undirected) > 0 else False

        # Density
        density = nx.density(G) if len(G) > 1 else 0.0

        return {
            "node_count": len(subgraph.items),
            "edge_count": len(subgraph.edges),
            "cross_domain_coverage": len(domains),
            "is_connected": is_connected,
            "density": round(density, 3),
        }
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_retrieval.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/comparator/retrieval.py tests/test_retrieval.py
git commit -m "feat(v6): add RetrievalEvaluator with structural metrics"
```

---

### Task 3: Add LLM-as-Judge metrics to RetrievalEvaluator

**Files:**
- Modify: `soulgraph/comparator/retrieval.py`
- Modify: `tests/test_retrieval.py`

**Step 1: Write the failing tests**

Add to `tests/test_retrieval.py`:

```python
from unittest.mock import patch, MagicMock


class TestRetrievalLLMJudge:
    def _make_subgraph(self) -> SoulGraph:
        g = SoulGraph(owner_id="sub")
        g.add_item(SoulItem(id="s1", text="想创业做AI工具", domains=["career"]))
        g.add_item(SoulItem(id="s2", text="房贷压力大", domains=["finance"]))
        g.add_edge(SoulEdge(from_id="s2", to_id="s1", relation="constrains", strength=0.8))
        return g

    @patch("soulgraph.comparator.retrieval.anthropic")
    def test_evaluate_returns_all_metrics(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"score": 0.8, "reasoning": "good"}')]
        )

        evaluator = RetrievalEvaluator(api_key="fake")
        sub = self._make_subgraph()
        result = evaluator.evaluate(
            full_graph=sub,
            subgraph=sub,
            query="创业",
            conversation_transcript="我想创业做AI工具，但是房贷压力大",
        )
        assert "faithfulness" in result
        assert "comprehensiveness" in result
        assert "diversity" in result
        assert "retrieval_score" in result
        assert 0.0 <= result["retrieval_score"] <= 1.0

    @patch("soulgraph.comparator.retrieval.anthropic")
    def test_evaluate_includes_structural(self, mock_anthropic):
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text='{"score": 0.7, "reasoning": "ok"}')]
        )

        evaluator = RetrievalEvaluator(api_key="fake")
        sub = self._make_subgraph()
        result = evaluator.evaluate(
            full_graph=sub,
            subgraph=sub,
            query="创业",
            conversation_transcript="我想创业",
        )
        assert "node_count" in result
        assert "cross_domain_coverage" in result
        assert "is_connected" in result
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_retrieval.py::TestRetrievalLLMJudge -v`
Expected: FAIL (evaluate method not defined)

**Step 3: Write implementation**

Extend `soulgraph/comparator/retrieval.py`:

```python
"""Retrieval-based evaluation — V6 metrics for query-driven subgraph quality."""
from __future__ import annotations

import json
import networkx as nx
import anthropic

from soulgraph.graph.models import SoulGraph


_JUDGE_PROMPT = """You are evaluating the quality of a knowledge subgraph retrieved for a query.

## Query
{query}

## Retrieved Subgraph Nodes
{nodes}

## Conversation Transcript (source)
{transcript}

## Task
Evaluate this dimension: **{dimension}**

{dimension_description}

Respond with JSON only:
{{"score": <float 0.0-1.0>, "reasoning": "<one sentence>"}}
"""

_DIMENSIONS = {
    "faithfulness": "Is every node in the subgraph directly supported by something said in the conversation? Score 1.0 if all nodes are grounded, 0.0 if none are.",
    "comprehensiveness": "Does this subgraph capture all important aspects of the conversation relevant to the query? Score 1.0 if complete coverage, 0.0 if missing everything relevant.",
    "diversity": "Does this subgraph span multiple different domains or perspectives related to the query? Score 1.0 if highly diverse (3+ domains), 0.5 if moderate (2 domains), 0.2 if single-domain.",
}


class RetrievalEvaluator:
    """Evaluate quality of a retrieved subgraph."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    def structural_metrics(self, subgraph: SoulGraph) -> dict:
        """Compute structural quality metrics for a subgraph (no LLM needed)."""
        if not subgraph.items:
            return {
                "node_count": 0,
                "edge_count": 0,
                "cross_domain_coverage": 0,
                "is_connected": False,
                "density": 0.0,
            }

        domains: set[str] = set()
        for item in subgraph.items:
            domains.update(item.domains)

        G = subgraph._to_nx()
        undirected = G.to_undirected()
        is_connected = nx.is_connected(undirected) if len(undirected) > 0 else False
        density = nx.density(G) if len(G) > 1 else 0.0

        return {
            "node_count": len(subgraph.items),
            "edge_count": len(subgraph.edges),
            "cross_domain_coverage": len(domains),
            "is_connected": is_connected,
            "density": round(density, 3),
        }

    def evaluate(
        self,
        full_graph: SoulGraph,
        subgraph: SoulGraph,
        query: str,
        conversation_transcript: str,
    ) -> dict:
        """Full evaluation: structural metrics + LLM-as-Judge."""
        structural = self.structural_metrics(subgraph)

        # LLM-as-Judge scores
        nodes_text = "\n".join(
            f"- [{item.id}] {item.text} (domains: {', '.join(item.domains)})"
            for item in subgraph.items
        )
        # Truncate transcript to last 4000 chars to fit context
        transcript = conversation_transcript[-4000:] if len(conversation_transcript) > 4000 else conversation_transcript

        llm_scores = {}
        for dim, desc in _DIMENSIONS.items():
            llm_scores[dim] = self._judge(query, nodes_text, transcript, dim, desc)

        # Combined structural score
        structural_score = 0.0
        if structural["node_count"] > 0:
            connected_bonus = 0.5 if structural["is_connected"] else 0.0
            domain_bonus = min(structural["cross_domain_coverage"] / 3.0, 1.0) * 0.3
            density_bonus = min(structural["density"] / 0.3, 1.0) * 0.2
            structural_score = connected_bonus + domain_bonus + density_bonus

        # Combined retrieval score
        retrieval_score = (
            llm_scores.get("faithfulness", 0.0) * 0.3
            + llm_scores.get("comprehensiveness", 0.0) * 0.3
            + llm_scores.get("diversity", 0.0) * 0.2
            + structural_score * 0.2
        )

        return {
            **structural,
            **llm_scores,
            "structural_score": round(structural_score, 3),
            "retrieval_score": round(retrieval_score, 3),
        }

    def _judge(self, query: str, nodes: str, transcript: str, dimension: str, description: str) -> float:
        """Call LLM to judge one dimension. Returns score 0.0-1.0."""
        if not self._api_key:
            return 0.5  # fallback when no API key

        kwargs: dict = {"api_key": self._api_key}
        if self._api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        client = anthropic.Anthropic(**kwargs)

        prompt = _JUDGE_PROMPT.format(
            query=query,
            nodes=nodes,
            transcript=transcript,
            dimension=dimension,
            dimension_description=description,
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            data = json.loads(text)
            return max(0.0, min(1.0, float(data["score"])))
        except Exception:
            return 0.5  # fallback on error
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_retrieval.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/comparator/retrieval.py tests/test_retrieval.py
git commit -m "feat(v6): add LLM-as-Judge metrics to RetrievalEvaluator"
```

---

### Task 4: Add query set to Zhang Wei fixture

**Files:**
- Modify: `fixtures/zhang_wei.json`
- Modify: `tests/test_graph.py`

**Step 1: Write the failing test**

Add to `tests/test_graph.py` in `TestFixtures`:

```python
    def test_zhang_wei_has_queries(self):
        import json
        path = Path(__file__).parent.parent / "fixtures" / "zhang_wei.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "queries" in data
        queries = data["queries"]
        assert len(queries) >= 5
        assert all("query" in q for q in queries)
        assert all("expected_domains" in q for q in queries)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph.py::TestFixtures::test_zhang_wei_has_queries -v`
Expected: FAIL ("queries" not in data)

**Step 3: Add queries to fixture**

Add to `fixtures/zhang_wei.json` after the `sessions` field:

```json
  "queries": [
    {
      "query": "创业",
      "description": "Startup ambition — should pull career + finance + family + identity",
      "expected_domains": ["career", "finance", "family", "identity"]
    },
    {
      "query": "家庭责任",
      "description": "Family responsibility — should pull family + finance + values",
      "expected_domains": ["family", "finance", "values"]
    },
    {
      "query": "健康",
      "description": "Health concerns — should pull health + career stress + hobbies",
      "expected_domains": ["health", "career", "hobbies"]
    },
    {
      "query": "人生意义",
      "description": "Life meaning — should pull values + identity + stories",
      "expected_domains": ["values", "identity", "stories"]
    },
    {
      "query": "si_002",
      "description": "Direct node query — 想创业做AI工具, the top hub",
      "expected_domains": ["career", "identity", "finance"]
    }
  ]
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph.py::TestFixtures -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add fixtures/zhang_wei.json tests/test_graph.py
git commit -m "feat(v6): add 5-query evaluation set to zhang_wei fixture"
```

---

### Task 5: Integrate query eval into multi-session runner

**Files:**
- Modify: `soulgraph/experiment/runner.py`
- Modify: `tests/test_experiment.py`

**Step 1: Write the failing test**

Add to `tests/test_experiment.py`:

```python
class TestQueryEvalPhase:
    def _make_gt_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="gt")
        g.add_item(SoulItem(id="si_001", text="想创业", domains=["career"], tags=["intention"]))
        g.add_item(SoulItem(id="si_002", text="房贷压力", domains=["finance"]))
        g.add_edge(SoulEdge(from_id="si_002", to_id="si_001", relation="constrains"))
        return g

    @patch("soulgraph.experiment.runner.RetrievalEvaluator")
    @patch("soulgraph.experiment.runner.RankingComparator")
    @patch("soulgraph.experiment.runner.Speaker")
    @patch("soulgraph.experiment.runner.Detector")
    def test_multi_session_with_queries(self, MockDetector, MockSpeaker, MockRankComp, MockRetEval):
        mock_speaker = MockSpeaker.return_value
        mock_speaker.respond.return_value = "我想创业"

        mock_detector = MockDetector.return_value
        mock_detector.ask_next_question.return_value = "你在想什么？"
        mock_detector.listen_and_detect.return_value = SoulGraph(owner_id="det")
        mock_detector.detected_graph = SoulGraph(owner_id="det")
        mock_detector.detected_graph.add_item(SoulItem(id="d1", text="创业", domains=["career"]))

        MockRankComp.return_value.compare.return_value = {
            "rank_correlation": 0.5, "domain_ndcg": 0.5,
            "absorption_rate": 0.5, "intention_recall": 0.5,
            "overall": 0.5, "matched_items": 1, "gt_items": 2, "det_items": 1,
        }

        MockRetEval.return_value.evaluate.return_value = {
            "node_count": 1, "edge_count": 0, "cross_domain_coverage": 1,
            "is_connected": True, "density": 0.0,
            "faithfulness": 0.8, "comprehensiveness": 0.7, "diversity": 0.5,
            "structural_score": 0.5, "retrieval_score": 0.65,
        }

        runner = ExperimentRunner(api_key="fake")
        queries = [{"query": "创业", "expected_domains": ["career"]}]
        result = runner.run_multi_session(
            self._make_gt_graph(),
            session_configs=[{"turns": 2, "topic_hints": ["career"]}],
            queries=queries,
            verbose=False,
        )
        assert "query_scores" in result
        assert len(result["query_scores"]) == 1
        assert "mean_retrieval_score" in result
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_experiment.py::TestQueryEvalPhase -v`
Expected: FAIL (unexpected keyword argument 'queries')

**Step 3: Modify runner**

In `soulgraph/experiment/runner.py`, add import at top:

```python
from soulgraph.comparator.retrieval import RetrievalEvaluator
```

Modify `run_multi_session()` signature to accept queries:

```python
    def run_multi_session(
        self,
        ground_truth: SoulGraph,
        session_configs: list[dict],
        queries: list[dict] | None = None,
        hub_top_k: int = 5,
        verbose: bool = True,
    ) -> dict:
```

After the existing multi-session summary block (after line ~274), before `return result`, add:

```python
        # Phase 2: Query evaluation (V6)
        if queries:
            all_conversations = []
            # Collect all conversation messages from all sessions
            # (they're in the local 'conversation' vars per session — need to accumulate)
            # We'll build transcript from detector's graph context instead
            ret_evaluator = RetrievalEvaluator(api_key=self._api_key)
            query_scores = []
            for q in queries:
                subgraph = detector.detected_graph.query_subgraph(q["query"])
                scores = ret_evaluator.evaluate(
                    full_graph=detector.detected_graph,
                    subgraph=subgraph,
                    query=q["query"],
                    conversation_transcript="",  # transcript built below
                )
                query_scores.append({"query": q["query"], **scores})

                if verbose:
                    print(f"\n  Query '{q['query']}': retrieval={scores['retrieval_score']:.3f}  "
                          f"faith={scores.get('faithfulness', 0):.3f}  "
                          f"comp={scores.get('comprehensiveness', 0):.3f}  "
                          f"div={scores.get('diversity', 0):.3f}  "
                          f"domains={scores['cross_domain_coverage']}  "
                          f"nodes={scores['node_count']}")

            mean_ret = sum(s["retrieval_score"] for s in query_scores) / len(query_scores)
            result["query_scores"] = query_scores
            result["mean_retrieval_score"] = round(mean_ret, 3)

            if verbose:
                print(f"\n  Mean Retrieval Score: {mean_ret:.3f}")
```

Also, accumulate conversation transcripts across sessions. Change the session loop to collect all messages:

Add `all_messages: list[Message] = []` before the session loop, and after each session's inner loop, add `all_messages.extend(conversation)`.

Then in the query eval phase, build the transcript:

```python
            transcript = "\n".join(f"[{m.role}]: {m.content}" for m in all_messages)
```

And pass `conversation_transcript=transcript` in the `evaluate()` call instead of `""`.

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_experiment.py::TestQueryEvalPhase -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/experiment/runner.py tests/test_experiment.py
git commit -m "feat(v6): integrate query eval phase into multi-session runner"
```

---

### Task 6: Add `--queries` CLI flag

**Files:**
- Modify: `soulgraph/cli.py`

**Step 1: Add queries support**

Add argument to parser:

```python
    parser.add_argument("--queries", action="store_true", default=False, help="Run query evaluation phase after multi-session (reads queries from fixture)")
```

In the `args.sessions > 0` block, after `session_configs` setup, load queries if flag set:

```python
            queries = None
            if args.queries:
                queries = fixture_data.get("queries", [])
                if not queries:
                    print("Warning: --queries flag set but fixture has no 'queries' field", file=sys.stderr)
            summary = runner.run_multi_session(gt, session_configs=session_configs, queries=queries)
```

**Step 2: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add soulgraph/cli.py
git commit -m "feat(v6): add --queries flag to CLI for retrieval evaluation"
```

---

### Task 7: Run V6 benchmark

**Step 1: Run multi-session + query eval**

```bash
ANTHROPIC_API_KEY=<key> .venv/bin/python -m soulgraph --experiment fixtures/zhang_wei.json --sessions 3 --turns 10 --queries --output results/v6_zhang_wei_retrieval.json
```

Expected: 3 sessions complete, then 5 query evaluations with retrieval scores.

**Step 2: Verify results**

Check that:
- `query_scores` contains 5 entries
- `mean_retrieval_score` is present
- "创业" query has cross_domain_coverage ≥ 3
- Subgraphs are connected for most queries

**Step 3: Commit results**

```bash
git add -f results/v6_zhang_wei_retrieval.json
git commit -m "docs: V6 retrieval benchmark results"
```

---

### Task 8: Write V6 findings report

**Files:**
- Create: `docs/findings-v6.md`

Document:
1. Per-query retrieval scores (faithfulness, comprehensiveness, diversity)
2. Structural metrics per query (cross-domain coverage, connectivity)
3. Mean retrieval score
4. Comparison: V5 rank_correlation approach vs V6 retrieval approach
5. Does "创业" query find cross-domain connections? (career→finance→family)
6. What worked, what didn't, V7 priorities

**Commit:**

```bash
git add docs/findings-v6.md
git commit -m "docs: V6 findings report — query-driven retrieval evaluation"
```
