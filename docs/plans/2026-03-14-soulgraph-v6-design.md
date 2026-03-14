# SoulGraph V6 Design — Query-Driven Retrieval + Retrieval Evaluation

## Vision

Ground truth doesn't exist in absolute form. A soul graph's quality = how well it serves queries. V6 adds query-driven subgraph retrieval (PPR with embedding-based seed resolution) and a new evaluation framework that measures retrieval quality instead of comparing against a pre-defined GT.

## Architecture

```
Multi-session pipeline (V5):
  Speaker ↔ Detector → builds graph across 3 sessions → accumulated graph

Query evaluation phase (V6, NEW):
  accumulated graph + query set → query_subgraph() per query → RetrievalEvaluator

RetrievalEvaluator:
  structural metrics (cross-domain, connectivity, density)
  + LLM-as-Judge (faithfulness to conversation, comprehensiveness, diversity)
  → retrieval quality scores
```

## Core Principles

1. **No absolute ground truth** — quality is measured by retrieval usefulness, not match to a fixture
2. **Cross-domain discovery** — "insurance" query should find "bike accident" through bridge nodes
3. **PPR for retrieval** — Personalized PageRank with query-biased restart vector
4. **Dual evaluation** — structural metrics (fast, deterministic) + LLM-as-Judge (semantic quality)
5. **Integrated pipeline** — query eval runs after multi-session graph building, reuses conversation transcript

## Component Design

### 1. Query-Driven Subgraph Retrieval

**New method on SoulGraph: `query_subgraph()`**

```python
def query_subgraph(
    self,
    query: str,              # text like "insurance" or node ID like "si_010"
    top_k: int = 15,         # max nodes in subgraph
    alpha: float = 0.85,     # PPR damping
    seed_k: int = 5,         # number of seed nodes from embedding search
) -> SoulGraph:
```

**Flow:**
1. If `query` matches an existing node ID → use that node as sole seed
2. Otherwise → embed query with sentence-transformers (all-MiniLM-L6-v2), find `seed_k` nearest nodes by cosine similarity (threshold ≥ 0.3)
3. Build personalization vector: seed nodes split probability mass equally, all others get 0
4. Run `nx.pagerank(G, personalization=pv, alpha=alpha, weight="weight")`
5. Take top-K nodes by PPR score
6. Extract induced subgraph: all edges between top-K nodes
7. Return as new `SoulGraph`

**Embedding helper:** Add `_embed_text()` and `_find_seed_nodes()` private methods. Use the same sentence-transformers model already in EmbeddingMatcher. Load model lazily (cached at class level).

### 2. RetrievalEvaluator

**New class in `soulgraph/comparator/retrieval.py`**

```python
class RetrievalEvaluator:
    def __init__(self, api_key: str = ""):
        # LLM client for judge calls

    def evaluate(
        self,
        full_graph: SoulGraph,
        subgraph: SoulGraph,
        query: str,
        conversation_transcript: str,
    ) -> dict:
```

**Returns:**

```python
{
    # Structural metrics (no LLM needed)
    "node_count": int,
    "edge_count": int,
    "cross_domain_coverage": int,     # number of distinct domains in subgraph
    "is_connected": bool,             # subgraph forms single connected component
    "density": float,                 # edges / max_possible_edges
    "seed_distance_mean": float,      # avg shortest path from seed to subgraph nodes

    # LLM-as-Judge metrics
    "faithfulness": float,            # 0-1, are subgraph nodes grounded in conversation?
    "comprehensiveness": float,       # 0-1, does subgraph cover all relevant aspects?
    "diversity": float,               # 0-1, does subgraph span multiple domains/perspectives?

    # Combined
    "retrieval_score": float,         # weighted combination
}
```

**Structural metrics** — pure Python, no API calls:
- `cross_domain_coverage`: `len(set(d for item in subgraph.items for d in item.domains))`
- `is_connected`: convert to undirected nx graph, check `nx.is_connected()`
- `density`: `nx.density(subgraph._to_nx())`
- `seed_distance_mean`: BFS from seed nodes, mean distance to all subgraph nodes

**LLM-as-Judge metrics** — one API call per metric:
- `faithfulness`: Send subgraph nodes + conversation transcript to LLM. "For each node, is it directly supported by something said in the conversation? Score 0-1."
- `comprehensiveness`: Send query + subgraph + conversation. "Does this subgraph capture all aspects of the conversation relevant to '{query}'? Score 0-1."
- `diversity`: Send query + subgraph. "Does this subgraph cover multiple different domains/perspectives related to '{query}'? Score 0-1."

**Combined score:**
`retrieval_score = faithfulness × 0.3 + comprehensiveness × 0.3 + diversity × 0.2 + structural_score × 0.2`
where `structural_score = (is_connected × 0.5 + min(cross_domain / 3, 1.0) × 0.3 + min(density / 0.3, 1.0) × 0.2)`

### 3. Query Set for Zhang Wei Fixture

Add `queries` field to `fixtures/zhang_wei.json`:

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

5 queries covering different aspects. `expected_domains` is a soft hint for structural eval (not hard GT — just what we'd expect a good subgraph to touch).

### 4. Integrated Experiment Pipeline

**Extend `run_multi_session()` return value:**

After the existing 3-session conversation phase, add a query evaluation phase:

```python
# After all sessions complete...
# Phase 2: Query evaluation
query_scores = []
transcript = "\n".join(f"[{m.role}]: {m.content}" for m in all_conversations)
evaluator = RetrievalEvaluator(api_key=self._api_key)

for q in queries:
    subgraph = detector.detected_graph.query_subgraph(q["query"])
    scores = evaluator.evaluate(
        full_graph=detector.detected_graph,
        subgraph=subgraph,
        query=q["query"],
        conversation_transcript=transcript,
    )
    query_scores.append({"query": q["query"], **scores})

result["query_scores"] = query_scores
result["mean_retrieval_score"] = mean(s["retrieval_score"] for s in query_scores)
```

**CLI:** Add `--queries` flag (default: use queries from fixture if present).

### 5. What Changes from V5

| Aspect | V5 | V6 |
|--------|----|----|
| Primary metric | rank_correlation vs GT | retrieval_score (faithfulness + comprehensiveness + diversity) |
| What we measure | "does ranking match our fixture?" | "does query retrieval return useful subgraphs?" |
| Graph quality proxy | Spearman ρ | LLM-as-Judge + structural metrics |
| New capability | multi-session | query_subgraph() with PPR |
| GT philosophy | pre-defined fixture is truth | conversation transcript is truth |

**V5 metrics kept:** absorption_rate (still valuable — "did we capture what was said?"), intention_recall (still useful). rank_correlation demoted from primary to secondary/diagnostic.

### 6. Code Changes

| File | Change |
|------|--------|
| `graph/models.py` | Add `query_subgraph()`, `_embed_text()`, `_find_seed_nodes()` |
| `comparator/retrieval.py` | NEW — `RetrievalEvaluator` class |
| `fixtures/zhang_wei.json` | Add `queries` field |
| `experiment/runner.py` | Add query eval phase to `run_multi_session()` |
| `cli.py` | Add `--queries` flag |
| `tests/test_graph.py` | Tests for `query_subgraph()` |
| `tests/test_retrieval.py` | NEW — Tests for `RetrievalEvaluator` |
| `tests/test_experiment.py` | Test query eval phase in multi-session |

### What Stays Unchanged

- V5 multi-session framework (Speaker↔Detector, persistent graph, topic steering)
- SoulGraph append-only semantics
- PageRank and domain_pagerank (query_subgraph builds on top)
- V4 RankingComparator (kept as secondary/diagnostic metric)
- EmbeddingMatcher for V3 node matching
- Zhang Wei fixture items/edges/sessions

## Success Criteria

- `query_subgraph()` returns connected subgraphs for ≥ 4/5 queries
- Cross-domain coverage ≥ 2 domains per query on average
- Faithfulness ≥ 0.8 (subgraph nodes grounded in conversation)
- Comprehensiveness ≥ 0.7 (covers relevant aspects)
- Mean retrieval_score ≥ 0.7 across all queries
- "创业" query subgraph includes nodes from ≥ 3 domains (career + finance + family/identity)
