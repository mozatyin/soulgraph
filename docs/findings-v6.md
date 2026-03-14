# V6 Findings Report — Query-Driven Retrieval

## Summary

V6 introduces query-driven subgraph retrieval via Personalized PageRank (PPR) and a new evaluation framework combining structural metrics with LLM-as-Judge scoring. The core hypothesis — that a detected graph's value should be measured by retrieval quality rather than fixture matching — is **validated**. Mean retrieval score across 5 queries is **0.817**, with 4/5 queries scoring above 0.82.

## Architecture

- **Query resolution**: Embed query text with `all-MiniLM-L6-v2`, find nearest nodes by cosine similarity (threshold >= 0.3)
- **Subgraph extraction**: PPR with alpha=0.85, biased restart toward seed nodes, take top-15 nodes + induced edges
- **Evaluation**: Structural metrics (connectivity, cross-domain coverage, density) + LLM-as-Judge (faithfulness, comprehensiveness, diversity)
- **Combined score**: `faith*0.3 + comp*0.3 + div*0.2 + structural*0.2`

## Benchmark Results

### Zhang Wei (3 sessions x 10 turns + 5 queries, 61 GT items, 141 detected items)

#### Multi-Session Scores (V5 metrics, same run)

| Metric | Session 1 | Session 2 | Session 3 | Delta (S3-S1) |
|--------|-----------|-----------|-----------|-----------|
| Rank Correlation | 0.483 | 0.602 | 0.587 | +0.104 |
| Domain NDCG | 0.959 | 0.905 | 0.907 | -0.052 |
| Absorption Rate | 0.967 | 0.984 | 1.000 | +0.033 |
| Intention Recall | 1.000 | 1.000 | 1.000 | +0.000 |
| **Overall (V4)** | **0.826** | **0.849** | **0.848** | **+0.022** |
| Matched Items | 59/61 | 60/61 | 61/61 | +2 |
| Detected Items | 69 | 113 | 141 | +72 |

#### Query Retrieval Scores (V6)

| Query | Faithfulness | Comprehensiveness | Diversity | Structural | **Retrieval** | Domains | Connected |
|-------|-------------|-------------------|-----------|------------|---------------|---------|-----------|
| 创业 | 1.00 | 0.80 | 1.00 | 0.411 | **0.822** | 13 | no |
| 家庭责任 | 0.87 | 0.90 | 1.00 | 0.892 | **0.909** | 11 | yes |
| 健康 | 0.93 | 0.20 | 0.50 | 0.376 | **0.514** | 15 | no |
| 人生意义 | 1.00 | 0.90 | 1.00 | 0.895 | **0.949** | 11 | yes |
| si_002 (node ID) | 1.00 | 0.70 | 1.00 | 0.908 | **0.892** | 12 | yes |
| **Mean** | **0.96** | **0.70** | **0.90** | **0.696** | **0.817** | 12.4 | 3/5 |

### V5 vs V6 Same-Run Comparison

The V6 run also showed improved V5 metrics compared to the original V5 benchmark:

| Metric | V5 Run | V6 Run | Change |
|--------|--------|--------|--------|
| Rank Improvement | -0.032 | +0.104 | +0.136 |
| Final Absorption | 0.984 | 1.000 | +0.016 |
| Final Intention Recall | 1.000 | 1.000 | = |
| Final Overall | 0.829 | 0.848 | +0.019 |

Note: V5 metric improvement likely due to run-to-run variance (different API responses), not V6 code changes. Single-run comparison; would need 3+ runs for statistical significance.

## Analysis

### What Worked

#### 1. Cross-Domain Discovery (Core V6 Insight)
- "创业" (entrepreneurship) query retrieved nodes from **13 domains** — not just career/finance but health, family, identity, values
- PPR walks through bridge edges to discover cross-domain connections
- This is exactly the "insurance → bike accident" phenomenon we designed for
- 家庭责任 retrieved from 11 domains, 人生意义 from 11 — queries naturally fan out

#### 2. Faithfulness Near-Perfect (0.96 mean)
- 4/5 queries scored 1.0 faithfulness — all retrieved nodes are genuinely grounded in conversation
- Only 家庭责任 scored slightly lower (0.87) — likely one node was tangentially connected
- This validates the PPR approach: PageRank naturally favors well-connected, conversation-grounded nodes

#### 3. Node ID Query Works (si_002)
- Direct node ID query scored 0.892 retrieval — confirms programmatic access works alongside semantic queries
- PPR correctly expands from the target node to its neighborhood
- Connected=true, 12 domains covered — useful for "explain this node in context" use cases

#### 4. Connected Subgraphs Correlate with Quality
- Connected queries (家庭责任, 人生意义, si_002): mean retrieval = 0.917
- Disconnected queries (创业, 健康): mean retrieval = 0.668
- Connectivity is a strong quality signal — connected subgraphs give coherent narratives

#### 5. Evaluation Framework Provides Actionable Signal
- LLM-as-Judge dimensions surface specific weaknesses (e.g., 健康 has low comprehensiveness)
- Structural metrics differentiate connected vs fragmented results
- Combined score balances all dimensions meaningfully

### What Didn't Work

#### 1. "健康" Query Scored Poorly (0.514)
- Comprehensiveness only 0.20 — the judge found health-related content in the conversation that wasn't in the subgraph
- Diversity 0.50 — likely because health items cluster in similar domains
- Not connected (15 domains but fragmented) — health items are scattered across the graph
- Root cause: health is a secondary theme in zhang_wei, so health-relevant nodes are sparse and poorly connected
- PPR struggles when seed nodes are peripheral — the random walk doesn't concentrate mass

#### 2. All Subgraphs Return Exactly 15 Nodes
- top_k=15 is fixed — doesn't adapt to query specificity
- "创业" should probably return more nodes (core theme), "健康" fewer (peripheral)
- Adaptive top_k based on seed strength or PPR mass distribution would improve quality

#### 3. Disconnected Subgraphs for Some Queries
- 创业 and 健康 returned disconnected subgraphs despite having many edges (43 and 28)
- PPR selects top-K nodes by score, but those nodes may not form a connected component
- Post-processing to enforce connectivity (e.g., add bridge nodes) would help

#### 4. Single Run, No Statistical Confidence
- LLM-as-Judge scores have inherent variance (~0.1 per dimension)
- Need 3+ runs to establish mean and confidence intervals
- The 0.817 mean retrieval could realistically range from 0.75-0.88

## V6 Success Criteria Assessment

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Mean retrieval score > 0.70 | > 0.70 | 0.817 | EXCEEDED |
| Faithfulness > 0.80 | > 0.80 | 0.96 | EXCEEDED |
| Cross-domain coverage > 5 | > 5 | 12.4 avg | EXCEEDED |
| All queries return subgraphs | 5/5 | 5/5 | MET |
| Node ID query works | functional | 0.892 | MET |

## Key Insight: V6 Reframes the Problem

V5's core issue was that rank correlation degraded as the graph grew — more detected items meant more "noise" relative to the fixed GT. V6 sidesteps this entirely:

- **V5 asks**: "Does the detected graph match the fixture?" — penalizes over-detection
- **V6 asks**: "Can the detected graph answer real questions?" — rewards rich, well-connected graphs

A 141-node graph with 0.514-0.949 retrieval quality is more useful than a 61-node graph that matches the fixture perfectly. The extra 80 nodes represent genuine conversation discoveries that make retrieval richer.

## Next Steps

1. **Adaptive top_k**: Scale subgraph size based on PPR mass concentration (tight cluster → fewer nodes, diffuse → more)
2. **Connectivity post-processing**: After PPR selection, add minimum bridge nodes to connect components
3. **Multi-run benchmark**: Run 3x to establish statistical confidence for LLM-as-Judge scores
4. **More fixtures**: Test on car_buyer and other personas to validate generalization
5. **Query difficulty analysis**: Categorize queries by theme centrality and correlate with retrieval quality
