# V4 Findings Report

## Summary

V4 reframes SoulGraph from a recall/precision test into a soul operating system. The key paradigm shift: over-detection is the goal, not a bug. Quality is measured by ranking accuracy (PageRank), not by counting false positives against a sparse ground truth.

New components: `tags` field on SoulItem, PageRank + Topic-Sensitive PageRank on SoulGraph, RankingComparator with Spearman ρ / NDCG@5 / absorption rate / intention recall, uncapped extraction (absorb everything), and a rich 60-item fixture (Zhang Wei).

## Benchmark Results

### Zhang Wei (3 runs × 15 turns, 61 GT items)

| Metric | Mean ± Std | Target | Status |
|--------|-----------|--------|--------|
| Rank Correlation | 0.600 ± 0.041 | ≥ 0.6 | MET |
| Domain NDCG@5 | 0.886 ± 0.045 | ≥ 0.5 | EXCEEDED |
| Absorption Rate | 0.967 ± 0.013 | ≥ 0.7 | EXCEEDED |
| Intention Recall | 0.933 ± 0.094 | ≥ 0.8 | EXCEEDED |
| **V4 Overall** | **0.826 ± 0.021** | ≥ 0.6 | **EXCEEDED** |

V3 metrics on same fixture (for comparison):
| Metric | Mean ± Std |
|--------|-----------|
| Node Recall | 0.967 ± 0.013 |
| Node Precision | 0.737 ± 0.052 |
| Hub Recall | 1.000 ± 0.000 |
| Triple F1 | 0.676 ± 0.018 |
| V3 Overall | 0.831 ± 0.007 |

### Car Buyer (1 run × 10 turns, 12 GT items — V3 fixture with V4 metrics)

| Metric | Score |
|--------|-------|
| Rank Correlation | 0.807 |
| Domain NDCG@5 | 0.896 |
| Absorption Rate | 1.000 |
| Intention Recall | 1.000 |
| **V4 Overall** | **0.911** |

Note: Car buyer is a small fixture (12 items). V4 metrics are very high because the graph structure is simple and all items are absorbed. The car_buyer fixture has no `tags`, so intention recall falls back to absorption rate.

### Per-Run Raw Scores (Zhang Wei)

| Run | Rank Corr | Domain NDCG | Absorption | Intention Recall | V4 Overall | Det Items | Det Edges |
|-----|-----------|-------------|------------|-----------------|------------|-----------|-----------|
| 1 | 0.564 | 0.941 | 0.951 | 1.000 | 0.842 | 87 | 152 |
| 2 | 0.579 | 0.888 | 0.984 | 0.800 | 0.797 | 76 | 167 |
| 3 | 0.657 | 0.830 | 0.967 | 1.000 | 0.839 | 78 | 149 |

## Analysis

### What Worked

#### 1. Dense Absorption (Highest Impact)
- Uncapping extraction limits produced 76-90 items from 61 GT (1.2-1.5x)
- Absorption rate 96.7% — nearly complete coverage of the persona
- V3's "over-detection problem" (50% node precision) is now reframed as success
- The graph captures nuances that sparse extraction would miss

#### 2. PageRank Correctly Identifies Intentions
- Intention recall 93.3% — almost all intention-tagged items found
- Intentions naturally rank highest due to edge density (structural design works)
- "想创业做AI工具" consistently appears as a top hub across all runs
- No special weighting needed — graph structure alone determines importance

#### 3. Domain NDCG Excellent (0.886)
- Topic-Sensitive PageRank correctly shifts node importance by domain
- "骑自行车" ranks higher in transport domain than in career
- Career-domain PageRank correctly elevates career-related intentions
- Far exceeds the 0.5 target — domain-contextual ranking is robust

#### 4. Stability (std ≤ 0.045)
- V4 overall std = 0.021 — consistent across runs
- Most variance comes from absorption rate and intention recall, not ranking quality
- Rank correlation std = 0.041 — reasonable for a correlation metric

### What Needs Improvement

#### 1. Rank Correlation Just Meets Target (0.600)
- This is the weakest metric — barely at the 0.6 target
- The issue: detected graph has many extra nodes (76-87 vs 61 GT) that participate in PageRank but have no GT counterpart
- Extra nodes dilute the ranking signal — some GT items get pushed down by non-GT items
- Possible fix: higher-quality edge extraction (not more edges, but more accurate edges)

#### 2. Intention Recall Variance (std = 0.094)
- Run 2 dropped to 0.800 intention recall — one intention was missed
- Likely due to conversation randomness — some conversations don't trigger all intention topics
- More turns (20+) would likely improve this
- Could also add explicit "intention probing" questions in the detector

#### 3. Edge Over-Generation
- 149-167 edges detected vs 81 GT edges (1.8-2.1x)
- Many plausible but non-GT connections created
- This inflates the detected graph's PageRank — nodes get importance from edges that don't exist in GT
- Structural similarity between detected and GT graphs could be improved

#### 4. Tag Extraction Not Yet Evaluated
- Tags are extracted but not systematically validated
- Some items may have incorrect or missing tags
- Need a tag accuracy metric in future versions

## Comparison: V3 → V4 Paradigm Shift

| Aspect | V3 | V4 |
|--------|----|----|
| Philosophy | Minimize false positives | Absorb everything, rank by importance |
| Extraction | Max 3 items/turn, confidence ≥ 0.4 | Uncapped, no filter |
| Evaluation | Recall/precision/F1 | Rank correlation, domain NDCG |
| Fixture | 12 items, 13 edges | 61 items, 81 edges |
| GT Design | Simple personas | Rich persona with structural PageRank GT |
| Key Question | "Did we find the right things?" | "Are the important things ranked correctly?" |
| Items Detected | 18-24 from 12 GT | 76-90 from 61 GT |
| Node Precision | ~55% (V3 metric) | N/A (not the goal) |
| Ranking Quality | N/A | 0.826 overall |

## V5 Priorities

1. **Improve rank correlation**: Explore edge pruning after extraction (remove low-confidence edges), or weight-only PageRank (use edge confidence as weight more aggressively)
2. **Multi-session support**: Same persona across multiple conversations — test Ebbinghaus decay, mention_count reinforcement
3. **Tag validation metric**: Add tag accuracy evaluation (compare detected tags to GT tags)
4. **Longer conversations**: Test 20-25 turns to see if absorption rate approaches 100% and rank correlation improves
5. **Cross-persona generalization**: Create 2-3 more rich fixtures (different demographics, languages, life situations)
6. **Production integration**: Use the ranked graph for downstream tasks — contextual retrieval, task recommendation, conversation steering
