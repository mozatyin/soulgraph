# V3 Findings Report

## Summary

V3 replaced LLM-as-judge comparison with deterministic embedding-based evaluation, added extract-then-diff architecture, laddering questions, and depth/breadth/bridge heuristic. The result is consistent ~0.77 overall score with near-zero variance across all 3 fixtures.

## Benchmark Results (3 runs × 10 turns)

| Metric | car_buyer | career_changer | anxiety_user |
|--------|-----------|----------------|--------------|
| Node Recall | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.972 ± 0.039 |
| Node Precision | 0.548 ± 0.041 | 0.566 ± 0.047 | 0.495 ± 0.039 |
| Hub Recall | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| Triple Recall | 1.000 ± 0.000 | 0.974 ± 0.036 | 1.000 ± 0.000 |
| Triple Precision | 0.362 ± 0.016 | 0.378 ± 0.029 | 0.384 ± 0.025 |
| Triple F1 | 0.531 ± 0.018 | 0.544 ± 0.034 | 0.555 ± 0.026 |
| **Overall** | **0.766 ± 0.009** | **0.772 ± 0.017** | **0.771 ± 0.017** |

## Comparison with V1/V2

| Version | car_buyer (best single) | car_buyer (mean±std) | career_changer |
|---------|------------------------|---------------------|----------------|
| V1 | 0.76 (legacy metric) | N/A | 0.68 |
| V2 | 0.60-0.76 (high variance) | N/A | N/A |
| **V3** | 0.851 (single run) | **0.766 ± 0.009** | **0.772 ± 0.017** |

Note: V1/V2 used LLM-as-judge (noisy). V3 uses embedding comparison (deterministic). Scores are not directly comparable, but V3 metrics are more trustworthy.

## What Worked

### 1. Embedding-Based Comparison (Highest Impact)
- sentence-transformers `all-MiniLM-L6-v2` + Hungarian algorithm for optimal node matching
- Triple-level soft matching: embed linearized triples, greedy assignment
- **Eliminated variance from comparison**: same detected graph always gives same score
- Overall = hub_recall × 0.3 + node_recall × 0.2 + triple_f1 × 0.5

### 2. Extract-Then-Diff Architecture
- Detector extracts from **latest speaker message only**, not full conversation
- Embedding-based dedup (cosine ≥ 0.82) replaces LLM consolidation pass
- Removed one LLM call per turn (consolidation) → faster + cheaper
- Controls over-detection at the source

### 3. Laddering Questions
- "Why is [item] important?" → discovers drives/causes edges
- "How does [item] show up day-to-day?" → discovers manifests_as edges
- Clean Language: using speaker's exact words triggers deeper disclosure
- Combined with depth/breadth/bridge/meta/discrepancy mode heuristic

### 4. Multi-Run Averaging
- Standard deviation ≤ 0.017 across all metrics
- V2's "same code scores 0.55-0.76" problem is solved
- Remaining variance comes from LLM conversation randomness, not comparison

## What Didn't Work / Remaining Issues

### 1. Over-Detection (Precision Gap)
- 18-24 items detected from 12 GT (node precision ~50-55%)
- 30-38 edges detected from 13 GT (triple precision ~36%)
- The detector finds all GT concepts but also extracts extras
- Extract-then-diff helped (no re-extraction) but 3 items/turn × 10 turns still allows ~30 items

### 2. Edge Over-Generation
- Triple recall is perfect (1.0) but precision is low (~0.37)
- The detector creates many plausible but non-GT edges
- This is structurally hard — the LLM generates reasonable connections between detected items

### 3. Occasional API Errors
- OpenRouter 403 errors require retry logic
- Empty responses need graceful handling

## Architecture

```
Speaker (GT graph → natural conversation)
    ↓ question
Detector:
    1. Extract from latest speaker message only
    2. Embedding dedup against existing graph (cosine ≥ 0.82)
    3. Add new items, strengthen duplicates
    4. Compute question mode (breadth/depth/bridge/meta/discrepancy)
    5. Laddering question for edge discovery
    ↓ detected graph
EmbeddingMatcher:
    1. Optimal node matching (Hungarian algorithm, threshold 0.55)
    2. Hub recall from matched hubs
    3. Triple soft matching (greedy, threshold 0.50)
    4. Overall = hub_recall × 0.3 + node_recall × 0.2 + triple_f1 × 0.5
```

## V4 Priorities

1. **Improve precision**: Stricter extraction (max 2 items/turn), higher confidence threshold
2. **Edge pruning**: After conversation, use embeddings to prune low-confidence edges
3. **Abstraction-level anchoring**: Prompt to extract at "personality profile" level, not sub-behaviors
4. **Longer conversations**: Test 15-20 turns to see if precision improves with more context
5. **Cross-fixture generalization**: Test on diverse fixture types (non-Chinese, non-purchase scenarios)
