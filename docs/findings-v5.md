# V5 Findings Report — Multi-Session Memory

## Summary

V5 adds multi-session support: the detector's graph persists across conversations, the speaker resets each session with topic hints, and mention reinforcement boosts PageRank edge weights. The core hypothesis — that rankings improve over multiple sessions — was **partially validated**. Absorption and intention recall improve dramatically across sessions, but rank correlation slightly degrades due to graph noise growth.

## Benchmark Results

### Zhang Wei (3 sessions × 10 turns, 61 GT items)

| Metric | Session 1 | Session 2 | Session 3 | Δ (S3-S1) |
|--------|-----------|-----------|-----------|-----------|
| Rank Correlation | 0.597 | 0.580 | 0.565 | -0.032 |
| Domain NDCG@5 | 0.698 | 0.878 | 0.878 | +0.180 |
| Absorption Rate | 0.902 | 0.967 | 0.984 | +0.082 |
| Intention Recall | 0.600 | 1.000 | 1.000 | +0.400 |
| **Overall (V4)** | **0.689** | **0.831** | **0.829** | **+0.140** |
| Matched Items | 55/61 | 59/61 | 60/61 | +5 |
| Detected Items | 57 | 91 | 106 | +49 |

### Graph Growth

| Metric | Session 1 | Session 2 | Session 3 |
|--------|-----------|-----------|-----------|
| Detected Items | 57 | 91 | 106 |
| Detected Edges | 100 | 219 | 284 |
| Noise Ratio (det/gt) | 0.93x | 1.49x | 1.74x |

### Success Criteria Check

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Rank improvement > 0 | > 0 | -0.032 | ❌ NOT MET |
| Final absorption ≥ 0.85 | ≥ 0.85 | 0.984 | ✅ EXCEEDED |
| Final rank correlation ≥ 0.65 | ≥ 0.65 | 0.565 | ❌ NOT MET |
| Intention recall ≥ 0.90 | ≥ 0.90 | 1.000 | ✅ EXCEEDED |
| Overall improvement | > 0 | +0.140 | ✅ MET |

## Analysis

### What Worked

#### 1. Absorption Grows Across Sessions (Highest Impact)
- Session 1 absorbed 55/61 items (90.2%), Session 3 reached 60/61 (98.4%)
- Multi-session accumulation nearly achieves complete persona coverage
- The append-only graph correctly preserves all discoveries from prior sessions

#### 2. Intention Recall Recovers Perfectly
- Session 1 only found 60% of intentions (3/5) — limited by 10-turn conversation window
- Sessions 2-3 found all intentions (100%) as the accumulated graph gives context
- This validates that multi-session is essential for complex personas — single session is insufficient

#### 3. Domain NDCG Improves Significantly (+0.180)
- Session 1 NDCG was only 0.698 — with incomplete coverage, domain rankings suffer
- Sessions 2-3 stabilized at 0.878 — once most items are absorbed, domain-contextual ranking works well
- Topic-steered sessions successfully cover different domains across conversations

#### 4. Overall Score Improves (+0.140)
- Overall jumped from 0.689 → 0.831 between sessions 1 and 2 (the biggest gains)
- Session 3 maintained at 0.829 — diminishing returns after coverage is high
- Multi-session V5 overall (0.829) matches V4 single-session overall (0.826)

#### 5. Topic-Steered Speaker Works
- Session 1 (career/family/finance): conversation stayed on financial anxiety and career choices
- Session 2 (identity/social/values): shifted to deeper identity exploration, values
- Session 3 (health/hobbies/stories/career): touched health concerns, hobbies, and revisited career
- Topic hints successfully bias conversation without feeling forced

### What Didn't Work

#### 1. Rank Correlation Degraded (-0.032)
- This is the key finding: **more items doesn't mean better ranking**
- Session 1: 57 detected items (0.93x GT) → rank_corr 0.597
- Session 3: 106 detected items (1.74x GT) → rank_corr 0.565
- Extra non-GT items participate in PageRank, diluting the ranking signal
- The noise grows faster than the mention reinforcement can compensate

#### 2. Mention Reinforcement Too Weak
- The 0.2 coefficient boost (`strength × (1 + 0.2 × mention_count)`) isn't enough
- With 106 detected items vs 61 GT, the extra ~45 nodes absorb PageRank mass
- Mention-reinforced items get marginal boost but it's overwhelmed by structural noise
- Need either: stronger coefficient (0.5+), or edge pruning to reduce noise

#### 3. Session 3 Detection Stalled
- Session 3 got stuck at 106 items / 284 edges for the last 5 turns
- Partly due to transient 403 API errors (OpenRouter rate limits)
- Also because with 100+ items already in the graph, the detector prompt becomes very long
- Context window pressure may prevent the detector from finding new items

#### 4. No Rank Correlation Target Met
- Neither the final target (0.65) nor improvement target (>0) was achieved
- This suggests rank correlation is fundamentally limited by the over-detection paradigm
- More data helps coverage but hurts precision of ranking

## Comparison: V4 Single-Session vs V5 Multi-Session

| Metric | V4 (15 turns) | V5 Session 1 (10t) | V5 Final (30t) |
|--------|---------------|---------------------|-----------------|
| Rank Correlation | 0.600 ± 0.041 | 0.597 | 0.565 |
| Domain NDCG | 0.886 ± 0.045 | 0.698 | 0.878 |
| Absorption | 0.967 ± 0.013 | 0.902 | 0.984 |
| Intention Recall | 0.933 ± 0.094 | 0.600 | 1.000 |
| Overall | 0.826 ± 0.021 | 0.689 | 0.829 |
| Detected Items | 76-87 | 57 | 106 |

Key insight: V5 with 30 total turns achieves similar overall to V4 with 15 turns, but with better absorption/intention recall and worse rank correlation. The multi-session architecture works for coverage but the ranking signal doesn't scale.

## V6 Priorities

1. **Edge pruning / noise reduction**: After each session, prune low-confidence edges or remove items with no GT match candidates. This is the #1 priority — rank correlation can't improve while noise grows unchecked.

2. **Stronger mention reinforcement**: Increase coefficient from 0.2 to 0.5+, or use exponential weighting (`strength × 2^(mention_count/3)`) so repeated items dominate PageRank more aggressively.

3. **Graph compaction**: Merge semantically similar detected items (cosine ≥ 0.90) to reduce node count. Currently 106 detected from 61 GT — many are near-duplicates that fragment the ranking.

4. **Adaptive detection**: When graph has 80+ items, switch detector to "refinement mode" — focus on confirming/deepening existing items rather than extracting new ones. Reduces noise in later turns.

5. **Multi-run stability**: This report is from a single run. Need 3+ runs for statistical confidence. 403 API errors affected Session 3 detection — need more robust retry/backoff.

6. **Longer sessions or more sessions**: Test 4-5 sessions to see if there's an inflection point where mention reinforcement overcomes noise.
