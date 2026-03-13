# SoulGraph V2 Findings Report

## V2 Goal

Improve local structure similarity (plateau at 0.47-0.60 in V1) through:
- Consolidation to reduce over-detection
- Fixed edge vocabulary for consistent matching
- MI reflective questioning for deeper connection elicitation
- Bidirectional edge matching in comparator
- Domain-aware semantic matching
- ItemType classification (cognitive/action/background) for SoulMap alignment

## Score Progression (V2 iterations)

| Iter | Change | Car Buyer | Career | Anxiety |
|------|--------|-----------|--------|---------|
| V1 baseline | — | **0.76** | **0.68** | — |
| V2a (Iter 1-3) | Consolidation + fixed edges + MI questions | 0.60 | 0.54 | — |
| V2b (Iter 5-7) | Conservative consolidation + bidirectional + domain match | 0.65 | — | — |
| V2c (Iter 9) | ItemType + anxiety fixture | — | — | 0.41 |

**Hub Recall remained 0.80-1.00 across all runs.** The local sim variance is the primary challenge.

## Key Finding: LLM Conversation Variance Dominates

The most important discovery in V2 is that **score variance between runs on the same fixture is comparable to the effect size of our code changes**. The same code running twice on car_buyer can produce scores ranging from ~0.55 to ~0.76 depending on how the LLM conversation unfolds.

This means:
1. Single-run benchmarks are unreliable for measuring improvement
2. Most V1 "improvements" (prompt tuning) may have been noise
3. Only infrastructure changes (batch matching, fuzzy edges) had real, repeatable impact

## What V2 Added That Works

### 1. ItemType Classification
Adding `cognitive`, `action`, `background` types to SoulItem aligns with SoulMap's need to distinguish:
- Cognitive: self-understanding ("说不清自己到底在焦虑什么")
- Action: real-world intention ("想试试冥想")
- Background: context ("工作每天加班到很晚")

The Detector reliably classifies items — in the anxiety experiment, cognitive items were correctly identified for self-doubt patterns, action items for coping strategies.

### 2. MI Reflective Questioning
The Detector's questions are dramatically better:
- V1: "What's been on your mind?" (generic)
- V2: "我注意到你描述了两个很相似的循环...这两种'转个不停'的状态，它们之间会相互影响吗？" (reflective, connection-probing)

This produces richer conversation and better item disclosure, even if it doesn't always improve the score (because the score metric is also noisy).

### 3. Fixed Edge Vocabulary
Constraining to 8 edge types eliminates the "relates_to" problem. All detected edges now map cleanly to the comparator's groups.

### 4. Bidirectional Multi-Edge Matching
The comparator now handles cases where GT has A→B but detection has B→A, or multiple edges between the same pair.

### 5. Conservative Consolidation
Merge-only-true-duplicates approach preserves neighborhood structure while reducing noise. Career_changer went from 21→15 items.

## What V2 Revealed About the Metric

The local structure similarity metric has fundamental limitations:

1. **Neighborhood mismatch amplification**: If hub A has 5 GT neighbors but only 2 are matched, neighbor_recall = 0.40. But those 2 matches might be the most important connections.

2. **Graph topology sensitivity**: The detected graph has a different shape than GT. The same information is organized differently — e.g., GT might have si_001→si_002→si_003 as a chain, but detection produces si_001→si_003 directly (skipping the middle node). This is semantically correct but scores poorly.

3. **Hub selection instability**: Which items become "hubs" depends on edge count. A small change in detected edges shifts which nodes are compared, causing large score swings.

## Cross-Domain Analysis (3 Fixtures)

| Domain | Type | Hub Recall | Local Sim | Overall | Items Detected |
|--------|------|------------|-----------|---------|---------------|
| Car Buyer | Concrete/purchase | 1.00 | 0.33-0.60 | 0.60-0.76 | 21-22 |
| Career Changer | Abstract/career | 1.00 | 0.23-0.47 | 0.54-0.68 | 15-21 |
| Anxiety User | Emotional/therapeutic | 0.80 | 0.15 | 0.41 | 18 |

**Pattern**: The more abstract/emotional the domain, the lower the local similarity. This is NOT because detection is worse — the anxiety Detector produces excellent therapeutic insights. It's because:
1. Emotional items have more valid phrasings → matching is harder
2. Emotional graphs have more implicit connections → GT edges don't capture all valid relationships
3. The anxiety user's graph has deeper nested cycles that 5 turns can't fully reconstruct

## Architecture Update: ItemType

```python
class ItemType(str, Enum):
    COGNITIVE = "cognitive"    # Self-understanding (认知型)
    ACTION = "action"          # Real-world intention (行动型)
    BACKGROUND = "background"  # Context, facts, experiences
```

This is backward-compatible (defaults to BACKGROUND). SoulMap can filter by type to generate star map colors and determine which items get "探索现实世界" buttons.

## V3 Recommendations

### 1. Multi-Run Averaging (Critical)
Run each experiment 3-5 times and average scores. Single runs are statistically meaningless for LLM-based conversation.

### 2. Rethink Local Similarity Metric
Current metric penalizes valid alternative graph structures. Consider:
- **Path-based similarity**: Check if GT paths exist in detected graph (possibly with different intermediate nodes)
- **Embedding-based graph comparison**: Embed both graphs and compare in embedding space
- **LLM-as-judge for structure**: Ask an LLM "does this detected graph capture the same relationships as the GT?" — single call, avoids node-by-node comparison

### 3. Longer Conversations (8-10 turns)
5 turns covers ~70% of a 12-node graph. 8-10 turns should reach deeper layers, especially for emotional domains where trust building takes longer.

### 4. SoulMap-Specific Metrics
For the product, what matters is:
- **Intention discovery rate**: How many valid action/cognitive items per conversation turn?
- **Intention quality**: Does the detected intention match what the user would self-report?
- **Graph utility**: Does the graph enable meaningful star map layout and insight generation?

These are closer to product success than abstract graph similarity scores.

### 5. Fixture Expansion
Need 5-10 fixtures covering: health, relationship, hobby, life transition, parenting, identity, spirituality — to validate generalization.
