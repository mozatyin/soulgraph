# Dual Soul Architecture — Deep Soul + Surface Soul

> **Inspired by Kahneman's Thinking Fast and Slow**
> Surface Soul = System 1 (fast, current, ephemeral)
> Deep Soul = System 2 (slow, enduring, compressed)

**Goal:** Two independent SoulGraph instances that model a person's mind at two timescales — Surface captures the present moment, Deep captures enduring personality.

**Architecture:** Periodic batch processing — Think Fast rebuilds Surface every N utterances, Think Slow consolidates Surface→Deep every M utterances. Deep uses aggressive merging (cosine 0.65) to compress; Surface uses standard extraction (cosine 0.82) for precision.

---

## Design Version: V1 (Initial)

### 1. Data Structure

```python
class DualSoul:
    surface: SoulGraph          # Current state — small, fresh
    deep: SoulGraph             # Long-term personality — compressed, stable

    surface_window: int = 20    # N: Surface covers last N utterances
    deep_cycle: int = 100       # M: Deep consolidation every M utterances
    merge_threshold: float = 0.65   # Deep merging cosine threshold
    decay_rate: float = 0.9     # Per-cycle decay for unmentioned Deep nodes

    total_utterances: int = 0
    _messages: list[Message]    # Full conversation history
```

**SoulItem additions:**
- `last_consolidated: datetime` — when Deep node was last updated by consolidation
- `decay_score: float = 1.0` — decays each cycle if not reinforced by Surface

### 2. Surface Soul — Think Fast

**Trigger:** Every N=20 utterances, full rebuild from recent window.

```
Flow:
1. Create fresh empty SoulGraph
2. Take last 20 messages from _messages
3. Run Detector on them sequentially (reuse existing extraction)
4. Result = new Surface Soul (~80-100 nodes)
5. Discard old Surface
```

**Why full rebuild (not incremental):**
- Guarantees internal consistency — no orphaned edges
- Fresh Detector context = no accumulated drift
- 20 LLM calls ≈ 30 seconds — acceptable

### 3. Deep Soul — Think Slow

**Trigger:** Every M=100 utterances, batch consolidation.

```
Flow for each Surface node:
1. Compute cosine similarity against all Deep nodes
2. If max_sim >= 0.65:
   → LLM merges Surface text into Deep node text
   → Deep node: mention_count++, confidence += 0.05
   → Record mapping: surface_id → deep_id
3. If max_sim < 0.65:
   → Copy Surface node into Deep as new node
   → Record mapping: surface_id → new_deep_id

Then for Surface edges:
4. Remap edge endpoints using mapping table
5. If Deep already has same (from, to, relation): strengthen
6. If not: add new edge

Then decay:
7. For each Deep node NOT hit by any Surface node this cycle:
   → decay_score *= 0.9
   → Nodes with decay_score < 0.1 are "dormant" (low PageRank weight, never deleted)
```

**LLM text merge prompt:**
```
Merge these two descriptions of the same concept into one concise sentence.
Keep the core meaning. Incorporate any new details from the second description.

Existing: "{deep_text}"
New observation: "{surface_text}"

Return only the merged description, one sentence.
```

### 4. Query — Dual PPR

```python
def query(self, question: str, top_k: int = 10) -> str:
    # Run PPR on both graphs
    surface_sub = self.surface.query_subgraph(question, top_k=top_k//2)
    deep_sub = self.deep.query_subgraph(question, top_k=top_k//2)

    # Merge results for LLM synthesis
    # Surface nodes tagged as [current], Deep nodes tagged as [enduring]
    # LLM gets both and synthesizes answer
```

**Query routing heuristic:**
- "what is she thinking right now" → Surface weight 0.8, Deep weight 0.2
- "what kind of person is she" → Surface weight 0.2, Deep weight 0.8
- Default: Surface 0.5, Deep 0.5

### 5. Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| N (surface_window) | 20 | ~80-100 nodes, manageable; covers recent context |
| M (deep_cycle) | 100 | 5 Surface cycles per Deep cycle; enough data to find patterns |
| merge_threshold | 0.65 | More aggressive than Mem0's 0.7; Deep needs compression |
| confidence_boost | 0.05 | Small increment per merge; caps at 1.0 |
| decay_rate | 0.9 | 10% per cycle; after 10 cycles without mention → 0.35 |
| decay_floor | 0.1 | Never delete; dormant nodes can be revived |

---

## Iteration Log

### V1 Weaknesses Identified

1. **N=20 full rebuild is wasteful** — 20 LLM calls every 20 utterances = 1 call per utterance on average (same as no batching). The "Think Fast" name implies speed but it's not faster.

2. **merge_threshold 0.65 is a guess** — No empirical basis. Too low = over-merging (loss of nuance). Too high = no compression. Need to test.

3. **No query routing mechanism** — "Surface weight 0.8 vs 0.2" is hand-wavy. How does the system know which type of question it is?

4. **Edge migration is fragile** — Surface edges mapped to Deep nodes via a mapping table. What about edges where one endpoint merged into Deep node A and the other into Deep node B, but A and B already had a different edge? Edge conflicts.

5. **Decay is simplistic** — Linear 0.9^n decay treats all unmentioned nodes equally. A core belief ("I'm religious") shouldn't decay the same as a transient observation.

6. **M=100 means first Deep consolidation happens very late** — User needs 100 utterances before any Deep Soul exists. For short conversations (<100 messages), Deep Soul is empty.

7. **Surface Soul is empty between rebuilds** — After utterance 1-19, Surface hasn't been built yet. After rebuild at 20, it's stale by utterance 39.

---

### V2: Addressing Weaknesses

**Fix #1: Surface is incremental, not full rebuild**
- Surface uses the SAME Detector as before — each new utterance → extract into Surface
- Every N utterances, Surface gets COMPRESSED (merge similar nodes within itself)
- This is faster (1 LLM call per utterance, same as current) and Surface is never empty

**Fix #6: Deep consolidation starts earlier**
- First Deep consolidation at M=50 (or even M=20)
- Subsequent consolidations every M=100
- Or: adaptive — consolidate when Surface reaches a node threshold (e.g., 150 nodes)

**Fix #7: Surface is always live**
- Surface continuously ingests (like current SoulEngine)
- Periodic compression keeps it manageable
- No "empty between rebuilds" problem

**Revised Flow:**
```
Utterance 1:  → Extract to Surface (normal)
Utterance 2:  → Extract to Surface
...
Utterance 20: → COMPRESS Surface (merge similar nodes, threshold 0.72)
              → Surface shrinks from ~80 nodes to ~40-50
Utterance 21: → Extract to Surface (continues)
...
Utterance 40: → COMPRESS Surface again
...
Utterance 100: → CONSOLIDATE Surface → Deep (Think Slow)
               → RESET Surface (start fresh)
```

### V2 Weaknesses

1. **Surface compression threshold (0.72) is yet another magic number** — Now we have THREE thresholds: Surface extraction dedup (0.82), Surface compression (0.72), Deep merge (0.65). Too many knobs.

2. **"RESET Surface" after Deep consolidation** — Loses the last 0-19 utterances of context. Should it carry forward?

3. **Still no principled query routing**

---

### V3: Simplify — Two Thresholds Max

**Key insight from cognitive science research:** The brain doesn't have separate "compression" and "merge" operations. It has ONE mechanism — replay during sleep — that strengthens strong connections and lets weak ones fade.

**Simplification:**
- **Surface**: Standard Detector extraction (0.82 dedup). No compression. Just a rolling window.
- **Deep**: One merge threshold (0.65). Consolidation = the "sleep" phase.
- **Rolling window, not periodic rebuild**: Surface keeps last N utterances' worth of nodes. When a node's source utterance falls outside the window, it's evicted → triggers Deep update.

```
Surface Soul = sliding window of size W nodes (not W utterances)
- New extraction → add to Surface (normal Detector, 0.82 dedup within Surface)
- When Surface.items > W:
  → Evict oldest nodes (by source_turn)
  → For each evicted node: merge into Deep (0.65 threshold)

Deep Soul = append-only with merge
- Receives evicted Surface nodes
- merge if similar (0.65) or add if novel
- decay_score for unmentioned nodes (0.9 per eviction batch)
```

**Why this is better:**
- Only 2 thresholds (0.82 for Surface dedup, 0.65 for Deep merge)
- Surface is ALWAYS live — never empty, never stale
- Deep grows organically — no arbitrary M=100 cycle
- Eviction is node-count driven, not utterance-count driven
- Natural backpressure: more redundant input → more dedup → slower Surface growth → less frequent Deep updates

### V3 Weaknesses

1. **Per-node eviction = per-node LLM call for Deep merge text** — If 10 nodes are evicted at once, that's 10 LLM calls for text merging. Could batch them.

2. **Source_turn ordering may not reflect importance** — Evicting "oldest" means evicting first-mentioned concepts, which might be the MOST important (primacy effect).

3. **W (window size) is the new magic number** — What should it be? 100? 200?

4. **No periodic "sleep" phase** — User said "Think Slow = every M utterances". V3 lost the periodic rhythm the user wanted.

---

### V4: Honor the User's Design — Periodic Dual Rhythm + Best of V3

**Reconcile:** The user explicitly said:
- Think Fast → 每隔 N 句话来构建更新 Surface Soul
- Think Slow → 每个 M 句话来更新 Deep Soul

The periodicity is intentional. It models the brain's sleep cycle — not continuous, but rhythmic.

**V4 Design:**

```
Surface Soul (Think Fast):
- Continuous extraction: every utterance → Detector adds to Surface (0.82 dedup)
- Every N=20 utterances: Surface COMPACTION
  → Merge nodes within Surface that are now similar (0.72 threshold)
  → This handles the "repeated themes" problem within Surface
  → Surface shrinks, stays manageable

Deep Soul (Think Slow):
- Every M=100 utterances: CONSOLIDATION
  → Take ALL Surface nodes
  → For each: find best Deep match (embedding similarity)
    - sim >= 0.65 → LLM merge text + mention_count++ + confidence boost
    - sim < 0.65 → add as new Deep node
  → Migrate edges (remap via mapping table, strengthen existing)
  → Apply decay to Deep nodes not matched this round
  → CLEAR Surface, start fresh cycle

- First consolidation at M₀=50 (earlier bootstrap)
- Subsequent consolidations every M=100

Query:
- Always search BOTH graphs
- Surface results tagged [recent], Deep results tagged [enduring]
- LLM synthesizes from both, weighing recency vs permanence
```

**Parameter consolidation:**

| Parameter | Value | Role |
|-----------|-------|------|
| Surface dedup | 0.82 | Within-Surface, per-utterance (existing) |
| Surface compaction | 0.72 | Within-Surface, every N utterances (new) |
| Deep merge | 0.65 | Surface→Deep, every M utterances (new) |
| N | 20 | Surface compaction cycle |
| M₀ | 50 | First Deep consolidation |
| M | 100 | Subsequent Deep consolidation interval |
| decay_rate | 0.9 | Per-cycle decay for unmatched Deep nodes |

### V4 Weaknesses

1. **Three thresholds is still too many** — Can we eliminate Surface compaction?
2. **M₀=50 vs M=100 is ad-hoc** — Why not just M=50 always?
3. **CLEAR Surface after Deep consolidation** — Last few utterances' context lost
4. **LLM text merge is expensive** — Every matched node needs an LLM call during consolidation. For 80 Surface nodes, that's 80 LLM calls in the "sleep" phase.

---

### V5: Batch LLM Merge + Eliminate Surface Compaction

**Insight:** Surface compaction (0.72) is redundant. The Detector already deduplicates at 0.82 within Surface. If themes truly repeat, the Detector will strengthen existing Surface nodes, not create duplicates. The real compression happens at Deep consolidation.

**Drop Surface compaction entirely.** Two thresholds only: 0.82 (Surface dedup) and 0.65 (Deep merge).

**Batch LLM merge:** Instead of one LLM call per node, batch all merge operations into a single call:

```python
CONSOLIDATION_PROMPT = """
You are consolidating a knowledge graph. For each pair below, merge the
"new observation" into the "existing concept" to create an updated description.

{pairs}

Return JSON array:
[
  {"deep_id": "...", "merged_text": "one concise sentence"},
  ...
]
"""
```

One LLM call handles all merges (up to ~40-50 pairs per call, well within context).

**Don't CLEAR Surface after consolidation.** Instead:
- After Deep consolidation, Surface continues accumulating
- Surface capacity is soft-limited: if Surface > 200 nodes, trigger early consolidation
- This means Surface is never empty

**V5 Design (final candidate):**

```
Surface Soul:
- Every utterance → Detector extracts into Surface (0.82 dedup)
- Surface grows continuously
- Surface > 200 nodes → trigger early Deep consolidation

Deep Soul:
- Every M=100 utterances (or Surface > 200 nodes): CONSOLIDATION
  1. Embed all Surface nodes + all Deep nodes
  2. Compute similarity matrix (Surface × Deep)
  3. Partition Surface nodes:
     - MERGE group: max_sim >= 0.65 → pair with best Deep match
     - NEW group: max_sim < 0.65 → will become new Deep nodes
  4. Batch LLM call: merge all MERGE pairs → updated Deep texts
  5. Apply updates: text, mention_count++, confidence boost, last_consolidated
  6. Add NEW group nodes to Deep
  7. Migrate edges (remap, strengthen/add)
  8. Decay unmatched Deep nodes (0.9×)
  9. CLEAR Surface (restart fresh)

Query:
- Search both Surface and Deep with PPR
- Merge results, Surface items get recency boost
- LLM synthesizes answer from combined subgraph

First consolidation: when Surface hits 150 nodes (adaptive, not fixed M₀)
```

**Two thresholds: 0.82 (Surface) and 0.65 (Deep). Clean.**

### V5 Weaknesses

1. **CLEAR Surface after consolidation still loses context** — The utterances that were just ingested (after the last Detector call but before consolidation) are fine, but the graph structure of "how current topics relate" is lost.

2. **Decay 0.9 is uniform** — High-confidence Deep nodes (beliefs, values) should decay slower than low-confidence ones (observations, moods).

3. **No mechanism for Deep→Surface feedback** — If querying Deep reveals something relevant to current Surface context, there's no way to bring it back.

4. **200-node threshold for early consolidation is arbitrary.**

---

### V6: Confidence-Weighted Decay + Carry Forward

**Fix decay:**
```python
effective_decay = decay_rate ** (1 / item.confidence)
# confidence=0.9 → decay^1.11 = 0.988 (almost no decay)
# confidence=0.3 → decay^3.33 = 0.717 (fast decay)
```

High-confidence items (core beliefs, strong patterns) resist decay. Low-confidence items (weak inferences, one-time observations) fade faster. This is biologically accurate — strongly encoded memories are more resistant to interference.

**Fix Surface clear — carry forward top nodes:**
```python
def consolidate_deep(self):
    # ... merge/add/decay as V5 ...

    # Don't fully clear Surface — keep top-K nodes by PageRank
    # These are the "active threads" that should persist
    pr = self.surface.pagerank()
    top_ids = sorted(pr, key=pr.get, reverse=True)[:10]
    carry_items = [i for i in self.surface.items if i.id in top_ids]
    carry_edges = [e for e in self.surface.edges
                   if e.from_id in top_ids and e.to_id in top_ids]

    # Reset Surface with carried-forward nodes
    self.surface = SoulGraph(owner_id=self.owner_id,
                             items=carry_items, edges=carry_edges)
```

This preserves the most important current threads across consolidation boundaries.

**V6 = V5 + confidence-weighted decay + top-K carry-forward.**

### V6 Weaknesses

1. **Carry-forward nodes get double-counted** — They're in Surface AND were just merged into Deep. Next consolidation, they'll merge into Deep again (boosting mention_count artificially).

2. **Still no query routing** — "Is she thinking about X right now" vs "Is she the kind of person who X" both query both graphs equally.

---

### V7: Dedup Carry-Forward + Simple Query Routing

**Fix double-counting:** Mark carried-forward nodes with a flag `carried_forward=True`. During next consolidation, skip these unless they were strengthened by new extractions (mention_count increased since carry-forward).

**Simple query routing via keywords:**
```python
RECENCY_KEYWORDS = {"now", "right now", "currently", "today", "lately", "recent"}
ENDURING_KEYWORDS = {"always", "kind of person", "generally", "core", "deep", "usually", "personality"}

def _route_query(self, question: str) -> tuple[float, float]:
    """Returns (surface_weight, deep_weight)."""
    q_lower = question.lower()
    has_recency = any(kw in q_lower for kw in RECENCY_KEYWORDS)
    has_enduring = any(kw in q_lower for kw in ENDURING_KEYWORDS)

    if has_recency and not has_enduring:
        return (0.7, 0.3)
    elif has_enduring and not has_recency:
        return (0.3, 0.7)
    else:
        return (0.5, 0.5)
```

This is simple, deterministic, and good enough. Can upgrade to LLM-based routing later if needed.

**V7 = V6 + carry-forward dedup + keyword query routing.**

### V7 Self-Assessment

**Strengths:**
- Two thresholds only (0.82, 0.65) — minimal knobs
- Continuous Surface extraction — never empty
- Batch LLM merge — efficient consolidation
- Confidence-weighted decay — biologically inspired
- Carry-forward preserves context
- Simple query routing

**Remaining concerns:**
- merge_threshold 0.65 needs empirical validation
- Consolidation cost: 1 batch LLM call + 1 embedding computation ≈ 10 seconds. Acceptable.
- Deep Soul growth rate: unknown until tested. If 0.65 is right, Deep should converge.

---

## FINAL DESIGN: V7

### Architecture Summary

```
                    ┌─────────────────────────┐
                    │      DualSoul           │
                    │                         │
  ingest(text) ───→ │  Surface Soul (live)    │
                    │  - Detector extracts    │
                    │  - 0.82 dedup           │
                    │  - ~80-200 nodes        │
                    │                         │
  every M utts ───→ │  ══ CONSOLIDATION ══    │
  or >200 nodes     │  1. Embed all nodes     │
                    │  2. Similarity matrix   │
                    │  3. Batch LLM merge     │
                    │  4. Decay unmatched     │
                    │  5. Carry forward top-K │
                    │                         │
                    │  Deep Soul (enduring)   │
                    │  - Compressed, stable   │
                    │  - 0.65 merge threshold │
                    │  - Grows slowly         │
                    │  - Confidence-weighted  │
                    │    decay                │
                    │                         │
  query(q) ───────→ │  PPR on both graphs     │
                    │  Route by keywords      │
                    │  Merge + synthesize     │
                    └─────────────────────────┘
```

### Parameters (Final)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| surface_dedup | 0.82 | Existing, proven for within-conversation dedup |
| deep_merge | 0.65 | Aggressive compression for long-term; lower than Mem0's 0.7 |
| deep_cycle (M) | 100 | ~5 minutes of conversation; enough for pattern detection |
| max_surface_nodes | 200 | Soft cap; triggers early consolidation |
| carry_forward_k | 10 | Top-10 Surface nodes persist across consolidation |
| decay_rate | 0.9 | Base rate; modulated by confidence |
| confidence_boost | 0.05 | Per-merge increment |
| first_consolidation | 150 nodes | Adaptive; not fixed utterance count |

### API Design

```python
class DualSoul:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514",
                 deep_cycle: int = 100, merge_threshold: float = 0.65):
        ...

    def ingest(self, text: str) -> None:
        """Add utterance, extract to Surface. Auto-consolidates when needed."""

    def query(self, question: str, top_k: int = 10) -> str:
        """Query both souls, route by keywords, synthesize answer."""

    def consolidate(self) -> dict:
        """Force Deep consolidation. Returns stats (merged, added, decayed)."""

    def save(self, path: str) -> None:
        """Save both graphs to JSON."""

    def load(self, path: str) -> None:
        """Load both graphs from JSON."""

    @property
    def surface(self) -> SoulGraph: ...

    @property
    def deep(self) -> SoulGraph: ...

    @property
    def stats(self) -> dict:
        """Return node counts, edge counts, consolidation history."""
```

### Persistence Format

```json
{
  "owner_id": "scarlett",
  "total_utterances": 257,
  "consolidation_count": 2,
  "surface": { /* SoulGraph JSON */ },
  "deep": { /* SoulGraph JSON */ },
  "config": {
    "deep_cycle": 100,
    "merge_threshold": 0.65,
    "decay_rate": 0.9,
    "carry_forward_k": 10,
    "max_surface_nodes": 200
  }
}
```

---

## Research-Informed Iterations (V8-V10)

### Research Findings Applied

**From FSRS (spaced repetition):**
- Memory stability S grows with each successful reinforcement
- Retrievability R = e^(-t/S) — exponential decay governed by stability
- Items reviewed successfully have INCREASING intervals between reviews
- Key insight: **stability is not fixed** — it grows with reinforcement

**From entity resolution research:**
- Productive threshold range: 0.7-0.95
- Start high for precision, tune down for recall
- **Adaptive thresholds based on graph density** — RELATER uses average neighborhood similarity
- LLM-hybrid: use embedding threshold as coarse filter, LLM as fine judge

**From Cowan's working memory (cognitive science):**
- Focus of attention: 4±1 chunks (not 7)
- Three tiers: Hot (4-7), Warm (20-50), Cold (unbounded)
- Chunks, not items — aggregated units count as one

**From fixed-capacity knowledge stores:**
- Best: combine recency + frequency + importance scoring
- GraphRAG: hierarchical communities via Leiden clustering
- Don't delete — summarize into higher-level community nodes

---

### V8: FSRS-Inspired Stability Model for Deep Decay

**Problem with V7:** `decay_rate ** (1/confidence)` is a one-shot heuristic. It doesn't model how memories actually strengthen over time.

**FSRS-inspired approach:** Each Deep node has a `stability` value (S). Stability determines how slowly the node decays. Each time a node is reinforced (matched by Surface during consolidation), stability INCREASES.

```python
class DeepSoulItem(SoulItem):
    stability: float = 1.0        # S: days-equivalent until retrievability drops to 90%
    retrievability: float = 1.0   # R: current probability of being "active"
    last_reinforced: int = 0      # utterance count at last reinforcement
    reinforcement_count: int = 0  # total times reinforced

def update_retrievability(self, current_utterance: int):
    """FSRS-inspired: R = e^(-t/S) where t = utterances since last reinforcement."""
    t = current_utterance - self.last_reinforced
    self.retrievability = math.exp(-t / max(self.stability, 0.1))

def reinforce(self, current_utterance: int):
    """When Surface matches this Deep node during consolidation."""
    self.last_reinforced = current_utterance
    self.reinforcement_count += 1
    self.retrievability = 1.0
    # Stability grows with each reinforcement (like FSRS)
    # First reinforcement: S=1→2, second: 2→4.5, third: 4.5→10...
    self.stability *= (1 + 0.5 * math.log(1 + self.reinforcement_count))
```

**Why this is better than flat decay:**
- A node reinforced 10 times has stability ~50 → barely decays over 100 utterances
- A node reinforced once has stability ~1.7 → drops to 0.56 retrievability after 1 cycle
- Core beliefs get reinforced repeatedly → near-permanent
- One-time observations → fade naturally
- No arbitrary `decay_rate` parameter needed

**Impact on PageRank:** Use `retrievability` as a weight multiplier:
```python
# In _to_nx for Deep Soul:
weight = edge.strength * (1 + 0.2 * target.mention_count) * target.retrievability
```

Dormant nodes (R < 0.1) contribute almost nothing to PageRank but aren't deleted.

### V8 Weaknesses

1. **Stability formula is borrowed from spaced repetition for flashcards** — soul graph concepts aren't flashcards. A concept can be "stable" without being repeatedly mentioned (e.g., deeply held belief mentioned once with high confidence).

2. **Initial stability should depend on extraction confidence** — A directly stated fact (confidence 0.9) should start with higher stability than a weak inference (confidence 0.3).

---

### V9: Confidence-Seeded Stability + Adaptive Merge Threshold

**Fix stability initialization:**
```python
# Initial stability seeded by extraction confidence
initial_stability = 1.0 + 4.0 * item.confidence
# confidence=0.9 → stability=4.6 (decays slowly from start)
# confidence=0.3 → stability=2.2 (decays faster)
# confidence=0.5 → stability=3.0 (moderate)
```

**Adaptive merge threshold based on Deep graph density:**

Research shows threshold should adapt to graph state. When Deep is sparse, be conservative (avoid premature merging). When Deep is dense, merge more aggressively.

```python
def _adaptive_merge_threshold(self) -> float:
    """Threshold decreases as Deep Soul grows — more aggressive compression for larger graphs."""
    n = len(self.deep.items)
    if n < 30:
        return 0.78  # Conservative: few nodes, keep them distinct
    elif n < 100:
        return 0.72  # Moderate
    elif n < 300:
        return 0.65  # Aggressive
    else:
        return 0.58  # Very aggressive for large graphs
```

**Why adaptive:**
- Early conversations: few concepts, each is unique → high threshold preserves nuance
- After 500+ utterances: many overlapping concepts → low threshold forces compression
- Natural convergence: as Deep grows, threshold drops, merge rate increases, growth slows
- This is the mechanism that MAKES Deep Soul converge

**Eliminates the fixed 0.65 magic number.** The threshold is now a function of graph size.

### V9 Self-Assessment

**Strengths over V7:**
- FSRS-inspired stability replaces arbitrary decay_rate
- Confidence seeds initial stability (biologically sound)
- Adaptive threshold eliminates magic number and drives convergence
- Only ONE explicit threshold left: 0.82 for Surface dedup (proven, keep it)

**Remaining concern:**
- The step-function for adaptive threshold (0.78/0.72/0.65/0.58) has arbitrary breakpoints
- Should it be a smooth function instead?

---

### V10: Smooth Adaptive Threshold + Final Architecture

**Smooth threshold function:**
```python
def _adaptive_merge_threshold(self) -> float:
    """Smooth logarithmic curve: starts high, asymptotes low."""
    n = max(len(self.deep.items), 1)
    # threshold = 0.82 - 0.06 * ln(n)
    # n=1   → 0.82  (same as Surface — no extra merging)
    # n=10  → 0.68
    # n=50  → 0.59
    # n=100 → 0.54
    # n=500 → 0.45
    return max(0.40, 0.82 - 0.06 * math.log(n))
```

This provides a smooth, principled curve that:
- Starts conservative (0.82) when Deep is nearly empty
- Gradually becomes more aggressive as concepts accumulate
- Floors at 0.40 to prevent over-merging (things that are only 40% similar shouldn't merge)
- Uses ONE formula with ONE coefficient (0.06) instead of step-function breakpoints

**Convergence analysis:**
- At threshold 0.82: ~3.1 nodes/line (our convergence experiment data)
- At threshold 0.65: estimated ~1.5 nodes/line (Mem0 is lower at 0.7)
- At threshold 0.50: estimated <0.5 nodes/line (mostly merging)
- Deep Soul growth rate decreases as graph grows → natural convergence

---

## FINAL DESIGN: V10

### Architecture

```
┌──────────────────────────────────────────────────────┐
│                    DualSoul                           │
│                                                      │
│  ┌─────────────────────┐                             │
│  │   Surface Soul      │ ← ingest(text) every turn  │
│  │   Think Fast        │                             │
│  │   - Live Detector   │                             │
│  │   - 0.82 dedup      │                             │
│  │   - Grows freely    │                             │
│  └────────┬────────────┘                             │
│           │                                          │
│     every M utterances                               │
│     or Surface > 200 nodes                           │
│           │                                          │
│  ┌────────▼────────────┐                             │
│  │   CONSOLIDATION     │  "Sleep Phase"              │
│  │   1. Embed all      │                             │
│  │   2. Adaptive       │ threshold = f(|Deep|)       │
│  │      threshold      │ 0.82 - 0.06·ln(n)          │
│  │   3. Batch LLM      │ merge matched pairs         │
│  │      merge          │                             │
│  │   4. FSRS decay     │ R = e^(-t/S)               │
│  │   5. Carry fwd      │ top-10 by PageRank          │
│  │      top-K          │                             │
│  └────────┬────────────┘                             │
│           │                                          │
│  ┌────────▼────────────┐                             │
│  │   Deep Soul         │                             │
│  │   Think Slow        │                             │
│  │   - Compressed      │                             │
│  │   - FSRS stability  │ S grows with reinforcement  │
│  │   - Converges       │ via adaptive threshold      │
│  │   - Never deletes   │ low-R nodes go dormant      │
│  └─────────────────────┘                             │
│                                                      │
│  query(q) ──→ PPR on both ──→ route by keywords      │
│               ──→ merge results ──→ LLM synthesize   │
└──────────────────────────────────────────────────────┘
```

### Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| surface_dedup | 0.82 | Existing, empirically validated |
| deep_merge | `0.82 - 0.06·ln(n)` | Adaptive; n = Deep node count |
| deep_merge_floor | 0.40 | Minimum threshold to prevent over-merging |
| deep_cycle (M) | 100 | Periodic consolidation interval |
| max_surface_nodes | 200 | Soft cap triggers early consolidation |
| carry_forward_k | 10 | Top-K Surface nodes persist |
| initial_stability | `1 + 4·confidence` | FSRS-inspired, seeded by extraction confidence |
| stability_growth | `S *= (1 + 0.5·ln(1 + reinforcement_count))` | FSRS-inspired |
| retrievability | `e^(-t/S)` | FSRS exponential decay |
| dormant_threshold | 0.1 | R < 0.1 = dormant (not deleted) |

### Key Design Decisions

1. **Surface is always live** — continuous extraction, never empty, never stale
2. **Deep consolidation is periodic** — "sleep phase" every M utterances (honors user's Think Fast/Slow rhythm)
3. **Adaptive merge threshold** — smooth curve drives natural convergence as Deep grows
4. **FSRS stability** — replaces flat decay; reinforced concepts resist fading
5. **Batch LLM merge** — one call per consolidation, not per node
6. **Carry-forward top-K** — preserves active threads across consolidation
7. **Never delete** — dormant nodes can revive if user revisits topic
8. **Keyword query routing** — simple, deterministic, upgradeable

### What Makes This Different From Mem0/MemGPT

| Feature | Mem0 | MemGPT | DualSoul (Ours) |
|---------|------|--------|-----------------|
| Memory tiers | 1 (flat) | 2 (context/archival) | 2 (Surface/Deep) with distinct semantics |
| Merge mechanism | LLM per-fact | Recursive summarization | Batch LLM + embedding similarity |
| Threshold | Fixed 0.7 | N/A | Adaptive f(graph_size) |
| Decay model | None | None | FSRS stability + retrievability |
| Graph structure | Entity-relation | None (text chunks) | Full SoulGraph (items + edges + PageRank) |
| Query method | Vector similarity | Context paging | Dual PPR + keyword routing |
| Periodicity | Per-message | Per-message | Periodic "sleep" cycles (cognitively inspired) |

### Implementation Complexity

- **DualSoul class**: ~200 lines (wraps two SoulGraphs + consolidation logic)
- **FSRS decay**: ~20 lines (3 formulas)
- **Adaptive threshold**: ~5 lines (1 formula)
- **Batch merge prompt**: ~15 lines (1 LLM call template)
- **Query routing**: ~15 lines (keyword matching)
- **Tests**: ~150 lines (consolidation, decay, merge, routing)
- **Total new code**: ~400 lines

### Open Questions for Empirical Testing

1. **Does carry-forward-K=10 cause drift?** Compare query quality with K=0 vs K=10.
2. **Does batch LLM merge preserve nuance?** Audit merged texts for information loss.
3. **Optimal coefficient for adaptive threshold** — 0.06 gives smooth curve, but needs A/B testing.

---

## Convergence Experiment Results (Empirical Data)

**Run:** 220 lines of Scarlett dialogue (57 original + 163 generated with repeated themes)

```
Lines  1- 57 (original):     3.1 items/line → 175 items
Lines 58-107 (extended 1):   3.1 items/line → 329 items
Lines 108-157 (extended 2):  1.7 items/line → 413 items  ← deceleration
Lines 158-220 (extended 3):  0.0 items/line → 416 items  ← SATURATED
```

**Key finding:** Even at 0.82 threshold, the graph SATURATES when themes repeat. After ~160 lines, zero new items were added. The graph converged at 416 items for ~10 core themes.

**Implications for Dual Soul:**
- Surface Soul (0.82 dedup) naturally converges within its topic window
- Deep Soul (adaptive threshold starting at 0.82, dropping to ~0.55) will converge FASTER
- Convergence is theme-driven, not just threshold-driven — repeated themes ARE caught
- With 0.65 threshold, estimated convergence at ~80-100 lines (half the current)
- 1000 lines with mixed new+repeated themes → log fit projects ~615 items (not 4600)

**Growth rate by segment:**
```
Lines   1- 10: 3.9/line (exploring)
Lines  41- 50: 2.2/line (decelerating)
Lines  81- 90: 2.2/line (stable)
Lines 111-120: 1.6/line (converging)
Lines 131-140: 1.5/line (near-saturated)
Lines 151-160: 0.7/line (almost done)
Lines 161-220: 0.0/line (SATURATED)
```

---

## V11: Final Design — Simplified by Empirical Evidence

### Key Simplification

The convergence experiment proves that dedup WORKS — the graph saturates naturally. This means:
1. **FSRS decay model is overkill** — nodes don't need complex stability tracking
2. **Adaptive threshold is still valuable** — it accelerates convergence for Deep Soul
3. **Simple priority score replaces FSRS** — `mention_count × confidence × recency_factor`

### V11 Changes from V10

1. **Drop FSRS entirely** — replace with simple priority score
2. **Keep adaptive merge threshold** — empirically justified
3. **Add convergence data** — we now know the real growth curve

### Priority Score (replaces FSRS)

```python
def _node_priority(self, item: SoulItem, current_cycle: int) -> float:
    """Simple importance score for Deep Soul nodes.
    Used as PageRank weight multiplier and dormancy check."""
    cycles_since = current_cycle - item.last_reinforced_cycle
    recency = 1.0 / (1.0 + cycles_since)
    return item.mention_count * item.confidence * recency
```

**Behavior:**
- 10 mentions, confidence 0.9, just reinforced → priority 9.0 (core belief)
- 1 mention, confidence 0.3, 10 cycles ago → priority 0.03 (dormant)
- No new fields needed — uses existing `mention_count` and `confidence`
- One new field: `last_reinforced_cycle: int` (which cycle last touched this node)

### ACTUAL FINAL Architecture

```
┌──────────────────────────────────────────────────────┐
│                    DualSoul                           │
│                                                      │
│  ┌─────────────────────┐                             │
│  │   Surface Soul      │ ← ingest(text) every turn  │
│  │   Think Fast        │                             │
│  │   - Live Detector   │                             │
│  │   - 0.82 dedup      │ (proven: converges at ~400  │
│  │   - Grows until     │  items for 10 themes)       │
│  │     saturated       │                             │
│  └────────┬────────────┘                             │
│           │                                          │
│     every M=100 utterances                           │
│     or Surface > 200 nodes                           │
│           │                                          │
│  ┌────────▼────────────┐                             │
│  │   CONSOLIDATION     │  "Sleep Phase"              │
│  │   1. Embed Surface  │                             │
│  │      + Deep nodes   │                             │
│  │   2. Sim matrix     │                             │
│  │   3. Adaptive       │ threshold=0.82-0.06·ln(n)   │
│  │      threshold      │ (n=|Deep|, floor=0.40)      │
│  │   4. Partition:     │                             │
│  │      MERGE / NEW    │                             │
│  │   5. Batch LLM      │ one call for all merges     │
│  │      text merge     │                             │
│  │   6. Update Deep:   │ mention_count++,             │
│  │      merge/add/     │ confidence boost,            │
│  │      edge migrate   │ last_reinforced_cycle=now    │
│  │   7. Carry forward  │ top-10 Surface nodes         │
│  │      top-K          │ (skip if already in Deep)    │
│  └────────┬────────────┘                             │
│           │                                          │
│  ┌────────▼────────────┐                             │
│  │   Deep Soul         │                             │
│  │   Think Slow        │                             │
│  │   - Compressed      │                             │
│  │   - Priority score: │ mention×conf×recency         │
│  │     as PageRank wt  │                             │
│  │   - Converges via   │ adaptive threshold           │
│  │     adaptive merge  │                             │
│  │   - Never deletes   │ low-priority = dormant       │
│  └─────────────────────┘                             │
│                                                      │
│  query(q) ──→ PPR on both ──→ keyword routing         │
│               ──→ merge results ──→ LLM synthesize   │
└──────────────────────────────────────────────────────┘
```

### Final Parameters

| Parameter | Value | Evidence |
|-----------|-------|----------|
| surface_dedup | 0.82 | Proven: saturates at ~400 items/10 themes |
| deep_merge | `max(0.40, 0.82 - 0.06·ln(n))` | Smooth curve; starts conservative, gets aggressive |
| deep_cycle (M) | 100 | User's design: Think Slow period |
| max_surface_nodes | 200 | Soft cap; half of saturation point |
| carry_forward_k | 10 | Cowan's 4±1 × 2 (buffer for connected edges) |
| confidence_boost | 0.05 | Per-merge; caps at 1.0 |
| priority_dormant | 0.05 | Below this = dormant (not deleted) |

### New SoulItem Fields

```python
# Add to existing SoulItem:
last_reinforced_cycle: int = 0    # consolidation cycle when last matched
```

That's it. ONE new field. Everything else reuses existing infrastructure.

### Implementation Size Estimate

| Component | Lines | Notes |
|-----------|-------|-------|
| `DualSoul` class | ~150 | Two SoulGraphs + ingest/query/consolidate |
| `consolidate()` | ~80 | Embed, sim matrix, partition, batch merge, edge migrate |
| `_batch_merge_prompt()` | ~20 | LLM prompt template for text merging |
| `_adaptive_threshold()` | ~3 | One-liner formula |
| `_node_priority()` | ~4 | One-liner formula |
| `_route_query()` | ~15 | Keyword matching |
| Tests | ~200 | Consolidation, decay, merge, routing, carry-forward |
| **Total** | **~470** | Minimal, focused |
