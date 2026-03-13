# SoulGraph V5 Design — Multi-Session Memory

## Vision

A soul isn't captured in one conversation. V5 adds multi-session support: the graph persists across conversations, items get reinforced through repeated mention, and rankings improve as the system learns more about the person over time.

## Architecture

```
Session 1 (career/family/finance):
  Speaker(GT, topic_hints) ↔ Detector(fresh graph) → 10 turns → Evaluate → session_1_scores

Session 2 (identity/social/values):
  Speaker(GT, topic_hints) ↔ Detector(SAME graph) → 10 turns → Evaluate → session_2_scores

Session 3 (health/hobbies/stories + revisit career):
  Speaker(GT, topic_hints) ↔ Detector(SAME graph) → 10 turns → Evaluate → session_3_scores

Key: Speaker resets (new conversation), Detector persists (accumulated memory)
```

## Core Principles

1. **Graph persists across sessions** — Detector keeps its accumulated graph between conversations
2. **Mention reinforcement, not decay** — Repeated items get `mention_count++`, boosting PageRank weight. Nothing decays or gets deleted. Append-only.
3. **Rankings improve over time** — More data = better ranking. Session 3 should have better rank correlation than session 1.
4. **Topic-steered sessions** — Speaker gets `topic_hints` per session to simulate natural conversation flow across days/weeks.

## Component Design

### 1. Session-Aware Graph Model

**SoulItem changes:**
- `source_session` (already exists, default `0`) — set to current session number by detector
- `last_referenced` (already exists) — updated when mention_count is bumped
- No new fields needed

**SoulEdge changes:**
- Add `source_session: int = 0` — tracks when edge was created

**Mention reinforcement in detector:**
- When embedding dedup finds match (cosine ≥ 0.82): bump `mention_count += 1`, update `last_referenced`
- Already happens in V4 — no change needed, just verify it works across sessions

**PageRank edge weight (tuned):**
- `effective_weight = strength × (1 + 0.2 × mention_count)`
- Changed from V4's `0.1` factor to `0.2` — more aggressive reinforcement for multi-session
- Items mentioned in 3 sessions: weight multiplied by 1.6x vs single-mention items

### 2. Multi-Session ExperimentRunner

**New method: `run_multi_session()`**
```python
def run_multi_session(
    self,
    ground_truth: SoulGraph,
    session_configs: list[dict],  # [{turns: 10, topic_hints: [...]}]
    hub_top_k: int = 5,
    verbose: bool = True,
) -> dict:
```

- Iterates through session configs
- Session 1: Fresh detector, run N turns
- Session 2+: Same detector (graph preserved), new Speaker (fresh conversation)
- After each session: compute V4 ranking metrics against full GT
- Returns: per-session scores, cross-session improvement, final scores

**Speaker session reset:**
- New Speaker instance per session (fresh message history)
- Speaker gets `topic_hints` in system prompt to steer conversation toward session-appropriate topics
- Simulates "user comes back tomorrow, talks about different things"

**Multi-run support:**
- `run_multi_session_multi()` wraps N runs, aggregates per-session metrics with mean±std

### 3. Sequential Fixture

**Extend `fixtures/zhang_wei.json`** with session metadata:

```json
{
  "owner_id": "zhang_wei",
  "items": [...],
  "edges": [...],
  "sessions": [
    {
      "session": 1,
      "topic_hints": ["career", "family", "finance"],
      "description": "Career frustration, family responsibility, startup ambition"
    },
    {
      "session": 2,
      "topic_hints": ["identity", "social", "values"],
      "description": "Deeper identity, social influences, value system"
    },
    {
      "session": 3,
      "topic_hints": ["health", "hobbies", "stories", "career"],
      "description": "Health, hobbies, life stories, PLUS revisit career (mention reinforcement)"
    }
  ]
}
```

Session 3 revisits "career" — tests that mention reinforcement works. "想创业做AI工具" should get re-mentioned, bumping its mention_count → higher PageRank.

### 4. Topic-Steered Speaker

**Speaker system prompt addition:**
```
This session focuses on: {topic_hints}
Steer the conversation naturally toward these topics.
You may touch on other topics if they come up naturally, but prioritize these areas.
```

Speaker still uses the full GT graph for responses — topic_hints just bias which aspects get discussed first.

### 5. Evaluation

**Per-session metrics** (all V4 metrics, computed after each session):
- rank_correlation, domain_ndcg, absorption_rate, intention_recall, overall

**Cross-session metrics (new):**
- `rank_improvement`: session_3.rank_correlation - session_1.rank_correlation
- `absorption_growth`: [session_1.absorption, session_2.absorption, session_3.absorption]
- `mention_reinforcement_effect`: average mention_count of top-5 ranked items vs bottom-5

**Success criteria:**
- Rank correlation improves across sessions (improvement > 0)
- Final session absorption ≥ 0.85
- Final session rank correlation ≥ 0.65 (better than V4's 0.60)
- Intention recall ≥ 0.90 across all sessions
- Mention-reinforced items rank higher than single-mention items

### 6. Code Changes

| File | Change |
|------|--------|
| `graph/models.py` | Add `source_session` to SoulEdge. Tune mention_count coefficient to 0.2. |
| `experiment/runner.py` | Add `run_multi_session()` and `run_multi_session_multi()`. |
| `experiment/speaker.py` | Accept `topic_hints` param, add to system prompt. |
| `experiment/detector.py` | Accept `session_number` param, set `source_session` on new items/edges. |
| `experiment/models.py` | Add `MultiSessionResult` model. |
| `fixtures/zhang_wei.json` | Add `sessions` field with topic_hints per session. |
| `comparator/ranking.py` | Add `mention_reinforcement_effect()` method. |
| `cli.py` | Add `--sessions` flag. |
| `tests/` | Tests for multi-session runner, session-aware detector, topic-steered speaker. |

### What Stays Unchanged

- SoulGraph append-only semantics
- V4 RankingComparator (reused as-is for per-session eval)
- V4 detector prompt (absorb everything)
- V4 PageRank methods (just coefficient change)
- V4 embedding dedup (cosine ≥ 0.82)
- EmbeddingMatcher for node matching

## Success Criteria

- Rank improvement > 0 (graph gets better over sessions)
- Final absorption ≥ 0.85 on 61-item fixture across 3 × 10 turns
- Final rank correlation ≥ 0.65
- Intention recall ≥ 0.90
- Stable across 3 runs (std ≤ 0.05)
- Top-5 ranked items have higher average mention_count than bottom-5
