# SoulGraph V4 Design — Soul Operating System

## Vision

SoulGraph stores a human soul. It absorbs everything about a person through conversation — facts, intentions, preferences, emotions, memories, relationships, behaviors, beliefs — and ranks information by importance using graph structure (PageRank). The graph then serves as the foundation for AI to support every task in human life.

**Key paradigm shift from V1-V3**: Over-detection is the goal, not a bug. The graph should be dense and comprehensive. Quality is measured by ranking accuracy, not recall/precision against a sparse ground truth.

## Architecture

```
Conversation → Absorption Engine → Dense Soul Graph → PageRank Ranking → Domain-Contextual Retrieval
                                         ↑
                                    Evaluation:
                                    - Rank correlation (Spearman ρ)
                                    - Domain NDCG@K
                                    - Absorption rate
                                    - Intention recall
```

## Core Principles

1. **Absorb everything** — no extraction caps, no confidence filters. Store all information.
2. **Intentions are the crown jewels** — they naturally rank highest because everything else orbits around them (most edges → highest PageRank).
3. **Ranking = importance** — PageRank on graph structure determines what matters. No manual weighting.
4. **Domain-contextual** — same node has different importance depending on the lens. "Owns a bike" is high-rank in transportation, low-rank in relationships. Topic-Sensitive PageRank.
5. **Append-only** — never delete. Importance decay through time weighting, not deletion.

## Component Design

### 1. Rich Persona Fixture ("Zhang Wei")

~60 items, ~80 edges. A complete person, not a quiz answer key.

| Domain | Example Items | Count |
|--------|--------------|-------|
| career | "8年程序员", "想创业做AI工具", "35岁焦虑", "技术到了天花板" | ~10 |
| family | "女儿5岁", "妻子是老师", "小时候父母严格", "每晚陪女儿读书" | ~10 |
| identity | "技术人骨子里", "不服输", "害怕平庸", "从为自己活到为爱而活" | ~8 |
| finance | "房贷压力", "存款够18个月", "IT行业收入稳定" | ~6 |
| health | "失眠", "运动太少", "体检有脂肪肝" | ~5 |
| social | "同事老王创业成功", "大学室友现在做投资" | ~5 |
| hobbies | "喜欢骑自行车", "偶尔打篮球", "喜欢开车兜风" | ~4 |
| values | "时间比钱重要", "家人安全感第一", "爱家人不等于委屈自己" | ~6 |
| stories | "大学做食堂项目赚第一桶金", "邻居跑车的童年记忆", "女儿问我为什么不开心" | ~6 |

**Structural design**: Intentions are hubs by construction. "想创业做AI工具" has 7+ edges (driven by frustration, constrained by finance, enabled by wife's support, conflicted by fear, etc.) → naturally high PageRank. "喜欢骑自行车" has 1-2 edges → low PageRank.

**Tags**: Each item gets `tags: list[str]` from: `fact`, `intention`, `preference`, `memory`, `emotion`, `belief`, `relationship`, `behavior`. Core ItemType (cognitive/action/background) stays unchanged.

### 2. Absorption Engine (Detector Changes)

**Remove limits:**
- No "max 3 items per turn" cap
- No confidence ≥ 0.4 filter (store everything, ranking sorts it)
- Extract-then-diff stays (extract from latest turn, embedding dedup ≥ 0.82)

**Add tag extraction:**
- LLM assigns tags to each extracted item
- Tags are metadata, not types — multiple tags per item allowed

**More aggressive edge extraction:**
- Current: edges only for new items
- V4: also discover edges between existing items when new context reveals connections

**Expected output**: 30-50 items after 15 turns (vs current 18-24). Denser edges.

### 3. PageRank Ranking

**Global PageRank** (`networkx.pagerank`):
- Directed graph: `from_id → to_id`, weighted by `edge.strength`
- Returns: `{item_id: importance_score}` for all items
- Intentions naturally rank highest due to edge density

**Topic-Sensitive PageRank** (per domain):
- `networkx.pagerank(G, personalization=domain_bias)`
- `domain_bias`: higher teleportation probability for nodes in target domain
- Pre-computed for all domains in the graph
- Same node, different importance depending on domain context

**Time decay** (simple first):
- Edge weight = `strength × (1 + 0.1 × mention_count)`
- Frequently reinforced edges boost connected nodes
- Full Ebbinghaus decay deferred to multi-session support

### 4. Evaluation Framework

**Metric 1: Rank Correlation (Spearman's ρ)** — weight 0.3
1. PageRank on GT graph → importance ranking
2. Match detected items to GT items (EmbeddingMatcher)
3. PageRank on detected graph → importance ranking
4. Spearman ρ between matched item rankings
5. Range: -1.0 to 1.0 (0.8+ = excellent)

**Metric 2: Domain NDCG@5** — weight 0.3
1. For each domain: GT top-5 by domain-PPR = ideal ranking
2. Detected graph top-5 by domain-PPR = system ranking
3. NDCG@5 per domain, averaged
4. Tests: does "bike" rank high in transportation? Does it drop in relationships?

**Metric 3: Absorption Rate** — weight 0.2
- `matched_items / total_GT_items`
- Target: 80%+ on 60-item fixture

**Metric 4: Intention Recall** — weight 0.2
- `matched_intentions / total_GT_intentions`
- Separate from general absorption because missing an intention is worse than missing a fact

**Overall V4 Score:**
```
overall = rank_correlation × 0.3 + domain_ndcg × 0.3 + absorption_rate × 0.2 + intention_recall × 0.2
```

### 5. Code Changes

| File | Change |
|------|--------|
| `graph/models.py` | Add `tags: list[str] = []` to SoulItem. Add `pagerank()` and `domain_pagerank(domain)` methods. |
| `experiment/detector.py` | Remove item cap + confidence filter. Add tag extraction to prompt. Add inter-existing-item edge discovery. |
| `comparator/ranking.py` | **NEW** — `RankingComparator`: Spearman ρ, domain NDCG@5, absorption rate, intention recall, overall V4 score. |
| `experiment/runner.py` | Add RankingComparator to `run()`. Report V4 metrics alongside V3 metrics. |
| `experiment/models.py` | Add `ranking_scores: dict | None` to ExperimentResult. |
| `fixtures/zhang_wei.json` | **NEW** — Rich persona (~60 items, ~80 edges, tagged). |
| `cli.py` | Show V4 metrics in output. |
| `tests/` | Tests for PageRank, RankingComparator, tag extraction. |

### What Stays Unchanged

- SoulGraph append-only semantics
- Speaker↔Detector conversation loop
- EmbeddingMatcher for node matching
- 3 core ItemTypes (cognitive/action/background)
- Embedding-based dedup in Detector (cosine ≥ 0.82)
- Multi-run averaging
- Retry logic for API errors

## Research References

- **Topic-Sensitive PageRank**: Haveliwala 2002 (Stanford)
- **Personalized PageRank**: Efficient Algorithms survey (arXiv 2403.05198, 2024)
- **Node Importance in KGs**: GENI (Park et al., KDD 2019)
- **Personal Knowledge Graphs**: PKG API (Balog et al., ACM 2024)
- **Incremental KG Construction**: iText2KG (WISE 2024), KGGen (2025)
- **User Profile Evaluation**: PersonaMem (COLM 2025)
- **Memory Systems**: EverMemOS (2026), MemoryBank (2023), Mem0

## Success Criteria

- Rank correlation ≥ 0.6 on first attempt
- Domain NDCG@5 ≥ 0.5 across all domains
- Absorption rate ≥ 0.7 on 60-item fixture
- Intention recall ≥ 0.8
- Overall V4 score ≥ 0.6
- Stable across 3 runs (std ≤ 0.05)
