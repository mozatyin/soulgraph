# V3 Research Notes

Research from web search + 3 specialized agents. These inform the V3 roadmap.

## 1. Graph Comparison (Replace Current Metric)

### Triple-Level Soft Matching (Highest Priority)
- Embed each triple `(subject, relation, object)` as a sentence using sentence-transformers
- Compute pairwise cosine similarity between all GT and detected triples
- Use Hungarian algorithm for optimal assignment (threshold cosine > 0.65)
- Gives **Soft Precision**, **Soft Recall**, **Soft F1**
- Directly solves "drives" vs "motivates" problem without manual synonym mapping

### Graph Edit Distance (GED)
- Feasible for our scale (12-21 nodes) — NP-hard but fine under ~30
- Node substitution cost = `1 - cosine_similarity(embed(A), embed(B))`
- Normalize by `max(|V1|+|E1|, |V2|+|E2|)` for 0-1 score
- NetworkX has `graph_edit_distance` and `optimize_graph_edit_distance`

### Multi-Level Decomposition (Replace Single Score)
| Level | Metric | How |
|-------|--------|-----|
| Node coverage | Soft node recall | Embed labels, best-match |
| Edge coverage | Soft triple recall | Embed triples |
| Structural role | Degree distribution | KL-divergence or EMD |
| Hub structure | PageRank correlation | Correlate ranks |
| Neighborhood | Jaccard on k-hop | Compare neighbor sets |

### Asymmetric Evaluation
- 21 detected from 12 GT is OK if recall is high
- Evaluate recall and precision separately
- Use F-beta with recall bias

## 2. Extraction Precision (Reduce Over-Detection)

### Abstraction-Level Anchoring (Prompt-Only, High Impact)
- Add: "Extract at personality-profile level, not sub-behaviors"
- "Would a psychologist list this as a separate item, or a sub-point?"
- Expected: 30-40% reduction in over-extraction

### Extract-Then-Diff Architecture (Medium-Term)
1. Extract from latest turn ONLY (not full conversation)
2. Diff candidates against existing graph programmatically (embedding similarity)
3. Route: new → add, duplicate → strengthen
- Removes dedup burden from LLM entirely

### Embedding-Based Dedup
- sentence-transformers `all-MiniLM-L6-v2` or Voyage AI `voyage-3`
- Cosine threshold ~0.85-0.90 for semantic equivalence
- Run after each extraction, before adding to graph

### CoT Justification
- Force LLM to quote evidence, check existing items, then decide ADD/STRENGTHEN/SKIP
- "Answer cleansing" — reasoning forces self-filtering

## 3. Conversation Strategy (Improve Edge Detection)

### Laddering (Single Highest-Value Technique)
- **UP**: "Why is [item] important to you?" → drives/causes edges
- **DOWN**: "How does [item] show up day-to-day?" → manifests_as edges
- Every "why" answer IS an edge in the graph

### Discrepancy Questioning (MI)
- "You mentioned [A] and [B] — how do those fit together?"
- Directly elicits conflicts_with, compensates, enables edges
- Wrong guesses yield MORE structural info than right ones

### Clean Language
- Use person's EXACT words when referencing items
- "And what kind of [X] is that [X]?" → attribute expansion
- "And what happens just before [X]?" → causal chain
- "And where does [X] come from?" → origin/drives edges

### Meta-Questions (Hub Self-Report)
- "Of everything we've discussed, what feels most central?"
- "If you had to pick the one thing that drives most of the others?"
- Person self-reports graph topology

### Depth vs Breadth Heuristic
```
density = edges / max(1, items*(items-1)/2)
if density < 0.15: DEPTH mode (laddering on item with fewest edges)
if density >= 0.15 but low coverage: BREADTH mode (new domains)
if disconnected clusters: BRIDGE mode (connect clusters)
if items >= 10: META mode (self-report structure)
```

## Sources

### Graph Comparison
- NetworkX GED: networkx.org/documentation/stable/reference/algorithms/similarity
- KG Entity Alignment Survey 2024 (Zhu et al.): springer.com/article/10.1007/s10462-024-10866-4
- Sentence-BERT/STS: sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity
- GED4py: github.com/Jacobe2169/ged4py

### Extraction
- iText2KG (incremental KG): arxiv.org/html/2409.03284v1
- KGGen (extract + cluster): arxiv.org/html/2502.09956v1
- LLM-empowered KG Construction Survey: arxiv.org/html/2510.20345v1

### Conversation
- MI (OARS): motivationalinterviewing.org/understanding-motivational-interviewing
- Laddering: uxmatters.com/mt/archives/2009/07/laddering-a-research-interview-technique
- Clean Language: en.wikipedia.org/wiki/Clean_language_interviewing
- Can LLMs Assess Personality? arxiv.org/pdf/2602.15848
- Socratic Questioning: positivepsychology.com/socratic-questioning
