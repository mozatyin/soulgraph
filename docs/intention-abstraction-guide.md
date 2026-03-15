# Intention Abstraction Layer — Implementation Guide

## What Was Built

A **multi-framework meta-consolidation layer** for DualSoul that discovers abstract root intentions from concrete dialogue-extracted intentions.

### Core Idea

When someone says "I want a pretty dress" (Phase 1), "I'll never be hungry again" (Phase 4), and "I need to marry for money" (Phase 7) — these are all **manifestations of the same root: survival/security need**. The embedding similarity between these texts is too low for the regular consolidation merge to catch them. We use **LLM laddering** (Gutman means-end chain theory) to discover the abstract root.

### Architecture

```
Layer 0: Raw Utterances     → "I want a pretty dress"
Layer 1: Concrete Intentions → si_001: "wants pretty dress" (Surface)
Layer 2: Deep Intentions     → di_001: "wants pretty dress" (Deep, after consolidation)
Layer 3: Root Intentions     → ri_001: "Survival and material security" (Deep, after meta-consolidation)
                               ↑ manifests-as edges connect ri_001 → di_001, di_005, di_020...
```

### 6 Psychological Frameworks (51 categories)

Every root intention is tagged with ALL frameworks simultaneously:

| Framework | Key | Categories |
|-----------|-----|-----------|
| Maslow's Hierarchy | `maslow` | physiological, safety, love, esteem, self_actualization |
| Self-Determination Theory | `sdt` | autonomy, competence, relatedness |
| Reiss 16 Basic Desires | `reiss` | power, independence, curiosity, acceptance, order, saving, honor, idealism, social_contact, family, status, vengeance, romance, eating, physical_activity, tranquility |
| Attachment Theory | `attachment` | secure, anxious, avoidant, disorganized |
| Young's 18 Schemas | `schema` | abandonment, mistrust, emotional_deprivation, defectiveness, social_isolation, dependence, vulnerability, enmeshment, failure, entitlement, insufficient_self_control, subjugation, self_sacrifice, approval_seeking, negativity, emotional_inhibition, unrelenting_standards, punitiveness |
| Emotion-Focused Therapy | `eft` | boundary_need, connection_need, safety_need, acceptance_need, distance_need |

### Key Files

| File | Purpose |
|------|---------|
| `soulgraph/frameworks.py` | Framework registry, validation, prompt generation |
| `soulgraph/dual_soul.py` | DualSoul with `meta_consolidate()` method |
| `soulgraph/graph/models.py` | SoulItem with `abstraction_level`, `motivation_tags` |
| `tests/test_dual_soul.py` | 31 tests for DualSoul including meta-consolidation |
| `tests/test_frameworks.py` | 13 tests for framework registry |
| `tests/test_models.py` | Tests for SoulItem new fields |

### How It Works

1. **Regular consolidation** (every 25 utterances): embedding similarity merge of Surface → Deep
2. **Meta-consolidation** (every 2-3 regular consolidations): LLM analyzes top-40 Deep nodes, discovers root motivations, creates `manifests-as` edges
3. **Query**: system prompt includes root motivations alongside Surface and Deep nodes

### Configuration

```python
ds = DualSoul(
    api_key="...",
    deep_cycle=25,        # Regular consolidation every N utterances
    max_surface_nodes=80, # Surface cap triggers consolidation
    carry_forward_k=8,    # Top-K surface nodes persist
    meta_cycle=2,         # Meta-consolidation every N regular consolidations
)

# Force meta-consolidation with specific frameworks
ds.meta_consolidate(frameworks=["maslow", "sdt", "attachment"])

# Or use all frameworks (default)
ds.meta_consolidate()
```

### Scarlett Experiment Results

V2 (with intention abstraction) discovered **20 root intentions** with multi-framework tags. Key improvements:
- Phase 1: 0.85 → 0.92 (+0.07)
- Phase 5: 0.85 → 0.95 (+0.10)
- Phase 7: 0.75 → 0.80 (+0.05)

### Theoretical Foundations

Research sources (from web search):
- **Maslow's Hierarchy** — prepotency principle for root classification
- **Kruglanski Goal Systems Theory** — equifinality (many means → one goal), multifinality (one means → many goals)
- **Gutman Means-End Chain** — iterative "why?" laddering from concrete to abstract
- **CAPS Model (Mischel)** — situation-dependent activation patterns
- **Self-Determination Theory** — autonomy/competence/relatedness needs
- **Reiss 16 Basic Desires** — empirically derived motivation profile
- **Attachment Theory** — relational style patterns
- **Young's Schema Therapy** — maladaptive belief patterns
- **Emotion-Focused Therapy** — emotion-to-unmet-need mapping

### Next Steps (PDCA)

1. Fix Phase 4 JSON parsing error (add retry/fallback)
2. Compare V1 vs V2 root intentions qualitatively
3. Test with therapy dialogue data (different domain)
4. Tune meta_cycle and framework selection per use case
5. Add `counterfinality` edges (means that undermine goals)

### Running Tests

```bash
.venv/bin/pytest --tb=short -q  # 142 tests
.venv/bin/pytest tests/test_dual_soul.py -v  # DualSoul-specific
.venv/bin/pytest tests/test_frameworks.py -v  # Framework registry
```

### Running Experiment

```bash
export ANTHROPIC_API_KEY=sk-or-...
.venv/bin/python scripts/scarlett_intention_experiment.py
# Results in results/scarlett_intention_experiment_v2/
```
