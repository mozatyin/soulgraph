# Intention Abstraction Layer — Implementation Guide

## What Was Built

A **multi-framework meta-consolidation layer** for DualSoul that discovers abstract root intentions from concrete dialogue-extracted intentions, with **emotional authenticity assessment** to distinguish false positives from genuine disclosures.

### Core Idea

When someone says "I want a pretty dress" (Phase 1), "I'll never be hungry again" (Phase 4), and "I need to marry for money" (Phase 7) — these are all **manifestations of the same root: survival/security need**. The embedding similarity between these texts is too low for the regular consolidation merge to catch them. We use **LLM laddering** (Gutman means-end chain theory) to discover the abstract root.

### Architecture

```
Layer 0: Raw Utterances     → "I want a pretty dress"
Layer 1: Concrete Intentions → si_001: "wants pretty dress" (Surface, with emotional_valence + authenticity_hint)
Layer 2: Deep Intentions     → di_001: "wants pretty dress" (Deep, after consolidation, emotional fields preserved)
Layer 3: Root Intentions     → ri_001: "Survival and material security" (Deep, after meta-consolidation)
                               ↑ manifests-as edges connect ri_001 → di_001, di_005, di_020...
                               ↑ emotional authenticity used to weight concrete→root grouping
```

### 7 Psychological Frameworks (55 categories)

Every root intention is tagged with ALL frameworks simultaneously:

| Framework | Key | Categories |
|-----------|-----|-----------|
| Maslow's Hierarchy | `maslow` | physiological, safety, love, esteem, self_actualization |
| Self-Determination Theory | `sdt` | autonomy, competence, relatedness |
| Reiss 16 Basic Desires | `reiss` | power, independence, curiosity, acceptance, order, saving, honor, idealism, social_contact, family, status, vengeance, romance, eating, physical_activity, tranquility |
| Attachment Theory | `attachment` | secure, anxious, avoidant, disorganized |
| Young's 18 Schemas | `schema` | abandonment, mistrust, emotional_deprivation, defectiveness, social_isolation, dependence, vulnerability, enmeshment, failure, entitlement, insufficient_self_control, subjugation, self_sacrifice, approval_seeking, negativity, emotional_inhibition, unrelenting_standards, punitiveness |
| Emotion-Focused Therapy | `eft` | boundary_need, connection_need, safety_need, acceptance_need, distance_need |
| Existential Psychology | `existential` | meaning, mortality, freedom, isolation |

### Emotional Authenticity Dimension (V3)

Each SoulItem carries two emotional context fields:

| Field | Values | Purpose |
|-------|--------|---------|
| `emotional_valence` | `neutral`, `aroused`, `extreme` | How emotionally charged the utterance was |
| `authenticity_hint` | `consistent`, `slip`, `amplified`, `unknown` | Whether the intention is genuine or emotion-amplified |

**Key insight**: High emotion is a bifurcation point:
- **False positive** (`extreme` + `amplified`): "I never want to see him again!" said in rage — not a real intention
- **Disclosure** (`extreme` + `slip`): "I've always been afraid of being abandoned" — barrier breakdown reveals hidden truth
- **Validated** (`aroused` + `consistent`): Emotion confirms the intention is genuine

These fields are:
1. **Extracted** at ingestion (Detector LLM call, zero extra cost)
2. **Preserved** through Surface→Deep consolidation
3. **Used** during meta-consolidation to weight root intention grouping

Future: Emotion Detector project will provide precise emotion classification → RL fusion via `emotional_valence` anchor point.

### Key Files

| File | Purpose |
|------|---------|
| `soulgraph/frameworks.py` | 7 framework registry, validation, prompt generation |
| `soulgraph/dual_soul.py` | DualSoul with `meta_consolidate()` + emotional authenticity |
| `soulgraph/graph/models.py` | SoulItem with `abstraction_level`, `motivation_tags`, `emotional_valence`, `authenticity_hint` |
| `soulgraph/experiment/detector.py` | Detector extracts emotional fields during ingestion |
| `tests/test_dual_soul.py` | DualSoul tests including meta-consolidation + emotional fields |
| `tests/test_frameworks.py` | Framework registry tests (7 frameworks) |
| `tests/test_models.py` | SoulItem field tests |
| `tests/test_detector_emotional.py` | Detector emotional extraction tests |
| `tests/test_experiment_json_fix.py` | JSON parse fallback tests |

### How It Works

1. **Ingestion**: Detector extracts items with `emotional_valence` and `authenticity_hint`
2. **Regular consolidation** (every 25 utterances): embedding similarity merge of Surface → Deep, emotional fields preserved
3. **Meta-consolidation** (every 2-3 regular consolidations): LLM analyzes top-40 Deep nodes with emotional context, discovers root motivations, creates `manifests-as` edges
4. **Query**: system prompt includes root motivations alongside Surface and Deep nodes

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
ds.meta_consolidate(frameworks=["maslow", "sdt", "attachment", "existential"])

# Or use all frameworks (default — all 7)
ds.meta_consolidate()
```

### Scarlett Experiment Results

V2 (with intention abstraction) discovered **20 root intentions** with multi-framework tags. Key improvements:
- Phase 1: 0.85 → 0.92 (+0.07)
- Phase 5: 0.85 → 0.95 (+0.10)
- Phase 7: 0.75 → 0.80 (+0.05)

V3 adds emotional authenticity + existential framework + JSON fix. Run experiment to compare.

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
- **Existential Psychology (Frankl/Yalom)** — meaning, mortality, freedom, isolation

### Next Steps (PDCA)

1. ~~Fix Phase 4 JSON parsing error~~ ✅ Done in V3
2. Run V3 Scarlett experiment and compare V1 vs V2 vs V3
3. Test with therapy dialogue data (different domain)
4. Tune meta_cycle and framework selection per use case
5. Add `counterfinality` edges (means that undermine goals)
6. RL fusion with Emotion Detector project for precise emotional classification

### Running Tests

```bash
.venv/bin/pytest --tb=short -q  # 163 tests
.venv/bin/pytest tests/test_dual_soul.py -v  # DualSoul-specific
.venv/bin/pytest tests/test_frameworks.py -v  # Framework registry
.venv/bin/pytest tests/test_detector_emotional.py -v  # Emotional extraction
```

### Running Experiment

```bash
export ANTHROPIC_API_KEY=sk-or-...
.venv/bin/python scripts/scarlett_intention_experiment.py
# Results in results/scarlett_intention_experiment_v3/
```
