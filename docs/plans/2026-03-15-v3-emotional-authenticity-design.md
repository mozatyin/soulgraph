# V3: Emotional Authenticity + Existential Framework + JSON Fix

## Goal

Add an emotional authenticity dimension to SoulGraph that distinguishes false positives (emotion amplification) from genuine disclosures (barrier breakthrough), add an existential psychology framework to cover meaning/mortality gaps, and fix the Phase 4 JSON parsing error.

## Architecture

Three independent improvements that touch different parts of the pipeline:

```
Improvement 1 (Emotional Authenticity):
  Detector prompt → SoulItem fields → Consolidation passthrough → Meta-consolidation weighting

Improvement 2 (Existential Framework):
  frameworks.py → meta_consolidate prompt (automatic via existing machinery)

Improvement 3 (JSON Fix):
  scarlett_intention_experiment.py → compare_phase() retry + error marking
```

## Improvement 1: Emotional Authenticity Dimension

### Rationale

When a person is emotionally aroused, their expressed intentions bifurcate:
- **False Positive (amplified)**: "I never want to see him again!" said in anger, not a real intention
- **Disclosure (slip/consistent)**: "I've always been afraid of being abandoned" — barrier breakdown reveals hidden truth

V2 treated all intentions equally. V3 captures emotional context at ingestion and uses it during meta-consolidation to adjust root intention confidence.

### Design: A+C (Ingestion + Meta-consolidation)

**Why A+C, not B (consolidation)?**
- A parallel system (Emotion Detector project) will handle precise emotion analysis
- SoulGraph only needs a lightweight signal, leaving a clean interface for future RL fusion
- Zero extra API calls: piggyback on existing Detector LLM call

### A-Layer: SoulItem New Fields

```python
class SoulItem(BaseModel):
    ...
    emotional_valence: str = "neutral"    # "neutral" | "aroused" | "extreme"
    authenticity_hint: str = "unknown"    # "consistent" | "slip" | "amplified" | "unknown"
```

- `emotional_valence`: How emotionally charged was the utterance when this item was extracted?
  - `neutral`: calm, factual
  - `aroused`: noticeable emotion (frustration, excitement, sadness)
  - `extreme`: intense emotion (rage, breakdown, euphoria)
- `authenticity_hint`: Initial LLM assessment of whether this intention is genuine
  - `consistent`: aligns with established patterns — likely real
  - `slip`: contradicts stated values, immediately corrected — likely disclosure
  - `amplified`: hyperbolic, emotion-driven, no pattern support — likely false positive
  - `unknown`: insufficient context to judge

### A-Layer: Detector Prompt Change

Add to `_DETECT_SYSTEM` JSON output format:
```json
{
  "new_items": [{
    "id": "si_NNN", "text": "...", ...,
    "emotional_valence": "neutral|aroused|extreme",
    "authenticity_hint": "consistent|slip|amplified|unknown"
  }]
}
```

Add rule to prompt:
> "For each item, assess the speaker's emotional state when expressing it (emotional_valence) and whether the intention is genuine or emotion-amplified (authenticity_hint). A 'slip' is something the speaker immediately corrects or denies — treat as likely genuine disclosure."

### Consolidation Passthrough

Surface→Deep consolidation preserves these fields on new Deep items. For merged items, take the max valence and most informative hint.

### C-Layer: Meta-consolidation Weighting

Add to `_ROOT_DISCOVERY_PROMPT` rules:
> "8. For concrete items marked `extreme` + `amplified`: reduce their weight when grouping into roots (likely false positive).
> 9. For concrete items marked `extreme` + `slip`: increase their weight — this is a genuine disclosure the person usually hides.
> 10. Items marked `aroused` + `consistent` are high-signal — the emotion validates the intention."

### Future RL Interface

The `emotional_valence` field serves as the anchor point for Emotion Detector fusion. When RL merges the two systems, Emotion Detector provides precise emotion classification → SoulGraph uses it to refine `authenticity_hint`.

## Improvement 2: Existential Psychology Framework

### Rationale

V2's `transformations_missed` identified:
- "The essential connection between her identity and Tara as more than just safety — as spiritual/existential anchor"
- "The spiritual/transcendent aspects of her connection to the land"

None of the 6 existing frameworks cover existential meaning. Adding Frankl/Yalom fills this gap.

### Design

Add to `frameworks.py`:

```python
"existential": {
    "name": "Existential Psychology (Frankl/Yalom)",
    "description": "4 ultimate concerns of human existence that shape motivation beneath need-satisfaction",
    "values": {
        "meaning": "Search for purpose, significance, coherence in life — why one exists",
        "mortality": "Awareness of death, legacy, finite time — drives urgency and priority",
        "freedom": "Burden of choice, responsibility, groundlessness — anxiety of unlimited options",
        "isolation": "Fundamental aloneness, unbridgeable gap between self and others",
    },
}
```

This automatically integrates via existing machinery:
- `DEFAULT_FRAMEWORKS` picks it up (it's `list(FRAMEWORKS.keys())`)
- `framework_prompt_section()` generates the prompt text
- `meta_consolidate()` sends it to LLM
- No code changes needed beyond the registry entry

Total categories: 51 → 55 (4 new).

## Improvement 3: JSON Parse Error Fix

### Rationale

Phase 4 scored 0.0 in V2 due to JSON parse error in `compare_phase()`. The root intentions were correctly discovered (15 roots), but the evaluation LLM returned malformed JSON.

### Design

In `compare_phase()`:
1. First attempt: existing `raw[start:end+1]` JSON parse
2. On failure: regex extraction of `overall_understanding` field
3. On total failure: return `overall_understanding: -1` (not 0.0)
4. In summary calculation: exclude phases with score -1, note as "parse_error"

```python
# Fallback regex
import re
match = re.search(r'"overall_understanding"\s*:\s*([\d.]+)', raw)
if match:
    return {"phase": phase_num, "overall_understanding": float(match.group(1)),
            "commentary": "partial parse — regex fallback"}
return {"phase": phase_num, "overall_understanding": -1, "commentary": "parse failed"}
```

## Testing Strategy

- **Improvement 1**: Unit tests for new SoulItem fields, test Detector extracts valence/hint, test meta-consolidation prompt includes authenticity rules
- **Improvement 2**: Test new framework in registry, validate_tag works, prompt section generated
- **Improvement 3**: Test regex fallback, test -1 exclusion in summary

## Success Criteria

Re-run Scarlett experiment as V3:
- Phase 4 should have a real score (not 0.0)
- Phase 9 should improve (emotional authenticity helps distinguish crisis disclosures from amplification)
- Existential framework should tag Tara-related root intentions with `existential: meaning`
- Overall score target: >= 0.85 (vs V2's 0.82)
