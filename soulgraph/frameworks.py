"""Psychological framework registry for motivation tagging.

Each framework defines a set of valid classification values.
The meta-consolidation step uses active frameworks to annotate root intentions.
Frameworks are pluggable — different use cases activate different combinations.
"""

FRAMEWORKS: dict[str, dict] = {
    "maslow": {
        "name": "Maslow's Hierarchy of Needs",
        "description": "5-layer need hierarchy (lower needs dominate until satisfied)",
        "values": {
            "physiological": "Basic survival: food, water, shelter, sleep, health",
            "safety": "Security, stability, money, protection, predictability",
            "love": "Connection, acceptance, intimacy, belonging, family",
            "esteem": "Status, recognition, competence, respect, independence",
            "self_actualization": "Purpose, meaning, creativity, personal growth",
        },
    },
    "sdt": {
        "name": "Self-Determination Theory (Deci & Ryan)",
        "description": "3 universal psychological needs; also classifies motivation quality",
        "values": {
            "autonomy": "Sense of volition, ownership of behavior, acting from genuine interest",
            "competence": "Feeling effective, capable, mastery of challenges",
            "relatedness": "Feeling connected, belonging, caring and being cared for",
        },
    },
    "reiss": {
        "name": "Reiss's 16 Basic Desires",
        "description": "Empirically derived motivation profile (each is a continuous dimension)",
        "values": {
            "power": "Desire to influence others, leadership",
            "independence": "Desire for self-reliance, self-sufficiency",
            "curiosity": "Desire for knowledge, truth",
            "acceptance": "Desire for positive self-regard",
            "order": "Desire for organization, stability, cleanliness",
            "saving": "Desire to collect, accumulate, not waste",
            "honor": "Desire to be loyal, obey moral codes",
            "idealism": "Desire for social justice, fairness",
            "social_contact": "Desire for companionship, peer interaction",
            "family": "Desire to raise children, family time",
            "status": "Desire for social standing, prestige",
            "vengeance": "Desire to get even, compete, win",
            "romance": "Desire for sex, beauty, erotic experience",
            "eating": "Desire for food, dining experience",
            "physical_activity": "Desire for exercise, movement",
            "tranquility": "Desire for emotional calm, freedom from anxiety",
        },
    },
    "attachment": {
        "name": "Attachment Theory (Bowlby/Ainsworth)",
        "description": "4 relational styles shaping how needs are expressed interpersonally",
        "values": {
            "secure": "Comfortable with intimacy and independence, balanced",
            "anxious": "Fear of abandonment, reassurance-seeking, hyperactivation",
            "avoidant": "Emotional distance, discomfort with closeness, self-reliance",
            "disorganized": "Contradictory approach-avoidance, confused relational patterns",
        },
    },
    "schema": {
        "name": "Young's 18 Early Maladaptive Schemas",
        "description": "Core belief patterns (from Schema Therapy) that distort need-pursuit",
        "values": {
            "abandonment": "Expectation that significant others will leave",
            "mistrust": "Expectation that others will hurt, abuse, or take advantage",
            "emotional_deprivation": "Belief that emotional needs won't be met by others",
            "defectiveness": "Feeling fundamentally flawed, unlovable, inferior",
            "social_isolation": "Feeling different from others, not belonging",
            "dependence": "Belief that one cannot handle daily life without help",
            "vulnerability": "Exaggerated fear of imminent catastrophe",
            "enmeshment": "Excessive emotional involvement with others, no separate identity",
            "failure": "Belief that one is inadequate and will inevitably fail",
            "entitlement": "Belief that one is special and above normal rules",
            "insufficient_self_control": "Difficulty with self-discipline and frustration tolerance",
            "subjugation": "Surrendering control to others to avoid anger or abandonment",
            "self_sacrifice": "Excessive focus on meeting others' needs at expense of own",
            "approval_seeking": "Excessive need for approval, attention, recognition",
            "negativity": "Pervasive focus on negative aspects of life",
            "emotional_inhibition": "Excessive suppression of emotions and spontaneity",
            "unrelenting_standards": "Relentless striving to meet very high internalized standards",
            "punitiveness": "Belief that people should be harshly punished for mistakes",
        },
    },
    "eft": {
        "name": "Emotion-Focused Therapy (Greenberg)",
        "description": "Maps emotions to unmet needs; distinguishes primary/secondary/instrumental emotions",
        "values": {
            "boundary_need": "Unmet need for boundaries, respect (beneath anger)",
            "connection_need": "Unmet need for connection, comfort (beneath sadness)",
            "safety_need": "Unmet need for safety, protection (beneath fear)",
            "acceptance_need": "Unmet need for acceptance, worthiness (beneath shame)",
            "distance_need": "Unmet need for distance from what is toxic (beneath disgust)",
        },
    },
}

# Default: use ALL frameworks for annotation
DEFAULT_FRAMEWORKS = list(FRAMEWORKS.keys())


def validate_tag(framework: str, value: str) -> bool:
    """Check if a framework/value pair is valid."""
    fw = FRAMEWORKS.get(framework)
    if not fw:
        return False
    return value in fw["values"]


def framework_prompt_section(framework_names: list[str] | None = None) -> str:
    """Generate the framework description section for LLM prompts."""
    names = framework_names or DEFAULT_FRAMEWORKS
    sections = []
    for name in names:
        fw = FRAMEWORKS.get(name)
        if not fw:
            continue
        values_text = "\n".join(
            f"    - {v}: {desc}" for v, desc in fw["values"].items()
        )
        sections.append(
            f"### {fw['name']} (key: \"{name}\")\n"
            f"{fw['description']}\n"
            f"  Values:\n{values_text}"
        )
    return "\n\n".join(sections)
