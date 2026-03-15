"""DualSoul — Deep Soul + Surface Soul architecture.

Inspired by Kahneman's Thinking Fast and Slow:
- Surface Soul (Think Fast): live extraction, captures current state
- Deep Soul (Think Slow): compressed long-term personality, periodic consolidation
"""
from __future__ import annotations

import json
import math
import time

import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message
from soulgraph.frameworks import DEFAULT_FRAMEWORKS, framework_prompt_section
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge

_EMB_MODEL: SentenceTransformer | None = None

def _get_emb_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB_MODEL


class DualSoul:
    """Two-graph architecture: Surface (live) + Deep (compressed)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        deep_cycle: int = 100,
        max_surface_nodes: int = 200,
        carry_forward_k: int = 10,
        meta_cycle: int = 3,
    ):
        self._api_key = api_key
        self._model = model
        self.deep_cycle = deep_cycle
        self.max_surface_nodes = max_surface_nodes
        self.carry_forward_k = carry_forward_k
        self.meta_cycle = meta_cycle

        self._detector = Detector(api_key=api_key, model=model)
        self._messages: list[Message] = []
        self.total_utterances: int = 0
        self._consolidation_count: int = 0
        self._deep = SoulGraph(owner_id="unknown")

        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)

    @property
    def surface(self) -> SoulGraph:
        return self._detector.detected_graph

    @property
    def deep(self) -> SoulGraph:
        return self._deep

    @property
    def stats(self) -> dict:
        return {
            "total_utterances": self.total_utterances,
            "consolidation_count": self._consolidation_count,
            "surface_items": len(self.surface.items),
            "surface_edges": len(self.surface.edges),
            "deep_items": len(self._deep.items),
            "deep_edges": len(self._deep.edges),
        }

    def ingest(self, text: str) -> None:
        """Add utterance, extract to Surface. Auto-consolidates when needed."""
        self._messages.append(Message(role="speaker", content=text))
        self.total_utterances += 1
        self._detector.listen_and_detect(self._messages)

        # Auto-consolidate triggers
        if (self.total_utterances % self.deep_cycle == 0
                or len(self.surface.items) > self.max_surface_nodes):
            self.consolidate()

    def _adaptive_merge_threshold(self) -> float:
        """Merge threshold decreases as Deep grows. Smooth log curve."""
        n = max(len(self._deep.items), 1)
        return max(0.40, 0.82 - 0.06 * math.log(n))

    def consolidate(self) -> dict:
        """Consolidate Surface → Deep. The 'sleep phase'."""
        surface_items = self.surface.items
        if not surface_items:
            return {"merged": 0, "added": 0, "decayed": 0}

        threshold = self._adaptive_merge_threshold()
        self._consolidation_count += 1

        # Embedding similarity: Surface × Deep
        model = _get_emb_model()
        surface_texts = [i.text for i in surface_items]
        surface_embs = model.encode(surface_texts, normalize_embeddings=True)

        merge_pairs: list[tuple[SoulItem, SoulItem]] = []  # (surface, deep)
        new_items: list[SoulItem] = []
        id_remap: dict[str, str] = {}  # surface_id → deep_id

        if self._deep.items:
            deep_texts = [i.text for i in self._deep.items]
            deep_embs = model.encode(deep_texts, normalize_embeddings=True)
            sim_matrix = surface_embs @ deep_embs.T

            for si, s_item in enumerate(surface_items):
                max_sim = float(np.max(sim_matrix[si]))
                best_di = int(np.argmax(sim_matrix[si]))
                if max_sim >= threshold:
                    merge_pairs.append((s_item, self._deep.items[best_di]))
                    id_remap[s_item.id] = self._deep.items[best_di].id
                else:
                    new_items.append(s_item)
                    new_id = f"di_{len(self._deep.items) + len(new_items):04d}"
                    id_remap[s_item.id] = new_id
        else:
            new_items = list(surface_items)
            for idx, item in enumerate(new_items):
                new_id = f"di_{idx + 1:04d}"
                id_remap[item.id] = new_id

        # Batch LLM merge for matched pairs
        merged_count = 0
        if merge_pairs:
            merged_texts = self._batch_merge(merge_pairs)
            for (s_item, d_item), merged_text in zip(merge_pairs, merged_texts):
                if merged_text:
                    d_item.text = merged_text
                d_item.mention_count += s_item.mention_count + 1
                d_item.confidence = min(1.0, d_item.confidence + 0.05)
                d_item.last_reinforced_cycle = self._consolidation_count
                for dom in s_item.domains:
                    if dom not in d_item.domains:
                        d_item.domains.append(dom)
                merged_count += 1

        # Add new items to Deep
        added_count = 0
        for item in new_items:
            new_id = id_remap[item.id]
            deep_item = SoulItem(
                id=new_id,
                text=item.text,
                domains=item.domains,
                item_type=item.item_type,
                confidence=item.confidence,
                specificity=item.specificity,
                tags=item.tags,
                mention_count=item.mention_count + 1,
                last_reinforced_cycle=self._consolidation_count,
            )
            self._deep.add_item(deep_item)
            added_count += 1

        # Migrate edges
        self._migrate_edges(id_remap)

        # Decay unreinforced Deep nodes
        decayed = self._apply_decay()

        # Carry forward top-K and reset Surface
        self._carry_forward_and_reset()

        # Trigger meta-consolidation periodically
        if self._consolidation_count % self.meta_cycle == 0:
            meta_result = self.meta_consolidate()
            return {
                "merged": merged_count, "added": added_count,
                "decayed": decayed, **meta_result,
            }

        return {"merged": merged_count, "added": added_count, "decayed": decayed}

    _ROOT_DISCOVERY_PROMPT = """\
You are analyzing a person's soul graph to discover their deepest root motivations.

Below are the person's detected intentions/beliefs/values from conversation analysis.
Your task: discover the ABSTRACT ROOT MOTIVATIONS that drive clusters of these concrete intentions.

## Available Psychological Frameworks
{frameworks_section}

## Concrete Intentions (from Deep Soul graph)
{items_text}

## Existing Root Intentions (already discovered)
{existing_roots_text}

## Rules
1. Group concrete intentions that serve the SAME underlying need, even if surface text is very different.
   Example: "wants pretty dress" and "runs lumber mill for money" both serve safety/survival.
2. For each group, name the abstract root motivation in one sentence.
3. Tag each root with classifications from ALL applicable frameworks above.
4. If a concrete intention maps to an EXISTING root, reference it by root_id instead of creating a new one.
5. A concrete intention can map to MULTIPLE roots (multifinality).
6. Only create roots backed by 2+ concrete intentions (equifinality signal).
7. For Maslow: prefer lower layers when ambiguous (prepotency principle).

Return JSON array:
[
  {{
    "root_id": "ri_XXXX" or null (null = create new root),
    "root_text": "one sentence describing the root motivation",
    "motivation_tags": {{"maslow": "safety", "sdt": "autonomy", ...}},
    "concrete_ids": ["di_0001", "di_0002", ...],
    "confidence": 0.0-1.0
  }}
]"""

    def meta_consolidate(self, frameworks: list[str] | None = None) -> dict:
        """Discover abstract root intentions from Deep graph. The 'dream phase'."""
        concrete_items = [
            i for i in self._deep.items
            if i.abstraction_level == 0 and i.confidence > 0.3
        ]
        if len(concrete_items) < 3:
            return {"roots_created": 0, "roots_updated": 0, "edges_created": 0}

        fw_names = frameworks or DEFAULT_FRAMEWORKS
        frameworks_section = framework_prompt_section(fw_names)

        # Sort by importance
        concrete_items.sort(
            key=lambda i: i.mention_count * i.confidence, reverse=True
        )
        top_items = concrete_items[:40]

        existing_roots = [i for i in self._deep.items if i.abstraction_level == 1]

        items_text = "\n".join(
            f'- id: "{i.id}", text: "{i.text}", domains: {i.domains}, '
            f'mentions: {i.mention_count}, confidence: {i.confidence:.2f}'
            for i in top_items
        )
        existing_roots_text = "\n".join(
            f'- root_id: "{r.id}", text: "{r.text}", tags: {r.motivation_tags}, '
            f'mentions: {r.mention_count}'
            for r in existing_roots
        ) or "(none yet)"

        prompt = self._ROOT_DISCOVERY_PROMPT.format(
            frameworks_section=frameworks_section,
            items_text=items_text,
            existing_roots_text=existing_roots_text,
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            start = raw.find("[")
            end = raw.rfind("]")
            if start < 0 or end < 0:
                return {"roots_created": 0, "roots_updated": 0, "edges_created": 0}
            discoveries = json.loads(raw[start:end + 1])
        except Exception:
            return {"roots_created": 0, "roots_updated": 0, "edges_created": 0}

        roots_created = 0
        roots_updated = 0
        edges_created = 0
        deep_ids = {i.id for i in self._deep.items}

        for disc in discoveries:
            root_id = disc.get("root_id")
            root_text = disc.get("root_text", "")
            tags = disc.get("motivation_tags", {})
            concrete_ids = disc.get("concrete_ids", [])
            conf = float(disc.get("confidence", 0.7))

            if not root_text or not concrete_ids:
                continue
            concrete_ids = [cid for cid in concrete_ids if cid in deep_ids]
            if not concrete_ids:
                continue

            existing_root = None
            if root_id:
                existing_root = next(
                    (i for i in self._deep.items if i.id == root_id), None
                )

            if existing_root:
                existing_root.text = root_text
                existing_root.mention_count += len(concrete_ids)
                existing_root.confidence = min(1.0, existing_root.confidence + 0.05)
                existing_root.last_reinforced_cycle = self._consolidation_count
                existing_root.motivation_tags.update(tags)
                roots_updated += 1
                actual_root_id = existing_root.id
            else:
                ri_idx = len([i for i in self._deep.items if i.abstraction_level == 1]) + 1
                actual_root_id = f"ri_{ri_idx:04d}"
                root_item = SoulItem(
                    id=actual_root_id,
                    text=root_text,
                    domains=list({d for cid in concrete_ids
                                  for i in self._deep.items if i.id == cid
                                  for d in i.domains}),
                    confidence=conf,
                    abstraction_level=1,
                    motivation_tags=tags,
                    mention_count=len(concrete_ids),
                    last_reinforced_cycle=self._consolidation_count,
                )
                self._deep.add_item(root_item)
                roots_created += 1

            for cid in concrete_ids:
                already = any(
                    e.from_id == actual_root_id and e.to_id == cid
                    and e.relation == "manifests-as"
                    for e in self._deep.edges
                )
                if not already:
                    self._deep.add_edge(SoulEdge(
                        from_id=actual_root_id, to_id=cid,
                        relation="manifests-as", strength=conf, confidence=conf,
                    ))
                    edges_created += 1

        return {
            "roots_created": roots_created,
            "roots_updated": roots_updated,
            "edges_created": edges_created,
        }

    _MERGE_PROMPT = """\
You are consolidating a knowledge graph. For each pair, merge the "new observation" \
into the "existing concept" to create a single updated description.

{pairs_text}

Return JSON array only:
[
  {{"deep_id": "...", "merged_text": "one concise sentence"}},
  ...
]"""

    def _batch_merge(self, pairs: list[tuple[SoulItem, SoulItem]]) -> list[str]:
        """Batch LLM call to merge Surface texts into Deep texts."""
        if not pairs:
            return []

        pairs_text = "\n".join(
            f'- deep_id: "{d.id}", existing: "{d.text}", new observation: "{s.text}"'
            for s, d in pairs
        )
        prompt = self._MERGE_PROMPT.format(pairs_text=pairs_text)

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
            start = raw.find("[")
            end = raw.rfind("]")
            if start >= 0 and end >= 0:
                results = json.loads(raw[start:end + 1])
                id_to_text = {r["deep_id"]: r["merged_text"] for r in results}
                return [id_to_text.get(d.id, d.text) for _, d in pairs]
        except Exception:
            pass

        # Fallback: keep existing deep texts
        return [d.text for _, d in pairs]

    def _migrate_edges(self, id_remap: dict[str, str]) -> None:
        """Migrate Surface edges to Deep, remapping IDs."""
        deep_ids = {item.id for item in self._deep.items}
        for edge in self.surface.edges:
            deep_from = id_remap.get(edge.from_id)
            deep_to = id_remap.get(edge.to_id)
            if not deep_from or not deep_to:
                continue
            if deep_from not in deep_ids or deep_to not in deep_ids:
                continue
            if deep_from == deep_to:
                continue
            existing = any(
                e.from_id == deep_from and e.to_id == deep_to and e.relation == edge.relation
                for e in self._deep.edges
            )
            if existing:
                self._deep.strengthen_edge(deep_from, deep_to, 0.1)
            else:
                self._deep.add_edge(SoulEdge(
                    from_id=deep_from,
                    to_id=deep_to,
                    relation=edge.relation,
                    strength=edge.strength,
                    confidence=edge.confidence,
                ))

    def _apply_decay(self) -> int:
        """Decay unreinforced Deep nodes. Returns count of decayed nodes."""
        decayed = 0
        for item in self._deep.items:
            if item.last_reinforced_cycle < self._consolidation_count:
                cycles_since = self._consolidation_count - item.last_reinforced_cycle
                decay_amount = 0.02 * cycles_since
                item.confidence = max(0.05, item.confidence - decay_amount)
                decayed += 1
        return decayed

    def _carry_forward_and_reset(self) -> None:
        """Keep top-K Surface nodes by PageRank, reset the rest."""
        if not self.surface.items:
            return

        pr = self.surface.pagerank()
        if not pr:
            self._detector.detected_graph = SoulGraph(owner_id=self.surface.owner_id)
            return

        sorted_ids = sorted(pr, key=pr.get, reverse=True)[:self.carry_forward_k]
        selected = set(sorted_ids)

        carry_items = [i for i in self.surface.items if i.id in selected]
        carry_edges = [
            e for e in self.surface.edges
            if e.from_id in selected and e.to_id in selected
        ]

        new_surface = SoulGraph(
            owner_id=self.surface.owner_id,
            items=carry_items,
            edges=carry_edges,
        )
        self._detector.detected_graph = new_surface

    # ── Query routing ──────────────────────────────────────────────

    _RECENCY_KEYWORDS = {"now", "right now", "currently", "today", "lately", "recent", "这会", "现在", "最近", "当下"}
    _ENDURING_KEYWORDS = {"always", "kind of person", "generally", "core", "deep", "usually", "personality", "一直", "本质", "性格", "一般", "通常"}

    _DUAL_QUERY_SYSTEM = """\
You are answering a question about a person based on two layers of their soul graph.

## Surface Soul (current state, recent observations)
{surface_nodes}

## Deep Soul (enduring personality, long-term patterns)
{deep_nodes}

## Rules
1. Surface items reflect what the person is thinking/doing NOW.
2. Deep items reflect WHO the person IS at a fundamental level.
3. Weigh Surface vs Deep based on the question type.
4. Be concise: 2-4 sentences.
5. Respond in the same language as the query."""

    def _route_query(self, question: str) -> tuple[float, float]:
        """Returns (surface_weight, deep_weight) based on keywords."""
        q_lower = question.lower()
        has_recency = any(kw in q_lower for kw in self._RECENCY_KEYWORDS)
        has_enduring = any(kw in q_lower for kw in self._ENDURING_KEYWORDS)

        if has_recency and not has_enduring:
            return (0.7, 0.3)
        elif has_enduring and not has_recency:
            return (0.3, 0.7)
        return (0.5, 0.5)

    def query(self, question: str, top_k: int = 10) -> str:
        """Query both souls, route by keywords, synthesize answer."""
        if not self.surface.items and not self._deep.items:
            return "Both graphs are empty — ingest some conversation first."

        sw, dw = self._route_query(question)
        surface_k = max(1, int(top_k * sw))
        deep_k = max(1, int(top_k * dw))

        surface_sub = (
            self.surface.query_subgraph(question, top_k=surface_k)
            if self.surface.items else SoulGraph(owner_id="")
        )
        deep_sub = (
            self._deep.query_subgraph(question, top_k=deep_k)
            if self._deep.items else SoulGraph(owner_id="")
        )

        surface_nodes = "\n".join(
            f"- [recent] {i.text} (domains: {', '.join(i.domains)}, confidence: {i.confidence:.1f})"
            for i in surface_sub.items
        ) or "(empty)"
        deep_nodes = "\n".join(
            f"- [enduring] {i.text} (domains: {', '.join(i.domains)}, confidence: {i.confidence:.1f}, mentions: {i.mention_count})"
            for i in deep_sub.items
        ) or "(empty)"

        system = self._DUAL_QUERY_SYSTEM.format(
            surface_nodes=surface_nodes,
            deep_nodes=deep_nodes,
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=1024,
                system=system,
                messages=[{"role": "user", "content": question}],
            )
            if response.content:
                return response.content[0].text
        except Exception:
            pass
        return ""

    # ── Persistence ────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save both graphs + state to JSON."""
        from pathlib import Path
        data = {
            "owner_id": self.surface.owner_id,
            "total_utterances": self.total_utterances,
            "consolidation_count": self._consolidation_count,
            "surface": json.loads(self.surface.model_dump_json()),
            "deep": json.loads(self._deep.model_dump_json()),
            "config": {
                "deep_cycle": self.deep_cycle,
                "max_surface_nodes": self.max_surface_nodes,
                "carry_forward_k": self.carry_forward_k,
                "meta_cycle": self.meta_cycle,
            },
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        """Load both graphs + state from JSON."""
        from pathlib import Path
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._deep = SoulGraph.model_validate(data["deep"])
        self._detector.detected_graph = SoulGraph.model_validate(data["surface"])
        self.total_utterances = data.get("total_utterances", 0)
        self._consolidation_count = data.get("consolidation_count", 0)
        config = data.get("config", {})
        if "deep_cycle" in config:
            self.deep_cycle = config["deep_cycle"]
        if "max_surface_nodes" in config:
            self.max_surface_nodes = config["max_surface_nodes"]
        if "carry_forward_k" in config:
            self.carry_forward_k = config["carry_forward_k"]
        if "meta_cycle" in config:
            self.meta_cycle = config["meta_cycle"]
