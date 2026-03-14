# Dual Soul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the DualSoul class — two independent SoulGraph instances (Surface Soul + Deep Soul) with periodic consolidation, adaptive merge threshold, and dual-graph query.

**Architecture:** Surface Soul extracts continuously via existing Detector (0.82 dedup). Every M=100 utterances, a "sleep phase" consolidates Surface→Deep using adaptive merge threshold (0.82-0.06·ln(n)), batch LLM text merge, edge migration, and priority-based decay. Query searches both graphs via PPR with keyword-based routing.

**Tech Stack:** Python 3.12, pydantic v2, anthropic SDK, sentence-transformers (all-MiniLM-L6-v2), numpy, networkx, pytest

**Design doc:** `docs/plans/2026-03-15-dual-soul-design.md` (V11 is the final design)

---

### Task 1: Add `last_reinforced_cycle` field to SoulItem

**Files:**
- Modify: `soulgraph/graph/models.py:40-68`
- Test: `tests/test_graph.py`

**Step 1: Write the failing test**

Add to `tests/test_graph.py`:

```python
class TestSoulItemReinforcedCycle:
    def test_default_last_reinforced_cycle(self):
        item = SoulItem(id="si_001", text="test", domains=["x"])
        assert item.last_reinforced_cycle == 0

    def test_set_last_reinforced_cycle(self):
        item = SoulItem(id="si_001", text="test", domains=["x"], last_reinforced_cycle=5)
        assert item.last_reinforced_cycle == 5
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph.py::TestSoulItemReinforcedCycle -v`
Expected: FAIL with "unexpected keyword argument 'last_reinforced_cycle'"

**Step 3: Write minimal implementation**

In `soulgraph/graph/models.py`, add to `SoulItem` class after `tags: list[str] = []` (line 54):

```python
    last_reinforced_cycle: int = 0
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_graph.py::TestSoulItemReinforcedCycle -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/test_graph.py -v`
Expected: All existing tests still PASS

**Step 6: Commit**

```bash
git add soulgraph/graph/models.py tests/test_graph.py
git commit -m "feat: add last_reinforced_cycle field to SoulItem"
```

---

### Task 2: Create DualSoul class skeleton with ingest

**Files:**
- Create: `soulgraph/dual_soul.py`
- Create: `tests/test_dual_soul.py`

**Step 1: Write the failing tests**

Create `tests/test_dual_soul.py`:

```python
"""Tests for DualSoul — Dual Soul architecture."""
import math
from unittest.mock import MagicMock, patch

from soulgraph.dual_soul import DualSoul
from soulgraph.graph.models import SoulGraph, SoulItem, SoulEdge


class TestDualSoulInit:
    @patch("soulgraph.dual_soul.Detector")
    def test_creates_two_graphs(self, MockDetector):
        ds = DualSoul(api_key="fake")
        assert isinstance(ds.surface, SoulGraph)
        assert isinstance(ds.deep, SoulGraph)

    @patch("soulgraph.dual_soul.Detector")
    def test_default_config(self, MockDetector):
        ds = DualSoul(api_key="fake")
        assert ds.deep_cycle == 100
        assert ds.max_surface_nodes == 200
        assert ds.carry_forward_k == 10
        assert ds.total_utterances == 0

    @patch("soulgraph.dual_soul.Detector")
    def test_custom_config(self, MockDetector):
        ds = DualSoul(api_key="fake", deep_cycle=50, max_surface_nodes=100)
        assert ds.deep_cycle == 50
        assert ds.max_surface_nodes == 100


class TestDualSoulIngest:
    @patch("soulgraph.dual_soul.Detector")
    def test_ingest_calls_detector(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="unknown")
        mock_det.listen_and_detect.return_value = mock_det.detected_graph

        ds = DualSoul(api_key="fake")
        ds.ingest("I love hiking")

        assert mock_det.listen_and_detect.call_count == 1
        assert ds.total_utterances == 1

    @patch("soulgraph.dual_soul.Detector")
    def test_ingest_increments_counter(self, MockDetector):
        mock_det = MockDetector.return_value
        mock_det.detected_graph = SoulGraph(owner_id="unknown")
        mock_det.listen_and_detect.return_value = mock_det.detected_graph

        ds = DualSoul(api_key="fake")
        ds.ingest("First")
        ds.ingest("Second")
        ds.ingest("Third")

        assert ds.total_utterances == 3

    @patch("soulgraph.dual_soul.Detector")
    def test_surface_is_detector_graph(self, MockDetector):
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="hiking", domains=["hobby"]))
        mock_det = MockDetector.return_value
        mock_det.detected_graph = g
        mock_det.listen_and_detect.return_value = g

        ds = DualSoul(api_key="fake")
        ds.ingest("I love hiking")

        assert len(ds.surface.items) == 1
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestDualSoulInit -v`
Expected: FAIL with "No module named 'soulgraph.dual_soul'"

**Step 3: Write minimal implementation**

Create `soulgraph/dual_soul.py`:

```python
"""DualSoul — Deep Soul + Surface Soul architecture.

Inspired by Kahneman's Thinking Fast and Slow:
- Surface Soul (Think Fast): live extraction, captures current state
- Deep Soul (Think Slow): compressed long-term personality, periodic consolidation
"""
from __future__ import annotations

import math

import anthropic

from soulgraph.experiment.detector import Detector
from soulgraph.experiment.models import Message
from soulgraph.graph.models import SoulGraph


class DualSoul:
    """Two-graph architecture: Surface (live) + Deep (compressed)."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        deep_cycle: int = 100,
        max_surface_nodes: int = 200,
        carry_forward_k: int = 10,
    ):
        self._api_key = api_key
        self._model = model
        self.deep_cycle = deep_cycle
        self.max_surface_nodes = max_surface_nodes
        self.carry_forward_k = carry_forward_k

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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dual_soul.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: DualSoul class skeleton with ingest and auto-consolidate trigger"
```

---

### Task 3: Implement adaptive merge threshold

**Files:**
- Modify: `soulgraph/dual_soul.py`
- Test: `tests/test_dual_soul.py`

**Step 1: Write the failing test**

Add to `tests/test_dual_soul.py`:

```python
class TestAdaptiveThreshold:
    @patch("soulgraph.dual_soul.Detector")
    def test_threshold_empty_deep(self, MockDetector):
        ds = DualSoul(api_key="fake")
        # n=1 (min) → 0.82
        assert ds._adaptive_merge_threshold() >= 0.81

    @patch("soulgraph.dual_soul.Detector")
    def test_threshold_decreases_with_size(self, MockDetector):
        ds = DualSoul(api_key="fake")
        # Add items to Deep
        for i in range(100):
            ds._deep.add_item(SoulItem(
                id=f"si_{i:03d}", text=f"item {i}", domains=["test"]
            ))
        t100 = ds._adaptive_merge_threshold()
        # Should be lower than empty
        assert t100 < 0.60

    @patch("soulgraph.dual_soul.Detector")
    def test_threshold_has_floor(self, MockDetector):
        ds = DualSoul(api_key="fake")
        for i in range(10000):
            ds._deep.add_item(SoulItem(
                id=f"si_{i:05d}", text=f"item {i}", domains=["test"]
            ))
        assert ds._adaptive_merge_threshold() >= 0.40
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestAdaptiveThreshold -v`
Expected: FAIL with "has no attribute '_adaptive_merge_threshold'"

**Step 3: Write minimal implementation**

Add to `DualSoul` class in `soulgraph/dual_soul.py`:

```python
    def _adaptive_merge_threshold(self) -> float:
        """Merge threshold decreases as Deep grows. Smooth log curve."""
        n = max(len(self._deep.items), 1)
        return max(0.40, 0.82 - 0.06 * math.log(n))
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestAdaptiveThreshold -v`
Expected: PASS

**Step 5: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: adaptive merge threshold for Deep Soul consolidation"
```

---

### Task 4: Implement consolidation core — embedding similarity + partition

**Files:**
- Modify: `soulgraph/dual_soul.py`
- Test: `tests/test_dual_soul.py`

**Step 1: Write the failing test**

Add to `tests/test_dual_soul.py`:

```python
class TestConsolidation:
    @patch("soulgraph.dual_soul.Detector")
    def test_consolidate_empty_surface(self, MockDetector):
        """Consolidation with empty Surface should be a no-op."""
        ds = DualSoul(api_key="fake")
        result = ds.consolidate()
        assert result["merged"] == 0
        assert result["added"] == 0

    @patch("soulgraph.dual_soul.anthropic")
    @patch("soulgraph.dual_soul.Detector")
    def test_consolidate_adds_new_to_deep(self, MockDetector, mock_anthropic):
        """Surface nodes with no Deep match should be added as new."""
        mock_det = MockDetector.return_value
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="loves hiking", domains=["hobby"], confidence=0.8))
        g.add_item(SoulItem(id="si_002", text="fears poverty", domains=["emotion"], confidence=0.9))
        mock_det.detected_graph = g

        ds = DualSoul(api_key="fake")
        result = ds.consolidate()

        assert result["added"] == 2
        assert len(ds.deep.items) == 2

    @patch("soulgraph.dual_soul.anthropic")
    @patch("soulgraph.dual_soul.Detector")
    def test_consolidate_merges_similar(self, MockDetector, mock_anthropic):
        """Surface node similar to Deep node should merge, not add."""
        # Pre-populate Deep with an item
        mock_det = MockDetector.return_value
        surface = SoulGraph(owner_id="test")
        surface.add_item(SoulItem(
            id="si_s01", text="afraid of being hungry again",
            domains=["emotion"], confidence=0.8
        ))
        mock_det.detected_graph = surface

        ds = DualSoul(api_key="fake")
        # Add similar item to Deep
        ds._deep.add_item(SoulItem(
            id="di_001", text="fears hunger and poverty",
            domains=["emotion"], confidence=0.7, mention_count=2
        ))

        # Mock LLM merge response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"deep_id": "di_001", "merged_text": "deeply fears hunger and poverty, afraid of being hungry again"}]')]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = ds.consolidate()

        # Should merge, not add
        assert result["merged"] >= 1
        assert len(ds.deep.items) == 1  # Still 1 item, not 2
        assert ds.deep.items[0].mention_count >= 3  # Was 2, now 3+
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestConsolidation -v`
Expected: FAIL with various attribute errors

**Step 3: Write minimal implementation**

Add to `DualSoul` class in `soulgraph/dual_soul.py`. Add this import at the top:

```python
import json
import numpy as np
from sentence_transformers import SentenceTransformer

_EMB_MODEL: SentenceTransformer | None = None

def _get_emb_model() -> SentenceTransformer:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        _EMB_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMB_MODEL
```

Add the `consolidate` method to the class:

```python
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
                # Merge domains
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

        return {"merged": merged_count, "added": added_count, "decayed": decayed}
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestConsolidation -v`
Expected: PASS (some tests may need `_batch_merge`, `_migrate_edges`, `_apply_decay`, `_carry_forward_and_reset` stubs)

**Step 5: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: consolidation core — embedding similarity, partition, merge"
```

---

### Task 5: Implement batch LLM merge

**Files:**
- Modify: `soulgraph/dual_soul.py`
- Test: `tests/test_dual_soul.py`

**Step 1: Write the failing test**

Add to `tests/test_dual_soul.py`:

```python
class TestBatchMerge:
    @patch("soulgraph.dual_soul.anthropic")
    @patch("soulgraph.dual_soul.Detector")
    def test_batch_merge_returns_texts(self, MockDetector, mock_anthropic):
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='[{"deep_id": "di_001", "merged_text": "merged concept A and B"}]')]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        ds = DualSoul(api_key="fake")

        s_item = SoulItem(id="si_001", text="concept A", domains=["test"])
        d_item = SoulItem(id="di_001", text="concept B", domains=["test"])
        pairs = [(s_item, d_item)]

        result = ds._batch_merge(pairs)
        assert len(result) == 1
        assert "merged" in result[0].lower()

    @patch("soulgraph.dual_soul.anthropic")
    @patch("soulgraph.dual_soul.Detector")
    def test_batch_merge_empty(self, MockDetector, mock_anthropic):
        ds = DualSoul(api_key="fake")
        result = ds._batch_merge([])
        assert result == []
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestBatchMerge -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `DualSoul` class:

```python
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
                # Map by deep_id
                id_to_text = {r["deep_id"]: r["merged_text"] for r in results}
                return [id_to_text.get(d.id, d.text) for _, d in pairs]
        except Exception:
            pass

        # Fallback: keep existing deep texts
        return [d.text for _, d in pairs]
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestBatchMerge -v`
Expected: PASS

**Step 5: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: batch LLM merge for Deep Soul consolidation"
```

---

### Task 6: Implement edge migration, decay, and carry-forward

**Files:**
- Modify: `soulgraph/dual_soul.py`
- Test: `tests/test_dual_soul.py`

**Step 1: Write the failing tests**

Add to `tests/test_dual_soul.py`:

```python
class TestEdgeMigration:
    @patch("soulgraph.dual_soul.Detector")
    def test_edges_remapped_to_deep(self, MockDetector):
        ds = DualSoul(api_key="fake")
        # Simulate id_remap
        remap = {"si_001": "di_001", "si_002": "di_002"}
        # Add surface edges
        ds._detector.detected_graph = SoulGraph(owner_id="test")
        ds.surface.add_item(SoulItem(id="si_001", text="a", domains=["x"]))
        ds.surface.add_item(SoulItem(id="si_002", text="b", domains=["x"]))
        ds.surface.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        # Add deep items
        ds._deep.add_item(SoulItem(id="di_001", text="a", domains=["x"]))
        ds._deep.add_item(SoulItem(id="di_002", text="b", domains=["x"]))

        ds._migrate_edges(remap)

        assert len(ds._deep.edges) >= 1
        e = ds._deep.edges[0]
        assert e.from_id == "di_001"
        assert e.to_id == "di_002"


class TestDecay:
    @patch("soulgraph.dual_soul.Detector")
    def test_unreinforced_nodes_decay(self, MockDetector):
        ds = DualSoul(api_key="fake")
        ds._consolidation_count = 5
        ds._deep.add_item(SoulItem(
            id="di_001", text="old item", domains=["x"],
            confidence=0.8, last_reinforced_cycle=1
        ))
        ds._deep.add_item(SoulItem(
            id="di_002", text="recent item", domains=["x"],
            confidence=0.8, last_reinforced_cycle=5
        ))

        decayed = ds._apply_decay()

        # di_001 should have lower confidence, di_002 unchanged
        old = next(i for i in ds._deep.items if i.id == "di_001")
        recent = next(i for i in ds._deep.items if i.id == "di_002")
        assert old.confidence < 0.8
        assert recent.confidence == 0.8
        assert decayed >= 1


class TestCarryForward:
    @patch("soulgraph.dual_soul.Detector")
    def test_carry_forward_keeps_top_k(self, MockDetector):
        mock_det = MockDetector.return_value
        g = SoulGraph(owner_id="test")
        # Add 20 items, some with edges
        for i in range(20):
            g.add_item(SoulItem(id=f"si_{i:03d}", text=f"item {i}", domains=["x"]))
        # Add edges to make si_000 a hub
        for i in range(1, 15):
            g.add_edge(SoulEdge(from_id=f"si_{i:03d}", to_id="si_000", relation="drives"))
        mock_det.detected_graph = g

        ds = DualSoul(api_key="fake", carry_forward_k=5)
        ds._carry_forward_and_reset()

        # Should keep ≤ 5 items
        assert len(ds.surface.items) <= 5
        # si_000 (hub) should be among carried forward
        carried_ids = {i.id for i in ds.surface.items}
        assert "si_000" in carried_ids
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestEdgeMigration tests/test_dual_soul.py::TestDecay tests/test_dual_soul.py::TestCarryForward -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add these methods to `DualSoul` class:

```python
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
            # Check if Deep already has this edge
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
                # Priority-based decay: low mention + low confidence = faster decay
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestEdgeMigration tests/test_dual_soul.py::TestDecay tests/test_dual_soul.py::TestCarryForward -v`
Expected: PASS

**Step 5: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: edge migration, priority decay, carry-forward for consolidation"
```

---

### Task 7: Implement dual-graph query with keyword routing

**Files:**
- Modify: `soulgraph/dual_soul.py`
- Test: `tests/test_dual_soul.py`

**Step 1: Write the failing tests**

Add to `tests/test_dual_soul.py`:

```python
class TestQueryRouting:
    @patch("soulgraph.dual_soul.Detector")
    def test_recency_keywords(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("What is she thinking right now?")
        assert sw > dw

    @patch("soulgraph.dual_soul.Detector")
    def test_enduring_keywords(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("What kind of person is she generally?")
        assert dw > sw

    @patch("soulgraph.dual_soul.Detector")
    def test_default_routing(self, MockDetector):
        ds = DualSoul(api_key="fake")
        sw, dw = ds._route_query("Tell me about her")
        assert sw == dw == 0.5


class TestDualQuery:
    @patch("soulgraph.dual_soul.anthropic")
    @patch("soulgraph.dual_soul.Detector")
    def test_query_empty_graphs(self, MockDetector, mock_anthropic):
        ds = DualSoul(api_key="fake")
        result = ds.query("What drives this person?")
        assert "empty" in result.lower() or "no" in result.lower()

    @patch("soulgraph.dual_soul.anthropic")
    @patch("soulgraph.dual_soul.Detector")
    def test_query_with_items(self, MockDetector, mock_anthropic):
        mock_det = MockDetector.return_value
        surface = SoulGraph(owner_id="test")
        surface.add_item(SoulItem(id="si_001", text="thinking about dinner", domains=["daily"]))
        mock_det.detected_graph = surface

        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(id="di_001", text="core belief in family", domains=["values"]))

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="She is currently thinking about dinner but deeply values family.")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        result = ds.query("What is she thinking?")
        assert len(result) > 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestQueryRouting tests/test_dual_soul.py::TestDualQuery -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `DualSoul` class:

```python
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

        # PPR on each graph
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestQueryRouting tests/test_dual_soul.py::TestDualQuery -v`
Expected: PASS

**Step 5: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: dual-graph query with keyword routing"
```

---

### Task 8: Implement save/load for DualSoul

**Files:**
- Modify: `soulgraph/dual_soul.py`
- Test: `tests/test_dual_soul.py`

**Step 1: Write the failing test**

Add to `tests/test_dual_soul.py`:

```python
class TestDualSoulPersistence:
    @patch("soulgraph.dual_soul.Detector")
    def test_save_and_load(self, MockDetector, tmp_path):
        mock_det = MockDetector.return_value
        surface = SoulGraph(owner_id="test")
        surface.add_item(SoulItem(id="si_001", text="surface item", domains=["x"]))
        mock_det.detected_graph = surface

        ds = DualSoul(api_key="fake")
        ds._deep.add_item(SoulItem(id="di_001", text="deep item", domains=["y"]))
        ds.total_utterances = 50
        ds._consolidation_count = 2

        path = str(tmp_path / "dual_soul.json")
        ds.save(path)

        # Load into new instance
        ds2 = DualSoul(api_key="fake")
        ds2.load(path)

        assert len(ds2.deep.items) == 1
        assert ds2.deep.items[0].text == "deep item"
        assert len(ds2.surface.items) == 1
        assert ds2.total_utterances == 50
        assert ds2._consolidation_count == 2
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestDualSoulPersistence -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `DualSoul` class:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_dual_soul.py::TestDualSoulPersistence -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add soulgraph/dual_soul.py tests/test_dual_soul.py
git commit -m "feat: save/load persistence for DualSoul"
```

---

### Task 9: Add DualSoul to CLI interactive mode

**Files:**
- Modify: `soulgraph/cli.py`

**Step 1: Read current CLI**

Read `soulgraph/cli.py` to understand the existing `--interact` flag.

**Step 2: Add `--dual` flag**

Add a `--dual` flag that uses `DualSoul` instead of `SoulEngine` in interactive mode:

```python
# In argument parser:
parser.add_argument("--dual", action="store_true", help="Use DualSoul (Deep + Surface) architecture")

# In interactive mode:
if args.dual:
    from soulgraph.dual_soul import DualSoul
    engine = DualSoul(api_key=api_key, model=model)
else:
    engine = SoulEngine(api_key=api_key, model=model)
```

Add `/stats` command to show dual soul stats and `/consolidate` to force consolidation.

**Step 3: Test manually**

Run: `.venv/bin/python -m soulgraph --interact --dual`
Type a few messages, then `/stats` to see both graphs, `/consolidate` to trigger.

**Step 4: Commit**

```bash
git add soulgraph/cli.py
git commit -m "feat: add --dual flag to CLI for DualSoul interactive mode"
```

---

### Task 10: Integration test with GWTW data

**Files:**
- Create: `scripts/dual_soul_gwtw.py`

**Step 1: Write the script**

```python
"""Test DualSoul with Gone with the Wind dialogue."""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soulgraph.dual_soul import DualSoul

def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("Error: set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    # Load Scarlett's lines
    lines = []
    with open("fixtures/gone_with_wind.jsonl") as f:
        for line in f:
            data = json.loads(line.strip())
            if data["speaker"] == "scarlett":
                lines.append(data["text"])

    print(f"Scarlett lines: {len(lines)}")

    ds = DualSoul(api_key=api_key, deep_cycle=30, max_surface_nodes=100)

    for i, line in enumerate(lines):
        ds.ingest(line)
        if (i + 1) % 10 == 0:
            s = ds.stats
            print(f"  Line {i+1}: Surface={s['surface_items']} items, Deep={s['deep_items']} items")

    print(f"\nFinal: {ds.stats}")

    # Query both souls
    queries = [
        "What is Scarlett thinking about right now?",
        "What kind of person is Scarlett at her core?",
        "What drives Scarlett's decisions?",
    ]
    for q in queries:
        print(f"\nQ: {q}")
        answer = ds.query(q)
        print(f"A: {answer}")

    # Save
    ds.save("results/scarlett_dual_soul.json")
    print("\nSaved to results/scarlett_dual_soul.json")

if __name__ == "__main__":
    main()
```

**Step 2: Run it**

Run: `export ANTHROPIC_API_KEY=... && .venv/bin/python scripts/dual_soul_gwtw.py`

**Step 3: Analyze results**

Compare Surface vs Deep node counts. Verify Deep converges faster than single-graph.

**Step 4: Commit**

```bash
git add scripts/dual_soul_gwtw.py
git commit -m "feat: DualSoul integration test with GWTW Scarlett dialogue"
```

---

### Task 11: Final — run all tests, push to GitHub

**Step 1: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests PASS (existing 38 + new ~20 = ~58 tests)

**Step 2: Commit any final fixes**

**Step 3: Push**

```bash
git push origin feat/v4-soul-operating-system
```
