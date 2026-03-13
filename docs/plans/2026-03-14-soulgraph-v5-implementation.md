# SoulGraph V5 Multi-Session Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add multi-session memory — graph persists across conversations, mention reinforcement boosts PageRank, rankings improve over time.

**Architecture:** Extend V4 incrementally. Add `source_session` to SoulEdge, tune mention_count coefficient, add topic-steered Speaker, build `run_multi_session()` in runner, create 3-session sequential fixture, add cross-session evaluation metrics.

**Tech Stack:** Python 3.12, pydantic v2, networkx, scipy, scikit-learn, sentence-transformers, anthropic SDK, pytest

---

### Task 1: Add `source_session` to SoulEdge + tune mention coefficient

**Files:**
- Modify: `soulgraph/graph/models.py`
- Modify: `tests/test_graph.py`

**Step 1: Write the failing tests**

Add to `tests/test_graph.py`:

```python
class TestSessionAwareness:
    def test_edge_source_session_default(self):
        edge = SoulEdge(from_id="a", to_id="b", relation="drives")
        assert edge.source_session == 0

    def test_edge_source_session_set(self):
        edge = SoulEdge(from_id="a", to_id="b", relation="drives", source_session=2)
        assert edge.source_session == 2

    def test_mention_boost_in_pagerank(self):
        """Items with higher mention_count should rank higher via boosted edge weight."""
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="A", domains=["x"], mention_count=5))
        g.add_item(SoulItem(id="si_002", text="B", domains=["x"], mention_count=0))
        g.add_item(SoulItem(id="si_003", text="C", domains=["x"], mention_count=0))
        # Both have same edge structure but si_001 has high mention_count
        g.add_edge(SoulEdge(from_id="si_003", to_id="si_001", relation="drives", strength=0.5))
        g.add_edge(SoulEdge(from_id="si_003", to_id="si_002", relation="drives", strength=0.5))
        ranks = g.pagerank()
        # si_001 should rank higher than si_002 due to mention_count boost
        assert ranks["si_001"] > ranks["si_002"]
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_graph.py::TestSessionAwareness -v`
Expected: FAIL (source_session not a field on SoulEdge)

**Step 3: Write minimal implementation**

In `soulgraph/graph/models.py`, add to `SoulEdge` class after `confidence`:

```python
    source_session: int = 0
```

In `SoulGraph._to_nx()`, change the weight formula from:
```python
weight = edge.strength * (1 + 0.1 * self._mention_count(edge.from_id))
```
to:
```python
weight = edge.strength * (1 + 0.2 * self._mention_count(edge.from_id))
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph.py::TestSessionAwareness -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/graph/models.py tests/test_graph.py
git commit -m "feat(v5): add source_session to SoulEdge, tune mention coefficient to 0.2"
```

---

### Task 2: Add topic_hints to Speaker

**Files:**
- Modify: `soulgraph/experiment/speaker.py`
- Modify: `tests/test_speaker.py`

**Step 1: Write the failing test**

Add to `tests/test_speaker.py`:

```python
    def test_topic_hints_in_system_prompt(self):
        g = SoulGraph(owner_id="test")
        g.add_item(SoulItem(id="si_001", text="test", domains=["x"]))
        speaker = Speaker(soul_graph=g, api_key="fake", topic_hints=["career", "family"])
        # Verify topic_hints are stored
        assert speaker.topic_hints == ["career", "family"]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_speaker.py::TestSpeaker::test_topic_hints_in_system_prompt -v`
Expected: FAIL (unexpected keyword argument 'topic_hints')

**Step 3: Write minimal implementation**

In `soulgraph/experiment/speaker.py`, modify `Speaker.__init__`:

```python
    def __init__(
        self,
        soul_graph: SoulGraph,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        topic_hints: list[str] | None = None,
    ):
        self.soul = soul_graph
        self.disclosed: set[str] = set()
        self.topic_hints = topic_hints or []
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
```

In the `respond()` method, after the system prompt is built, add topic hints if present. Add this after `system = _SPEAKER_SYSTEM.format(...)`:

```python
        if self.topic_hints:
            system += (
                f"\n\n## Session Focus\n"
                f"This session focuses on: {', '.join(self.topic_hints)}.\n"
                f"Steer the conversation naturally toward these topics. "
                f"You may touch on other topics if they come up naturally, but prioritize these areas."
            )
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_speaker.py -v`
Expected: All PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/experiment/speaker.py tests/test_speaker.py
git commit -m "feat(v5): add topic_hints to Speaker for session-steered conversation"
```

---

### Task 3: Add session_number to Detector

**Files:**
- Modify: `soulgraph/experiment/detector.py`
- Modify: `tests/test_experiment.py`

**Step 1: Write the failing test**

Add to `tests/test_experiment.py`:

```python
class TestDetectorSession:
    def test_detector_session_number(self):
        from soulgraph.experiment.detector import Detector
        det = Detector(api_key="fake", session_number=2)
        assert det.session_number == 2

    def test_detector_default_session(self):
        from soulgraph.experiment.detector import Detector
        det = Detector(api_key="fake")
        assert det.session_number == 0
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_experiment.py::TestDetectorSession -v`
Expected: FAIL (unexpected keyword argument 'session_number')

**Step 3: Write minimal implementation**

In `soulgraph/experiment/detector.py`, modify `Detector.__init__`:

```python
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514", session_number: int = 0):
        self.detected_graph = SoulGraph(owner_id="unknown")
        self.session_number = session_number
        kwargs: dict = {"api_key": api_key}
        if api_key.startswith("sk-or-"):
            kwargs["base_url"] = "https://openrouter.ai/api"
        self._client = anthropic.Anthropic(**kwargs)
        self._model = model
```

In `_add_item()`, set `source_session`:

```python
        self.detected_graph.add_item(
            SoulItem(
                id=item_id,
                text=item_data["text"],
                domains=item_data.get("domains", ["general"]),
                item_type=item_type,
                confidence=confidence,
                specificity=item_data.get("specificity", 0.5),
                tags=tags,
                source_session=str(self.session_number),
            )
        )
```

In `_add_edge_safe()`, set `source_session`:

```python
        self.detected_graph.add_edge(
            SoulEdge(
                from_id=from_id,
                to_id=to_id,
                relation=edge_data.get("relation", "relates_to"),
                strength=edge_data.get("strength", 0.5),
                confidence=edge_data.get("confidence", 0.5),
                source_session=self.session_number,
            )
        )
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_experiment.py::TestDetectorSession -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/experiment/detector.py tests/test_experiment.py
git commit -m "feat(v5): add session_number to Detector, set source_session on items/edges"
```

---

### Task 4: Add MultiSessionResult model

**Files:**
- Modify: `soulgraph/experiment/models.py`
- Modify: `tests/test_experiment.py`

**Step 1: Write the failing test**

Add to `tests/test_experiment.py`:

```python
class TestMultiSessionResult:
    def test_multi_session_result(self):
        from soulgraph.experiment.models import MultiSessionResult
        result = MultiSessionResult(
            session_scores=[
                {"rank_correlation": 0.4, "absorption_rate": 0.3},
                {"rank_correlation": 0.5, "absorption_rate": 0.6},
                {"rank_correlation": 0.7, "absorption_rate": 0.9},
            ],
            rank_improvement=0.3,
            final_scores={"rank_correlation": 0.7, "absorption_rate": 0.9, "overall": 0.8},
            num_sessions=3,
            turns_per_session=10,
        )
        assert result.num_sessions == 3
        assert result.rank_improvement == 0.3
        assert len(result.session_scores) == 3
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_experiment.py::TestMultiSessionResult -v`
Expected: FAIL (cannot import MultiSessionResult)

**Step 3: Write minimal implementation**

Add to `soulgraph/experiment/models.py`:

```python
class MultiSessionResult(BaseModel):
    session_scores: list[dict[str, Any]]
    rank_improvement: float
    final_scores: dict[str, Any]
    num_sessions: int
    turns_per_session: int
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_experiment.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add soulgraph/experiment/models.py tests/test_experiment.py
git commit -m "feat(v5): add MultiSessionResult model"
```

---

### Task 5: Add `run_multi_session()` to ExperimentRunner

**Files:**
- Modify: `soulgraph/experiment/runner.py`
- Modify: `tests/test_experiment.py`

**Step 1: Write the failing test**

Add to `tests/test_experiment.py`:

```python
class TestMultiSessionRunner:
    def _make_gt_graph(self) -> SoulGraph:
        g = SoulGraph(owner_id="gt")
        g.add_item(SoulItem(id="si_001", text="重视家庭", domains=["family"], tags=["intention"]))
        g.add_item(SoulItem(id="si_002", text="想买SUV", domains=["purchase"]))
        g.add_edge(SoulEdge(from_id="si_001", to_id="si_002", relation="drives"))
        return g

    @patch("soulgraph.experiment.runner.RankingComparator")
    @patch("soulgraph.experiment.runner.Speaker")
    @patch("soulgraph.experiment.runner.Detector")
    def test_run_multi_session(self, MockDetector, MockSpeaker, MockRankComp):
        mock_speaker = MockSpeaker.return_value
        mock_speaker.respond.return_value = "我最近在想买车"

        mock_detector = MockDetector.return_value
        mock_detector.ask_next_question.return_value = "你在想什么？"
        mock_detector.listen_and_detect.return_value = SoulGraph(owner_id="det")
        mock_detector.detected_graph = SoulGraph(owner_id="det")

        MockRankComp.return_value.compare.return_value = {
            "rank_correlation": 0.5, "domain_ndcg": 0.5,
            "absorption_rate": 0.5, "intention_recall": 0.5,
            "overall": 0.5, "matched_items": 1, "gt_items": 2, "det_items": 0,
        }

        runner = ExperimentRunner(api_key="fake")
        session_configs = [
            {"turns": 3, "topic_hints": ["family"]},
            {"turns": 3, "topic_hints": ["career"]},
        ]
        result = runner.run_multi_session(
            self._make_gt_graph(), session_configs=session_configs, verbose=False
        )
        assert result["num_sessions"] == 2
        assert len(result["session_scores"]) == 2
        assert "rank_improvement" in result
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_experiment.py::TestMultiSessionRunner -v`
Expected: FAIL (run_multi_session not defined)

**Step 3: Write minimal implementation**

Add to `soulgraph/experiment/runner.py`:

```python
    def run_multi_session(
        self,
        ground_truth: SoulGraph,
        session_configs: list[dict],
        hub_top_k: int = 5,
        verbose: bool = True,
    ) -> dict:
        """Run multiple sessions with persistent detector graph."""
        detector = Detector(api_key=self._api_key, model=self._detector_model, session_number=0)
        session_scores: list[dict] = []
        ranking_comp = RankingComparator()

        for si, config in enumerate(session_configs):
            turns = config.get("turns", 10)
            topic_hints = config.get("topic_hints", [])
            detector.session_number = si + 1

            speaker = Speaker(
                soul_graph=ground_truth,
                api_key=self._api_key,
                model=self._speaker_model,
                topic_hints=topic_hints,
            )
            conversation: list[Message] = []

            if verbose:
                print(f"\n{'='*60}")
                print(f"SESSION {si + 1}/{len(session_configs)} — topics: {topic_hints}")
                print(f"{'='*60}")
                print(f"Graph state: {len(detector.detected_graph.items)} items, {len(detector.detected_graph.edges)} edges")

            question = detector.ask_next_question(conversation)

            for turn in range(turns):
                if verbose:
                    print(f"\n--- Session {si + 1}, Turn {turn + 1}/{turns} ---")
                    print(f"Detector asks: {question[:80]}...")

                response = speaker.respond(question, conversation)
                conversation.append(Message(role="speaker", content=response))

                if verbose:
                    print(f"Speaker says: {response[:80]}...")

                detector.listen_and_detect(conversation)

                if verbose:
                    print(f"Detected so far: {len(detector.detected_graph.items)} items, {len(detector.detected_graph.edges)} edges")

                question = detector.ask_next_question(conversation)
                conversation.append(Message(role="detector", content=question))

            # Evaluate after this session
            scores = ranking_comp.compare(ground_truth, detector.detected_graph)
            session_scores.append(scores)

            if verbose:
                print(f"\n--- Session {si + 1} Results ---")
                print(f"Rank Correlation: {scores['rank_correlation']:.3f}")
                print(f"Domain NDCG:      {scores['domain_ndcg']:.3f}")
                print(f"Absorption Rate:  {scores['absorption_rate']:.3f}")
                print(f"Intention Recall: {scores['intention_recall']:.3f}")
                print(f"Overall (V4):     {scores['overall']:.3f}")

        # Cross-session metrics
        rank_improvement = (
            session_scores[-1]["rank_correlation"] - session_scores[0]["rank_correlation"]
            if len(session_scores) >= 2 else 0.0
        )

        result = {
            "session_scores": session_scores,
            "rank_improvement": round(rank_improvement, 3),
            "final_scores": session_scores[-1] if session_scores else {},
            "num_sessions": len(session_configs),
            "turns_per_session": [c.get("turns", 10) for c in session_configs],
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"MULTI-SESSION SUMMARY")
            print(f"{'='*60}")
            for si, scores in enumerate(session_scores):
                print(f"  Session {si + 1}: rank_corr={scores['rank_correlation']:.3f}  absorption={scores['absorption_rate']:.3f}  overall={scores['overall']:.3f}")
            print(f"  Rank Improvement: {rank_improvement:+.3f}")

        return result
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_experiment.py::TestMultiSessionRunner -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add soulgraph/experiment/runner.py tests/test_experiment.py
git commit -m "feat(v5): add run_multi_session() to ExperimentRunner"
```

---

### Task 6: Add session metadata to Zhang Wei fixture

**Files:**
- Modify: `fixtures/zhang_wei.json`
- Modify: `tests/test_graph.py`

**Step 1: Write the failing test**

Add to `tests/test_graph.py`:

```python
    def test_zhang_wei_has_sessions(self):
        import json
        path = Path(__file__).parent.parent / "fixtures" / "zhang_wei.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "sessions" in data
        sessions = data["sessions"]
        assert len(sessions) == 3
        assert all("topic_hints" in s for s in sessions)
        assert all("session" in s for s in sessions)
        # Session 3 should revisit career
        assert "career" in sessions[2]["topic_hints"]
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_graph.py::TestFixtures::test_zhang_wei_has_sessions -v`
Expected: FAIL ("sessions" not in data)

**Step 3: Add sessions metadata to fixture**

Load `fixtures/zhang_wei.json`, add at the top level:

```json
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
```

**Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_graph.py::TestFixtures -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add fixtures/zhang_wei.json tests/test_graph.py
git commit -m "feat(v5): add 3-session metadata to zhang_wei fixture"
```

---

### Task 7: Add `--sessions` flag to CLI

**Files:**
- Modify: `soulgraph/cli.py`

**Step 1: Add sessions support to CLI**

Add argument to parser:
```python
    parser.add_argument("--sessions", type=int, default=0, help="Number of sessions for multi-session experiment (reads session configs from fixture)")
```

Add handling in the experiment block (after `elif args.experiment:` and before the existing `if args.runs > 1:`):

```python
        if args.sessions > 0:
            import json as json_mod
            fixture_data = json_mod.loads(gt_path.read_text(encoding="utf-8"))
            session_configs = fixture_data.get("sessions", [])[:args.sessions]
            if not session_configs:
                print("Error: fixture has no 'sessions' field", file=sys.stderr)
                sys.exit(1)
            for sc in session_configs:
                sc["turns"] = args.turns
            print(f"Running multi-session experiment ({args.sessions} sessions, {args.turns} turns each) on {gt_path.name}...")
            summary = runner.run_multi_session(gt, session_configs=session_configs)
            if args.output:
                Path(args.output).write_text(json_mod.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"\nSummary saved to {args.output}")
```

**Step 2: Run full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add soulgraph/cli.py
git commit -m "feat(v5): add --sessions flag to CLI for multi-session experiments"
```

---

### Task 8: Run V5 benchmark

**Step 1: Single multi-session run to verify pipeline**

Run:
```bash
ANTHROPIC_API_KEY=<key> .venv/bin/python -m soulgraph --experiment fixtures/zhang_wei.json --sessions 3 --turns 10
```

Expected: 3 sessions complete, per-session scores printed, rank improvement shown.

**Step 2: Check that rank correlation improves across sessions**

Verify output shows positive rank improvement (session 3 > session 1).

**Step 3: Save results**

Run:
```bash
ANTHROPIC_API_KEY=<key> .venv/bin/python -m soulgraph --experiment fixtures/zhang_wei.json --sessions 3 --turns 10 --output results/v5_zhang_wei_3sessions.json
```

**Step 4: Commit results**

```bash
git add -f results/v5_zhang_wei_3sessions.json
git commit -m "docs: V5 multi-session benchmark results"
```

---

### Task 9: Write V5 findings report

**Files:**
- Create: `docs/findings-v5.md`

Document:
1. Per-session V4 metrics (rank correlation, absorption, intention recall, overall)
2. Cross-session rank improvement
3. Does mention reinforcement work? (top items should have higher mention_count)
4. Absorption growth across sessions
5. Comparison with V4 single-session results
6. What worked, what didn't, V6 priorities

**Commit:**

```bash
git add docs/findings-v5.md
git commit -m "docs: V5 findings report"
```
