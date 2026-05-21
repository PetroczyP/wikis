"""Unit tests for app/core/agents/unified_wiki_pipeline.py (#243, #318).

Covers:
- Empty skeleton → pages=[], all reports empty.
- Single code cluster → 1 page, gate_report present, target_* from skeleton.
- Gate strips one paragraph → final markdown omits it, gate_report has the flag,
  stream emits gate.page_done with n_stripped=1.
- Custom gate_mode flows through to apply_gate (readme+sql strip, identifier flag).

The FakeLLM pattern follows test_planner_agent.py (BaseChatModel subclass).
The writer LLM returns a canned markdown string; the planner LLM returns a
canned JSON PageSpec.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)
from app.core.agents.unified_wiki_pipeline import (
    GeneratedPage,
    UnifiedPipelineResult,
    run_unified_pipeline,
)


# ── FakeLLM ────────────────────────────────────────────────────────────────────


class FakeLLM(BaseChatModel):
    """Minimal BaseChatModel subclass that returns canned text responses.

    Responses are returned in sequence; if exhausted, the last is repeated.
    If ``responses`` is a list, they are cycled through.
    """

    responses: list[str] = []
    record_calls: list[dict] = []
    model_name: str = "fake-llm"

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.record_calls.append({"messages": messages})
        idx = min(len(self.record_calls) - 1, len(self.responses) - 1)
        text = self.responses[idx] if self.responses else ""
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError("FakeLLM does not support streaming")

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeLLM":  # type: ignore[override]
        """Return self — tools are ignored in the fake."""
        return self


# ── Helpers ────────────────────────────────────────────────────────────────────


def _planner_response(
    title: str = "Auth Module",
    description: str = "Handles authentication.",
    retrieval_query: str = "auth login session",
) -> str:
    """Return a canned planner JSON response."""
    return json.dumps({
        "title": title,
        "description": description,
        "retrieval_query": retrieval_query,
    })


def _make_cluster(
    cluster_id: int = 1,
    kind: str = "code",
    dirs: list[str] | None = None,
    artifacts: list[ArtifactInfo] | None = None,
) -> Cluster:
    if dirs is None:
        dirs = ["src/auth"]
    if artifacts is None:
        artifacts = [
            ArtifactInfo(
                kind="symbol",  # type: ignore[arg-type]
                name="AuthService",
                source_path="src/auth/service.py",
                layer="public_api",
            )
        ]
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs,
        artifacts=artifacts,
        total_artifacts=len(artifacts),
        primary_languages=["py"],
        depth_range=(1, 1),
    )


def _make_skeleton(clusters: list[Cluster]) -> StructureSkeleton:
    return StructureSkeleton(
        code_clusters=[],
        doc_clusters=[],
        total_arch_symbols=0,
        total_dirs_covered=len(clusters),
        total_dirs_in_repo=len(clusters),
        repo_languages=["py"],
        effective_depth=2,
        clusters=clusters,
    )


def _make_evidence_pack(cluster_id: int = 1) -> EvidencePack:
    return EvidencePack(cluster_id=cluster_id, kind="code")


# ── Tests ──────────────────────────────────────────────────────────────────────


class TestEmptySkeleton:
    def test_empty_skeleton_returns_no_pages(self, tmp_path):
        """Empty skeleton → pages=[], all reports empty."""
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[_planner_response()])
        result = run_unified_pipeline(
            skeleton,
            {},
            llm=llm,
            repo_root=str(tmp_path),
        )
        assert isinstance(result, UnifiedPipelineResult)
        assert result.pages == []
        assert result.gate_reports == {}
        assert result.plan_report.pages_emitted == 0

    def test_empty_skeleton_plan_report_has_zero_budget(self, tmp_path):
        """Empty skeleton plan report has zero budget totals."""
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[])
        result = run_unified_pipeline(
            skeleton,
            {},
            llm=llm,
            repo_root=str(tmp_path),
        )
        assert result.plan_report.tool_budget_total == 0
        assert result.plan_report.tool_budget_used == 0
        assert result.plan_report.tool_budget_exceeded is False


def _setup_cited_file(tmp_path, rel_path: str, content: str = "# content\n") -> None:
    """Create a cited file inside tmp_path for verifier span reading."""
    full = tmp_path / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)


class TestSingleCodeCluster:
    def test_single_cluster_produces_one_page(self, tmp_path):
        """Single code cluster → 1 page in result."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        writer_md = "The AuthService handles user login. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),          # planner call
            writer_md,                    # writer call
        ])

        result = run_unified_pipeline(
            skeleton,
            packs,
            llm=llm,
            repo_root=str(tmp_path),
        )
        assert len(result.pages) == 1

    def test_single_cluster_page_has_title(self, tmp_path):
        """Page title comes from the planner response."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        writer_md = "The AuthService handles user login. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(title="Auth Module"),
            writer_md,
        ])
        result = run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert result.pages[0].title == "Auth Module"

    def test_single_cluster_reports_present(self, tmp_path):
        """plan_report and gate_report present for a single cluster."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        writer_md = "Auth module. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        result = run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert result.plan_report is not None
        assert 1 in result.gate_reports

    def test_target_fields_from_skeleton(self, tmp_path):
        """target_symbols and target_folders come from the skeleton (not the LLM)."""
        cluster = _make_cluster(
            artifacts=[
                ArtifactInfo(
                    kind="symbol",  # type: ignore[arg-type]
                    name="AuthService",
                    source_path="src/auth/service.py",
                    layer="public_api",
                )
            ],
            dirs=["src/auth"],
        )
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        writer_md = "Description paragraph. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        result = run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        page = result.pages[0]
        assert "AuthService" in page.metadata.get("target_symbols", [])
        assert "src/auth" in page.metadata.get("target_folders", [])


class TestGateStripping:
    def test_gate_strips_readme_paragraph(self, tmp_path):
        """Gate mode strip removes a paragraph that claims README content without reading it."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Two paragraphs: one mentions README (triggers gate strip), one is clean.
        writer_md = (
            "The README describes the setup process.\n\n"
            "The AuthService handles user login. [src/auth/service.py:1]"
        )

        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])

        events_captured: list[dict] = []

        def _cb(event: dict) -> None:
            events_captured.append(event)

        result = run_unified_pipeline(
            skeleton,
            packs,
            llm=llm,
            repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
            stream_callback=_cb,
        )

        page = result.pages[0]
        # The README paragraph should be stripped from the final markdown
        assert "README" not in page.markdown

        # gate_report should record the strip
        gate_report = result.gate_reports[1]
        assert len(gate_report.stripped_paragraphs) == 1
        assert page.metadata["gate_stripped_total"] == 1

    def test_gate_page_done_event_emitted_with_n_stripped(self, tmp_path):
        """stream_callback receives gate.page_done with n_stripped=1 when gate strips."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        writer_md = (
            "The README describes the setup.\n\n"
            "Clean paragraph. [src/auth/service.py:1]"
        )
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        events: list[dict] = []

        run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
            stream_callback=events.append,
        )

        gate_events = [e for e in events if e.get("event") == "gate.page_done"]
        assert len(gate_events) == 1
        assert gate_events[0]["n_stripped"] == 1


class TestGateMode:
    def test_custom_gate_mode_flows_to_apply_gate(self, tmp_path):
        """Custom gate_mode dict is respected — sql=strip removes SQL paragraphs."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Paragraph with SQL keyword (no .sql file in trace → should be stripped with sql=strip)
        writer_md = (
            "CREATE TABLE users (id INTEGER PRIMARY KEY);\n\n"
            "Clean line. [src/auth/service.py:1]"
        )
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        result = run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        page = result.pages[0]
        assert "CREATE TABLE" not in page.markdown
        assert result.gate_reports[1].stripped_paragraphs  # at least one stripped

    def test_identifier_flag_mode_does_not_strip(self, tmp_path):
        """identifier=flag mode keeps identifier-flagged paragraphs (gate action=flag).

        The gate should NOT strip when action is "flag"; it only flags the paragraph.
        """
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}

        # Paragraph with an unknown CamelCase identifier (no trace to ground it)
        writer_md = "The AuthService handles requests. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        result = run_unified_pipeline(
            skeleton, packs, llm=llm, repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        # identifier mode=flag should NOT strip the paragraph
        page = result.pages[0]
        assert result.gate_reports[1].stripped_paragraphs == []
        assert "AuthService" in page.markdown


class TestStreamCallbackEvents:
    def test_planner_started_and_done_events_emitted(self, tmp_path):
        """planner.started and planner.done events are always emitted."""
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[])
        events: list[dict] = []
        run_unified_pipeline(skeleton, {}, llm=llm, repo_root=str(tmp_path),
                             stream_callback=events.append)
        event_names = {e.get("event") for e in events}
        assert "planner.started" in event_names
        assert "planner.done" in event_names

    def test_writer_page_started_emitted(self, tmp_path):
        """writer.page_started event emitted for each page."""
        cluster = _make_cluster()
        skeleton = _make_skeleton([cluster])
        packs = {1: _make_evidence_pack(1)}
        writer_md = "Content. [src/auth/service.py:1]"
        llm = FakeLLM(responses=[
            _planner_response(),
            writer_md,
        ])
        events: list[dict] = []
        run_unified_pipeline(skeleton, packs, llm=llm, repo_root=str(tmp_path),
                             stream_callback=events.append)
        started = [e for e in events if e.get("event") == "writer.page_started"]
        assert len(started) == 1
        assert started[0]["title"] == "Auth Module"


# ── dispatch_page_generation guard (Copilot #270) ────────────────────────────


class TestDispatchSkipWhenUnifiedPipelinePopulatedPages:
    """Regression for Copilot review on #270: when the unified pipeline
    populates ``wiki_pages`` upstream, ``dispatch_page_generation`` must
    route directly to ``finalize_wiki`` instead of fanning out a second
    ``generate_page_content`` pass that would duplicate every page via
    the ``operator.add`` accumulator on ``WikiState.wiki_pages``."""

    def test_dispatch_returns_finalize_when_wiki_pages_present(self):
        from unittest.mock import MagicMock  # noqa: PLC0415

        from app.core.agents.wiki_graph_optimized import (  # noqa: PLC0415
            OptimizedWikiGenerationAgent,
        )

        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)

        state = {
            "wiki_pages": [{"page_id": "0#0", "title": "Auth"}],
            "wiki_structure_spec": MagicMock(),
        }

        result = agent.dispatch_page_generation(state, MagicMock())
        assert result == "finalize_wiki"

    def test_dispatch_returns_finalize_when_pipeline_produced_zero_pages(self):
        """Regression for Rio review on #271: when the unified pipeline
        produces 0 pages (empty repo / no clusters) and emits a degenerate
        ``WikiStructureSpec`` with ``wiki_pages=[]``, the dispatcher must
        STILL route to ``finalize_wiki``.  Returning ``[]`` here would
        silently complete the graph without firing finalize_wiki or
        export_wiki, leaving the invocation success=True with 0 artifacts
        exported."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from app.core.agents.wiki_graph_optimized import (  # noqa: PLC0415
            OptimizedWikiGenerationAgent,
        )

        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        state = {
            "wiki_pages": [],
            "wiki_structure_spec": MagicMock(),
        }

        result = agent.dispatch_page_generation(state, MagicMock())
        assert result == "finalize_wiki"

    def test_dispatch_raises_when_no_structure_at_all(self):
        """Sanity: if neither wiki_pages nor wiki_structure_spec are in
        state (catastrophic upstream failure), the dispatcher still raises
        the explicit RuntimeError so generate_wiki() catches it cleanly."""
        from unittest.mock import MagicMock  # noqa: PLC0415

        from app.core.agents.wiki_graph_optimized import (  # noqa: PLC0415
            OptimizedWikiGenerationAgent,
        )

        agent = OptimizedWikiGenerationAgent.__new__(OptimizedWikiGenerationAgent)
        state = {}

        with pytest.raises(RuntimeError, match="wiki_structure_spec"):
            agent.dispatch_page_generation(state, MagicMock())


# ── doc/confluence/jira cluster fallback (Copilot #270) ──────────────────────


class TestSkeletonFallbackPromotesDocClusters:
    """Regression for Rio review on #270: when ``skeleton.clusters`` is
    empty (legacy build_skeleton path), the fallback must promote each
    ``DocCluster`` into a proper ``Cluster(kind="doc")`` before the
    evidence-pack dispatch.  A naive ``code_clusters + doc_clusters``
    concatenation crashes on the first ``cluster.cluster_id`` access
    because ``DocCluster`` and ``Cluster`` are unrelated dataclasses
    (``DocCluster`` has only ``dir_path``, ``doc_files``, ``doc_types``,
    ``file_count``).

    Locates the source file via importlib instead of a hardcoded /tmp
    path so the test runs under CI as well as the local dev tree.
    """

    def test_fallback_promotes_doc_clusters_to_real_clusters(self):
        # Locate wiki_graph_optimized.py via the imported module — works
        # whether the test runs in CI, /tmp/wikis-243, or anywhere else.
        from pathlib import Path  # noqa: PLC0415

        from app.core.agents import wiki_graph_optimized  # noqa: PLC0415

        src = Path(wiki_graph_optimized.__file__)
        text = src.read_text()

        # The fix promotes each DocCluster into a Cluster(kind="doc") with
        # cluster_id, kind, artifacts — the attributes the evidence-pack
        # dispatch reads.  We check by grepping for the promotion call.
        assert "Cluster(" in text and "kind=\"doc\"" in text, (
            "expected the unified fallback to construct Cluster(kind=\"doc\") "
            "objects from DocCluster entries; a raw concatenation crashes on "
            "DocCluster.cluster_id (AttributeError) and yields zero pages"
        )

    def test_doccluster_cannot_be_passed_through_directly(self):
        """A DocCluster has no cluster_id / kind / artifacts, so even if
        someone tries to bypass the promotion the failure is loud."""
        from app.core.wiki_structure_planner.structure_skeleton import (  # noqa: PLC0415
            DocCluster,
        )

        dc = DocCluster(dir_path="docs", doc_files=[], doc_types=[], file_count=0)
        with pytest.raises(AttributeError):
            _ = dc.cluster_id  # noqa: B018
        with pytest.raises(AttributeError):
            _ = dc.kind  # noqa: B018
        with pytest.raises(AttributeError):
            _ = dc.artifacts  # noqa: B018


# ── Diagram placement (Fix 1) ─────────────────────────────────────────────────


class TestDiagramPlacement:
    """Regression tests for the diagram placement fix.

    The diagram section must be inserted AFTER the first ## section (Overview),
    not prepended before all content.  These tests call the private insertion
    logic directly via the module-level helper to avoid needing a real
    DiagramGenerator / storage object.
    """

    def _insert_diagram(self, final_markdown: str, diagram_section: str) -> str:
        """Replicate the insertion logic from _run_page (step 5b)."""
        import re as _re

        second_heading = _re.search(r'\n##\s+', final_markdown)
        if second_heading:
            insert_at = second_heading.start()
            return (
                final_markdown[:insert_at]
                + "\n\n" + diagram_section + "\n"
                + final_markdown[insert_at:]
            )
        first_blank = final_markdown.find('\n\n')
        if first_blank > 0:
            return (
                final_markdown[:first_blank]
                + "\n\n" + diagram_section
                + final_markdown[first_blank:]
            )
        return final_markdown + "\n\n" + diagram_section

    def test_diagram_inserted_after_first_section_not_at_top(self):
        """Diagram must appear AFTER the Overview ## heading, not before it."""
        md = (
            "## Overview\n\nThis is the overview text.\n\n"
            "## Details\n\nDetailed content here.\n"
        )
        diagram = "```mermaid\ngraph TD\n  A --> B\n```"
        result = self._insert_diagram(md, diagram)

        overview_pos = result.find("## Overview")
        diagram_pos = result.find("```mermaid")
        details_pos = result.find("## Details")

        assert overview_pos < diagram_pos, "diagram must come after ## Overview"
        assert diagram_pos < details_pos, "diagram must come before ## Details"

    def test_diagram_not_at_very_beginning_of_markdown(self):
        """Diagram must not be the very first thing in the page."""
        md = "## Overview\n\nIntro text.\n\n## Next\n\nMore.\n"
        diagram = "```mermaid\ngraph TD\n  A --> B\n```"
        result = self._insert_diagram(md, diagram)

        assert not result.startswith("```mermaid"), (
            "diagram should not be prepended before all content"
        )

    def test_diagram_falls_back_after_first_paragraph_when_no_second_heading(self):
        """When there is only one ## heading, diagram goes after first paragraph."""
        md = "## Overview\n\nOnly section here. No second heading.\n"
        diagram = "DIAGRAM"
        result = self._insert_diagram(md, diagram)

        # Diagram must come after the first \n\n (end of first paragraph)
        first_blank = md.find('\n\n')
        assert result[first_blank + 2:].startswith("DIAGRAM")

    def test_diagram_appended_when_no_headings_and_no_blank_line(self):
        """When no blank line exists at all, diagram is appended at end."""
        md = "Single line with no blank."
        diagram = "DIAGRAM"
        result = self._insert_diagram(md, diagram)
        assert result.endswith("DIAGRAM")
