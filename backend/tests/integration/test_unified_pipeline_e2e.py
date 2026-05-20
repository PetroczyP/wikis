"""End-to-end integration tests for the unified wiki pipeline (#243, #318).

Tests run the full pipeline (planner → writer → gate) against tiny fixture
repos in backend/tests/fixtures/unified_pipeline/.  All LLM calls use a
deterministic ``FakeLLM`` that returns well-formed JSON for both the planner
and the writer roles, detected via the system-prompt prefix.

Fixture repos
-------------
tiny_code_repo/     — 5-file Python auth module (code cluster)
tiny_doc_repo/      — 4-file Markdown doc repo with internal links
tiny_confluence_export/ — 3 Confluence-style markdown pages
tiny_jira_export/   — 2 epics with 1 story each

Each fixture test asserts:
- ``len(result.pages) > 0``
- Every page has non-empty ``title`` and ``markdown``
- No ``target_*`` metadata field is entirely absent (list may be empty for doc repos)

Adversarial tests
-----------------
- Code repo: writer hallucinates README content without reading a README →
  gate strips the paragraph (readme=strip) → final markdown does NOT contain it.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.agents.unified_wiki_pipeline import (
    UnifiedPipelineResult,
    run_unified_pipeline,
)
from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)

# ── Fixtures path ─────────────────────────────────────────────────────────────

_FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "unified_pipeline"


# ── FakeLLM ───────────────────────────────────────────────────────────────────


class _FakeMsg:
    def __init__(self, content: str) -> None:
        self.content = content


class RoleAwareFakeLLM(BaseChatModel):
    """FakeLLM that inspects the system prompt to decide which role is calling.

    Role detection (in order):
    1. System prompt starts with "You are a technical documentation writer"
       → writer role: return ``writer_response``.
    2. Otherwise → planner role: return ``planner_response``.

    Pydantic fields required for BaseChatModel subclass.
    """

    planner_response: str = ""
    writer_response: str = ""
    record_calls: list[dict] = []
    model_name: str = "role-aware-fake"

    @property
    def _llm_type(self) -> str:
        return "role-aware-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.record_calls.append({"messages": messages})

        # Detect role from system prompt
        system_content = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "system":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
            if msg.__class__.__name__ == "SystemMessage":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        if system_content.startswith("You are a technical documentation writer"):
            text = self.writer_response
        else:
            text = self.planner_response

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError

    def bind_tools(self, tools: Any, **kwargs: Any) -> "RoleAwareFakeLLM":  # type: ignore[override]
        return self


# ── Helpers ───────────────────────────────────────────────────────────────────


def _planner_json(
    title: str = "Module",
    description: str = "Covers the module.",
    retrieval_query: str = "module components",
) -> str:
    return json.dumps({"title": title, "description": description, "retrieval_query": retrieval_query})


def _writer_md(content: str, cite_path: str = "src/auth/service.py") -> str:
    """Return markdown with a citation token."""
    return f"{content} [{cite_path}:1-5]"


def _make_cluster(
    cluster_id: int,
    kind: str,
    dirs: list[str],
    artifact_name: str,
    artifact_path: str,
    artifact_kind: str,
    layer: str = "public_api",
) -> Cluster:
    artifact = ArtifactInfo(
        kind=artifact_kind,  # type: ignore[arg-type]
        name=artifact_name,
        source_path=artifact_path,
        layer=layer,
    )
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs,
        artifacts=[artifact],
        total_artifacts=1,
        primary_languages=["py"],
        depth_range=(1, 2),
    )


def _make_skeleton(clusters: list[Cluster]) -> StructureSkeleton:
    return StructureSkeleton(
        code_clusters=[],
        doc_clusters=[],
        total_arch_symbols=0,
        total_dirs_covered=len(clusters),
        total_dirs_in_repo=len(clusters),
        repo_languages=["py"],
        effective_depth=3,
        clusters=clusters,
    )


def _make_pack(cluster_id: int, kind: str = "code") -> EvidencePack:
    return EvidencePack(cluster_id=cluster_id, kind=kind)


def _run(
    skeleton: StructureSkeleton,
    packs: dict[int, EvidencePack],
    repo_root: str,
    llm: RoleAwareFakeLLM,
    gate_mode: dict[str, str] | None = None,
) -> UnifiedPipelineResult:
    return run_unified_pipeline(
        skeleton,
        packs,
        llm=llm,
        repo_root=repo_root,
        gate_mode=gate_mode,
    )


# ── tiny_code_repo ─────────────────────────────────────────────────────────────


class TestTinyCodeRepo:
    _REPO = str(_FIXTURES_DIR / "tiny_code_repo")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Module", description="Authentication logic."),
            writer_response=_writer_md("AuthService manages login.", "src/auth/service.py"),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=1, kind="code",
            dirs=["src/auth"],
            artifact_name="AuthService",
            artifact_path="src/auth/service.py",
            artifact_kind="symbol",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip(), f"Page {page.page_id} has empty title"

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip(), f"Page {page.page_id} has empty markdown"

    def test_target_symbols_populated_from_skeleton(self):
        result = _run(self._skeleton(), {1: _make_pack(1)}, self._REPO, self._llm())
        page = result.pages[0]
        # target_symbols comes from skeleton artifacts with layer in {entry_point, public_api}
        # our fixture uses layer=public_api so it should be populated
        assert "target_symbols" in page.metadata
        assert "target_folders" in page.metadata
        assert "target_docs" in page.metadata


class TestTinyCodeRepoAdversarial:
    """Writer hallucinates README content; gate strips the paragraph (readme=strip)."""

    _REPO = str(_FIXTURES_DIR / "tiny_code_repo")

    HALLUCINATED_README = "The README explains how to set up the project."

    def _llm(self) -> RoleAwareFakeLLM:
        # Writer emits two paragraphs: one real, one that hallucinates README content.
        writer_md = (
            f"AuthService manages login. [src/auth/service.py:1-5]\n\n"
            f"{self.HALLUCINATED_README}"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Module"),
            writer_response=writer_md,
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=1, kind="code",
            dirs=["src/auth"],
            artifact_name="AuthService",
            artifact_path="src/auth/service.py",
            artifact_kind="symbol",
        )
        return _make_skeleton([cluster])

    def test_readme_hallucination_stripped_by_gate(self):
        result = _run(
            self._skeleton(),
            {1: _make_pack(1)},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        final_md = result.pages[0].markdown
        assert "README" not in final_md, (
            "README hallucination survived into final markdown — gate should have stripped it"
        )

    def test_gate_stripped_count_positive(self):
        result = _run(
            self._skeleton(),
            {1: _make_pack(1)},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        total_stripped = sum(len(r.stripped_paragraphs) for r in result.gate_reports.values())
        assert total_stripped >= 1


# ── tiny_doc_repo ─────────────────────────────────────────────────────────────


class TestTinyDocRepo:
    _REPO = str(_FIXTURES_DIR / "tiny_doc_repo")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="System Docs", description="System documentation overview."),
            writer_response=_writer_md("This system handles requests.", "docs/overview.md"),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=2, kind="doc",
            dirs=["docs"],
            artifact_name="overview",
            artifact_path="docs/overview.md",
            artifact_kind="doc_section",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip()

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip()

    def test_target_docs_in_metadata(self):
        result = _run(self._skeleton(), {2: _make_pack(2, "doc")}, self._REPO, self._llm())
        page = result.pages[0]
        assert "target_docs" in page.metadata


class TestTinyDocRepoAdversarial:
    """Writer claims README content without reading it; gate strips the paragraph."""

    _REPO = str(_FIXTURES_DIR / "tiny_doc_repo")

    def _llm(self) -> RoleAwareFakeLLM:
        writer_md = (
            "The README describes this component in detail.\n\n"
            "System documentation for deployment. [docs/deployment.md:1-5]"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="System Docs"),
            writer_response=writer_md,
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=2, kind="doc",
            dirs=["docs"],
            artifact_name="deployment",
            artifact_path="docs/deployment.md",
            artifact_kind="doc_section",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_readme_paragraph_stripped_by_gate(self):
        result = _run(
            self._skeleton(),
            {2: _make_pack(2, "doc")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        page = result.pages[0]
        assert "README" not in page.markdown


# ── tiny_confluence_export ────────────────────────────────────────────────────


class TestTinyConfluenceExport:
    _REPO = str(_FIXTURES_DIR / "tiny_confluence_export")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(
                title="Engineering Confluence",
                description="Engineering team documentation.",
            ),
            writer_response=_writer_md(
                "Engineering processes documented here.",
                "spaces/ENG/auth_design.md",
            ),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=3, kind="confluence",
            dirs=["spaces/ENG"],
            artifact_name="Authentication Design",
            artifact_path="spaces/ENG/auth_design.md",
            artifact_kind="confluence_page",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {3: _make_pack(3, "confluence")}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {3: _make_pack(3, "confluence")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip()

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {3: _make_pack(3, "confluence")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip()


class TestTinyConfluenceAdversarial:
    """Writer hallucinates a SQL schema without reading a .sql file.  Gate strips it."""

    _REPO = str(_FIXTURES_DIR / "tiny_confluence_export")

    HALLUCINATED_SQL = "CREATE TABLE users (id INTEGER PRIMARY KEY)"

    def _llm(self) -> RoleAwareFakeLLM:
        writer_md = (
            f"{self.HALLUCINATED_SQL};\n\n"
            "Auth design documented. [spaces/ENG/auth_design.md:1-5]"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Engineering Confluence"),
            writer_response=writer_md,
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=3, kind="confluence",
            dirs=["spaces/ENG"],
            artifact_name="auth_design",
            artifact_path="spaces/ENG/auth_design.md",
            artifact_kind="confluence_page",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_sql_paragraph_stripped(self):
        result = _run(
            self._skeleton(),
            {3: _make_pack(3, "confluence")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        page = result.pages[0]
        assert "CREATE TABLE" not in page.markdown


# ── tiny_jira_export ─────────────────────────────────────────────────────────


class TestTinyJiraExport:
    _REPO = str(_FIXTURES_DIR / "tiny_jira_export")

    def _llm(self) -> RoleAwareFakeLLM:
        return RoleAwareFakeLLM(
            planner_response=_planner_json(
                title="Auth Epic",
                description="Authentication system epics and stories.",
            ),
            writer_response=_writer_md(
                "Authentication epic AUTH-1 captures the full auth scope.",
                "epics/AUTH-1.md",
            ),
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=4, kind="jira",
            dirs=["epics"],
            artifact_name="AUTH-1",
            artifact_path="epics/AUTH-1.md",
            artifact_kind="jira_issue",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_produces_at_least_one_page(self):
        result = _run(self._skeleton(), {4: _make_pack(4, "jira")}, self._REPO, self._llm())
        assert len(result.pages) > 0

    def test_every_page_has_non_empty_title(self):
        result = _run(self._skeleton(), {4: _make_pack(4, "jira")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.title.strip()

    def test_every_page_has_non_empty_markdown(self):
        result = _run(self._skeleton(), {4: _make_pack(4, "jira")}, self._REPO, self._llm())
        for page in result.pages:
            assert page.markdown.strip()


class TestTinyJiraAdversarial:
    """Writer hallucinates SQL; gate strips the SQL paragraph (sql=strip)."""

    _REPO = str(_FIXTURES_DIR / "tiny_jira_export")

    HALLUCINATED_SQL = "SELECT * FROM issues WHERE project = 'AUTH'"

    def _llm(self) -> RoleAwareFakeLLM:
        writer_md = (
            f"{self.HALLUCINATED_SQL};\n\n"
            "Epic AUTH-1 covers authentication. [epics/AUTH-1.md:1-5]"
        )
        return RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Epic"),
            writer_response=writer_md,
        )

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=4, kind="jira",
            dirs=["epics"],
            artifact_name="AUTH-1",
            artifact_path="epics/AUTH-1.md",
            artifact_kind="jira_issue",
            layer="",
        )
        return _make_skeleton([cluster])

    def test_sql_paragraph_stripped_by_gate(self):
        result = _run(
            self._skeleton(),
            {4: _make_pack(4, "jira")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        assert len(result.pages) > 0
        final_md = result.pages[0].markdown
        assert "SELECT" not in final_md, (
            "SQL hallucination survived into final markdown — gate should have stripped it"
        )

    def test_at_least_one_paragraph_kept(self):
        """The legitimate paragraph (AUTH-1) should remain after SQL stripping."""
        result = _run(
            self._skeleton(),
            {4: _make_pack(4, "jira")},
            self._REPO,
            self._llm(),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )
        page = result.pages[0]
        assert "AUTH-1" in page.markdown, (
            "The legitimate AUTH-1 paragraph should survive after SQL stripping"
        )


# ── Chaptered pipeline (#290) ─────────────────────────────────────────────────


def _chaptered_planner_json(
    chapter_title: str,
    chapter_description: str,
    subpage_titles: list[str],
) -> str:
    """Return a JSON string matching the chaptered planner's expected format."""
    subpages = [
        {
            "title": t,
            "description": f"Description for {t}.",
            "page_order": i + 1,
            "retrieval_query": t.lower().replace(" ", " "),
            "target_symbols": [],
            "target_folders": [],
            "target_docs": [],
        }
        for i, t in enumerate(subpage_titles)
    ]
    return json.dumps({
        "chapter_title": chapter_title,
        "chapter_description": chapter_description,
        "subpages": subpages,
    })


class ChapteredFakeLLM(BaseChatModel):
    """FakeLLM for the chaptered pipeline.

    The chaptered planner receives a system prompt that does NOT start with
    "You are a technical documentation writer", so we detect roles by checking
    whether the system prompt contains "writer".

    - Planner call → returns ``planner_response`` (chaptered JSON).
    - Writer call → returns ``writer_response`` (sub-page markdown).
    - Chapter index call → returns ``index_response`` (index markdown).
    """

    planner_response: str = ""
    writer_response: str = ""
    index_response: str = ""
    record_calls: list[dict] = []
    model_name: str = "chaptered-fake"

    @property
    def _llm_type(self) -> str:
        return "chaptered-fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.record_calls.append({"messages": messages})

        system_content = ""
        for msg in messages:
            if hasattr(msg, "type") and msg.type == "system":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break
            if msg.__class__.__name__ == "SystemMessage":
                system_content = msg.content if isinstance(msg.content, str) else str(msg.content)
                break

        # Writer role detection (system prompt starts with the writer prefix).
        if system_content.startswith("You are a technical documentation writer"):
            text = self.writer_response
        # Chapter index: no system prompt (direct HumanMessage to LLM).
        elif not system_content:
            text = self.index_response
        else:
            text = self.planner_response

        msg = AIMessage(content=text)
        msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ChapteredFakeLLM":
        return self


class TestChapteredPipeline:
    """Integration: WIKIS_CHAPTERED_PAGES=1 → 1 index page + N sub-pages per chapter."""

    _REPO = str(_FIXTURES_DIR / "tiny_code_repo")

    def _skeleton(self) -> StructureSkeleton:
        cluster = _make_cluster(
            cluster_id=1, kind="code",
            dirs=["src/auth"],
            artifact_name="AuthService",
            artifact_path="src/auth/service.py",
            artifact_kind="symbol",
        )
        return _make_skeleton([cluster])

    def _llm(self, subpage_titles: list[str] | None = None) -> ChapteredFakeLLM:
        titles = subpage_titles or ["Session Management", "Token Validation"]
        return ChapteredFakeLLM(
            planner_response=_chaptered_planner_json(
                chapter_title="Auth Chapter",
                chapter_description="Authentication and authorization.",
                subpage_titles=titles,
            ),
            writer_response="AuthService manages login. [src/auth/service.py:1-5]",
            index_response="Auth Chapter overview text. Brief summary of the chapter.",
        )

    def _run_chaptered(self, skeleton=None, packs=None, llm=None) -> UnifiedPipelineResult:
        sk = skeleton or self._skeleton()
        pk = packs or {1: _make_pack(1)}
        lm = llm or self._llm()
        env_patch = {"WIKIS_CHAPTERED_PAGES": "1"}
        old_env = {k: os.environ.get(k) for k in env_patch}
        try:
            os.environ.update(env_patch)
            return run_unified_pipeline(sk, pk, llm=lm, repo_root=self._REPO)
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_chaptered_pipeline_produces_pages(self):
        """WIKIS_CHAPTERED_PAGES=1 path produces at least one page."""
        result = self._run_chaptered()
        assert len(result.pages) > 0

    def test_chaptered_pipeline_produces_index_plus_subpages(self):
        """Each chapter produces 1 index page + N sub-pages.

        The tiny cluster fixture has 1 artifact, so the chaptered planner caps
        it at 1 sub-page (tiny cluster rule: <=5 total_artifacts → max 1 subpage).
        Total: 1 index + 1 sub-page = 2 pages.
        """
        result = self._run_chaptered()
        # One cluster → one chapter → 1 index + (1..2) sub-pages.
        # Tiny cluster (1 artifact) → 1 sub-page → 2 total.
        assert len(result.pages) >= 2

    def test_chaptered_pipeline_has_index_and_at_least_one_subpage(self):
        """First page is the index, remaining pages are sub-pages."""
        result = self._run_chaptered()
        assert len(result.pages) >= 2
        index_page = result.pages[0]
        assert index_page.metadata.get("is_chapter_index") is True
        subpages = result.pages[1:]
        assert len(subpages) >= 1

    def test_chaptered_index_page_has_chapter_title(self):
        """The chapter index page uses the chapter title."""
        result = self._run_chaptered()
        index_page = result.pages[0]
        assert index_page.title == "Auth Chapter"

    def test_chaptered_subpages_have_non_empty_titles(self):
        """Sub-pages have non-empty titles."""
        result = self._run_chaptered()
        for page in result.pages[1:]:
            assert page.title.strip(), f"Sub-page has empty title"

    def test_chaptered_index_contains_wikilinks(self):
        """Chapter index markdown contains [[wikilink]] TOC entries."""
        result = self._run_chaptered()
        index_md = result.pages[0].markdown
        # The index must contain at least one [[wikilink]]
        assert "[[" in index_md, f"No wikilinks in chapter index: {index_md!r}"

    def test_chaptered_subpages_have_non_empty_markdown(self):
        """All sub-pages have non-empty markdown content."""
        result = self._run_chaptered()
        for page in result.pages[1:]:
            assert page.markdown.strip(), f"Sub-page {page.title!r} has empty markdown"

    def test_flat_pipeline_unaffected_when_flag_off(self):
        """When WIKIS_CHAPTERED_PAGES=0, the flat pipeline runs as before."""
        lm = RoleAwareFakeLLM(
            planner_response=_planner_json(title="Auth Module", description="Auth logic."),
            writer_response=_writer_md("AuthService manages login.", "src/auth/service.py"),
        )
        # Ensure flag is off
        env_patch = {"WIKIS_CHAPTERED_PAGES": "0"}
        old_env = {k: os.environ.get(k) for k in env_patch}
        try:
            os.environ.update(env_patch)
            result = run_unified_pipeline(
                self._skeleton(), {1: _make_pack(1)}, llm=lm, repo_root=self._REPO,
            )
        finally:
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # Flat pipeline: exactly 1 page per cluster
        assert len(result.pages) == 1
        assert result.pages[0].title == "Auth Module"
