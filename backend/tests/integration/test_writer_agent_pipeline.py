"""Integration tests for the real WikiContentWriter inside the unified pipeline (#285).

Regression guard: after a pipeline run, each generated page must have at least
one paragraph that survives the gate + verifier.  This test closes the
"silent empty page" regression described in issue #285.

The FakeLLM pattern mirrors test_unified_pipeline_e2e.py (RoleAwareFakeLLM).
We use a single-cluster fixture so the test is fast and deterministic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.agents.unified_wiki_pipeline import run_unified_pipeline
from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)


# ── RoleAwareFakeLLM (same pattern as e2e integration test) ──────────────────


class RoleAwareFakeLLM(BaseChatModel):
    """Detects writer / planner / verifier role from system prompt prefix."""

    planner_response: str = ""
    writer_response: str = ""
    verifier_response: str = ""
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
        elif "verifier" in system_content.lower():
            text = self.verifier_response
        else:
            text = self.planner_response

        msg = AIMessage(content=text)
        msg.tool_calls = []  # type: ignore[attr-defined]
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        raise NotImplementedError


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_cluster(cluster_id: int = 1, kind: str = "code") -> Cluster:
    artifact = ArtifactInfo(
        kind="symbol",  # type: ignore[arg-type]
        name="AuthService",
        source_path="src/auth/service.py",
        layer="public_api",
    )
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=["src/auth"],
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


def _planner_json(title: str = "Auth Module", description: str = "Handles auth.") -> str:
    return json.dumps({"title": title, "description": description, "retrieval_query": "auth login"})


def _verifier_supported(n: int) -> str:
    return json.dumps([{"verdict": "supported", "reason": "Grounded."} for _ in range(n)])


# ── Regression guard: pages must have cited, surviving paragraphs ─────────────


class TestWriterProducesNonEmptyPages:
    """Each generated page must have at least one paragraph surviving gate + verifier.

    This is the regression guard for #285: the stub produced zero citations, every
    paragraph was stripped, and pages were reduced to 32-line frontmatter skeletons.
    """

    def _setup_cited_file(self, tmp_path: Path) -> None:
        cited = tmp_path / "src" / "auth" / "service.py"
        cited.parent.mkdir(parents=True, exist_ok=True)
        cited.write_text("class AuthService:\n    def login(self, user, pwd): ...\n")

    def test_page_markdown_non_empty_after_gate_and_verifier(self, tmp_path):
        """The final page markdown is non-empty after gate and verifier pass."""
        self._setup_cited_file(tmp_path)

        writer_md = "The AuthService handles user login. [src/auth/service.py:1-2]"
        llm = RoleAwareFakeLLM(
            planner_response=_planner_json(),
            writer_response=writer_md,
            verifier_response=_verifier_supported(1),
        )

        result = run_unified_pipeline(
            _make_skeleton([_make_cluster()]),
            {1: _make_pack(1)},
            llm=llm,
            verifier_llm=llm,
            repo_root=str(tmp_path),
            gate_mode={"readme": "strip", "sql": "strip", "identifier": "flag"},
        )

        assert len(result.pages) == 1
        page = result.pages[0]
        assert page.markdown.strip(), (
            f"Page {page.page_id!r} has empty markdown — "
            "citation pipeline stripped everything (regression from #285)"
        )

    def test_page_has_at_least_one_kept_cited_claim(self, tmp_path):
        """cited_claims on the page must be non-empty after verifier."""
        self._setup_cited_file(tmp_path)

        writer_md = "AuthService validates JWT tokens. [src/auth/service.py:1-2]"
        llm = RoleAwareFakeLLM(
            planner_response=_planner_json(),
            writer_response=writer_md,
            verifier_response=_verifier_supported(1),
        )

        result = run_unified_pipeline(
            _make_skeleton([_make_cluster()]),
            {1: _make_pack(1)},
            llm=llm,
            verifier_llm=llm,
            repo_root=str(tmp_path),
        )

        page = result.pages[0]
        assert len(page.cited_claims) > 0, (
            "cited_claims is empty — all paragraphs stripped; regression from #285"
        )

    def test_zero_citation_writer_emits_warning_event(self, tmp_path):
        """When the writer produces no citations, a writer.zero_citations event is emitted."""
        self._setup_cited_file(tmp_path)

        # Writer returns content with no citation tokens
        llm = RoleAwareFakeLLM(
            planner_response=_planner_json(),
            writer_response="This paragraph has no citation at all.",
            verifier_response=_verifier_supported(1),
        )

        events: list[dict] = []
        run_unified_pipeline(
            _make_skeleton([_make_cluster()]),
            {1: _make_pack(1)},
            llm=llm,
            verifier_llm=llm,
            repo_root=str(tmp_path),
            stream_callback=events.append,
        )

        zero_cite_events = [e for e in events if e.get("event") == "writer.zero_citations"]
        assert len(zero_cite_events) == 1
        assert zero_cite_events[0]["page_id"] == 1

    def test_multiple_clusters_all_pages_non_empty(self, tmp_path):
        """Every page survives when multiple clusters are processed."""
        # Set up cited files for both clusters
        for rel in ["src/auth/service.py", "src/admin/controller.py"]:
            p = tmp_path / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("class Placeholder: pass\n")

        cluster1 = _make_cluster(cluster_id=1, kind="code")
        cluster2 = Cluster(
            cluster_id=2,
            kind="code",  # type: ignore[arg-type]
            dirs=["src/admin"],
            artifacts=[
                ArtifactInfo(
                    kind="symbol",  # type: ignore[arg-type]
                    name="AdminController",
                    source_path="src/admin/controller.py",
                    layer="public_api",
                )
            ],
            total_artifacts=1,
            primary_languages=["py"],
            depth_range=(1, 2),
        )

        # FakeLLM cycles: planner response repeated, writer response repeated
        call_idx = [0]
        planner_resp = _planner_json()
        writer_resp = "Component handles requests. [src/auth/service.py:1]"
        verifier_resp = _verifier_supported(1)

        class CyclingFakeLLM(BaseChatModel):
            model_name: str = "cycling-fake"

            @property
            def _llm_type(self) -> str:
                return "cycling-fake"

            def _generate(
                self,
                messages: list[BaseMessage],
                stop: list[str] | None = None,
                run_manager: Any = None,
                **kwargs: Any,
            ) -> ChatResult:
                system_content = ""
                for msg in messages:
                    if msg.__class__.__name__ == "SystemMessage":
                        system_content = msg.content if isinstance(msg.content, str) else ""
                        break
                if system_content.startswith("You are a technical documentation writer"):
                    text = writer_resp
                elif "verifier" in system_content.lower():
                    text = verifier_resp
                else:
                    text = planner_resp
                msg = AIMessage(content=text)
                msg.tool_calls = []  # type: ignore[attr-defined]
                return ChatResult(generations=[ChatGeneration(message=msg)])

            def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
                raise NotImplementedError

        llm = CyclingFakeLLM()
        result = run_unified_pipeline(
            _make_skeleton([cluster1, cluster2]),
            {1: _make_pack(1), 2: _make_pack(2)},
            llm=llm,
            verifier_llm=llm,
            repo_root=str(tmp_path),
            gate_mode={"readme": "flag", "sql": "flag", "identifier": "flag"},
        )

        assert len(result.pages) == 2
        for page in result.pages:
            assert page.markdown.strip(), (
                f"Page {page.page_id!r} is empty after pipeline — regression from #285"
            )
