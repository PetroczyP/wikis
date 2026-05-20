"""Unit tests for ChapterSpec / SubPageSpec models + run_planner_chaptered (#288).

Covers:
- ChapterSpec / SubPageSpec dataclasses have the right fields
- run_planner_chaptered returns list[ChapterSpec] with >=1 sub-page per chapter
- Back-compat: flat (old-format) LLM response wraps into 1-sub-page ChapterSpec
- Tiny cluster (<=5 symbols) -> 1 sub-page max
- Sub-page cap: planner tries to emit 8 sub-pages -> capped to 6
- Feature flag WIKIS_CHAPTERED_PAGES defaults to off
- run_planner (old) still returns list[PageSpec] (no regression)
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.planner_agent import (
    ChapterSpec,
    PageSpec,
    SubPageSpec,
    run_planner,
    run_planner_chaptered,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)

# ---- FakeLLM -----------------------------------------------------------------


class FakeLLM(BaseChatModel):
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
        raise NotImplementedError

    def bind_tools(self, tools: Any, **kwargs: Any) -> "FakeLLM":
        return self


# ---- Helpers -----------------------------------------------------------------


def _make_artifact(
    kind: str = "symbol",
    name: str = "MyClass",
    source_path: str = "src/module.py",
    layer: str = "public_api",
    connections: int = 5,
) -> ArtifactInfo:
    return ArtifactInfo(
        kind=kind,  # type: ignore[arg-type]
        name=name,
        source_path=source_path,
        layer=layer,
        connections=connections,
    )


def _make_cluster(
    cluster_id: int = 1,
    kind: str = "code",
    dirs: list[str] | None = None,
    artifacts: list[ArtifactInfo] | None = None,
    n_symbols: int = 10,
) -> Cluster:
    arts = artifacts or [_make_artifact(name=f"Sym{i}") for i in range(n_symbols)]
    return Cluster(
        cluster_id=cluster_id,
        kind=kind,  # type: ignore[arg-type]
        dirs=dirs or ["src/module"],
        artifacts=arts,
        total_artifacts=len(arts),
        primary_languages=["py"],
        depth_range=(1, 2),
    )


def _make_skeleton(clusters: list[Cluster]) -> StructureSkeleton:
    return StructureSkeleton(
        code_clusters=[],
        doc_clusters=[],
        total_arch_symbols=0,
        total_dirs_covered=0,
        total_dirs_in_repo=0,
        repo_languages=["py"],
        effective_depth=3,
        repo_name="test-repo",
        clusters=clusters,
    )


def _make_pack(cluster_id: int = 1, kind: str = "code") -> EvidencePack:
    return EvidencePack(
        cluster_id=cluster_id,
        kind=kind,
        signatures=[("MyClass", "src/module.py", "public_api", "Main class.")],
        file_heads=[("src/module.py", "class MyClass:\n    pass")],
        readme_excerpt="",
        sql_blocks=[],
    )


def _chapter_json(
    cluster_id: int,
    n_subpages: int = 3,
    chapter_title: str = "Authentication Service",
    chapter_description: str = "Handles authentication.",
) -> str:
    subpages = [
        {
            "title": f"SubPage {i}",
            "description": f"Description {i}.",
            "page_order": i,
            "retrieval_query": f"query {i}",
            "target_symbols": [f"Sym{i}"],
            "target_folders": ["src/module"],
            "target_docs": [],
            "evidence_filter_symbols": [],
            "evidence_filter_paths": [],
        }
        for i in range(1, n_subpages + 1)
    ]
    return json.dumps({
        "cluster_id": cluster_id,
        "chapter_title": chapter_title,
        "chapter_description": chapter_description,
        "subpages": subpages,
    })


def _flat_json(
    cluster_id: int,
    title: str = "Auth Module",
    description: str = "Does auth.",
    retrieval_query: str = "auth module",
) -> str:
    return json.dumps({
        "cluster_id": cluster_id,
        "title": title,
        "description": description,
        "retrieval_query": retrieval_query,
    })


# ---- Tests: SubPageSpec / ChapterSpec dataclasses ----------------------------


class TestDataclassFields:
    def test_subpage_spec_has_required_fields(self):
        sp = SubPageSpec(
            title="Intro",
            description="Overview section.",
            page_order=1,
            retrieval_query="auth overview",
            target_symbols=["Foo"],
            target_folders=["src/auth"],
            target_docs=[],
            evidence_filter_symbols=["Foo"],
            evidence_filter_paths=["src/auth/core.py"],
        )
        assert sp.title == "Intro"
        assert sp.page_order == 1
        assert sp.evidence_filter_symbols == ["Foo"]
        assert sp.evidence_filter_paths == ["src/auth/core.py"]

    def test_chapter_spec_has_required_fields(self):
        sp = SubPageSpec(
            title="Intro",
            description="Overview.",
            page_order=1,
            retrieval_query="auth",
            target_symbols=[],
            target_folders=[],
            target_docs=[],
            evidence_filter_symbols=[],
            evidence_filter_paths=[],
        )
        ch = ChapterSpec(
            cluster_id=5,
            chapter_title="Auth Layer",
            chapter_description="Handles all auth concerns.",
            subpages=[sp],
        )
        assert ch.cluster_id == 5
        assert ch.chapter_title == "Auth Layer"
        assert len(ch.subpages) == 1

    def test_chapter_spec_subpages_insertion_order_preserved(self):
        sps = [
            SubPageSpec("B", "desc", 2, "b", [], [], [], [], []),
            SubPageSpec("A", "desc", 1, "a", [], [], [], [], []),
        ]
        ch = ChapterSpec(cluster_id=1, chapter_title="T", chapter_description="D", subpages=sps)
        assert ch.subpages[0].page_order == 2


# ---- Tests: run_planner_chaptered happy path ---------------------------------


class TestRunPlannerChapteredHappyPath:
    def test_returns_list_of_chapter_spec(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=3)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert isinstance(chapters, list)
        assert len(chapters) == 1
        assert isinstance(chapters[0], ChapterSpec)

    def test_chapter_has_subpages(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=3)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters[0].subpages) >= 1

    def test_subpages_are_subpage_spec_instances(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=3)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        for sp in chapters[0].subpages:
            assert isinstance(sp, SubPageSpec)

    def test_cluster_id_preserved(self, tmp_path):
        cluster = _make_cluster(cluster_id=42)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=42)
        llm = FakeLLM(responses=[_chapter_json(42, n_subpages=3)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {42: pack}, llm=llm, repo_root=str(tmp_path))
        assert chapters[0].cluster_id == 42

    def test_chapter_title_from_llm(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(
            responses=[_chapter_json(1, chapter_title="Token Auth Pipeline")],
            record_calls=[],
        )
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert chapters[0].chapter_title == "Token Auth Pipeline"

    def test_empty_skeleton_returns_empty(self, tmp_path):
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {}, llm=llm, repo_root=str(tmp_path))
        assert chapters == []

    def test_multiple_clusters_each_get_chapter(self, tmp_path):
        clusters = [_make_cluster(cluster_id=i) for i in range(1, 4)]
        skeleton = _make_skeleton(clusters)
        packs = {c.cluster_id: _make_pack(c.cluster_id) for c in clusters}
        responses = [_chapter_json(i, n_subpages=3) for i in range(1, 4)]
        llm = FakeLLM(responses=responses, record_calls=[])
        chapters = run_planner_chaptered(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert len(chapters) == 3
        assert {ch.cluster_id for ch in chapters} == {1, 2, 3}


# ---- Tests: back-compat (flat LLM response -> 1-subpage ChapterSpec) ---------


class TestBackCompatFlatResponse:
    def test_flat_response_wraps_in_one_subpage_chapter(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_flat_json(1, title="Auth Module")], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters) == 1
        assert isinstance(chapters[0], ChapterSpec)
        assert len(chapters[0].subpages) == 1

    def test_flat_response_chapter_title_from_flat_title(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_flat_json(1, title="Auth Module")], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert "Auth Module" in chapters[0].chapter_title

    def test_flat_response_subpage_has_page_order_one(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_flat_json(1)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert chapters[0].subpages[0].page_order == 1


# ---- Tests: tiny cluster (<=5 symbols) -> 1 sub-page max --------------------


class TestTinyCluster:
    def test_tiny_cluster_capped_at_one_subpage(self, tmp_path):
        tiny = _make_cluster(cluster_id=1, n_symbols=3)
        skeleton = _make_skeleton([tiny])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=3)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters[0].subpages) == 1

    def test_boundary_five_symbols_is_tiny(self, tmp_path):
        tiny = _make_cluster(cluster_id=1, n_symbols=5)
        skeleton = _make_skeleton([tiny])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=4)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters[0].subpages) == 1

    def test_six_symbols_not_tiny(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, n_symbols=6)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=3)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters[0].subpages) == 3


# ---- Tests: sub-page cap at 6 -----------------------------------------------


class TestSubPageCap:
    def test_eight_subpages_capped_to_six(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, n_symbols=20)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=8)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters[0].subpages) <= 6

    def test_exactly_six_subpages_not_capped(self, tmp_path):
        cluster = _make_cluster(cluster_id=1, n_symbols=20)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=6)], record_calls=[])
        chapters = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert len(chapters[0].subpages) == 6


# ---- Tests: run_planner (old API) still works unchanged ----------------------


class TestOldPlannerUnchanged:
    def test_run_planner_returns_page_spec_list(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1)
        llm = FakeLLM(
            responses=[
                json.dumps({
                    "cluster_id": 1,
                    "title": "Auth",
                    "description": "Does auth.",
                    "retrieval_query": "auth",
                })
            ],
            record_calls=[],
        )
        pages = run_planner(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert isinstance(pages, list)
        assert len(pages) == 1
        assert isinstance(pages[0], PageSpec)


# ---- Tests: feature flag -----------------------------------------------------


class TestFeatureFlag:
    def test_flag_off_by_default(self):
        import importlib

        old = os.environ.pop("WIKIS_CHAPTERED_PAGES", None)
        try:
            import app.core.feature_flags as ff_mod
            importlib.reload(ff_mod)
            flags = ff_mod.get_feature_flags()
            assert flags.chaptered_pages is False
        finally:
            if old is not None:
                os.environ["WIKIS_CHAPTERED_PAGES"] = old
            importlib.reload(ff_mod)

    def test_flag_on_when_env_set_to_one(self):
        import importlib

        os.environ["WIKIS_CHAPTERED_PAGES"] = "1"
        try:
            import app.core.feature_flags as ff_mod
            importlib.reload(ff_mod)
            flags = ff_mod.get_feature_flags()
            assert flags.chaptered_pages is True
        finally:
            os.environ.pop("WIKIS_CHAPTERED_PAGES", None)
            importlib.reload(ff_mod)

    def test_flag_off_when_env_set_to_zero(self):
        import importlib

        os.environ["WIKIS_CHAPTERED_PAGES"] = "0"
        try:
            import app.core.feature_flags as ff_mod
            importlib.reload(ff_mod)
            flags = ff_mod.get_feature_flags()
            assert flags.chaptered_pages is False
        finally:
            os.environ.pop("WIKIS_CHAPTERED_PAGES", None)
            importlib.reload(ff_mod)
