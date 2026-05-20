"""Unit tests for EvidenceSlice + slice_evidence_pack (#289).

Covers:
- EvidenceSlice dataclass has all required fields
- slice_evidence_pack splits 6 symbols across 2 subpages when target_symbols match
- No overlap in symbols between slices (unless edges connect both sides)
- empty target_symbols -> even-split fallback (round-robin)
- call_graph_edges filtered: only edges touching slice symbols are included
- cross_cluster_refs filtered by symbol membership
- doc_paths shares up to 3 doc_snippets from pack
- summary field populated when storage returns llm_summary values
- run_planner_chaptered returns (list[ChapterSpec], dict[int, list[EvidenceSlice]])
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.core.wiki_structure_planner.evidence import (
    EvidencePack,
    EvidenceSlice,
    slice_evidence_pack,
)
from app.core.wiki_structure_planner.planner_agent import (
    ChapterSpec,
    SubPageSpec,
    run_planner_chaptered,
)
from app.core.wiki_structure_planner.structure_skeleton import (
    ArtifactInfo,
    Cluster,
    StructureSkeleton,
)


# ---- Helpers -----------------------------------------------------------------


def _make_artifact(name: str, source_path: str = "src/mod.py", layer: str = "public_api") -> ArtifactInfo:
    return ArtifactInfo(
        kind="symbol",
        name=name,
        source_path=source_path,
        layer=layer,
        connections=5,
    )


def _make_cluster(cluster_id: int = 1, artifacts: list[ArtifactInfo] | None = None) -> Cluster:
    arts = artifacts or [_make_artifact(f"Sym{i}") for i in range(6)]
    return Cluster(
        cluster_id=cluster_id,
        kind="code",
        dirs=["src/module"],
        artifacts=arts,
        total_artifacts=len(arts),
        primary_languages=["py"],
        depth_range=(1, 2),
    )


def _make_pack(
    cluster_id: int = 1,
    artifact_names: list[str] | None = None,
    call_graph_edges: list[tuple] | None = None,
    cross_cluster_refs: list[str] | None = None,
    path_leaves: list[str] | None = None,
    doc_snippets: list[str] | None = None,
) -> EvidencePack:
    names = artifact_names or [f"Sym{i}" for i in range(6)]
    pack = EvidencePack(cluster_id=cluster_id, kind="code")
    pack.artifact_names = names
    pack.call_graph_edges = call_graph_edges or []
    pack.cross_cluster_refs = cross_cluster_refs or []
    pack.path_leaves = path_leaves or [f"src/{n.lower()}.py" for n in names]
    pack.doc_snippets = doc_snippets or []
    return pack


def _make_chapter_spec(
    cluster_id: int,
    subpages: list[SubPageSpec] | None = None,
) -> ChapterSpec:
    sps = subpages or [
        SubPageSpec(
            title=f"SubPage {i}",
            description=f"Desc {i}",
            page_order=i,
            retrieval_query=f"query {i}",
            target_symbols=[f"Sym{i*3}", f"Sym{i*3+1}", f"Sym{i*3+2}"],
            target_folders=["src/module"],
            target_docs=[],
            evidence_filter_symbols=[],
            evidence_filter_paths=[],
        )
        for i in range(2)
    ]
    return ChapterSpec(
        cluster_id=cluster_id,
        chapter_title="Test Chapter",
        chapter_description="A test chapter.",
        subpages=sps,
    )


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


def _chapter_json(cluster_id: int, n_subpages: int = 2) -> str:
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
        for i in range(n_subpages)
    ]
    return json.dumps({
        "cluster_id": cluster_id,
        "chapter_title": "Test Chapter",
        "chapter_description": "Handles test logic.",
        "subpages": subpages,
    })


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


# ---- Tests: EvidenceSlice dataclass -----------------------------------------


class TestEvidenceSliceDataclass:
    def test_has_required_fields(self):
        sl = EvidenceSlice(
            subpage_index=0,
            subpage_title="Intro",
            symbols=["Foo", "Bar"],
            file_paths=["src/foo.py"],
            doc_paths=["docs/guide.md"],
            call_graph_edges=[("Foo", "Bar")],
            cross_cluster_refs=["ExternalSym"],
            summary="Foo handles auth. Bar manages sessions.",
        )
        assert sl.subpage_index == 0
        assert sl.subpage_title == "Intro"
        assert sl.symbols == ["Foo", "Bar"]
        assert sl.file_paths == ["src/foo.py"]
        assert sl.doc_paths == ["docs/guide.md"]
        assert sl.call_graph_edges == [("Foo", "Bar")]
        assert sl.cross_cluster_refs == ["ExternalSym"]
        assert "Foo" in sl.summary

    def test_default_empty_fields(self):
        sl = EvidenceSlice(
            subpage_index=1,
            subpage_title="Overview",
            symbols=[],
            file_paths=[],
            doc_paths=[],
            call_graph_edges=[],
            cross_cluster_refs=[],
            summary="",
        )
        assert sl.symbols == []
        assert sl.summary == ""


# ---- Tests: slice_evidence_pack with target_symbols matching ----------------


class TestSliceEvidencePackWithTargetSymbols:
    def test_returns_one_slice_per_subpage(self):
        pack = _make_pack(artifact_names=["Sym0", "Sym1", "Sym2", "Sym3", "Sym4", "Sym5"])
        chapter = _make_chapter_spec(1)  # 2 subpages, each with 3 target_symbols
        slices = slice_evidence_pack(pack, chapter)
        assert len(slices) == 2

    def test_slice_symbols_match_subpage_target_symbols(self):
        pack = _make_pack(artifact_names=["Sym0", "Sym1", "Sym2", "Sym3", "Sym4", "Sym5"])
        chapter = _make_chapter_spec(1)
        slices = slice_evidence_pack(pack, chapter)
        # subpage 0: target_symbols = [Sym0, Sym1, Sym2]
        assert set(slices[0].symbols) == {"Sym0", "Sym1", "Sym2"}
        # subpage 1: target_symbols = [Sym3, Sym4, Sym5]
        assert set(slices[1].symbols) == {"Sym3", "Sym4", "Sym5"}

    def test_no_symbol_overlap_between_slices(self):
        pack = _make_pack(artifact_names=["Sym0", "Sym1", "Sym2", "Sym3", "Sym4", "Sym5"])
        chapter = _make_chapter_spec(1)
        slices = slice_evidence_pack(pack, chapter)
        set0 = set(slices[0].symbols)
        set1 = set(slices[1].symbols)
        assert set0.isdisjoint(set1)

    def test_subpage_index_matches_page_order(self):
        pack = _make_pack(artifact_names=["Sym0", "Sym1", "Sym2", "Sym3", "Sym4", "Sym5"])
        chapter = _make_chapter_spec(1)
        slices = slice_evidence_pack(pack, chapter)
        assert slices[0].subpage_index == chapter.subpages[0].page_order
        assert slices[1].subpage_index == chapter.subpages[1].page_order

    def test_subpage_title_copied_from_subpage_spec(self):
        pack = _make_pack(artifact_names=["Sym0", "Sym1", "Sym2", "Sym3", "Sym4", "Sym5"])
        chapter = _make_chapter_spec(1)
        slices = slice_evidence_pack(pack, chapter)
        assert slices[0].subpage_title == chapter.subpages[0].title
        assert slices[1].subpage_title == chapter.subpages[1].title


# ---- Tests: empty target_symbols -> even-split fallback ----------------------


class TestSliceEvidencePackEvenSplitFallback:
    def test_empty_target_symbols_triggers_even_split(self):
        pack = _make_pack(artifact_names=["A", "B", "C", "D", "E", "F"])
        sp0 = SubPageSpec("P0", "desc", 0, "q0", [], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", [], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        # 6 symbols split into 2 subpages -> 3 each
        assert len(slices[0].symbols) == 3
        assert len(slices[1].symbols) == 3

    def test_even_split_covers_all_symbols(self):
        pack = _make_pack(artifact_names=["A", "B", "C", "D", "E", "F"])
        sp0 = SubPageSpec("P0", "desc", 0, "q0", [], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", [], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        all_assigned = set(slices[0].symbols) | set(slices[1].symbols)
        assert all_assigned == {"A", "B", "C", "D", "E", "F"}

    def test_uneven_split_doesnt_lose_symbols(self):
        # 5 symbols, 2 subpages: first gets 2, second gets 3 (or similar)
        pack = _make_pack(artifact_names=["A", "B", "C", "D", "E"])
        sp0 = SubPageSpec("P0", "desc", 0, "q0", [], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", [], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        all_assigned = set(slices[0].symbols) | set(slices[1].symbols)
        assert all_assigned == {"A", "B", "C", "D", "E"}

    def test_single_subpage_gets_all_symbols(self):
        pack = _make_pack(artifact_names=["A", "B", "C"])
        sp = SubPageSpec("P0", "desc", 0, "q0", [], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        assert len(slices) == 1
        assert set(slices[0].symbols) == {"A", "B", "C"}

    def test_single_subpage_page_order_one_does_not_produce_empty_slice(self):
        """Regression for Rio #328 review: page_order=1 with n=1 caused
        start=len, end=2*len → empty slice. The fix uses enumerate idx."""
        pack = _make_pack(artifact_names=["A", "B", "C"])
        # page_order=1 (1-based, as flat-spec wrapping produces)
        sp = SubPageSpec("P0", "desc", 1, "q0", [], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        assert len(slices) == 1
        assert set(slices[0].symbols) == {"A", "B", "C"}, (
            "page_order=1 must not produce empty slice for n=1 chapter"
        )


# ---- Tests: call_graph_edges filtering --------------------------------------


class TestCallGraphEdgeFiltering:
    def test_edges_filtered_to_slice_symbols(self):
        pack = _make_pack(
            artifact_names=["A", "B", "C", "D"],
            call_graph_edges=[("A", "B"), ("C", "D"), ("A", "D"), ("B", "C")],
        )
        sp0 = SubPageSpec("P0", "desc", 0, "q0", ["A", "B"], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", ["C", "D"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        # A, B are in slice0 — edges where caller or callee is A or B
        edges_0 = set(slices[0].call_graph_edges)
        assert ("A", "B") in edges_0
        # C, D are in slice1 — edges where caller or callee is C or D
        edges_1 = set(slices[1].call_graph_edges)
        assert ("C", "D") in edges_1

    def test_cross_slice_edge_appears_in_both(self):
        # Edge (A, D): A in slice0, D in slice1 -> appears in both
        pack = _make_pack(
            artifact_names=["A", "B", "C", "D"],
            call_graph_edges=[("A", "D")],
        )
        sp0 = SubPageSpec("P0", "desc", 0, "q0", ["A", "B"], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", ["C", "D"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        assert ("A", "D") in slices[0].call_graph_edges
        assert ("A", "D") in slices[1].call_graph_edges

    def test_irrelevant_edges_excluded(self):
        # Only A and B in pack; edge C->D not in slice
        pack = _make_pack(
            artifact_names=["A", "B"],
            call_graph_edges=[("A", "B"), ("X", "Y")],
        )
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A", "B"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        assert ("A", "B") in slices[0].call_graph_edges
        assert ("X", "Y") not in slices[0].call_graph_edges


# ---- Tests: cross_cluster_refs filtering ------------------------------------


class TestCrossClusterRefsFiltering:
    def test_cross_refs_filtered_by_slice_symbols(self):
        pack = _make_pack(
            artifact_names=["Auth", "Session"],
            cross_cluster_refs=["Auth.login", "DB.query", "Session.start"],
        )
        sp0 = SubPageSpec("P0", "desc", 0, "q0", ["Auth"], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", ["Session"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        # "auth" case-insensitive match in "Auth.login"
        assert any("Auth" in r or "auth" in r.lower() for r in slices[0].cross_cluster_refs)
        # "session" match in "Session.start"
        assert any("Session" in r or "session" in r.lower() for r in slices[1].cross_cluster_refs)
        # "DB.query" has no slice symbol substring -> not in either
        assert "DB.query" not in slices[0].cross_cluster_refs
        assert "DB.query" not in slices[1].cross_cluster_refs

    def test_no_cross_refs_returns_empty_list(self):
        pack = _make_pack(artifact_names=["A", "B"], cross_cluster_refs=[])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A", "B"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        assert slices[0].cross_cluster_refs == []


# ---- Tests: doc_paths from doc_snippets -------------------------------------


class TestDocPaths:
    def test_doc_paths_shared_from_pack_snippets(self):
        pack = _make_pack(
            artifact_names=["A", "B"],
            doc_snippets=["docs/overview.md", "docs/api.md", "docs/guide.md", "docs/extra.md"],
        )
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A", "B"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        # At most 3 doc snippets shared
        assert len(slices[0].doc_paths) <= 3
        assert slices[0].doc_paths == ["docs/overview.md", "docs/api.md", "docs/guide.md"]

    def test_fewer_than_three_snippets_all_included(self):
        pack = _make_pack(artifact_names=["A"], doc_snippets=["docs/readme.md"])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        assert slices[0].doc_paths == ["docs/readme.md"]

    def test_no_doc_snippets_returns_empty_list(self):
        pack = _make_pack(artifact_names=["A"], doc_snippets=[])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        assert slices[0].doc_paths == []


# ---- Tests: file_paths filtering --------------------------------------------


class TestFilePathsFiltering:
    def test_file_paths_filtered_by_symbol_name(self):
        pack = _make_pack(
            artifact_names=["Auth", "Session"],
            path_leaves=["src/auth/core.py", "src/session/manager.py", "src/utils.py"],
        )
        sp0 = SubPageSpec("P0", "desc", 0, "q0", ["Auth"], [], [], [], [])
        sp1 = SubPageSpec("P1", "desc", 1, "q1", ["Session"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp0, sp1])
        slices = slice_evidence_pack(pack, chapter)
        # auth path goes to slice0
        assert "src/auth/core.py" in slices[0].file_paths
        # session path goes to slice1
        assert "src/session/manager.py" in slices[1].file_paths

    def test_file_paths_fallback_to_all_when_no_match(self):
        pack = _make_pack(
            artifact_names=["WeirdName"],
            path_leaves=["src/totally_unrelated.py"],
        )
        sp = SubPageSpec("P0", "desc", 0, "q0", ["WeirdName"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        # fallback: all paths
        assert slices[0].file_paths == ["src/totally_unrelated.py"]


# ---- Tests: summary field with storage --------------------------------------


class TestSummaryWithStorage:
    def test_summary_populated_from_storage(self):
        """Storage returns llm_summary for symbols -> populated in slice."""
        pack = _make_pack(artifact_names=["Auth", "Session"])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["Auth", "Session"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])

        class FakeStorage:
            def get_node_summary(self, symbol: str) -> str:
                summaries = {
                    "Auth": "Handles authentication flow.",
                    "Session": "Manages user sessions.",
                }
                return summaries.get(symbol, "")

        slices = slice_evidence_pack(pack, chapter, storage=FakeStorage())
        assert "Auth" in slices[0].summary or "authentication" in slices[0].summary.lower()
        assert len(slices[0].summary) > 0

    def test_summary_empty_when_no_storage(self):
        pack = _make_pack(artifact_names=["Auth", "Session"])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["Auth", "Session"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])
        slices = slice_evidence_pack(pack, chapter)
        # No storage -> summary stays empty or is ""
        assert slices[0].summary == ""

    def test_summary_truncated_at_1200_chars(self):
        """Concatenated summaries over 1200 chars should be truncated."""
        pack = _make_pack(artifact_names=["A", "B"])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A", "B"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])

        class VerboseStorage:
            def get_node_summary(self, symbol: str) -> str:
                return "x" * 800  # each returns 800 chars

        slices = slice_evidence_pack(pack, chapter, storage=VerboseStorage())
        assert len(slices[0].summary) <= 1200

    def test_storage_exception_handled_gracefully(self):
        """If storage raises, summary should be empty (not crash)."""
        pack = _make_pack(artifact_names=["A"])
        sp = SubPageSpec("P0", "desc", 0, "q0", ["A"], [], [], [], [])
        chapter = ChapterSpec(1, "Ch", "Chapter", [sp])

        class BrokenStorage:
            def get_node_summary(self, symbol: str) -> str:
                raise RuntimeError("DB down")

        # Should not raise
        slices = slice_evidence_pack(pack, chapter, storage=BrokenStorage())
        assert slices[0].summary == ""


# ---- Tests: run_planner_chaptered returns (chapters, evidence) tuple --------


class TestRunPlannerChapteredReturnType:
    def test_returns_tuple_of_chapters_and_evidence_dict(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        # Build an EvidencePack with artifact_names set
        pack = _make_pack(cluster_id=1, artifact_names=["Sym0", "Sym1", "Sym2"])
        packs = {1: pack}
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=2)], record_calls=[])
        result = run_planner_chaptered(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_list_of_chapter_spec(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1, artifact_names=["Sym0", "Sym1", "Sym2"])
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=2)], record_calls=[])
        chapters, _ = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert isinstance(chapters, list)
        assert all(isinstance(c, ChapterSpec) for c in chapters)

    def test_second_element_is_dict_of_evidence_slices(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1, artifact_names=["Sym0", "Sym1", "Sym2"])
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=2)], record_calls=[])
        _, chapter_evidence = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        assert isinstance(chapter_evidence, dict)
        # key is cluster_id=1
        assert 1 in chapter_evidence
        assert isinstance(chapter_evidence[1], list)
        assert all(isinstance(sl, EvidenceSlice) for sl in chapter_evidence[1])

    def test_evidence_dict_has_one_slice_per_subpage(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        pack = _make_pack(cluster_id=1, artifact_names=["Sym0", "Sym1", "Sym2"])
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=2)], record_calls=[])
        chapters, chapter_evidence = run_planner_chaptered(skeleton, {1: pack}, llm=llm, repo_root=str(tmp_path))
        n_subpages = len(chapters[0].subpages)
        assert len(chapter_evidence[1]) == n_subpages

    def test_multiple_clusters_all_in_evidence_dict(self, tmp_path):
        clusters = [_make_cluster(cluster_id=i) for i in range(1, 4)]
        skeleton = _make_skeleton(clusters)
        packs = {i: _make_pack(cluster_id=i, artifact_names=["Sym0", "Sym1"]) for i in range(1, 4)}
        responses = [_chapter_json(i, n_subpages=2) for i in range(1, 4)]
        llm = FakeLLM(responses=responses, record_calls=[])
        chapters, chapter_evidence = run_planner_chaptered(skeleton, packs, llm=llm, repo_root=str(tmp_path))
        assert len(chapters) == 3
        assert set(chapter_evidence.keys()) == {1, 2, 3}

    def test_no_pack_for_cluster_excluded_from_evidence_dict(self, tmp_path):
        cluster = _make_cluster(cluster_id=1)
        skeleton = _make_skeleton([cluster])
        llm = FakeLLM(responses=[_chapter_json(1, n_subpages=2)], record_calls=[])
        # No pack provided
        chapters, chapter_evidence = run_planner_chaptered(skeleton, {}, llm=llm, repo_root=str(tmp_path))
        # cluster_id=1 should not be in chapter_evidence (no pack to slice)
        assert 1 not in chapter_evidence

    def test_empty_skeleton_returns_empty_tuple(self, tmp_path):
        skeleton = _make_skeleton([])
        llm = FakeLLM(responses=[], record_calls=[])
        result = run_planner_chaptered(skeleton, {}, llm=llm, repo_root=str(tmp_path))
        assert isinstance(result, tuple)
        chapters, chapter_evidence = result
        assert chapters == []
        assert chapter_evidence == {}
