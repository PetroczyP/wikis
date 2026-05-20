"""Tests for wiki_content_writer/writer_tools.py (#237).

Covers: tool function shapes, read_attachment_meta metadata-only guarantee,
and stub behaviour for graph-backed tools.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.core.wiki_content_writer.writer_tools import (
    AttachmentMeta,
    DocChunk,
    FileContent,
    GrepMatch,
    SymbolCallees,
    SymbolCallers,
    SymbolSignature,
    WriterTools,
)

# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_tools(
    repo_root: str = "/repo",
    storage: Any = None,
    code_graph: Any = None,
    graph_text_index: Any = None,
) -> WriterTools:
    return WriterTools(
        repo_root=repo_root,
        storage=storage,
        code_graph=code_graph,
        graph_text_index=graph_text_index,
    )


# ── WriterTools construction ─────────────────────────────────────────────────


class TestWriterToolsConstruction:
    def test_constructs_with_required_fields_only(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        assert tools is not None

    def test_repo_root_stored(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        assert tools.repo_root == str(tmp_path)

    def test_optional_deps_default_to_none(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        assert tools.storage is None
        assert tools.code_graph is None
        assert tools.graph_text_index is None

    def test_constructs_with_all_deps(self, tmp_path):
        storage = MagicMock()
        graph = MagicMock()
        fts = MagicMock()
        tools = WriterTools(
            repo_root=str(tmp_path),
            storage=storage,
            code_graph=graph,
            graph_text_index=fts,
        )
        assert tools.storage is storage
        assert tools.code_graph is graph
        assert tools.graph_text_index is fts


# ── FileContent return type ──────────────────────────────────────────────────


class TestReadFile:
    def test_returns_file_content_type(self, tmp_path):
        (tmp_path / "main.py").write_text("def hello():\n    pass\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("main.py")
        assert isinstance(result, FileContent)

    def test_has_path_and_lines_fields(self, tmp_path):
        (tmp_path / "app.py").write_text("x = 1\ny = 2\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("app.py")
        assert result.path == "app.py"
        assert isinstance(result.lines, list)

    def test_reads_file_content(self, tmp_path):
        (tmp_path / "cfg.py").write_text("A = 1\nB = 2\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("cfg.py")
        assert "A = 1" in "\n".join(result.lines)

    def test_line_range_filters_lines(self, tmp_path):
        content = "\n".join(f"line{i}" for i in range(1, 11))
        (tmp_path / "big.py").write_text(content)
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("big.py", line_range=(3, 5))
        # Should only include lines 3-5 (1-indexed)
        assert len(result.lines) == 3
        assert "line3" in result.lines[0]

    def test_missing_file_returns_empty_content(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("does_not_exist.py")
        assert result.lines == []
        assert result.error is not None

    def test_has_total_lines_field(self, tmp_path):
        (tmp_path / "x.py").write_text("a\nb\nc\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("x.py")
        assert isinstance(result.total_lines, int)
        assert result.total_lines == 3

    def test_parent_traversal_rejected(self, tmp_path):
        # Create a sibling file outside the simulated repo_root
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")
        repo = tmp_path / "repo"
        repo.mkdir()
        tools = WriterTools(repo_root=str(repo))
        result = tools.read_file("../outside.txt")
        assert result.lines == []
        assert result.error is not None
        assert "escapes" in result.error

    def test_absolute_path_rejected(self, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("nope")
        repo = tmp_path / "repo"
        repo.mkdir()
        tools = WriterTools(repo_root=str(repo))
        result = tools.read_file(str(outside))
        assert result.lines == []
        assert result.error is not None

    def test_symlink_escape_rejected(self, tmp_path):
        outside = tmp_path / "secret.txt"
        outside.write_text("nope")
        repo = tmp_path / "repo"
        repo.mkdir()
        # Create a symlink inside repo that points outside
        (repo / "link.txt").symlink_to(outside)
        tools = WriterTools(repo_root=str(repo))
        result = tools.read_file("link.txt")
        assert result.lines == []
        assert result.error is not None

    def test_inverted_line_range_returns_error(self, tmp_path):
        (tmp_path / "x.py").write_text("a\nb\nc\nd\ne\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_file("x.py", line_range=(5, 2))
        assert result.lines == []
        assert result.error is not None
        assert "inverted" in result.error


# ── SymbolSignature return type ──────────────────────────────────────────────


class TestGetSignature:
    def test_returns_symbol_signature_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("SomeClass")
        assert isinstance(result, SymbolSignature)

    def test_has_required_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("SomeClass")
        assert hasattr(result, "symbol")
        assert hasattr(result, "signature")
        assert hasattr(result, "file_path")
        assert hasattr(result, "layer")
        assert hasattr(result, "docstring")
        assert hasattr(result, "found")

    def test_not_found_when_no_graph(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("Nonexistent")
        assert result.found is False

    def test_symbol_echoed(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("MyClass")
        assert result.symbol == "MyClass"

    def test_stub_returns_found_false_even_with_graph(self, tmp_path):
        # Stub behaviour: until #243 wires GraphQueryService, the
        # presence of a code_graph must not flip `found` to True.
        graph = MagicMock()
        tools = WriterTools(repo_root=str(tmp_path), code_graph=graph)
        result = tools.get_signature("MyClass")
        assert result.found is False


# ── SymbolCallers / SymbolCallees return types ───────────────────────────────


class TestGetCallers:
    def test_returns_symbol_callers_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("some_func")
        assert isinstance(result, SymbolCallers)

    def test_has_symbol_and_callers_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert hasattr(result, "symbol")
        assert hasattr(result, "callers")
        assert isinstance(result.callers, list)

    def test_empty_callers_when_no_graph(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert result.callers == []

    def test_has_found_field_defaulting_false(self, tmp_path):
        # `found` discriminates "no callers" from "backend not wired".
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert hasattr(result, "found")
        assert result.found is False


class TestGetCallees:
    def test_returns_symbol_callees_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("some_func")
        assert isinstance(result, SymbolCallees)

    def test_has_symbol_and_callees_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert hasattr(result, "symbol")
        assert hasattr(result, "callees")
        assert isinstance(result.callees, list)

    def test_empty_callees_when_no_graph(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert result.callees == []

    def test_has_found_field_defaulting_false(self, tmp_path):
        # Symmetric with TestGetCallers — `found` discriminates "no
        # callees" from "backend not wired".
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert hasattr(result, "found")
        assert result.found is False


# ── GrepMatch return type ────────────────────────────────────────────────────


class TestGrep:
    def test_returns_list_of_grep_matches(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.grep("pattern")
        assert isinstance(result, list)

    def test_each_result_is_grep_match(self, tmp_path):
        (tmp_path / "a.py").write_text("def pattern_func(): pass\n")
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.grep("pattern_func")
        for item in result:
            assert isinstance(item, GrepMatch)

    def test_grep_match_has_required_fields(self, tmp_path):
        # Even if empty, the type is correct — verify by constructing one
        match = GrepMatch(file_path="x.py", line_number=1, line_text="x = 1", score=1.0)
        assert match.file_path == "x.py"
        assert match.line_number == 1
        assert match.line_text == "x = 1"
        assert match.score == 1.0

    def test_grep_reads_search_score_metadata_key(self, tmp_path):
        # GraphTextIndex / StorageTextIndex populate metadata["search_score"].
        # Verify grep picks up that key (with fallback to legacy "score").
        class _FakeDoc:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = meta

        class _FakeIndex:
            def __init__(self, docs):
                self._docs = docs

            def search(self, pattern, k=20):
                return self._docs

        docs = [
            _FakeDoc("hit one", {"rel_path": "a.py", "start_line": 5, "search_score": 0.92}),
            _FakeDoc("hit two", {"rel_path": "b.py", "start_line": 7, "score": 0.41}),
            _FakeDoc("hit three", {"rel_path": "c.py", "start_line": 9}),
        ]
        tools = WriterTools(repo_root=str(tmp_path), graph_text_index=_FakeIndex(docs))
        result = tools.grep("anything")
        assert len(result) == 3
        assert result[0].score == 0.92
        assert result[1].score == 0.41
        assert result[2].score == 0.0


# ── DocChunk return type ─────────────────────────────────────────────────────


class TestListDocChunks:
    def test_returns_list(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.list_doc_chunks("docs/README.md")
        assert isinstance(result, list)

    def test_each_result_is_doc_chunk(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.list_doc_chunks("docs/guide.md")
        for item in result:
            assert isinstance(item, DocChunk)

    def test_doc_chunk_has_required_fields(self):
        chunk = DocChunk(
            doc_path="docs/guide.md",
            chunk_index=0,
            heading="Overview",
            text="Some content",
        )
        assert chunk.doc_path == "docs/guide.md"
        assert chunk.chunk_index == 0
        assert chunk.heading == "Overview"
        assert chunk.text == "Some content"


# ── AttachmentMeta — never returns content ──────────────────────────────────


class TestReadAttachmentMeta:
    """read_attachment_meta must return only metadata — never file content."""

    def test_returns_attachment_meta_type(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("diagram.png")
        assert isinstance(result, AttachmentMeta)

    def test_has_name_mime_parent_fields(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("chart.png")
        assert hasattr(result, "name")
        assert hasattr(result, "mime")
        assert hasattr(result, "parent")

    def test_name_echoed(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("logo.svg")
        assert result.name == "logo.svg"

    def test_no_content_field(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("file.pdf")
        # The result type must NOT have a 'content' attribute
        assert not hasattr(result, "content")
        assert not hasattr(result, "data")
        assert not hasattr(result, "bytes")
        assert not hasattr(result, "raw")

    def test_returns_none_mime_when_unknown(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("file.unknownext12345")
        # mime should be None or a string, never content
        assert result.mime is None or isinstance(result.mime, str)

    def test_png_mime_detected(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("image.png")
        assert result.mime == "image/png"

    def test_pdf_mime_detected(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("document.pdf")
        assert result.mime == "application/pdf"

    def test_svg_mime_detected(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.read_attachment_meta("diagram.svg")
        assert result.mime == "image/svg+xml"


# ── Storage-wired tool tests ─────────────────────────────────────────────────
#
# Each test below mocks the storage protocol so no real DB is needed.
# The mock returns deterministic rows matching the repo_nodes schema.


def _make_node(
    node_id: str = "n1",
    symbol_name: str = "WorkerPool",
    rel_path: str = "app/worker.py",
    start_line: int = 10,
    end_line: int = 30,
    signature: str = "class WorkerPool:",
    symbol_type: str = "class",
    docstring: str = "Manages a pool of workers.",
    source_text: str = "class WorkerPool:\n    pass",
) -> dict:
    return {
        "node_id": node_id,
        "symbol_name": symbol_name,
        "rel_path": rel_path,
        "start_line": start_line,
        "end_line": end_line,
        "signature": signature,
        "symbol_type": symbol_type,
        "docstring": docstring,
        "source_text": source_text,
        "is_architectural": 1,
        "macro_cluster": None,
        "micro_cluster": None,
    }


def _make_storage(
    symbol_rows=None,
    fts_rows=None,
    edges_to=None,
    edges_from=None,
    nodes_by_prefix=None,
    node_by_id=None,
):
    storage = MagicMock()
    storage.search_fts_by_symbol_name.return_value = symbol_rows or []
    storage.search_fts_by_column.return_value = fts_rows or []
    storage.get_edges_to.return_value = edges_to or []
    storage.get_edges_from.return_value = edges_from or []
    storage.get_nodes_by_path_prefix.return_value = nodes_by_prefix or []
    storage.get_node.return_value = node_by_id
    return storage


class TestGetSignatureWithStorage:
    def test_found_true_when_storage_returns_match(self, tmp_path):
        node = _make_node()
        storage = _make_storage(symbol_rows=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_signature("WorkerPool")
        assert result.found is True

    def test_symbol_echoed_with_storage(self, tmp_path):
        node = _make_node()
        storage = _make_storage(symbol_rows=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_signature("WorkerPool")
        assert result.symbol == "WorkerPool"

    def test_file_path_and_lines_in_result(self, tmp_path):
        node = _make_node(rel_path="app/worker.py")
        storage = _make_storage(symbol_rows=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_signature("WorkerPool")
        assert result.file_path == "app/worker.py"

    def test_signature_text_populated(self, tmp_path):
        node = _make_node(signature="class WorkerPool:")
        storage = _make_storage(symbol_rows=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_signature("WorkerPool")
        assert result.signature == "class WorkerPool:"

    def test_docstring_populated(self, tmp_path):
        node = _make_node(docstring="Manages a pool of workers.")
        storage = _make_storage(symbol_rows=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_signature("WorkerPool")
        assert result.docstring == "Manages a pool of workers."

    def test_not_found_when_storage_returns_empty(self, tmp_path):
        storage = _make_storage(symbol_rows=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_signature("Nonexistent")
        assert result.found is False

    def test_storage_queried_with_symbol_name(self, tmp_path):
        storage = _make_storage(symbol_rows=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        tools.get_signature("MyClass")
        storage.search_fts_by_symbol_name.assert_called_once()
        call_args = storage.search_fts_by_symbol_name.call_args
        assert call_args[0][0] == "MyClass" or call_args[1].get("name") == "MyClass"

    def test_no_storage_returns_not_found(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_signature("WorkerPool")
        assert result.found is False


class TestGrepWithStorage:
    def test_returns_grep_matches_from_storage(self, tmp_path):
        row = _make_node(rel_path="src/queue.py", start_line=5, symbol_name="JobQueue")
        storage = _make_storage(fts_rows=[row])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.grep("JobQueue")
        assert len(result) == 1
        assert isinstance(result[0], GrepMatch)

    def test_file_path_and_line_in_grep_match(self, tmp_path):
        row = _make_node(rel_path="src/queue.py", start_line=5)
        storage = _make_storage(fts_rows=[row])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.grep("JobQueue")
        assert result[0].file_path == "src/queue.py"
        assert result[0].line_number == 5

    def test_empty_results_when_no_matches(self, tmp_path):
        storage = _make_storage(fts_rows=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.grep("NothingHere")
        assert result == []

    def test_storage_queried_with_pattern(self, tmp_path):
        storage = _make_storage(fts_rows=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        tools.grep("MyPattern")
        storage.search_fts_by_column.assert_called_once()

    def test_no_storage_returns_empty(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.grep("anything")
        assert result == []


class TestGetCallersWithStorage:
    def test_returns_callers_found_true(self, tmp_path):
        target_node = _make_node(node_id="target", symbol_name="WorkerPool")
        caller_node = _make_node(node_id="caller1", symbol_name="Main", rel_path="app/main.py")
        edge = {"source_id": "caller1", "target_id": "target", "rel_type": "calls"}
        storage = _make_storage(symbol_rows=[target_node], edges_to=[edge])
        storage.get_node.return_value = caller_node
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callers("WorkerPool")
        assert result.found is True

    def test_callers_list_populated(self, tmp_path):
        target_node = _make_node(node_id="target", symbol_name="WorkerPool")
        caller_node = _make_node(node_id="caller1", symbol_name="Main", rel_path="app/main.py")
        edge = {"source_id": "caller1", "target_id": "target", "rel_type": "calls"}
        storage = _make_storage(symbol_rows=[target_node], edges_to=[edge])
        storage.get_node.return_value = caller_node
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callers("WorkerPool")
        assert len(result.callers) == 1
        assert result.callers[0].symbol == "Main"
        assert result.callers[0].file_path == "app/main.py"

    def test_no_callers_when_edges_empty(self, tmp_path):
        target_node = _make_node(node_id="target", symbol_name="WorkerPool")
        storage = _make_storage(symbol_rows=[target_node], edges_to=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callers("WorkerPool")
        assert result.found is True
        assert result.callers == []

    def test_not_found_when_symbol_missing(self, tmp_path):
        storage = _make_storage(symbol_rows=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callers("Ghost")
        assert result.found is False

    def test_no_storage_returns_not_found(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callers("fn")
        assert result.found is False


class TestGetCalleesWithStorage:
    def test_returns_callees_found_true(self, tmp_path):
        source_node = _make_node(node_id="source", symbol_name="run")
        callee_node = _make_node(node_id="callee1", symbol_name="WorkerPool", rel_path="app/worker.py")
        edge = {"source_id": "source", "target_id": "callee1", "rel_type": "calls"}
        storage = _make_storage(symbol_rows=[source_node], edges_from=[edge])
        storage.get_node.return_value = callee_node
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callees("run")
        assert result.found is True

    def test_callees_list_populated(self, tmp_path):
        source_node = _make_node(node_id="source", symbol_name="run")
        callee_node = _make_node(node_id="callee1", symbol_name="WorkerPool", rel_path="app/worker.py")
        edge = {"source_id": "source", "target_id": "callee1", "rel_type": "calls"}
        storage = _make_storage(symbol_rows=[source_node], edges_from=[edge])
        storage.get_node.return_value = callee_node
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callees("run")
        assert len(result.callees) == 1
        assert result.callees[0].symbol == "WorkerPool"
        assert result.callees[0].file_path == "app/worker.py"

    def test_edges_filtered_by_calls_and_imports(self, tmp_path):
        source_node = _make_node(node_id="source", symbol_name="run")
        storage = _make_storage(symbol_rows=[source_node], edges_from=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        tools.get_callees("run")
        storage.get_edges_from.assert_called_once()
        call_kwargs = storage.get_edges_from.call_args
        rel_types = call_kwargs[1].get("rel_types") or (call_kwargs[0][1] if len(call_kwargs[0]) > 1 else None)
        assert rel_types is not None
        assert "calls" in rel_types
        assert "imports" in rel_types

    def test_not_found_when_symbol_missing(self, tmp_path):
        storage = _make_storage(symbol_rows=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.get_callees("Ghost")
        assert result.found is False

    def test_no_storage_returns_not_found(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.get_callees("fn")
        assert result.found is False


class TestListDocChunksWithStorage:
    def test_returns_doc_chunks_from_storage(self, tmp_path):
        node = _make_node(
            node_id="chunk1",
            symbol_name="Overview",
            rel_path="docs/README.md",
            source_text="# Overview\nSome intro text.",
        )
        storage = _make_storage(nodes_by_prefix=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.list_doc_chunks("docs/README.md")
        assert len(result) == 1
        assert isinstance(result[0], DocChunk)

    def test_doc_chunk_heading_from_symbol_name(self, tmp_path):
        node = _make_node(symbol_name="Overview", source_text="# Overview\nIntro.")
        storage = _make_storage(nodes_by_prefix=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.list_doc_chunks("docs/README.md")
        assert result[0].heading == "Overview"

    def test_doc_chunk_text_is_snippet(self, tmp_path):
        node = _make_node(source_text="Some long text content here.")
        storage = _make_storage(nodes_by_prefix=[node])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.list_doc_chunks("docs/README.md")
        assert "Some long text" in result[0].text

    def test_empty_when_no_nodes(self, tmp_path):
        storage = _make_storage(nodes_by_prefix=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        result = tools.list_doc_chunks("docs/nonexistent.md")
        assert result == []

    def test_no_storage_returns_empty(self, tmp_path):
        tools = WriterTools(repo_root=str(tmp_path))
        result = tools.list_doc_chunks("docs/README.md")
        assert result == []

    def test_storage_queried_with_doc_path(self, tmp_path):
        storage = _make_storage(nodes_by_prefix=[])
        tools = WriterTools(repo_root=str(tmp_path), storage=storage)
        tools.list_doc_chunks("docs/guide.md")
        storage.get_nodes_by_path_prefix.assert_called_once()
        call_args = storage.get_nodes_by_path_prefix.call_args
        assert call_args[0][0] == "docs/guide.md" or call_args[1].get("prefix") == "docs/guide.md"
