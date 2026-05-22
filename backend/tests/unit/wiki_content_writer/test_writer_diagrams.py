"""Unit tests for LLM-based diagram injection in WikiContentWriter.

Covers:
- _gather_cluster_source: mock storage + tools, file path gathering
- _llm_class_diagram: valid diagram, EMPTY response, invalid Mermaid
- _llm_data_model_diagram: same pattern
- _inject_cluster_diagrams: end-to-end, injection before ## Key Components
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from app.core.wiki_content_writer.writer_agent import WikiContentWriter
from app.core.wiki_content_writer.writer_tools import FileContent, WriterTools


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_writer(
    storage=None,
    llm=None,
    repo_root="/repo",
) -> WikiContentWriter:
    """Build a WikiContentWriter with a minimal mock WriterTools."""
    tools = MagicMock(spec=WriterTools)
    tools.storage = storage
    tools.repo_root = repo_root
    writer = WikiContentWriter(tools=tools, llm_client=llm)
    return writer


def _code_node(
    node_id: str,
    symbol_name: str,
    symbol_type: str = "class",
    rel_path: str = "src/mod.py",
    is_architectural: int = 1,
) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "symbol_name": symbol_name,
        "symbol_type": symbol_type,
        "rel_path": rel_path,
        "is_architectural": is_architectural,
    }


def _doc_node(node_id: str, symbol_name: str, rel_path: str = "README.md") -> dict[str, Any]:
    return {
        "node_id": node_id,
        "symbol_name": symbol_name,
        "symbol_type": "markdown_document",
        "rel_path": rel_path,
        "is_architectural": 0,
    }


VALID_CLASS_DIAGRAM = """\
```mermaid
classDiagram
    class Foo {
        +bar() str
    }
    class Baz {
        -_qux int
    }
    Foo --|> Baz : implements
```"""

VALID_ER_DIAGRAM = """\
```mermaid
erDiagram
    User {
        int id PK
    }
    Post {
        int id PK
        int user_id FK
    }
    User ||--o{ Post : "has"
```"""


# ── _gather_cluster_source ──────────────────────────────────────────────────


class TestGatherClusterSource:
    """_gather_cluster_source reads files from storage nodes."""

    def test_returns_empty_when_storage_is_none(self):
        writer = _make_writer(storage=None)
        result = writer._gather_cluster_source(cluster_id=1, storage=None, max_files=5, chars_per_file=3000)
        assert result == ""

    def test_returns_empty_when_no_nodes(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = []
        writer = _make_writer(storage=storage)
        result = writer._gather_cluster_source(1, storage, max_files=5, chars_per_file=3000)
        assert result == ""

    def test_doc_nodes_are_excluded(self):
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = [
            _doc_node("d1", "README", "README.md"),
        ]
        # tools.read_file should never be called — no code nodes
        writer = _make_writer(storage=storage)
        result = writer._gather_cluster_source(1, storage, max_files=5, chars_per_file=3000)
        writer.tools.read_file.assert_not_called()
        assert result == ""

    def test_code_nodes_trigger_read_file(self):
        storage = MagicMock()
        nodes = [_code_node("n1", "Foo", rel_path="src/foo.py")]
        storage.get_nodes_by_cluster.return_value = nodes

        # tools.read_file returns content for src/foo.py
        writer = _make_writer(storage=storage)
        writer.tools.read_file.return_value = FileContent(
            path="src/foo.py",
            lines=["class Foo:", "    pass"],
            total_lines=2,
        )

        result = writer._gather_cluster_source(1, storage, max_files=5, chars_per_file=3000)
        assert "src/foo.py" in result
        assert "class Foo" in result

    def test_max_files_respected(self):
        storage = MagicMock()
        # 10 nodes with distinct paths
        nodes = [_code_node(f"n{i}", f"Class{i}", rel_path=f"src/mod{i}.py") for i in range(10)]
        storage.get_nodes_by_cluster.return_value = nodes

        writer = _make_writer(storage=storage)
        writer.tools.read_file.return_value = FileContent(
            path="x",
            lines=["content"],
            total_lines=1,
        )

        writer._gather_cluster_source(1, storage, max_files=3, chars_per_file=3000)
        # read_file should be called at most 3 times
        assert writer.tools.read_file.call_count <= 3

    def test_architectural_paths_preferred(self):
        storage = MagicMock()
        # 1 architectural node + 5 non-architectural nodes
        nodes = [
            _code_node("arch1", "CoreClass", rel_path="src/core.py", is_architectural=1),
        ] + [
            _code_node(f"n{i}", f"Other{i}", rel_path=f"src/other{i}.py", is_architectural=0)
            for i in range(5)
        ]
        storage.get_nodes_by_cluster.return_value = nodes

        writer = _make_writer(storage=storage)
        writer.tools.read_file.return_value = FileContent(
            path="x",
            lines=["content"],
            total_lines=1,
        )

        # Only max_files=1, should pick the architectural one
        writer._gather_cluster_source(1, storage, max_files=1, chars_per_file=3000)
        call_args = [c[0][0] for c in writer.tools.read_file.call_args_list]
        assert "src/core.py" in call_args

    def test_chars_per_file_truncates_content(self):
        storage = MagicMock()
        nodes = [_code_node("n1", "Foo", rel_path="src/foo.py")]
        storage.get_nodes_by_cluster.return_value = nodes

        long_content = "x" * 10000
        writer = _make_writer(storage=storage)
        writer.tools.read_file.return_value = FileContent(
            path="src/foo.py",
            lines=[long_content],
            total_lines=1,
        )

        result = writer._gather_cluster_source(1, storage, max_files=5, chars_per_file=100)
        # The content for this file should be truncated to 100 chars
        file_section = result.split("=== src/foo.py ===")[-1]
        assert len(file_section) <= 101  # +1 for the leading newline

    def test_read_file_error_skips_file(self):
        storage = MagicMock()
        nodes = [
            _code_node("n1", "Foo", rel_path="src/foo.py"),
            _code_node("n2", "Bar", rel_path="src/bar.py"),
        ]
        storage.get_nodes_by_cluster.return_value = nodes

        writer = _make_writer(storage=storage)

        def _side_effect(path, **kwargs):
            if path == "src/foo.py":
                return FileContent(path=path, lines=[], total_lines=0, error="not found")
            return FileContent(path=path, lines=["class Bar: pass"], total_lines=1)

        writer.tools.read_file.side_effect = _side_effect

        result = writer._gather_cluster_source(1, storage, max_files=5, chars_per_file=3000)
        assert "src/bar.py" in result
        assert "src/foo.py" not in result


# ── _llm_class_diagram ─────────────────────────────────────────────────────────


class TestAgenticDiagram:
    """_agentic_diagram: tool loop produces validated Mermaid block."""

    def _make_llm(self, final_text: str):
        """Mock LLM that immediately returns final_text with no tool calls."""
        llm = MagicMock()
        resp = MagicMock()
        resp.content = final_text
        resp.tool_calls = []
        llm.invoke.return_value = resp
        llm.bind_tools = MagicMock(return_value=llm)
        return llm

    def test_valid_class_diagram_returned(self):
        llm = self._make_llm(VALID_CLASS_DIAGRAM)
        writer = _make_writer(llm=llm)
        result = writer._agentic_diagram(["src/foo.py"], "Foo Module", "class")
        assert "classDiagram" in result
        assert "```mermaid" in result

    def test_valid_arch_diagram_returned(self):
        arch = "```mermaid\nstateDiagram-v2\n    [*] --> Idle\n    Idle --> Running\n```"
        llm = self._make_llm(arch)
        writer = _make_writer(llm=llm)
        result = writer._agentic_diagram(["src/foo.py"], "Foo", "architecture")
        assert "stateDiagram-v2" in result

    def test_empty_sentinel_returns_empty(self):
        llm = self._make_llm("EMPTY")
        writer = _make_writer(llm=llm)
        result = writer._agentic_diagram(["src/foo.py"], "Foo", "class")
        assert result == ""

    def test_no_mermaid_block_returns_empty(self):
        llm = self._make_llm("Here is some prose without a diagram.")
        writer = _make_writer(llm=llm)
        result = writer._agentic_diagram(["src/foo.py"], "Foo", "class")
        assert result == ""

    def test_wrong_header_rejected(self):
        """classDiagram expected but LLM returns graph TD → rejected."""
        bad = "```mermaid\ngraph TD\n    A --> B\n```"
        llm = self._make_llm(bad)
        writer = _make_writer(llm=llm)
        result = writer._agentic_diagram(["src/foo.py"], "Foo", "class")
        assert result == ""

    def test_llm_none_returns_empty(self):
        writer = _make_writer(llm=None)
        result = writer._agentic_diagram(["src/foo.py"], "Foo", "class")
        assert result == ""

    def test_llm_exception_returns_empty(self):
        llm = MagicMock()
        llm.bind_tools = MagicMock(side_effect=RuntimeError("no tools"))
        writer = _make_writer(llm=llm)
        result = writer._agentic_diagram(["src/foo.py"], "Foo", "class")
        assert result == ""


# ── _inject_cluster_diagrams ───────────────────────────────────────────────────


class TestInjectClusterDiagrams:
    """_inject_cluster_diagrams: assembles sections from _agentic_diagram calls."""

    _ARCH = "```mermaid\nstateDiagram-v2\n    [*] --> Idle\n    Idle --> Running\n    Running --> [*]\n```"
    _CLASS = VALID_CLASS_DIAGRAM
    _ER = VALID_ER_DIAGRAM

    def _writer_with_agentic(self, arch="", class_d="", data_d=""):
        """Writer whose _agentic_diagram is mocked to return preset diagrams."""
        storage = MagicMock()
        storage.get_nodes_by_cluster.return_value = [
            _code_node("n1", "Foo", rel_path="src/foo.py")
        ]
        storage.get_nodes_by_path_prefix.return_value = [
            {"symbol_name": "Foo", "macro_cluster": 1, "rel_path": "src/foo.py"}
        ]
        writer = _make_writer(storage=storage, llm=MagicMock())

        def _fake_agentic(files, title, diagram_type):
            return {"architecture": arch, "class": class_d, "datamodel": data_d}.get(diagram_type, "")

        writer._agentic_diagram = _fake_agentic
        return writer

    def test_no_diagrams_returns_original(self):
        writer = self._writer_with_agentic()
        original = "## Overview\nFoo.\n\n## Key Components\n- Foo"
        result = writer._inject_cluster_diagrams(original, cluster_id=1, chapter_title="Foo")
        assert result == original

    def test_arch_diagram_injected_before_key_components(self):
        writer = self._writer_with_agentic(arch=self._ARCH)
        md = "## Overview\nText.\n\n## Key Components\n- Foo"
        result = writer._inject_cluster_diagrams(md, cluster_id=1, chapter_title="Foo")
        assert "## Architecture" in result
        assert result.find("## Architecture") < result.find("## Key Components")

    def test_sections_injected_before_sub_pages_when_no_key_components(self):
        writer = self._writer_with_agentic(arch=self._ARCH)
        md = "## Overview\nText.\n\n## Sub-pages\n- [[Foo]]"
        result = writer._inject_cluster_diagrams(md, cluster_id=1, chapter_title="Foo")
        assert "## Architecture" in result
        assert result.find("## Architecture") < result.find("## Sub-pages")

    def test_class_diagram_injected_when_available(self):
        writer = self._writer_with_agentic(class_d=self._CLASS)
        md = "## Overview\nText.\n\n## Key Components\n- Foo"
        result = writer._inject_cluster_diagrams(md, cluster_id=1, chapter_title="Foo")
        assert "## Class Structure" in result

    def test_all_three_sections_when_all_available(self):
        writer = self._writer_with_agentic(arch=self._ARCH, class_d=self._CLASS, data_d=self._ER)
        md = "## Overview\nText.\n\n## Key Components\n- Foo"
        result = writer._inject_cluster_diagrams(md, cluster_id=1, chapter_title="Foo")
        assert "## Architecture" in result
        assert "## Class Structure" in result
        assert "## Data Model" in result

    def test_no_storage_returns_original(self):
        writer = _make_writer(storage=None, llm=None)
        original = "## Overview\nFoo.\n\n## Key Components\n- Foo"
        result = writer._inject_cluster_diagrams(original, cluster_id=1, chapter_title="Foo")
        assert result == original

    def test_appended_to_end_when_no_marker_found(self):
        writer = self._writer_with_agentic(arch=self._ARCH)
        md = "## Overview\nJust overview text with no markers."
        result = writer._inject_cluster_diagrams(md, cluster_id=1, chapter_title="Foo")
        assert "## Architecture" in result
        assert "stateDiagram-v2" in result


# ── _validate_mermaid accepts classDiagram and erDiagram ──────────────────────


class TestValidateMermaidHeaders:
    """_validate_mermaid must accept classDiagram and erDiagram headers."""

    def test_class_diagram_is_valid(self):
        from app.core.wiki_content_writer.diagram_generator import _validate_mermaid
        diagram = "```mermaid\nclassDiagram\n    class Foo {\n        +bar() str\n    }\n```"
        ok, reason = _validate_mermaid(diagram)
        assert ok, f"Expected valid, got: {reason}"

    def test_er_diagram_is_valid(self):
        from app.core.wiki_content_writer.diagram_generator import _validate_mermaid
        diagram = "```mermaid\nerDiagram\n    User {\n        int id PK\n    }\n```"
        ok, reason = _validate_mermaid(diagram)
        assert ok, f"Expected valid, got: {reason}"

    def test_graph_lr_still_valid(self):
        from app.core.wiki_content_writer.diagram_generator import _validate_mermaid
        diagram = "```mermaid\ngraph LR\n    A[\"Node\"]\n```"
        ok, reason = _validate_mermaid(diagram)
        assert ok, f"Expected valid, got: {reason}"

    def test_invalid_header_rejected(self):
        from app.core.wiki_content_writer.diagram_generator import _validate_mermaid
        diagram = "```mermaid\nsequenceDiagram\n    A ->> B: Hello\n```"
        ok, reason = _validate_mermaid(diagram)
        assert not ok
