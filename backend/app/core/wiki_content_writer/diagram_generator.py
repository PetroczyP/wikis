"""Architecture diagram generator for wiki pages (#234).

Deterministic — no LLM calls.  Queries the storage graph and emits a
Mermaid ``graph LR`` block showing intra-cluster component relationships.

Public API
----------
``DiagramGenerator(storage)``
    ``generate_cluster_diagram(cluster_id, page_spec) -> str``
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Relationship types to include in the diagram (structural + call graph).
_INTRA_CLUSTER_REL_TYPES = {
    "calls", "imports", "references", "creates",
    "member_uses", "inheritance", "composition",
}


def _safe_id(name: str) -> str:
    """Convert a symbol name to a Mermaid-safe node identifier."""
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _sanitize_label(name: str) -> str:
    """Sanitize a symbol name for use as a Mermaid double-quoted node label.

    Mermaid double-quoted labels break on `"`, `#`, unmatched `<`/`>`, and
    raw HTML entities.  Strip or replace these before embedding in the diagram.
    """
    # Replace double quotes (would close the label string early)
    s = name.replace('"', "'")
    # Remove characters that confuse Mermaid's label parser
    s = re.sub(r"[<>{}|\\]", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s or _safe_id(name)


def _sanitize_member(text: str) -> str:
    """Sanitize a classDiagram member line.

    Preserves ``->`` return-type arrows (only strips bare ``<``/``>`` that
    would break Mermaid's parser, not the ``->`` sequence used for return types).
    """
    s = text.replace('"', "'")
    # Strip < not preceded by - and > not preceded by - (preserves -> arrows).
    s = re.sub(r"(?<!-)<|(?<!-)>", "", s)
    s = re.sub(r"[{}|\\]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or text


def _validate_mermaid(diagram: str) -> tuple[bool, str]:
    """Lightweight structural validation for a Mermaid diagram block.

    Returns (is_valid, error_reason).  Checks:
    - Fenced code block with mermaid tag
    - Valid diagram header (graph/flowchart, classDiagram, or erDiagram)
    - No obvious unclosed double-quotes
    """
    lines = diagram.strip().splitlines()
    if not lines or lines[0].strip() != "```mermaid":
        return False, "missing ```mermaid fence"
    if lines[-1].strip() != "```":
        return False, "unclosed code fence"

    body = [l for l in lines[1:-1] if l.strip()]
    if not body:
        return False, "empty diagram body"

    first = body[0].strip()
    if not re.match(
        r"^(graph|flowchart)\s+(LR|TD|TB|RL|BT)$"
        r"|^classDiagram$|^erDiagram$"
        r"|^stateDiagram(-v2)?$",
        first,
    ):
        return False, f"invalid header: {first!r}"

    for line in body[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        # Count unescaped double-quotes: should be even
        if stripped.count('"') % 2 != 0:
            return False, f"unmatched quote in: {stripped!r}"

    return True, ""


def _node_name(node: dict) -> str:
    """Extract display name from a node dict.

    The storage returns ``symbol_name`` (SQLite/Postgres real column); tests
    pass ``name`` (mocked dicts matching the spec).  Accept both.
    """
    return node.get("symbol_name") or node.get("name") or node["node_id"]


_MAX_REPO_CLUSTERS = 10
_MAX_LABEL_CHARS = 30

# Cross-cluster relationship types to include in the repo diagram.
_CROSS_CLUSTER_REL_TYPES = {
    "calls", "imports", "references", "creates",
    "member_uses", "inheritance", "composition",
}

# Symbol types considered "class-like" for classDiagram.
_CLASS_LIKE_TYPES = {"class", "interface", "struct", "enum", "trait"}

# Symbol types considered member-level (rendered inside the class body).
_MEMBER_TYPES = {"method", "function", "property", "field"}

# Source-text patterns that identify TRUE database/persistence model classes.
# Bug 3 fix: BaseModel and @dataclass are removed — they match Pydantic schemas
# (request/response DTOs) which are NOT ER data models. Only SQLAlchemy ORM
# constructs, Django models, TypeORM entities, GORM structs, and similar
# database-level declarations qualify.
_DATA_MODEL_PATTERNS = (
    "Column(",           # SQLAlchemy Column()
    "mapped_column(",    # SQLAlchemy 2.x mapped_column()
    "relationship(",     # SQLAlchemy relationship()
    "DeclarativeBase",   # SQLAlchemy DeclarativeBase
    "db.Model",          # Flask-SQLAlchemy
    "Table(",            # SQLAlchemy Table construct
    "@Entity",           # TypeORM / Java JPA entity
    "@Column",           # TypeORM / Java JPA column
    'gorm:"',            # Go GORM struct tag
)


class DiagramGenerator:
    def __init__(self, storage: Any, llm_client: Any = None) -> None:
        self.storage = storage
        self._llm = llm_client

    def generate_repo_diagram(
        self,
        cluster_ids: list[int],
        cluster_titles: dict[int, str],
    ) -> str:
        """Generate a Mermaid diagram showing cluster-level dependencies.

        Queries edges between clusters (cross-cluster edges only — intra-cluster
        edges are excluded).  Returns a ``graph LR`` flowchart where nodes are
        cluster titles (slugified, ≤30 chars) and edges represent cross-cluster
        dependencies (calls/imports/etc.).

        Parameters
        ----------
        cluster_ids:
            Ordered list of cluster IDs to include.  Capped at 10.
        cluster_titles:
            Mapping from cluster_id → human-readable title.

        Returns
        -------
        str
            Fenced Mermaid code block, or ``''`` on failure / empty input.
        """
        if self.storage is None:
            return ""
        if not cluster_ids:
            return ""

        # Cap at 10 clusters.
        selected = cluster_ids[:_MAX_REPO_CLUSTERS]

        try:
            return self._build_repo_diagram(selected, cluster_titles)
        except Exception as exc:
            logger.debug("generate_repo_diagram error: %s", exc)
            return ""

    def _build_repo_diagram(
        self,
        cluster_ids: list[int],
        cluster_titles: dict[int, str],
    ) -> str:
        """Internal: build the Mermaid graph LR for cluster-level deps.

        Raises on storage errors so the outer ``generate_repo_diagram`` can
        catch them and return '' gracefully.
        """
        # Build a mapping from cluster_id → set of node_ids in that cluster.
        # Let storage errors propagate — the outer method catches them.
        cluster_node_ids: dict[int, set[str]] = {}
        for cid in cluster_ids:
            nodes = self.storage.get_nodes_by_cluster(cid)
            cluster_node_ids[cid] = {n["node_id"] for n in nodes}

        # Invert: node_id → cluster_id (for quick lookup during edge scan).
        node_to_cluster: dict[str, int] = {}
        for cid, node_ids in cluster_node_ids.items():
            for nid in node_ids:
                node_to_cluster[nid] = cid

        # Collect cross-cluster edges: (src_cluster_id, tgt_cluster_id)
        seen: set[tuple[int, int]] = set()
        cross_edges: list[tuple[int, int]] = []

        for src_cid in cluster_ids:
            for src_nid in cluster_node_ids.get(src_cid, set()):
                try:
                    raw_edges = self.storage.get_edges_from(
                        src_nid, rel_types=list(_CROSS_CLUSTER_REL_TYPES)
                    )
                except Exception:
                    raw_edges = []
                for e in raw_edges:
                    tgt_nid = e.get("target_id", "")
                    tgt_cid = node_to_cluster.get(tgt_nid)
                    if tgt_cid is None or tgt_cid == src_cid:
                        continue  # skip intra-cluster or external
                    pair = (src_cid, tgt_cid)
                    if pair not in seen:
                        seen.add(pair)
                        cross_edges.append(pair)

        return _render_repo_clusters(cluster_ids, cluster_titles, cross_edges)

    def generate_cluster_diagram(self, cluster_id: int, page_spec: Any) -> str:
        """Return a ## Architecture section with Mermaid diagram + LLM context.

        The section includes:
        - A 2-3 sentence LLM-written explanation of the component relationships
          shown in the diagram (when an LLM client is provided).
        - The Mermaid graph block.

        Returns '' when no nodes are found for the cluster.
        """
        if self.storage is None:
            return ""

        try:
            mermaid = self._build_diagram(cluster_id)
        except Exception as exc:
            logger.debug("DiagramGenerator: cluster_id=%d error: %s", cluster_id, exc)
            return ""

        if not mermaid:
            return ""

        # Validate Mermaid syntax; ask LLM to fix if invalid.
        valid, reason = _validate_mermaid(mermaid)
        if not valid:
            logger.warning(
                "[DIAGRAM] cluster_id=%d invalid Mermaid (%s); attempting LLM fix",
                cluster_id, reason,
            )
            mermaid = self._fix_mermaid(mermaid, reason)
            if not mermaid:
                return ""

        # Build context: symbol names visible in the diagram
        try:
            nodes = self.storage.get_nodes_by_cluster(cluster_id)
            node_names = [_node_name(n) for n in _select_nodes(nodes)]
        except Exception:
            node_names = []

        explanation = self._explain(mermaid, node_names, page_spec)
        if explanation:
            return f"## Architecture\n\n{explanation}\n\n{mermaid}"
        return f"## Architecture\n\n{mermaid}"

    # ── Class diagram helper (internal) ────────────────────────────────────────

    def _build_class_diagram(self, cluster_id: int) -> str:
        all_nodes = self.storage.get_nodes_by_cluster(cluster_id)
        if not all_nodes:
            return ""

        # Split into class-like nodes and potential members.
        class_nodes = [
            n for n in all_nodes
            if (n.get("symbol_type") or n.get("kind", "")) in _CLASS_LIKE_TYPES
        ]
        if len(class_nodes) < 2:
            return ""

        class_node_ids: set[str] = {n["node_id"] for n in class_nodes}
        class_name_to_id: dict[str, str] = {
            _node_name(n): n["node_id"] for n in class_nodes
        }

        # Collect member nodes (methods, properties, fields) whose parent_symbol
        # matches one of the selected class nodes.
        # Bug 2 fix: DB stores parent_symbol as "module.ClassName"; normalize to
        # the short name (after the last dot) before matching against class_names.
        class_names: set[str] = set(class_name_to_id)
        member_map: dict[str, list[dict]] = {_node_name(n): [] for n in class_nodes}
        for n in all_nodes:
            if (n.get("symbol_type") or n.get("kind", "")) in _MEMBER_TYPES:
                parent = n.get("parent_symbol")
                if parent:
                    parent_short = parent.rsplit(".", 1)[-1] if "." in parent else parent
                    if parent_short in class_names:
                        member_map[parent_short].append(n)

        # Collect inheritance edges within the selected class nodes.
        inheritance_edges: list[tuple[str, str]] = []
        seen_inh: set[tuple[str, str]] = set()
        for n in class_nodes:
            nid = n["node_id"]
            try:
                raw = self.storage.get_edges_from(nid, rel_types=["inheritance"])
            except Exception:
                raw = []
            for e in raw:
                tgt_id = e.get("target_id", "")
                if tgt_id in class_node_ids:
                    pair = (nid, tgt_id)
                    if pair not in seen_inh:
                        seen_inh.add(pair)
                        inheritance_edges.append(pair)

        id_to_name: dict[str, str] = {n["node_id"]: _node_name(n) for n in class_nodes}

        return _render_class_diagram(class_nodes, member_map, inheritance_edges, id_to_name)

    # ── ER diagram helper (internal) ────────────────────────────────────────────

    def _build_data_model_diagram(self, cluster_id: int) -> str:
        all_nodes = self.storage.get_nodes_by_cluster(cluster_id)
        if not all_nodes:
            return ""

        # Filter to struct/class nodes that look like data models.
        model_nodes = [
            n for n in all_nodes
            if (n.get("symbol_type") or n.get("kind", "")) in {"class", "struct"}
            and _is_data_model(n)
        ]
        if len(model_nodes) < 2:
            return ""

        model_node_ids: set[str] = {n["node_id"] for n in model_nodes}

        # Collect composition/references edges between model nodes.
        rel_edges: list[tuple[str, str]] = []
        seen_rel: set[tuple[str, str]] = set()
        for n in model_nodes:
            nid = n["node_id"]
            try:
                raw = self.storage.get_edges_from(nid, rel_types=["composition", "references"])
            except Exception:
                raw = []
            for e in raw:
                tgt_id = e.get("target_id", "")
                if tgt_id in model_node_ids:
                    pair = (nid, tgt_id)
                    if pair not in seen_rel:
                        seen_rel.add(pair)
                        rel_edges.append(pair)

        id_to_name: dict[str, str] = {n["node_id"]: _node_name(n) for n in model_nodes}
        return _render_er_diagram(model_nodes, rel_edges, id_to_name)

    def generate_cluster_context_diagram(
        self,
        cluster_id: int,
        cluster_title: str,
        all_cluster_ids: list[int],
        cluster_titles: dict[int, str],
    ) -> str:
        """Generate a Mermaid graph LR showing the cluster as a subgraph.

        The cluster's own architectural nodes are rendered inside a ``subgraph``
        block.  When *all_cluster_ids* and *cluster_titles* are provided,
        cross-cluster edges are detected and neighbouring clusters are rendered
        as plain external nodes.

        Returns a fenced ``graph TD`` block, or ``''`` on empty cluster /
        storage error.
        """
        if self.storage is None:
            return ""
        try:
            return self._build_cluster_context_diagram(
                cluster_id, cluster_title, all_cluster_ids, cluster_titles
            )
        except Exception as exc:
            logger.debug(
                "generate_cluster_context_diagram cluster_id=%d error: %s", cluster_id, exc
            )
            return ""

    def _build_cluster_context_diagram(
        self,
        cluster_id: int,
        cluster_title: str,
        all_cluster_ids: list[int],
        cluster_titles: dict[int, str],
    ) -> str:
        nodes = self.storage.get_nodes_by_cluster(cluster_id)
        if not nodes:
            return ""

        nodes = _select_nodes(nodes)
        if not nodes:
            return ""

        node_ids: set[str] = {n["node_id"] for n in nodes}
        id_to_name: dict[str, str] = {n["node_id"]: _node_name(n) for n in nodes}

        # Collect intra-cluster edges.
        intra_edges: list[tuple[str, str]] = []
        seen_intra: set[tuple[str, str]] = set()

        # Build a node_id → external cluster_id lookup if cross-cluster info given.
        node_to_ext_cluster: dict[str, int] = {}
        if all_cluster_ids and cluster_titles:
            for ext_cid in all_cluster_ids:
                if ext_cid == cluster_id:
                    continue
                try:
                    ext_nodes = self.storage.get_nodes_by_cluster(ext_cid)
                except Exception:
                    ext_nodes = []
                for en in ext_nodes:
                    node_to_ext_cluster[en["node_id"]] = ext_cid

        cross_cluster_edges: list[tuple[str, int]] = []  # (src_node_id, tgt_cluster_id)
        seen_cross: set[tuple[str, int]] = set()

        for n in nodes:
            nid = n["node_id"]
            try:
                raw = self.storage.get_edges_from(nid, rel_types=list(_CROSS_CLUSTER_REL_TYPES))
            except Exception:
                raw = []
            for e in raw:
                tgt_id = e.get("target_id", "")
                if tgt_id in node_ids:
                    pair = (nid, tgt_id)
                    if pair not in seen_intra:
                        seen_intra.add(pair)
                        intra_edges.append(pair)
                elif tgt_id in node_to_ext_cluster:
                    ext_cid = node_to_ext_cluster[tgt_id]
                    cross_pair = (nid, ext_cid)
                    if cross_pair not in seen_cross:
                        seen_cross.add(cross_pair)
                        cross_cluster_edges.append(cross_pair)

        return _render_cluster_context(
            cluster_id=cluster_id,
            cluster_title=cluster_title,
            nodes=nodes,
            id_to_name=id_to_name,
            intra_edges=intra_edges,
            cross_cluster_edges=cross_cluster_edges,
            cluster_titles=cluster_titles,
        )

    def _fix_mermaid(self, mermaid: str, reason: str) -> str:
        """Ask the LLM to fix invalid Mermaid syntax. Returns fixed diagram or ''."""
        if self._llm is None:
            return ""
        try:
            from langchain_core.messages import HumanMessage  # noqa: PLC0415
            prompt = (
                f"The following Mermaid diagram has a syntax error: {reason}\n\n"
                f"{mermaid}\n\n"
                f"Fix the Mermaid syntax so it renders correctly. "
                f"Return ONLY the corrected ```mermaid...``` block, nothing else. "
                f"Keep the same nodes and edges — only fix syntax issues."
            )
            response = self._llm.invoke([HumanMessage(content=prompt)])
            fixed = response.content if hasattr(response, "content") else str(response)
            # Extract the mermaid block from the response
            m = re.search(r"```mermaid\n.*?```", fixed, re.DOTALL)
            if m:
                candidate = m.group(0)
                ok, _ = _validate_mermaid(candidate)
                if ok:
                    return candidate
            logger.debug("[DIAGRAM] LLM fix did not produce valid Mermaid")
            return ""
        except Exception as exc:
            logger.debug("[DIAGRAM] LLM fix call failed: %s", exc)
            return ""

    def _explain(self, mermaid: str, node_names: list[str], page_spec: Any) -> str:
        """Ask the LLM for a 2-3 sentence explanation of the diagram."""
        if self._llm is None:
            return ""
        try:
            from langchain_core.messages import HumanMessage  # noqa: PLC0415
            title = getattr(page_spec, "title", "") or (
                page_spec.get("title", "") if isinstance(page_spec, dict) else ""
            )
            prompt = (
                f"The following Mermaid diagram shows the architecture of "
                f'"{title}". '
                f"Components: {', '.join(node_names[:12])}.\n\n"
                f"{mermaid}\n\n"
                f"Write 2-3 sentences describing the key relationships and "
                f"architectural pattern shown. Be specific and concise. "
                f"Do not repeat the component names as a list."
            )
            response = self._llm.invoke([HumanMessage(content=prompt)])
            text = response.content if hasattr(response, "content") else str(response)
            return text.strip() if text and text.strip() else ""
        except Exception as exc:
            logger.debug("DiagramGenerator LLM explain failed: %s", exc)
            return ""

    # ── internals ──────────────────────────────────────────────────────────────

    def _build_diagram(self, cluster_id: int) -> str:
        nodes = self.storage.get_nodes_by_cluster(cluster_id)
        if not nodes:
            return ""

        nodes = _select_nodes(nodes)
        if not nodes:
            return ""
        node_ids: set[str] = {n["node_id"] for n in nodes}
        id_to_name: dict[str, str] = {n["node_id"]: _node_name(n) for n in nodes}

        seen_edges: set[tuple[str, str]] = set()
        edges: list[tuple[str, str]] = []
        for node in nodes:
            nid = node["node_id"]
            try:
                raw_edges = self.storage.get_edges_from(
                    nid, rel_types=list(_INTRA_CLUSTER_REL_TYPES)
                )
            except Exception:
                raw_edges = []
            for e in raw_edges:
                target = e.get("target_id", "")
                if target in node_ids:
                    pair = (nid, target)
                    if pair not in seen_edges:
                        seen_edges.add(pair)
                        edges.append(pair)

        return _render(id_to_name, edges)


# ── helpers ────────────────────────────────────────────────────────────────────


_DOC_TYPES: frozenset[str] = frozenset({
    "config_document",
    "markdown_document",
    "toml_document",
    "module_doc",
    "file_doc",
    "rst_document",
    "html_document",
    "asciidoc_document",
})


def _select_nodes(nodes: list[dict]) -> list[dict]:
    """Filter document-type nodes from a node list, keeping all code nodes."""
    return [
        n for n in nodes
        if (n.get("symbol_type") or n.get("kind", "")) not in _DOC_TYPES
    ]


def _render(id_to_name: dict[str, str], edges: list[tuple[str, str]]) -> str:
    """Build a Mermaid graph LR block from node names and edge pairs."""
    lines: list[str] = ["```mermaid", "graph LR"]

    # Emit node declarations — labels sanitized to prevent Mermaid parse errors.
    for nid, name in id_to_name.items():
        safe = _safe_id(name)
        label = _sanitize_label(name)
        lines.append(f'    {safe}["{label}"]')

    # Emit edges.
    for src_id, tgt_id in edges:
        src = _safe_id(id_to_name[src_id])
        tgt = _safe_id(id_to_name[tgt_id])
        lines.append(f"    {src} --> {tgt}")

    lines.append("```")
    return "\n".join(lines)


def _select_class_nodes(nodes: list[dict]) -> list[dict]:
    """Return all class-like nodes; prefer architectural ones when filtering."""
    return nodes


def _is_data_model(node: dict) -> bool:
    """Return True when a node's source_text looks like a data-model class."""
    source = node.get("source_text") or ""
    return any(pattern in source for pattern in _DATA_MODEL_PATTERNS)


def _is_abstract(node: dict) -> bool:
    """Heuristic: does the class appear to be abstract?"""
    src = node.get("source_text") or ""
    return "abstract" in src or "ABC" in src or "@abstractmethod" in src


def _method_visibility(name: str) -> str:
    """Return Mermaid visibility prefix: + public, - private."""
    if name.startswith("_"):
        return "-"
    return "+"


def _method_signature_line(member: dict) -> str:
    """Build a Mermaid member line like ``+process(x int) str``."""
    raw_name = _node_name(member)
    # Strip leading path segments (keep only the part after the last dot).
    display_name = raw_name.rsplit(".", 1)[-1]
    visibility = _method_visibility(display_name)

    sig = (member.get("signature") or "").strip()
    if sig:
        # Strip the leading 'def ' / 'func ' / 'fn ' keyword if present.
        sig = re.sub(r"^(def|func|fn|function)\s+", "", sig)
        # Strip leading path prefix from the signature name as well.
        sig = re.sub(r"^[A-Za-z0-9_.]+\.", "", sig)
        return f"{visibility}{_sanitize_member(sig)}"

    params = (member.get("parameters") or "").strip()
    ret = (member.get("return_type") or "").strip()
    if ret:
        return f"{visibility}{_sanitize_member(display_name)}({_sanitize_member(params)}) {_sanitize_member(ret)}"
    return f"{visibility}{_sanitize_member(display_name)}({_sanitize_member(params)})"


def _render_class_diagram(
    class_nodes: list[dict],
    member_map: dict[str, list[dict]],
    inheritance_edges: list[tuple[str, str]],
    id_to_name: dict[str, str],
) -> str:
    """Build a Mermaid classDiagram block."""
    lines: list[str] = ["```mermaid", "classDiagram"]

    for n in class_nodes:
        cname = _node_name(n)
        sym_type = n.get("symbol_type") or n.get("kind", "class")
        lines.append(f"    class {_safe_id(cname)} {{")
        # Stereotype annotation.
        if sym_type == "interface":
            lines.append(f"        <<interface>>")
        elif sym_type == "enum":
            lines.append(f"        <<enumeration>>")
        elif _is_abstract(n):
            lines.append(f"        <<abstract>>")
        # Member lines.
        for member in member_map.get(cname, []):
            lines.append(f"        {_method_signature_line(member)}")
        lines.append("    }")

    # Inheritance edges: Child --|> Parent
    for src_id, tgt_id in inheritance_edges:
        src_name = _safe_id(id_to_name[src_id])
        tgt_name = _safe_id(id_to_name[tgt_id])
        lines.append(f"    {src_name} --|> {tgt_name} : implements")

    lines.append("```")
    return "\n".join(lines)


def _render_er_diagram(
    model_nodes: list[dict],
    rel_edges: list[tuple[str, str]],
    id_to_name: dict[str, str],
) -> str:
    """Build a Mermaid erDiagram block."""
    lines: list[str] = ["```mermaid", "erDiagram"]

    for n in model_nodes:
        entity_name = _safe_id(_node_name(n))
        lines.append(f"    {entity_name} {{")
        lines.append("    }")

    for src_id, tgt_id in rel_edges:
        src = _safe_id(id_to_name[src_id])
        tgt = _safe_id(id_to_name[tgt_id])
        lines.append(f'    {src} ||--o{{ {tgt} : "has"')

    lines.append("```")
    return "\n".join(lines)


def _render_cluster_context(
    cluster_id: int,
    cluster_title: str,
    nodes: list[dict],
    id_to_name: dict[str, str],
    intra_edges: list[tuple[str, str]],
    cross_cluster_edges: list[tuple[str, int]],
    cluster_titles: dict[int, str],
) -> str:
    """Build a Mermaid graph TD with a subgraph for the cluster.

    Only nodes that participate in at least one edge are rendered — isolated
    nodes add visual clutter without conveying relationships.  Returns '' when
    fewer than 2 connected nodes exist (nothing useful to show).
    """
    # Collect node IDs that appear in at least one edge.
    connected: set[str] = set()
    for src_id, tgt_id in intra_edges:
        connected.add(src_id)
        connected.add(tgt_id)
    for src_nid, _ in cross_cluster_edges:
        connected.add(src_nid)

    # Filter to connected nodes only; bail if fewer than 2.
    connected_nodes = [n for n in nodes if n["node_id"] in connected]
    if len(connected_nodes) < 2 and not cross_cluster_edges:
        return ""

    lines: list[str] = ["```mermaid", "graph TD"]

    safe_title = _sanitize_label(cluster_title)
    lines.append(f'    subgraph "{safe_title}"')
    for n in connected_nodes:
        name = _node_name(n)
        safe = _safe_id(name)
        label = _sanitize_label(name)
        lines.append(f'        {safe}["{label}"]')
    lines.append("    end")

    # External cluster nodes (one node per neighbouring cluster).
    # Build a stable cid → safe_id map first to ensure edges reference the
    # same de-collided ID used in the node declaration.
    internal_safe_ids: set[str] = {_safe_id(_node_name(n)) for n in nodes}
    ext_cid_to_safe: dict[int, str] = {}
    seen_ext_safe: set[str] = set(internal_safe_ids)
    seen_ext: set[int] = set()
    for _src_nid, ext_cid in cross_cluster_edges:
        if ext_cid not in seen_ext:
            seen_ext.add(ext_cid)
            ext_title = cluster_titles.get(ext_cid) or f"cluster_{ext_cid}"
            ext_safe = _safe_id(ext_title[:_MAX_LABEL_CHARS])
            # Guarantee uniqueness against internal nodes and other external nodes.
            if ext_safe in seen_ext_safe:
                ext_safe = f"{ext_safe}_{ext_cid}"
            seen_ext_safe.add(ext_safe)
            ext_cid_to_safe[ext_cid] = ext_safe
            ext_label = _sanitize_label(ext_title[:_MAX_LABEL_CHARS])
            lines.append(f'    {ext_safe}["{ext_label}"]')

    # Intra-cluster edges.
    for src_id, tgt_id in intra_edges:
        src = _safe_id(id_to_name[src_id])
        tgt = _safe_id(id_to_name[tgt_id])
        lines.append(f"    {src} --> {tgt}")

    # Cross-cluster edges: source node → external cluster node.
    for src_nid, ext_cid in cross_cluster_edges:
        src_name = id_to_name.get(src_nid)
        ext_safe = ext_cid_to_safe.get(ext_cid)
        if src_name is None or ext_safe is None:
            continue
        src_safe = _safe_id(src_name)
        lines.append(f"    {src_safe} --> {ext_safe}")

    lines.append("```")
    return "\n".join(lines)


def _render_repo_clusters(
    cluster_ids: list[int],
    cluster_titles: dict[int, str],
    cross_edges: list[tuple[int, int]],
) -> str:
    """Build a Mermaid graph LR block for the repo-level cluster diagram.

    Node identifiers are derived from the cluster title (slugified).
    Labels are truncated to ``_MAX_LABEL_CHARS`` (30) chars so no node label
    overflows Mermaid's rendering box.

    When there are no clusters, returns ``''``.
    """
    if not cluster_ids:
        return ""

    lines: list[str] = ["```mermaid", "graph LR"]

    # Map cluster_id → (safe_node_id, display_label)
    cid_to_safe: dict[int, str] = {}
    for cid in cluster_ids:
        raw_title = cluster_titles.get(cid) or f"cluster_{cid}"
        # Truncate label to _MAX_LABEL_CHARS before sanitizing.
        label = _sanitize_label(raw_title[:_MAX_LABEL_CHARS])
        safe = _safe_id(raw_title[:_MAX_LABEL_CHARS])
        # Guarantee uniqueness by appending cluster id when safe id collides.
        if safe in cid_to_safe.values():
            safe = f"{safe}_{cid}"
        cid_to_safe[cid] = safe
        lines.append(f'    {safe}["{label}"]')

    # Emit cross-cluster edges.
    for src_cid, tgt_cid in cross_edges:
        src_safe = cid_to_safe.get(src_cid)
        tgt_safe = cid_to_safe.get(tgt_cid)
        if src_safe and tgt_safe:
            lines.append(f"    {src_safe} --> {tgt_safe}")

    lines.append("```")
    return "\n".join(lines)
