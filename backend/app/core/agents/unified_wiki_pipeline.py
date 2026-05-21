"""Unified wiki pipeline orchestrator (#243, default path since #242).

Wires the unified planner → writer → source-kind gate into a single
end-to-end pipeline.  This is the sole structure-planning path; the
legacy graph-first, deepagents, and cluster-based paths were removed
in #242.  The citation verifier was removed in #318.

Public API
----------
``run_unified_pipeline(skeleton, evidence_packs, *, llm, repo_root,
                        name_index, stream_callback)``
    → ``UnifiedPipelineResult``

``UnifiedPipelineResult``
    ``pages``          – list of ``GeneratedPage`` (final, post-gate)
    ``plan_report``    – ``PlanReport`` from the unified planner
    ``gate_reports``   – ``dict[int, GateReport]`` keyed by ``page_id``

``GeneratedPage``
    ``page_id``        – ``cluster_id`` from the originating ``PageSpec``
    ``title``          – page title (from planner or fallback)
    ``description``    – 1-2 sentence description
    ``markdown``       – final stripped markdown
    ``cited_claims``   – kept ``CitedClaim`` list (post-gate)
    ``metadata``       – dict with target_* fields + gate flags
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel

from app.core.wiki_content_writer.citation_extractor import CitedClaim, extract_citations
from app.core.wiki_content_writer.source_gate import GateReport, ToolCall, apply_gate
from app.core.wiki_content_writer.writer_agent import WikiContentWriter
from app.core.wiki_content_writer.writer_tools import WriterTools
from app.core.wiki_structure_planner.evidence import EvidencePack, EvidenceSlice
from app.core.feature_flags import get_feature_flags
from app.core.wiki_structure_planner.planner_agent import (
    ChapterSpec,
    PageSpec,
    PlanReport,
    run_planner_chaptered,
    run_planner_with_report,
)
from app.core.wiki_structure_planner.structure_skeleton import StructureSkeleton

logger = logging.getLogger(__name__)

# Default gate mode: README + SQL strip, identifier is flagged only.
_DEFAULT_GATE_MODE: dict[str, str] = {
    "readme": "strip",
    "sql": "strip",
    "identifier": "flag",
}


# ── Result dataclasses ────────────────────────────────────────────────────────


@dataclass
class GeneratedPage:
    """A single wiki page produced by the unified pipeline.

    Attributes
    ----------
    page_id : int
        Cluster ID of the originating ``PageSpec``.
    title : str
        Page title from the unified planner (or deterministic fallback).
    description : str
        1-2 sentence description.
    markdown : str
        Final markdown content after gate and verifier stripping.
    cited_claims : list[CitedClaim]
        Kept ``CitedClaim`` objects (post-gate) for downstream use.
    metadata : dict
        Carries ``target_symbols``, ``target_folders``, ``target_docs`` from
        the skeleton plus ``gate_flags_total``, ``gate_stripped_total``
        for observability.
    """

    page_id: int
    title: str
    description: str
    markdown: str
    cited_claims: list[CitedClaim] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedPipelineResult:
    """Result of :func:`run_unified_pipeline`.

    Attributes
    ----------
    pages : list[GeneratedPage]
        Final generated pages (one per cluster).
    plan_report : PlanReport
        Budget and coverage summary from the unified planner.
    gate_reports : dict[int, GateReport]
        Per-page gate reports keyed by ``cluster_id``.
    """

    pages: list[GeneratedPage]
    plan_report: PlanReport
    gate_reports: dict[int, GateReport] = field(default_factory=dict)


# (stub removed in #285 — real writer agent is WikiContentWriter.generate_page)


# ── Per-page pipeline ─────────────────────────────────────────────────────────


def _run_page(
    spec: PageSpec,
    *,
    writer: WikiContentWriter,
    repo_root: str,
    name_index: set[str] | None,
    name_to_signatures: dict[str, set[str]] | None = None,
    cluster_file_map: dict[str, int] | None = None,
    gate_mode: dict[str, str],
    stream_callback: Callable[[dict], None] | None,
    storage: Any = None,
    verifier_llm: "BaseChatModel | None" = None,
) -> tuple[GeneratedPage, GateReport]:
    """Run the writer → gate pipeline for one ``PageSpec``.

    Args:
        spec: Planner output for this page.
        writer: Pre-constructed WikiContentWriter (shared across pages).
        repo_root: Absolute path to the checked-out repository root.
        name_index: Set of known symbol names for identifier grounding.
        name_to_signatures: Optional class → {method names} mapping for dotted
            method/attribute grounding (Rule 3 extension, #315).
        cluster_file_map: Optional file path → cluster_id mapping for cross-
            cluster reference detection (Rule 4, #316).
        gate_mode: Per-rule gate actions (passed through to ``apply_gate``).
        stream_callback: Optional SSE event emitter.

    Returns:
        ``(page, gate_report)``
    """
    page_id = spec.cluster_id

    # ── 1. Emit writer.page_started ───────────────────────────────────
    if stream_callback:
        try:
            stream_callback({
                "event": "writer.page_started",
                "page_id": page_id,
                "title": spec.title,
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (writer.page_started): %s", cb_exc)

    # ── 2. Generate content ───────────────────────────────────────────
    logger.info("[UNIFIED] writer: page_id=%d title=%r", page_id, spec.title)
    raw_markdown, tool_trace = writer.generate_page(spec, stream_callback=stream_callback)

    # ── 3. Emit writer.page_done, then writer.zero_citations if applicable ────
    cited_claims_raw = extract_citations(raw_markdown)

    # page_done fires first — SPA/SSE consumers treat it as the completion signal.
    if stream_callback:
        try:
            stream_callback({
                "event": "writer.page_done",
                "page_id": page_id,
                "n_paragraphs": len(cited_claims_raw),
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (writer.page_done): %s", cb_exc)

    # Zero-citation alarm fires after page_done as a follow-up operator signal.
    if not cited_claims_raw or not any(c.citations for c in cited_claims_raw):
        logger.warning(
            "[UNIFIED] writer produced zero citations for page_id=%d title=%r",
            page_id,
            spec.title,
        )
        if stream_callback:
            try:
                stream_callback({"event": "writer.zero_citations", "page_id": page_id, "title": spec.title})
            except Exception:
                pass

    # ── 4. Source-kind gate ───────────────────────────────────────────
    logger.info(
        "[UNIFIED] gate: page_id=%d paragraphs=%d",
        page_id,
        len(cited_claims_raw),
    )
    kept_after_gate, gate_report = apply_gate(
        cited_claims_raw,
        tool_trace,
        known_symbols=name_index,
        name_to_signatures=name_to_signatures,
        cluster_file_map=cluster_file_map,
        current_cluster_id=spec.cluster_id,
        mode=gate_mode,
    )

    n_flagged = len(gate_report.flags)
    n_gate_stripped = len(gate_report.stripped_paragraphs)
    logger.info(
        "[UNIFIED] gate done: page_id=%d flagged=%d stripped=%d",
        page_id,
        n_flagged,
        n_gate_stripped,
    )

    if n_gate_stripped:
        _emit_stripped(
            stream_callback,
            event="gate.page_done",
            page_id=page_id,
            n_flagged=n_flagged,
            n_stripped=n_gate_stripped,
        )
    else:
        if stream_callback:
            try:
                stream_callback({
                    "event": "gate.page_done",
                    "page_id": page_id,
                    "n_flagged": n_flagged,
                    "n_stripped": 0,
                })
            except Exception as cb_exc:
                logger.debug("stream_callback error (gate.page_done): %s", cb_exc)

    kept_after_verify = kept_after_gate

    # ── 5. Reconstruct final markdown from kept paragraphs ────────────
    final_markdown = _claims_to_markdown(kept_after_verify)

    # ── 5b. Insert architecture diagram after the Overview section ────
    # The Overview paragraph (first ## section) should come first;
    # the diagram follows it so readers get context before the visual.
    if storage is not None:
        import re as _re  # noqa: PLC0415

        from app.core.wiki_content_writer.diagram_generator import DiagramGenerator  # noqa: PLC0415
        # Use verifier_llm (low-tier) for the diagram explanation — cheap call.
        dg = DiagramGenerator(storage, llm_client=verifier_llm)
        diagram_section = dg.generate_cluster_diagram(page_id, spec)
        if diagram_section:
            # Find the second ## heading — that marks the end of the first
            # (Overview) section.  Insert the diagram just before it.
            second_heading = _re.search(r'\n##\s+', final_markdown)
            if second_heading:
                insert_at = second_heading.start()
                final_markdown = (
                    final_markdown[:insert_at]
                    + "\n\n" + diagram_section + "\n"
                    + final_markdown[insert_at:]
                )
            else:
                # No second heading — insert after the first blank-line gap
                # so the opening paragraph stays on top.
                first_blank = final_markdown.find('\n\n')
                if first_blank > 0:
                    final_markdown = (
                        final_markdown[:first_blank]
                        + "\n\n" + diagram_section
                        + final_markdown[first_blank:]
                    )
                else:
                    final_markdown = final_markdown + "\n\n" + diagram_section

    # ── 7. Format for human readability ───────────────────────────────
    try:
        from app.core.wiki_content_writer.page_formatter import PageFormatter  # noqa: PLC0415
        formatter = PageFormatter(llm_client=verifier_llm)
        formatted = formatter.format_page(final_markdown, page_spec=spec)
        if formatted and formatted.strip():
            final_markdown = formatted
    except Exception as fmt_exc:
        logger.warning("[UNIFIED] formatter failed: %s", fmt_exc)
        # Keep unformatted markdown — formatter is enhancement, not required

    if stream_callback:
        try:
            stream_callback({"event": "formatter.page_done", "page_id": page_id})
        except Exception:
            pass

    page = GeneratedPage(
        page_id=page_id,
        title=spec.title,
        description=spec.description,
        markdown=final_markdown,
        cited_claims=kept_after_verify,
        metadata={
            "target_symbols": spec.target_symbols,
            "target_folders": spec.target_folders,
            "target_docs": spec.target_docs,
            "gate_flags_total": n_flagged,
            "gate_stripped_total": n_gate_stripped,
        },
    )
    return page, gate_report


# ── Markdown reconstruction ───────────────────────────────────────────────────


def _claims_to_markdown(claims: list[CitedClaim]) -> str:
    """Reconstruct markdown from kept ``CitedClaim`` objects.

    Paragraphs are joined with a blank line between them (CommonMark paragraph
    separator).  If the claim list is empty, an empty string is returned.
    """
    if not claims:
        return ""
    return "\n\n".join(c.paragraph_text for c in claims)


# ── Audit event helper ────────────────────────────────────────────────────────


def _emit_stripped(
    stream_callback: Callable[[dict], None] | None,
    *,
    event: str,
    page_id: int,
    **kwargs: Any,
) -> None:
    """Emit a stripped-paragraph audit event via *stream_callback*.

    Swallows all exceptions so audit events never break the pipeline.
    """
    if stream_callback is None:
        return
    payload: dict[str, Any] = {"event": event, "page_id": page_id, **kwargs}
    try:
        stream_callback(payload)
    except Exception as exc:
        logger.debug("stream_callback error (%s): %s", event, exc)


# ── Public entrypoint ─────────────────────────────────────────────────────────


def run_unified_pipeline(
    skeleton: StructureSkeleton,
    evidence_packs: dict[int, EvidencePack],
    *,
    llm: BaseChatModel,
    verifier_llm: BaseChatModel | None = None,  # used by PageFormatter (Pass 2) since #320
    repo_root: str,
    storage: Any | None = None,
    name_index: set[str] | None = None,
    name_to_signatures: dict[str, set[str]] | None = None,
    cluster_file_map: dict[str, int] | None = None,
    gate_mode: dict[str, str] | None = None,
    stream_callback: Callable[[dict], None] | None = None,
) -> UnifiedPipelineResult:
    """Run the unified planner → writer → gate pipeline.

    This is the single public entrypoint for the unified wiki generation
    pipeline.  It is deliberately synchronous — async orchestration is the
    responsibility of the caller (``OptimizedWikiGenerationAgent`` or a test
    harness).

    Parameters
    ----------
    skeleton :
        Completed ``StructureSkeleton`` with ``clusters`` (preferred) or
        ``code_clusters`` (backwards-compat fallback).
    evidence_packs :
        Pre-built evidence packs keyed by ``cluster_id``.  Missing packs
        cause the planner to use a fallback spec with a note in the report.
    llm :
        Primary LangChain ``BaseChatModel`` used for both the planner and the
        writer content generation.
    verifier_llm :
        Low-tier LLM used by ``PageFormatter`` (Pass 2 section reorganisation).
        Pass ``None`` to skip the LLM formatting pass.
    repo_root :
        Absolute path to the checked-out repository root.  Passed to the
        planner (tool IO) and the gate (identifier grounding).
    name_index :
        Set of known symbol names from the code graph.  Used by the source-
        kind gate (rule 3 — identifier grounding).  Pass ``None`` to rely
        solely on the tool trace for grounding.
    name_to_signatures :
        Optional class → {method/attribute names} mapping built from
        ``repo_nodes.parent_symbol``.  Enables the gate Rule 3 extension
        (#315) which flags invented method calls on real classes.  Pass
        ``None`` (default) to skip dotted-method grounding.
    cluster_file_map :
        Optional mapping of repo-relative file path → cluster_id.  When
        provided, enables gate Rule 4 (``cross_cluster_reference``, #316):
        paragraphs citing files from a different cluster than the page being
        written are flagged.  Default action is ``"warn"`` (non-destructive).
        Pass ``None`` (default) to skip cross-cluster detection.
    gate_mode :
        Per-rule action override for :func:`apply_gate`.  Defaults to
        ``{"readme": "strip", "sql": "strip", "identifier": "flag"}``.
    stream_callback :
        Optional function called with a ``dict`` event payload for each
        pipeline milestone.  Events: ``planner.started``, ``planner.done``,
        ``writer.page_started``, ``writer.page_done``, ``gate.page_done``.
        All callbacks are wrapped in try/except so a misbehaving callback
        never breaks the pipeline.

    Returns
    -------
    UnifiedPipelineResult
        ``pages`` — one ``GeneratedPage`` per cluster (empty when skeleton
        has no clusters).
        ``plan_report`` — planner budget / coverage summary.
        ``gate_reports`` — per-page gate reports (empty dict when no pages).

    Raises
    ------
    ValueError
        If *repo_root* is not a valid directory (propagated from the planner).
    """
    effective_gate_mode = gate_mode if gate_mode is not None else _DEFAULT_GATE_MODE

    # ── Planner phase ─────────────────────────────────────────────────
    if stream_callback:
        try:
            stream_callback({"event": "planner.started"})
        except Exception as cb_exc:
            logger.debug("stream_callback error (planner.started): %s", cb_exc)

    logger.info(
        "[UNIFIED] planner: clusters=%d evidence_packs=%d",
        len(skeleton.clusters or skeleton.code_clusters),
        len(evidence_packs),
    )

    flags = get_feature_flags()
    if flags.chaptered_pages:
        # Chaptered path (#290): generate_chapter runs N sub-pages in parallel
        # + synthesises a chapter index with [[wikilink]] TOC.
        chapters: list[ChapterSpec]
        chapter_evidence: dict[int, list[EvidenceSlice]]
        chapters, chapter_evidence = run_planner_chaptered(
            skeleton, evidence_packs, llm=llm, repo_root=repo_root, storage=storage,
        )

        # Build the plan_report before entering the writer phase.
        n_subpages_total = sum(len(ch.subpages) for ch in chapters)
        plan_report = PlanReport(
            tool_budget_total=3 * len(chapters),
            tool_budget_used=0,
            tool_budget_exceeded=False,
            # Each chapter emits 1 index page + N sub-pages.
            pages_emitted=len(chapters) + n_subpages_total,
            notes=[],
        )
        logger.info(
            "[UNIFIED] chaptered: %d chapters, %d clusters with evidence slices",
            len(chapters),
            len(chapter_evidence),
        )

        if stream_callback:
            try:
                stream_callback({
                    "event": "planner.done",
                    "pages_emitted": plan_report.pages_emitted,
                    "tool_budget_used": plan_report.tool_budget_used,
                    "tool_budget_total": plan_report.tool_budget_total,
                    "tool_budget_exceeded": plan_report.tool_budget_exceeded,
                    "notes": plan_report.notes,
                })
            except Exception as cb_exc:
                logger.debug("stream_callback error (planner.done chaptered): %s", cb_exc)

        # Build writer once — shared across all chapters.
        chapter_writer = WikiContentWriter(
            tools=WriterTools(repo_root=repo_root, storage=storage),
            llm_client=llm,
        )

        chapter_pages: list[GeneratedPage] = []
        chapter_gate_reports: dict[int, GateReport] = {}

        for ch in chapters:
            slices = chapter_evidence.get(ch.cluster_id, [])
            if len(slices) < len(ch.subpages):
                logger.warning(
                    "[UNIFIED] chapter cluster_id=%d has %d subpages but only %d slices; "
                    "trailing subpages will use empty evidence",
                    ch.cluster_id, len(ch.subpages), len(slices),
                )
            try:
                # Use new_event_loop / run_until_complete / close instead of
                # asyncio.run() — safe both inside and outside a running loop.
                _loop = asyncio.new_event_loop()
                try:
                    index_md, subpage_mds = _loop.run_until_complete(
                        chapter_writer.generate_chapter(
                            ch, slices, stream_callback=stream_callback,
                        )
                    )
                finally:
                    _loop.close()
            except Exception as exc:
                logger.error(
                    "[UNIFIED] generate_chapter failed for cluster_id=%d title=%r: %s",
                    ch.cluster_id,
                    ch.chapter_title,
                    exc,
                    exc_info=True,
                )
                index_md = f"# {ch.chapter_title}\n\n{ch.chapter_description}\n"
                subpage_mds = [
                    f"# {sp.title}\n\n{sp.description}\n"
                    for sp in ch.subpages
                ]

            # Chapter index page
            chapter_pages.append(GeneratedPage(
                page_id=ch.cluster_id,
                title=ch.chapter_title,
                description=ch.chapter_description,
                markdown=index_md,
                metadata={"is_chapter_index": True},
            ))
            chapter_gate_reports[ch.cluster_id] = GateReport()

            # Sub-pages — zip_longest so length mismatch produces a fallback
            # markdown rather than silently dropping trailing sub-pages.
            import itertools  # noqa: PLC0415
            _fallback_md = lambda sp: f"# {sp.title}\n\n{sp.description}\n"  # noqa: E731
            for i, (sp, md) in enumerate(itertools.zip_longest(
                ch.subpages,
                subpage_mds,
                fillvalue=None,
            )):
                if sp is None:
                    continue  # extra markdown with no subpage spec — skip
                if md is None:
                    logger.warning(
                        "[UNIFIED] no markdown for subpage %r cluster_id=%d — using fallback",
                        sp.title, ch.cluster_id,
                    )
                    md = _fallback_md(sp)
                sub_page_id = ch.cluster_id * 1000 + i
                chapter_pages.append(GeneratedPage(
                    page_id=sub_page_id,
                    title=sp.title,
                    description=sp.description,
                    markdown=md,
                    metadata={"chapter_cluster_id": ch.cluster_id, "subpage_index": i},
                ))
                chapter_gate_reports[sub_page_id] = GateReport()

        logger.info(
            "[UNIFIED] chaptered pipeline complete: chapters=%d pages=%d",
            len(chapters),
            len(chapter_pages),
        )

        return UnifiedPipelineResult(
            pages=chapter_pages,
            plan_report=plan_report,
            gate_reports=chapter_gate_reports,
        )
    else:
        page_specs, plan_report = run_planner_with_report(
            skeleton,
            evidence_packs,
            llm=llm,
            repo_root=repo_root,
        )

    logger.info(
        "[UNIFIED] planner done: pages=%d tool_calls=%d/%d exceeded=%s",
        plan_report.pages_emitted,
        plan_report.tool_budget_used,
        plan_report.tool_budget_total,
        plan_report.tool_budget_exceeded,
    )

    if stream_callback:
        try:
            stream_callback({
                "event": "planner.done",
                "pages_emitted": plan_report.pages_emitted,
                "tool_budget_used": plan_report.tool_budget_used,
                "tool_budget_total": plan_report.tool_budget_total,
                "tool_budget_exceeded": plan_report.tool_budget_exceeded,
                "notes": plan_report.notes,
            })
        except Exception as cb_exc:
            logger.debug("stream_callback error (planner.done): %s", cb_exc)

    if not page_specs:
        logger.info("[UNIFIED] no page specs produced — returning empty result")
        return UnifiedPipelineResult(
            pages=[],
            plan_report=plan_report,
            gate_reports={},
        )

    # ── Per-page writer → gate ────────────────────────────────────────
    pages: list[GeneratedPage] = []
    gate_reports: dict[int, GateReport] = {}

    # Build writer once — shared tools and LLM across all pages.
    writer = WikiContentWriter(
        tools=WriterTools(repo_root=repo_root, storage=storage),
        llm_client=llm,
    )

    for spec in page_specs:
        try:
            page, gate_report = _run_page(
                spec,
                writer=writer,
                repo_root=repo_root,
                name_index=name_index,
                name_to_signatures=name_to_signatures,
                cluster_file_map=cluster_file_map,
                gate_mode=effective_gate_mode,
                stream_callback=stream_callback,
                storage=storage,
                verifier_llm=verifier_llm,
            )
            pages.append(page)
            gate_reports[spec.cluster_id] = gate_report
        except Exception as exc:
            logger.error(
                "[UNIFIED] page pipeline failed for cluster_id=%d title=%r: %s",
                spec.cluster_id,
                spec.title,
                exc,
                exc_info=True,
            )
            # Emit a minimal page so the caller always gets one entry per spec.
            fallback_markdown = f"# {spec.title}\n\n{spec.description}\n"
            pages.append(
                GeneratedPage(
                    page_id=spec.cluster_id,
                    title=spec.title,
                    description=spec.description,
                    markdown=fallback_markdown,
                    metadata={
                        "target_symbols": spec.target_symbols,
                        "target_folders": spec.target_folders,
                        "target_docs": spec.target_docs,
                        "pipeline_error": str(exc),
                    },
                )
            )
            gate_reports[spec.cluster_id] = GateReport()

    logger.info(
        "[UNIFIED] pipeline complete: pages=%d gate_stripped_total=%d",
        len(pages),
        sum(len(r.stripped_paragraphs) for r in gate_reports.values()),
    )

    return UnifiedPipelineResult(
        pages=pages,
        plan_report=plan_report,
        gate_reports=gate_reports,
    )
