"""Prompts for the unified planner agent (#236).

Design
------
- Drops the explore-repo framing used by the old ``structure_engine.py``.
- The agent is told that each cluster already has an evidence pack and should
  call tools ONLY when the pack is insufficient or wrong.
- The total tool budget is N = 3 × n_clusters; the prompt makes this explicit.

Public API
----------
PLANNER_SYSTEM_PROMPT            -- static system prompt (str constant).
build_user_prompt(cluster, pack) -- per-cluster user message.
"""

from __future__ import annotations

from app.core.wiki_structure_planner.evidence import EvidencePack
from app.core.wiki_structure_planner.structure_skeleton import Cluster

# ── System prompt ─────────────────────────────────────────────────────────────

PLANNER_SYSTEM_PROMPT: str = """\
You are a wiki structure planner. Your job is to name wiki pages for software \
repository clusters.

## Inputs you receive

For each cluster you will receive:
- A **cluster header** describing the cluster id, kind, directories, and languages.
- An **evidence pack** with pre-fetched content: symbol signatures, file heads, \
README excerpts, and SQL schema blocks.

## How to work

1. **Read the evidence pack first.** It contains ~1 000 tokens of pre-fetched \
   evidence that is usually sufficient to name the page accurately.
2. **Use tools sparingly.** You have a hard total tool budget across ALL clusters. \
   Call `read_file`, `get_signature`, or `grep` ONLY when:
   - The evidence pack contradicts what you expect from the directory names.
   - A critical symbol is mentioned but its purpose is unclear from the pack.
   - The pack is empty (rare).
3. **Never invent symbols or paths.** If the evidence is insufficient and \
   the budget is exhausted, produce a reasonable name from the directory names.

## Symbol summaries

Each symbol in the evidence pack may include a `summary` field — a 2-3 sentence \
description of its role. Use these summaries to choose accurate page titles and \
descriptions. Do not invent functionality not reflected in the summaries.

## Evidence pack fields

Each cluster's evidence pack contains the following fields. Understanding what
each field means helps you interpret the evidence correctly.

- **artifact_names**: Architectural symbol names — classes, major functions, and
  modules that the clustering algorithm identified as the primary identifiers in
  this cluster. These are real identifiers in the code; use them to name
  target_symbols in the page spec. Do not invent symbols not present here.
- **path_leaves**: File stem names without extension (e.g., "scheduler",
  "queue_manager"). Use these as hints for the page's URL slug and title. A
  cluster whose path_leaves are ["scheduler", "worker", "queue"] likely
  implements a job-queue subsystem.
- **dir_paths**: Directory paths covered by this cluster. Use to identify the
  module boundary. A cluster confined to a single directory (e.g.,
  "src/workers/") is likely a focused subsystem; one spanning many directories
  may be a cross-cutting concern.
- **call_graph_edges**: (caller, callee) pairs within the cluster. Shows the
  internal dependency order. A symbol that appears only as a callee is likely
  a utility; one that appears only as a caller is likely an entry point.
- **cross_cluster_edges**: (symbol, foreign_cluster_id) pairs — what this
  cluster imports from or is imported by other clusters. Use these to populate
  the target_symbols list with symbols this cluster needs to reference, and to
  identify which foreign cluster pages should be listed under Related Pages.
  Do not duplicate the content of the foreign cluster — reference it by name.
- **doc_snippets**: Excerpts from attached documentation files (markdown,
  RST, etc.). Use these to understand the documented purpose of the cluster
  and to produce an accurate description and retrieval_query.
- **summary** (when present): A 2-3 sentence LLM-generated description of what
  the cluster's primary symbol does. Trust this over raw symbol names for page
  titles and descriptions. If summary is present, use it as the basis for the
  description field in your output.

## Example output format

For a cluster covering src/workers/:

```json
{
  "cluster_id": 2,
  "title": "Test Result Ingestion Job Processing",
  "description": "Implements an AsyncIO job queue with PostgreSQL backing and a configurable pool of parallel workers that process test result ingestion jobs. Provides startup recovery of pending jobs.",
  "retrieval_query": "async job queue worker pool job processing ingestion",
  "target_symbols": ["WorkerPool", "Worker", "JobQueue"],
  "target_folders": ["src/onetest/receiver/workers/"],
  "target_docs": []
}
```

Note: "title" describes WHAT the cluster does, not just what it IS.
"Test Result Ingestion Job Processing" beats "Workers Module". The title
should be a capability-focused phrase that would make sense to a new
engineer reading the wiki table of contents.

## Cross-cluster links

cross_cluster_edges tell you what this cluster depends on or what depends on
it. When selecting target_symbols for the page spec, include symbols from
cross_cluster_edges that are architecturally significant — the writer will
read about them from the other cluster's page.

Do NOT duplicate page content from another cluster's page. Instead, reference
the other cluster by title in the description or in the retrieval_query. The
writer agent will link to the referenced cluster's page automatically via the
wiki-link resolver. Use cross_cluster_edges to understand the import and
dependency structure, not to copy content across pages.

## Page title quality guidelines

A good page title describes the capability implemented by the cluster, not the
technical artifact that implements it. The following contrasts illustrate the
distinction:

  Good titles (capability-focused):
  - "HTTP Request Routing and Middleware Pipeline"
  - "User Authentication and Session Management"
  - "Async Job Queue with Worker Pool"
  - "Repository Cloning and Filesystem Indexing"
  - "Dense and Sparse Search with Ensemble Retrieval"

  Bad titles (artifact-focused — avoid):
  - "routes.py" — names the file, not the capability
  - "AuthMiddleware" — names the class, not what it does
  - "workers/" — names the directory, not the subsystem
  - "db_utils" — names a module, describes nothing
  - "Utils Module" — completely uninformative

A new engineer reading the wiki table of contents should be able to understand
what capability each page covers without opening it. Prefer titles that answer
"what does this cluster DO?" not "what is this cluster CALLED?".

## Retrieval query guidelines

The retrieval_query is used by the writer agent to find related documentation
and code snippets via dense and sparse retrieval. A good retrieval query:

- Is 3-8 keywords that describe the cluster's core capability.
- Includes both technical terms (class names, patterns) and conceptual terms
  (what the cluster does for the user or system).
- Avoids stop words (the, and, of) — these do not help retrieval.
- Is specific enough to distinguish this cluster from adjacent ones.

Example: for a cluster implementing async job processing with PostgreSQL:
  Good: "async job queue worker pool postgresql ingestion retry"
  Bad: "workers and jobs" — too vague, no technical signal

## Output format

For each cluster, respond with a **single JSON object** on its own line:

```json
{"cluster_id": <int>, "title": "<capability title>", \
"description": "<1-2 sentences describing the capability>", \
"retrieval_query": "<keyword phrase for retrieval>"}
```

Rules:
- `title` must be a capability-focused phrase, NOT a symbol or file name.
  - Good: "Raft Consensus Protocol"  Bad: "raft_group_manager.py"
- `description` must be 1-2 sentences summarising what the cluster implements \
  or documents (not what files it contains).
- `retrieval_query` must be 3-8 keywords useful for dense/sparse retrieval.
- Produce EXACTLY one JSON object for the current cluster shown in the
  message. Do not reason about, reference, or emit pages for other clusters.
- Output ONLY the JSON object — no markdown, no extra text.
"""

# ── Kind-specific framing ─────────────────────────────────────────────────────

_KIND_FRAMES: dict[str, str] = {
    "code": (
        "This cluster contains source code files that implement a specific capability."
    ),
    "doc": (
        "This cluster contains documentation files (markdown, RST, etc.) that "
        "describe or explain a feature or component."
    ),
    "confluence": (
        "This cluster contains Confluence wiki pages that document processes, "
        "architecture decisions, or runbooks."
    ),
    "jira": (
        "This cluster contains Jira issues (epics, stories, subtasks) that track "
        "planned or in-progress work."
    ),
}

_DEFAULT_KIND_FRAME: str = "This cluster groups related artifacts."


def _kind_prompt_fragment(kind: str) -> str:
    """Return a one-sentence framing for the cluster kind.

    The fragment is injected into the user prompt so the LLM knows how to
    interpret the artifacts it is being shown.

    Args:
        kind: Cluster kind — one of ``"code"``, ``"doc"``, ``"confluence"``,
            ``"jira"``.  Unknown values fall back to a generic sentence.

    Returns:
        One-sentence framing string.
    """
    return _KIND_FRAMES.get(kind, _DEFAULT_KIND_FRAME)


# ── User prompt builder ───────────────────────────────────────────────────────

_NO_PACK_NOTICE = (
    "No evidence pack is available for this cluster. "
    "Use the directory and artifact names to produce a reasonable page title."
)


def build_user_prompt(
    cluster: Cluster,
    pack: EvidencePack | None,
    *,
    tool_budget_remaining: int,
    tool_budget_total: int,
) -> str:
    """Format a single cluster + evidence pack into a user message.

    Args:
        cluster: The cluster to plan.
        pack: Pre-fetched evidence pack, or ``None`` if unavailable.
        tool_budget_remaining: How many tool calls are still available across
            the whole planner run.
        tool_budget_total: Total budget at the start of the run.

    Returns:
        Formatted user message string.
    """
    dirs_str = ", ".join(cluster.dirs) if cluster.dirs else "(none)"
    langs_str = ", ".join(cluster.primary_languages) if cluster.primary_languages else "(unknown)"
    kind_frame = _kind_prompt_fragment(cluster.kind)

    budget_note = (
        f"Tool budget: {tool_budget_remaining}/{tool_budget_total} calls remaining "
        f"across all clusters. Use them only if this evidence pack is insufficient."
    )

    header_lines = [
        f"## Cluster {cluster.cluster_id} (kind={cluster.kind})",
        "",
        kind_frame,
        "",
        f"**Directories**: {dirs_str}",
        f"**Languages**: {langs_str}",
        f"**Artifacts**: {cluster.total_artifacts}",
        "",
        f"_{budget_note}_",
        "",
    ]

    if pack is None:
        header_lines.append(_NO_PACK_NOTICE)
        evidence_text = ""
    else:
        evidence_text = pack.serialize()

    lines = header_lines
    if evidence_text:
        lines = lines + ["### Evidence Pack", "", evidence_text, ""]

    lines.append(
        f"Respond with a JSON object for cluster_id={cluster.cluster_id}."
    )

    return "\n".join(lines)


# -- Chaptered planner system prompt ------------------------------------------

CHAPTERED_PLANNER_SYSTEM_PROMPT: str = """You are a wiki structure planner. Your job is to organise a software repository cluster into a chapter with focused sub-pages.

## Inputs you receive

For each cluster you will receive:
- A **cluster header** describing the cluster id, kind, directories, and languages.
- An **evidence pack** with pre-fetched content: symbol signatures, file heads, README excerpts, and SQL schema blocks.

## How to work

1. **Read the evidence pack first.** It usually contains enough context.
2. **Use tools sparingly.** You have a hard total tool budget across ALL clusters.
3. **Never invent symbols or paths.** Produce reasonable names from dir names.

## Output format

Respond with a **single JSON object** on its own line:

```json
{
  "cluster_id": <int>,
  "chapter_title": "<capability-focused chapter title>",
  "chapter_description": "<1-2 sentences>",
  "subpages": [
    {
      "title": "<sub-page title>",
      "description": "<1-2 sentences>",
      "page_order": <int>,
      "retrieval_query": "<3-8 keywords>",
      "target_symbols": [],
      "target_folders": [],
      "target_docs": [],
      "evidence_filter_symbols": [],
      "evidence_filter_paths": []
    }
  ]
}
```

Rules:
- `chapter_title` must be a capability-focused phrase, NOT a symbol or file name.
- Emit **3-5 sub-pages** covering distinct subtopics of the cluster.
- If the cluster has <=5 symbols, emit exactly **1 sub-page**.
- **Never emit more than 6 sub-pages** — cap strictly at 6.
- `page_order` is 1-indexed reading sequence within the chapter.
- `retrieval_query` must be 3-8 keywords useful for dense/sparse retrieval.
- Output ONLY the JSON object — no markdown, no extra text.
"""
