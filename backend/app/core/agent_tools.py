"""Canonical tool schemas shared by wiki-generation agents.

Both the planner (``wiki_structure_planner.planner_agent``) and the writer
(``wiki_content_writer.writer_agent``) draw from the same catalog so a single
``create_llm(tools=WIKI_TOOL_SCHEMAS)`` call covers both. The agents narrow
which tools they actually invoke via their system prompts, not via separate
``bind_tools`` calls — duplicate tool names across separate bindings would
trip provider validators (Bedrock rejects, OpenAI silently picks last).

Future work tracked under issue #287's broader consolidation: the deep
research / ask / code-map services have analogous tool surfaces
(``read_source_file``, ``search_codebase``, ``search_graph``, ``get_code``,
``get_symbol_relationships``, etc.) that should fold into a unified registry.
For now this module covers only the wiki-generation pipeline.
"""

from __future__ import annotations

from typing import Any

WIKI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read lines from a repo-relative file. Use to find the exact "
                "line spans you will cite, or to verify evidence-pack claims."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Repo-relative file path."},
                    "start_line": {"type": "integer", "description": "1-indexed first line (optional)."},
                    "end_line": {"type": "integer", "description": "1-indexed last line (optional)."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_signature",
            "description": "Look up a symbol's signature, file path, and layer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol name to look up."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a text pattern across repository source files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search for."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callers",
            "description": "List symbols that call the given symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol name."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_callees",
            "description": "List symbols that the given symbol calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Symbol name."},
                },
                "required": ["symbol"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_doc_chunks",
            "description": "Return documentation chunks for a doc file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_path": {"type": "string", "description": "Documentation file path."},
                },
                "required": ["doc_path"],
            },
        },
    },
]
