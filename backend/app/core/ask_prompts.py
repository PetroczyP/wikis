"""
Ask Prompts - System and task prompts for the Agentic Ask engine.

These prompts are optimized for Q&A (not long-form research):
- Concise answer generation (not multi-page reports)
- Progressive disclosure workflow: search → relationships → code → answer
- Natural file path citations (not [N] numeric references)
- Hard iteration limits to prevent runaway execution

Unlike deep_research prompts, these do NOT include:
- Todo list management (overkill for Q&A)
- Filesystem offloading (answers are short)
- Multi-step research planning (Q&A is direct)
"""

from app.prompts import load_prompt

from datetime import datetime

# =============================================================================
# WORKFLOW INSTRUCTIONS
# =============================================================================

ASK_WORKFLOW_INSTRUCTIONS: str = load_prompt("ask/workflow_instructions.md")

# =============================================================================
# TOOL USAGE GUIDANCE
# =============================================================================

ASK_TOOL_INSTRUCTIONS: str = load_prompt("ask/tool_instructions.md")

# =============================================================================
# OUTPUT FORMAT
# =============================================================================

ASK_OUTPUT_INSTRUCTIONS: str = load_prompt("ask/output_instructions.md")

# =============================================================================
# COMBINED SYSTEM PROMPT
# =============================================================================


def get_ask_instructions(max_iterations: int = 8) -> str:
    """Generate the complete Ask system prompt.

    Args:
        max_iterations: Maximum tool call budget for the agent

    Returns:
        Complete system prompt for the Ask agent
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    return "\n\n".join(
        [
            ASK_WORKFLOW_INSTRUCTIONS.format(date=current_date, max_iterations=max_iterations),
            ASK_TOOL_INSTRUCTIONS,
            ASK_OUTPUT_INSTRUCTIONS,
        ]
    )


# Pre-built instance for default use
ASK_INSTRUCTIONS = get_ask_instructions()


# =============================================================================
# USER PROMPT BUILDER
# =============================================================================


def get_ask_prompt(question: str, repo_context: str = "", chat_history: str = "") -> str:
    """Build the user message for the Ask agent.

    Args:
        question: The user's question
        repo_context: Optional repository overview (from analysis store)
        chat_history: Optional formatted recent chat messages

    Returns:
        Formatted user prompt
    """
    parts = [f"## Question\n\n{question}\n"]

    if repo_context:
        parts.append(f"## Repository Context\n\n{repo_context}\n")

    if chat_history:
        parts.append(f"## Conversation History\n\n{chat_history}\n")

    parts.append(
        "## Getting Started\n\n"
        "Use `search_symbols` (call multiple in parallel) to find relevant code, "
        "then `get_relationships` for key symbols, then `get_code` for 2-4 critical "
        "symbols. Answer directly — do not create a research report.\n"
    )

    return "\n".join(parts)
