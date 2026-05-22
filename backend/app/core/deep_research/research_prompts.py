"""
Research Prompts - System and task prompts for DeepAgents

Streamlined prompts focused on:
- Clear workflow structure
- Parallel tool calling for efficiency
- Hard limits to prevent runaway execution
- Custom tool guidance (search_codebase, get_symbol_relationships, think)

Note: DeepAgents middleware already injects detailed tool documentation for
built-in tools (filesystem, todos). These prompts focus on repo-specific workflow.
"""

from app.prompts import load_prompt

from datetime import datetime

# =============================================================================
# MAIN RESEARCH WORKFLOW INSTRUCTIONS
# =============================================================================

RESEARCH_WORKFLOW_INSTRUCTIONS: str = load_prompt("research/workflow_instructions.md")

# =============================================================================
# TOOL USAGE GUIDELINES
# =============================================================================

TOOL_USAGE_INSTRUCTIONS: str = load_prompt("research/tool_instructions.md")

# =============================================================================
# STOPPING CRITERIA
# =============================================================================

STOPPING_CRITERIA: str = load_prompt("research/stopping_criteria.md")


# =============================================================================
# OUTPUT FORMAT
# =============================================================================

OUTPUT_FORMAT_INSTRUCTIONS: str = load_prompt("research/output_format.md")

# =============================================================================
# MAIN RESEARCH INSTRUCTIONS (COMBINED)
# =============================================================================


def get_research_instructions() -> str:
    """Generate the complete research instructions.

    Returns:
        Complete system prompt for the research agent
    """
    current_date = datetime.now().strftime("%Y-%m-%d")

    return "\n\n".join(
        [
            RESEARCH_WORKFLOW_INSTRUCTIONS.format(date=current_date),
            TOOL_USAGE_INSTRUCTIONS,
            STOPPING_CRITERIA,
            OUTPUT_FORMAT_INSTRUCTIONS,
        ]
    )


# Legacy export for backward compatibility
RESEARCH_INSTRUCTIONS = get_research_instructions()

# =============================================================================
# RESEARCH TYPE SPECIFIC GUIDANCE
# =============================================================================

RESEARCH_TYPE_PROMPTS = {
    "general": """Focus on understanding the system holistically:
- How components interact
- Key design decisions
- Implementation patterns used""",
    "architecture": """Focus on system design:
- Component boundaries and responsibilities
- Communication patterns between modules
- Data flow through the system
- Scalability and performance considerations
- Configuration and deployment aspects""",
    "implementation": """Focus on code details:
- Specific functions and their logic
- Data structures and algorithms
- Error handling approaches
- Edge cases and boundary conditions
- Testing patterns""",
    "security": """Focus on security aspects:
- Authentication mechanisms
- Authorization and access control
- Input validation and sanitization
- Sensitive data handling
- Configuration of security features
- Potential vulnerabilities""",
    "integration": """Focus on external interactions:
- API contracts and protocols
- External service integrations
- Data exchange formats
- Error handling and retry logic
- Configuration and credentials""",
    "debugging": """Focus on problem investigation:
- Error handling and logging
- State management and data flow
- Edge cases and race conditions
- Dependencies and their behavior
- Configuration and environment""",
    "feature": """Focus on implementation planning:
- Where the feature should be added
- Existing patterns to follow
- Required changes and additions
- Impact on existing functionality
- Testing requirements""",
}


def get_research_prompt(research_type: str, topic: str, context: str = "", chat_history: str = "") -> str:
    """
    Generate a research task prompt for the question.

    Args:
        research_type: Type of research (architecture, security, etc.)
        topic: The specific question or topic
        context: Additional context
        chat_history: Optional formatted conversation history string

    Returns:
        Formatted task prompt
    """
    type_guidance = RESEARCH_TYPE_PROMPTS.get(research_type, RESEARCH_TYPE_PROMPTS["general"])

    prompt_parts = [
        f"## Research Question\n\n{topic}\n",
    ]

    if context:
        prompt_parts.append(f"## Repository Context\n\n{context}\n")

    if chat_history:
        prompt_parts.append(f"## Conversation History\n\n{chat_history}\n\n"
                            "_Use the history above to resolve any references in the research "
                            "question (e.g. 'it', 'that module', 'the same service'). Focus on "
                            "what is new or different in the current question._\n")

    prompt_parts.append(f"## Research Focus\n\n{type_guidance}\n")

    prompt_parts.append("""## Getting Started

1. Search strategically using `search_codebase` (call multiple in parallel)
2. Use `get_symbol_relationships` for code connections
3. Use `think` briefly after searches to reflect
4. Return the comprehensive answer directly in your final message
""")

    return "\n".join(prompt_parts)
