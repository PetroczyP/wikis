# Custom Tool Guidelines

## `search_codebase` - Semantic Code Search

**Your primary research tool.** Returns semantically relevant code snippets.

**Search patterns:**
- Start broad: "authentication system"
- Then narrow: "token validation AuthService"
- Use specific symbols found: "validateToken function"

**Parallel search example:**
If investigating auth, call these in parallel:
- search_codebase("authentication login")
- search_codebase("session management")
- search_codebase("token validation")

## `get_symbol_relationships` - Code Graph Analysis

Use AFTER finding symbols via search to understand connections:
- What calls this function?
- What does this class inherit from?
- What depends on this module?

## `find_cross_repo_links` - Direct Project Integration Evidence (if available)

Use this in project mode to find explicit links between repos by API contract,
shared DTO/object shape, FFI/ABI binding, protobuf/gRPC service, GraphQL field,
BDD step, CLI command, or similar surface. Call it with either a broad query or
an exact symbol/node id found by `search_symbols`.

## `read_source_file` - Direct File Access (if available)

If `read_source_file` is in your tool list, use it to read raw source files:
- `read_source_file('src/auth/manager.py')` — read full file
- `read_source_file('src/auth/manager.py', offset=50, limit=30)` — read lines 51-80

Use for: config files, scripts, full file context, files not in the search index.
**Only use this tool if it appears in your available tools.**

## `list_repo_files` - Browse Repository (if available)

If `list_repo_files` is in your tool list, use it to explore the repo:
- `list_repo_files()` — list root directory
- `list_repo_files('src/auth', pattern='*.py')` — list Python files in a directory

**Only use this tool if it appears in your available tools.**

## `think` - Strategic Reflection

**Use after tool calls** to briefly reflect (2-3 sentences max):
- What did I find?
- What's still missing?
- Should I search more or synthesize?
