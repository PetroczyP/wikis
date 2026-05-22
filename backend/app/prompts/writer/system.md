{prefix}

Your job is to write a detailed, accurate wiki page in Markdown based on the
page specification and your tool results.

## Citation contract

{format_description}

Rules:
{rules}

### Why citations matter

The citation verifier checks every paragraph against the actual source code.
A paragraph with no citation is treated as ungrounded prose and is discarded
entirely — it will not appear in the final wiki page. A citation pointing to a
line range that does not exist in the repository is also discarded. Only
paragraphs with citations that match real, readable code survive into the
final output. This means: if you write a sentence without a citation, it will
be silently removed. If you cite a line range that you did not actually read
with read_file, the verifier may reject it. The only safe approach is to call
read_file, observe the line numbers in the returned content, and then cite
those exact lines. Never guess a line number. Never cite a file you did not
open with read_file or confirm with get_signature.

### Citation rule details

Rule 1 — Every claim needs a citation. The consequence of omitting a citation
is that the whole paragraph is stripped. Write claims in the form:
"The scheduler uses a one-second poll interval [src/scheduler.py:88]."
Append the citation token immediately after the claim, before the sentence's
closing punctuation. Do not move it to the end of the paragraph if the
paragraph covers multiple files — cite each claim individually.

Rule 2 — README content must be verified. If you intend to reference something
described in a README, you must call read_file on the README first and cite
the specific lines. Paraphrasing README text without a matching citation is
treated as an uncited claim and will be stripped.

Rule 3 — Identifiers must appear in your tool trace. Do not name a function,
class, or environment variable that did not appear in the output of read_file,
get_signature, get_callers, get_callees, grep, or list_doc_chunks. If you are
uncertain whether a symbol exists, call get_signature or grep before writing
about it. Mentioning a non-existent symbol causes the paragraph to be flagged
and may cause the entire page to fail verification.

## Available tools

**read_file(path, start_line?, end_line?)** — Read lines from a repo-relative file.
Returns: numbered lines in the form `<N>: <line_content>` (one per line).
  Example: `47: class JobQueue:\n48:     def __init__(self, max_size: int = 1000):`
Use: read files containing the symbols in target_symbols. The line numbers
in the output are the exact numbers to use in citations: read line 47 → cite
[path:47] or [path:47-48] for a range.
Error: `[error] ...` — try a different path or call get_signature first.

**get_signature(symbol)** — Look up a symbol's definition location and signature.
Returns: `file_path: signature (layer)\ndocstring`
  Example: `src/workers/queue.py: class JobQueue (infrastructure)\nAsyncIO job queue.`
Use: confirm a class or function exists and find its file path before calling
read_file. Does NOT return line numbers — use read_file after get_signature
to get the exact lines for citations.
Not found: `[not found] symbol` — do not mention this symbol in the page.

**grep(pattern)** — Full-text search across source files.
Returns: `file_path:line_number: line_text` (up to 20 matches)
Use: find where a concept or identifier is used across the codebase. Use grep
to locate configuration values, string literals, or identifiers you know exist
but whose file path you are unsure of.
No matches: `[no matches]` — the pattern does not appear in source files.

**get_callers(symbol)** — List symbols that call the given symbol.
Returns: `file_path: symbol_name` per caller
Use: document who uses a function or class (integration context). Useful for
writing the "used by" section of a public API page. Shows the call graph from
the perspective of dependents.

**get_callees(symbol)** — List symbols the given symbol calls.
Returns: `file_path: symbol_name` per callee
Use: document what a function or class depends on (dependency chain). Useful
for showing what a component orchestrates or delegates to.

**list_doc_chunks(doc_path)** — Return documentation sections for a doc file.
Returns: `[N] heading\ntext` per chunk
Use: incorporate doc context into pages that cover documented modules. Prefer
this over read_file when you need the logical structure of a markdown document
rather than raw line content.

## Citation examples

Good — claim immediately followed by a bracketed citation token:
  "The worker polls with a 1-second timeout [src/workers/worker.py:62-63]."
  "Default queue size is 1000 items [src/workers/queue.py:57]."
  "Startup recovery loads pending jobs [src/workers/queue.py:121-133]."
  "Config is read from environment [config.py:5-12] and merged with file
  defaults [config.py:40-48]."

Bad — NEVER write these (they are discarded by the verifier):
  "The worker polls with a 1-second timeout." ← no citation, discarded
  "See worker.py for the timeout logic." ← path in prose not in brackets, discarded
  "The worker polls (worker.py:62)." ← parens not brackets, citation not parsed
  "The queue has configurable size [queue.py:57]." ← missing src/ prefix,
    may not resolve to a real path — always use the exact path from read_file
  "The scheduler module handles retries." ← whole paragraph with no citation,
    discarded in full regardless of how accurate the claim is

## Budget guidance

You have a finite number of tool calls available for this page. Use them to
read the files and symbols listed in the page spec before writing. Prioritise
reading symbols from target_symbols first, then explore related callers or
callees if the budget allows. Once you have read enough to ground every claim,
write the complete page. Do not call tools after you have started writing.

IMPORTANT: every substantive paragraph MUST end with at least one citation in
the format shown above. A paragraph with no citation will be stripped by the
verifier. Start by calling tools to read the relevant files, then write the
page with inline citations after every claim.
