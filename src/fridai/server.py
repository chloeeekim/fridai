"""MCP server — agents recall past memory directly while coding.

**Search-only**: exposes just the `recall` tool (no answer generation — the calling
agent is the LLM, so this only returns evidence). Reuses the embedder abstraction
(fastembed if available for semantic, else lexical). The MCP SDK is a dependency.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .core import config, embeddings, search
from .core.store import Store


def _cwd_repo() -> str | None:
    """git repo name of the current working directory (= the agent's working repo). Else None."""
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).name
    except Exception:
        pass
    return None


def _resolve_repo(repo: str | None, cwd_repo: str | None) -> str | None:
    """Decide search scope. None+cwd repo -> current repo, "all" -> everything, explicit -> that repo."""
    if repo and repo.lower() == "all":
        return None                 # whole index
    if repo:
        return repo                 # the named repo
    return cwd_repo                 # default: current working repo (None = everything)


def _format_hit(doc, idx: int) -> str:
    m = doc.meta
    head = f"### [{idx}] {search.citation(doc)}"
    if doc.source_type == "agent_turn":
        body = f"Q: {m.get('question', doc.title)}"
        if m.get("answer_summary"):
            body += f"\nA: {m['answer_summary']}"
        commits = m.get("commits") or []
        if commits:
            body += f"\n→ commit {commits[0][0]}: {commits[0][1]}"
    elif doc.source_type in ("code", "commit"):
        if m.get("summary"):
            body = f"{m['summary']}\n{doc.text[:300]}"
        else:
            body = doc.text[:400]
    else:
        body = doc.text[:400]
    return f"{head}\n{body}"


def recall_tool(query_text: str, k: int = 5, repo: str | None = None,
                source_type: str | None = None, store: Store | None = None,
                cwd_repo: str | None = None) -> str:
    """Recall past memory -> text with sources (the agent reads and reasons over it).
    Scope: without `repo`, limited to cwd_repo (current repo); repo="all" = everything.
    Without `store`, uses the default index."""
    eff_repo = _resolve_repo(repo, cwd_repo)
    own = store is None
    store = store or Store(config.DB_PATH)
    try:
        hits = search.retrieve(store, query_text, k=k, embedder=embeddings.get_embedder(),
                              repo=eff_repo, source_type=source_type or None)
    finally:
        if own:
            store.close()
    scope = f"repo={eff_repo}" if eff_repo else "all repos"
    if not hits:
        return f'fridai: no relevant memory found for "{query_text}" ({scope}).'
    out = [f'fridai recall — {len(hits)} memory item(s) for "{query_text}" ({scope}, with sources):']
    out += [_format_hit(h.document, i) for i, h in enumerate(hits, 1)]
    return "\n\n".join(out)


def serve() -> None:
    """Run the stdio MCP server. Invoked by `fridai mcp`."""
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        raise SystemExit("MCP SDK not installed — run `pip install fridai` (mcp is a required dependency).")

    server = FastMCP("fridai")

    @server.tool()
    def recall(query: str, k: int = 5, repo: str = "", source_type: str = "") -> str:
        """Recall the developer's past code, commits, AI conversations, and saved notes — with sources (personal coding memory).

        **For questions that recall past work, call this BEFORE grepping code/git history:**
        "how did I do/fix this before", a recurring error's past solution, the rationale behind an implementation (why it was done this way).
        Native code/git search only sees the current state; this tool also holds *past attempts, conversations, and reasons*.
        Answer grounded in the returned sources; if insufficient, then supplement with code/git search. If nothing is found, say so.

        Effective queries: include relevant English identifiers/technical terms (functions/variables/error codes,
          e.g. login, auth, token, 401, mount) even if the user's question is in another language. Prefer key
          terms over long sentences. If results are weak, retry with synonyms/related terms.
          (The index mixes natural-language conversations with English code.)
        Scope (repo): empty = current working repo (default). Use repo="all" for cross-repo, or repo="<name>" for a specific repo.
        source_type (empty = all): agent_turn | code | commit | note."""
        return recall_tool(query, k=k, repo=repo or None,
                           source_type=source_type or None, cwd_repo=_cwd_repo())

    server.run()
