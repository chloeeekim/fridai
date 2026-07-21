"""MCP server — agents recall and record memory directly while coding.

Exposes two tools: `recall` (search past memory) and `remember` (save a durable note).
No answer generation — the calling agent is the LLM, so `recall` only returns evidence.
Reuses the embedder abstraction (fastembed if available for semantic, else lexical).
The MCP SDK is a dependency.
"""
from __future__ import annotations

import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .core import config, embeddings, search
from .core.sources import notes
from .core.store import Store


def _parse_since(s: str | None):
    """Relative time to a UTC cutoff: '7d' / '24h' / '2w'. None/invalid -> None (no filter)."""
    if not s:
        return None
    m = re.fullmatch(r"(\d+)\s*([dhw])", s.strip())
    if not m:
        return None
    n, u = int(m.group(1)), m.group(2)
    delta = {"d": timedelta(days=n), "h": timedelta(hours=n), "w": timedelta(weeks=n)}[u]
    return datetime.now(timezone.utc) - delta


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
                cwd_repo: str | None = None, since: str | None = None) -> str:
    """Recall past memory -> text with sources (the agent reads and reasons over it).
    Scope: without `repo`, limited to cwd_repo (current repo); repo="all" = everything.
    `since` (e.g. "7d"/"24h"/"2w") limits to recent memory. Without `store`, uses the default index."""
    eff_repo = _resolve_repo(repo, cwd_repo)
    own = store is None
    store = store or Store(config.DB_PATH)
    try:
        if store.stats()["total"] == 0:
            return "fridai: the index is empty — run `fridai index` first."
        hits = search.retrieve(store, query_text, k=k, embedder=embeddings.get_embedder(),
                              repo=eff_repo, source_type=source_type or None,
                              since=_parse_since(since))
    finally:
        if own:
            store.close()
    scope = f"repo={eff_repo}" if eff_repo else "all repos"
    if not hits:
        return f'fridai: no relevant memory found for "{query_text}" ({scope}).'
    out = [f'fridai recall — {len(hits)} memory item(s) for "{query_text}" ({scope}, with sources):']
    out += [_format_hit(h.document, i) for i, h in enumerate(hits, 1)]
    return "\n\n".join(out)


def remember_tool(text: str, repo: str | None = None, store: Store | None = None,
                  cwd_repo: str | None = None) -> str:
    """Persist a durable note to memory -> a confirmation string.
    Scope: without `repo`, attaches to cwd_repo (current repo) so recall finds it by
    default. Without `store`, uses the default index. Redaction/embedding apply on save."""
    text = (text or "").strip()
    if not text:
        return "fridai: nothing to remember (the note is empty)."
    eff_repo = repo if repo else (cwd_repo or "")
    own = store is None
    store = store or Store(config.DB_PATH)
    try:
        doc = notes.add_note(store, text, repo=eff_repo, embedder=embeddings.get_embedder())
    finally:
        if own:
            store.close()
    scope = f"repo={eff_repo}" if eff_repo else "all repos (global)"
    return f'fridai: remembered ({scope}) — "{doc.title}"'


def serve() -> None:
    """Run the stdio MCP server. Invoked by `fridai mcp`."""
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception:
        raise SystemExit("MCP SDK not installed — run `pip install fridai` (mcp is a required dependency).")

    server = FastMCP("fridai")

    @server.tool()
    def recall(query: str, k: int = 5, repo: str = "", source_type: str = "", since: str = "") -> str:
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
        source_type (empty = all): agent_turn | code | commit | note.
        since (empty = all time): limit to recent memory, e.g. "7d", "24h", "2w"."""
        return recall_tool(query, k=k, repo=repo or None,
                           source_type=source_type or None, since=since or None,
                           cwd_repo=_cwd_repo())

    @server.tool()
    def remember(text: str, repo: str = "") -> str:
        """Save a durable memory note to the developer's personal coding memory (retrievable later via `recall`).

        **Call this when something is worth remembering across sessions:** a decision and its rationale
        (why an approach was chosen over alternatives), a non-obvious gotcha or constraint, or a fix for a
        recurring problem. Prefer distilled, self-contained notes over raw transcript — write what a future
        session would need to know, including the *why*.
        This writes to persistent storage, so only save things genuinely worth keeping (not transient status).
        Scope (repo): empty = the current working repo (default, so `recall` finds it there); repo="<name>" pins it to a specific repo."""
        return remember_tool(text, repo=repo or None, cwd_repo=_cwd_repo())

    server.run()
