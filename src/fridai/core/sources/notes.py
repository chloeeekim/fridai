"""User notes — a memory the developer (or an agent) writes directly, not parsed
from disk like the other sources.

A note is authored input: one Document with source_type="note". It is indexed,
redacted, embedded, and recalled exactly like every other source. The agent-facing
entry point is the MCP `remember` tool; `fridai note` is the human/testing entry point.
Both call `add_note`.
"""
from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .. import embeddings
from ..models import Document, make_id


def _cwd_repo(path: str = ".") -> str:
    """git repo name of `path` (= the working repo). '' if not inside a git repo."""
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).name
    except Exception:
        pass
    return ""


def add_note(store, text: str, *, repo: str | None = None,
             embedder=None, when: datetime | None = None) -> Document | None:
    """Save one note. `repo=None` -> detect the current git repo (so recall's default
    cwd scoping finds it); `repo=""` -> a cross-repo (global) note. Returns the stored
    Document, or None if the text is empty. Redaction/embedding happen on upsert."""
    text = (text or "").strip()
    if not text:
        return None
    when = when or datetime.now(timezone.utc)
    if repo is None:
        repo = _cwd_repo()
    doc = Document(
        id=make_id("note", repo, when.isoformat(), text),
        source_type="note", repo=repo, path="note",
        title=text.splitlines()[0].strip()[:80], text=text, timestamp=when, meta={},
    )
    if embedder:
        embeddings.embed_documents([doc], embedder)
    store.upsert([doc])
    return doc
