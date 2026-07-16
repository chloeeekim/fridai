"""F1 — index git commit history.

One commit = Document(source_type="commit"): title + changed files as the body.
Incremental: store the last-indexed HEAD in index_state -> only new commits added.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .. import embeddings
from ..models import Document, make_id
from .code import repo_name

_H = "@@C@@"   # commit header marker
_F = "\x1f"    # field separator


def _head(repo_path) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _parse_log(raw: str, repo: str, root: str = "") -> list[Document]:
    docs, cur = [], None
    for ln in raw.splitlines():
        if ln.startswith(_H):
            if cur:
                docs.append(_build(repo, *cur, root))
            parts = ln[len(_H):].split(_F)
            cur = (parts + ["", "", ""])[:3] + [[]] if len(parts) >= 3 else None
        elif ln.strip() and cur:
            cur[3].append(ln.strip())
    if cur:
        docs.append(_build(repo, *cur, root))
    return docs


def _build(repo, full_sha, iso, subject, files, root="") -> Document:
    ts = None
    try:
        ts = datetime.fromisoformat(iso)
    except ValueError:
        pass
    short = full_sha[:9]
    body = subject + ("\n" + "\n".join(files) if files else "")
    return Document(
        id=make_id("commit", repo, full_sha), source_type="commit", repo=repo,
        path=short, title=subject, text=body, timestamp=ts,
        meta={"sha": short, "full_sha": full_sha, "files": files, "root": root},
    )


def _log(repo_path, rev_range: str | None) -> str:
    args = ["git", "-C", str(repo_path), "log", "--name-only",
            f"--pretty=format:{_H}%H{_F}%cI{_F}%s"]
    if rev_range:
        args.append(rev_range)
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return ""


def index_commits(repo_path: str | Path, store, *, reindex: bool = False,
                  embedder=None) -> dict:
    """Incrementally index commits. Generates vectors if embedder given. Returns {commits}."""
    repo = repo_name(repo_path)
    head = _head(repo_path)
    if head is None:
        return {"commits": 0}
    key = f"commits:{repo}"
    last = None if reindex else store.get_state(key)
    if last == head:
        return {"commits": 0}
    rev_range = f"{last}..{head}" if last else None
    docs = _parse_log(_log(repo_path, rev_range), repo, str(Path(repo_path).resolve()))
    if embedder:
        embeddings.embed_documents(docs, embedder)
    store.upsert(docs)
    store.set_state(key, head)
    return {"commits": len(docs)}


def changes(repo_path: str | Path, sha: str, max_lines: int = 40) -> str | None:
    """Return the commit's change summary (stat) + patch. Truncated past max_lines. None on failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "show", sha, "--stat", "--patch", "--format="],
            capture_output=True, text=True, timeout=15,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    lines = out.stdout.strip().splitlines()
    if not lines:
        return None
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... (+{len(lines) - max_lines} more lines truncated)"]
    return "\n".join(lines)


def documents(repo_path: str | Path) -> Iterator[Document]:
    """Non-incremental convenience function (all commits)."""
    repo = repo_name(repo_path)
    if _head(repo_path) is None:
        return
    yield from _parse_log(_log(repo_path, None), repo, str(Path(repo_path).resolve()))
