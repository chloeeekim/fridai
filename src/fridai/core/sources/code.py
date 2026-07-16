"""F1 — chunk and index git-tracked source files.

- only files tracked by `git ls-files` (respects .gitignore automatically, skips junk)
- excludes binary/large files, **chunks by function/class** (falls back to N-line windows for
  unsupported languages / huge symbols), preserves line ranges for citation
- incremental: stores each file's content hash in index_state, reindexing only changes
"""
from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Iterator

from .. import config, embeddings
from ..models import Document, make_id


def load_ignore(repo_path: str | Path) -> list[str]:
    """Load `.fridaiignore` patterns — repo root + global (~/.fridai/). Skips blank lines / `#` comments."""
    pats: list[str] = []
    for p in (Path(repo_path) / ".fridaiignore", config.HOME / ".fridaiignore"):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pats.append(line)
        except OSError:
            pass
    return pats


def is_ignored(relpath: str, patterns: list[str]) -> bool:
    """Simple gitignore-style glob match: `*.env`, `secrets/`, `path/to/x`, incl. subpaths."""
    base = Path(relpath).name
    for pat in patterns:
        if pat.endswith("/"):                       # directory -> everything under it
            if relpath == pat[:-1] or relpath.startswith(pat):
                return True
        elif (fnmatch.fnmatch(relpath, pat) or fnmatch.fnmatch(relpath, f"*/{pat}")
              or fnmatch.fnmatch(base, pat)):
            return True
    return False

WINDOW = 60          # lines per chunk
OVERLAP = 15         # overlap between adjacent chunks
MAX_BYTES = 400_000  # skip files larger than this

# binary/asset extensions (not code/docs)
SKIP_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".pdf", ".zip", ".gz",
    ".tar", ".jar", ".class", ".so", ".dylib", ".dll", ".bin", ".lock",
    ".woff", ".woff2", ".ttf", ".eot", ".mp4", ".mov", ".mp3", ".svg",
}
LANG_BY_EXT = {
    ".py": "python", ".kt": "kotlin", ".java": "java", ".js": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".go": "go", ".rs": "rust",
    ".rb": "ruby", ".c": "c", ".h": "c", ".cpp": "cpp", ".sh": "shell",
    ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".md": "markdown",
    ".sql": "sql", ".gradle": "gradle", ".kts": "kotlin",
}


def repo_name(repo_path: str | Path) -> str:
    return Path(repo_path).resolve().name


def tracked_files(repo_path: str | Path) -> list[str]:
    """List of relative paths for files tracked by git."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path), "ls-files"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except Exception:
        return []
    return [ln for ln in out.splitlines() if ln.strip()]


def is_indexable(abspath: Path) -> bool:
    if abspath.suffix.lower() in SKIP_EXTS:
        return False
    try:
        if abspath.stat().st_size > MAX_BYTES:
            return False
        with abspath.open("rb") as fh:
            if b"\x00" in fh.read(8192):   # null byte -> treat as binary
                return False
    except OSError:
        return False
    return True


def file_hash(abspath: Path) -> str:
    h = hashlib.sha1()
    with abspath.open("rb") as fh:
        for blk in iter(lambda: fh.read(65536), b""):
            h.update(blk)
    return h.hexdigest()


def chunk_lines(lines: list[str], window=WINDOW, overlap=OVERLAP) -> Iterator[tuple[int, int, str]]:
    """(start_line, end_line, text) chunks. Line numbers are 1-based."""
    n = len(lines)
    if n == 0:
        return
    step = max(1, window - overlap)
    i = 0
    while i < n:
        seg = lines[i:i + window]
        yield i + 1, i + len(seg), "\n".join(seg)
        if i + window >= n:
            break
        i += step


MAX_UNIT = 120   # symbol units larger than this get further split into N-line windows

# heuristic for function/class declaration starts (stdlib regex, zero deps). None for unsupported langs.
_DECL_PY = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+\w")
_DECL_BRACE = re.compile(
    r"^\s*(?:[\w@<>\[\]]+\s+){0,5}(?:fun|func|fn|class|interface|object|enum|struct|impl|trait)\s+\w"
)
_BRACE_LANGS = {"kotlin", "java", "javascript", "typescript", "go", "rust", "c", "cpp", "gradle"}


def _decl_re(lang: str):
    if lang == "python":
        return _DECL_PY
    if lang in _BRACE_LANGS:
        return _DECL_BRACE
    return None


def chunk_symbols(lines: list[str], lang: str) -> Iterator[tuple[int, int, str]]:
    """Split at function/class declaration boundaries. Unsupported lang / no boundary -> N-line window fallback.

    Huge symbols (>MAX_UNIT lines) are re-split into windows. Line numbers are 1-based absolute.
    """
    rx = _decl_re(lang)
    if rx is None or not lines:
        yield from chunk_lines(lines)
        return
    bounds = [i for i, ln in enumerate(lines) if rx.match(ln)]
    if not bounds:
        yield from chunk_lines(lines)
        return
    segs: list[tuple[int, int]] = []
    if bounds[0] > 0:
        segs.append((0, bounds[0]))            # preamble before first declaration (imports etc.)
    for j, b in enumerate(bounds):
        end = bounds[j + 1] if j + 1 < len(bounds) else len(lines)
        segs.append((b, end))
    for s, e in segs:
        seg = lines[s:e]
        if len(seg) > MAX_UNIT:                # too big -> re-split into windows
            for r1, r2, txt in chunk_lines(seg):
                yield s + r1, s + r2, txt
        else:
            yield s + 1, e, "\n".join(seg)


def file_documents(repo: str, relpath: str, text: str, root: str = "") -> list[Document]:
    docs = []
    lang = LANG_BY_EXT.get(Path(relpath).suffix.lower(), "text")
    for start, end, chunk in chunk_symbols(text.splitlines(), lang):
        if not chunk.strip():
            continue
        docs.append(Document(
            id=make_id("code", repo, relpath, start),
            source_type="code", repo=repo, path=relpath,
            title=f"{relpath}:{start}-{end}", text=chunk, timestamp=None,
            meta={"path": relpath, "start_line": start, "end_line": end,
                  "lang": lang, "root": root},
        ))
    return docs


def index_code(repo_path: str | Path, store, *, reindex: bool = False,
               embedder=None, prune: bool = True) -> dict:
    """Incrementally index a repo's code files. Generates vectors if embedder given.
    prune=True removes chunks of files gone from git.
    Returns {files, chunks, skipped, pruned}."""
    repo = repo_name(repo_path)
    base = Path(repo_path)
    root = str(base.resolve())
    ignore = load_ignore(repo_path)
    files = chunks = skipped = pruned = 0
    current: set[str] = set()
    for rel in tracked_files(repo_path):
        ab = base / rel
        if not is_indexable(ab) or is_ignored(rel, ignore):   # excluded by .fridaiignore
            continue
        current.add(rel)
        h = file_hash(ab)
        key = f"code:{repo}:{rel}"
        if not reindex and store.get_state(key) == h:
            skipped += 1
            continue
        store.delete_by_path("code", repo, rel)
        text = ab.read_text(encoding="utf-8", errors="replace")
        docs = file_documents(repo, rel, text, root=root)
        if embedder:
            embeddings.embed_documents(docs, embedder)
        store.upsert(docs)
        store.set_state(key, h)
        files += 1
        chunks += len(docs)

    if prune:
        for stale in store.paths("code", repo) - current:   # paths no longer tracked
            store.delete_by_path("code", repo, stale)
            store.delete_state(f"code:{repo}:{stale}")
            pruned += 1

    return {"files": files, "chunks": chunks, "skipped": skipped, "pruned": pruned}


def documents(repo_path: str | Path) -> Iterator[Document]:
    """Non-incremental convenience function (for tests / one-off indexing)."""
    repo = repo_name(repo_path)
    base = Path(repo_path)
    root = str(base.resolve())
    ignore = load_ignore(repo_path)
    for rel in tracked_files(repo_path):
        ab = base / rel
        if not is_indexable(ab) or is_ignored(rel, ignore):
            continue
        text = ab.read_text(encoding="utf-8", errors="replace")
        yield from file_documents(repo, rel, text, root=root)
