"""F4 shared base — common data structures & engine for coding-agent conversations
(agent-independent).

Per-agent parsers live in sibling modules: claude_recall / codex_recall / gemini_recall.
Each parser turns session files into a list of `Turn` and reuses the shared machinery here:
  - `Turn`             : normalized unit of question/answer/context (cwd·branch·files·commits)
  - `link_commits`     : match a question to its resulting commit (time window ∩ touched files)
  - `turn_to_document` : Turn -> indexed Document (records source agent in meta['agent'])
  - `index_sessions`   : mtime-incremental indexing engine over session files (parser callback)
  - `AgentAdapter`     : declares one agent source (dir, glob, parser, state key)
  - `adapters`         : the adapter registry — the single source of truth for known agents
  - `index_adapter`    : index one agent via its adapter (generic over all agents)
  - `index_all`        : sum indexing across every registered adapter

Adding a new agent means writing its `parse_session` + one `ADAPTER = AgentAdapter(...)`
and listing it in `adapters()`; the engine, index_all, and stats stay untouched.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

from .. import config, embeddings
from ..models import Document, make_id


@dataclass
class Turn:
    question: str
    answer: str = ""
    when: datetime | None = None
    cwd: str = ""
    repo: str = ""
    branch: str = ""
    session_id: str = ""
    session_title: str = ""
    tools: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    commits: list[tuple[str, str, str]] = field(default_factory=list)  # (hash, subject, kind)


def _clean(text: str) -> str:
    """Strip harness-injected blocks (system reminders / background task notifications). Shared by parsers."""
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    text = re.sub(r"<task-notification>.*?</task-notification>", "", text, flags=re.DOTALL)
    return text.strip()


# ── shared parsing helpers (used by claude/codex/gemini parsers) ──
_FILE_KEYS = ("path", "file_path", "filename", "notebook_path", "dir_path", "absolute_path")


def _ts(s):
    """Parse an ISO timestamp; 'Z' suffix supported on all versions (3.11+ needs normalizing)."""
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def _text(content) -> str:
    """Text from a message `content` that is either a plain string or a [{type, text}] array."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("text")).strip()


def _files_from_args(args) -> list[str]:
    """Best-effort file paths from a tool-call args dict or JSON string (path-like keys only)."""
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, TypeError):
            return []
    if not isinstance(args, dict):
        return []
    return [args[k] for k in _FILE_KEYS if isinstance(args.get(k), str)]


# ── question -> resulting-commit matching ──
_commit_cache: dict[str, list[tuple[datetime, str, str, set]]] = {}


def _repo_commits(cwd: str):
    if not cwd or cwd in _commit_cache:
        return _commit_cache.get(cwd, [])
    out, cur = [], None
    if (Path(cwd) / ".git").exists():
        try:
            raw = subprocess.run(
                ["git", "-C", cwd, "log", "--all", "--name-only",
                 "--pretty=format:@@@%cI%x09%h%x09%s"],
                capture_output=True, text=True, timeout=20,
            ).stdout
            for ln in raw.splitlines():
                if ln.startswith("@@@"):
                    if cur:
                        out.append(cur)
                    parts = ln[3:].split("\t", 2)
                    cur = None
                    if len(parts) == 3:
                        try:
                            # the 'Z' suffix is only accepted by fromisoformat on 3.11+ -> normalize
                            cur = (datetime.fromisoformat(parts[0].replace("Z", "+00:00")),
                                   parts[1], parts[2], set())
                        except ValueError:
                            cur = None
                elif ln.strip() and cur:
                    cur[3].add(Path(ln.strip()).name)
            if cur:
                out.append(cur)
        except Exception:
            pass
    _commit_cache[cwd] = out
    return out


def link_commits(turn: Turn, window_min: int | None = None) -> None:
    window_min = config.COMMIT_WINDOW_MIN if window_min is None else window_min
    if not turn.when:
        return
    lo, hi = turn.when, turn.when + timedelta(minutes=window_min)
    cands = [c for c in _repo_commits(turn.cwd) if lo <= c[0] <= hi]
    if not cands:
        turn.commits = []
        return
    turn_files = {Path(f).name for f in turn.files}
    strong = [(len(turn_files & c[3]), c[0] - turn.when, c) for c in cands if turn_files & c[3]]
    if strong:
        strong.sort(key=lambda x: (-x[0], x[1]))
        turn.commits = [(c[1], c[2], "file") for _, _, c in strong[:3]]
        return
    nonmerge = [c for c in cands if c[3]] or cands
    nonmerge.sort(key=lambda c: c[0] - turn.when)
    turn.commits = [(nonmerge[0][1], nonmerge[0][2], "time")]


# ── summarize / Document conversion ──
def summarize(answer: str, n: int = 130) -> str:
    if not answer:
        return ""
    text = " ".join(answer.split())
    parts = re.split(r"(?<=[.!?。])\s+|(?<=[다요죠음])\s+", text)
    s = next((p for p in parts if len(p.strip()) > 8), text)
    return s if len(s) <= n else s[: n - 1] + "…"


def turn_to_document(turn: Turn, agent: str = "claude") -> Document:
    title = turn.session_title or turn.question.splitlines()[0][:80]
    return Document(
        id=make_id("agent_turn", turn.session_id, str(turn.when), turn.question[:120]),
        source_type="agent_turn", repo=turn.repo, path=turn.session_id,
        title=title, text=f"{turn.question}\n\n{turn.answer}".strip(),
        timestamp=turn.when,
        meta={
            "question": turn.question, "answer": turn.answer,
            "answer_summary": summarize(turn.answer),
            "branch": turn.branch, "cwd": turn.cwd,
            "tools": turn.tools, "files": turn.files,
            "session_title": turn.session_title, "session_id": turn.session_id,
            "commits": turn.commits, "agent": agent,   # which agent this record came from
        },
    )


# ── shared incremental indexing engine ──
def index_sessions(store, files, parse, *, embedder=None, reindex: bool = False,
                   agent: str = "claude", state_prefix: str = "agent") -> dict:
    """Agent-independent indexing engine — index session files incrementally by mtime.

    `files`: session file paths. `parse`: (Path)->list[Turn], the per-adapter parser.
    Each file's mtime is stored in index_state (`{state_prefix}:<path>`) so unchanged files skip.
    Turn ids are stable so re-upsert is idempotent (no delete handling). Returns {turns, files, skipped}.
    """
    turns_total = files_done = skipped = 0
    for path in sorted(files):
        key = f"{state_prefix}:{path}"
        try:
            mtime = str(path.stat().st_mtime)
        except OSError:
            continue
        if not reindex and store.get_state(key) == mtime:
            skipped += 1
            continue
        docs = []
        for t in parse(path):
            link_commits(t)
            docs.append(turn_to_document(t, agent=agent))
        if embedder and docs:
            embeddings.embed_documents(docs, embedder)
        if docs:
            store.upsert(docs)
        store.set_state(key, mtime)
        files_done += 1
        turns_total += len(docs)
    return {"turns": turns_total, "files": files_done, "skipped": skipped}


# ── adapter registry: one declarative entry per coding-agent source ──
@dataclass(frozen=True)
class AgentAdapter:
    """Declares one coding-agent conversation source.

    name          : source tag stored in meta['agent'] (e.g. "claude"/"codex"/"gemini")
    default_dir   : where the agent keeps its sessions (config.*_SESSIONS)
    find_sessions : enumerate session files under a root (each agent globs differently)
    parse         : (Path) -> list[Turn], the per-agent parser
    state_prefix  : index_state key namespace for incremental mtime tracking
    """
    name: str
    default_dir: Path
    find_sessions: Callable[[Path], Iterable[Path]]
    parse: Callable[[Path], list[Turn]]
    state_prefix: str


def adapters() -> list[AgentAdapter]:
    """The registered agent adapters (single source of truth). Import is lazy so the
    parser modules can import this base module without a cycle."""
    from . import claude_recall, codex_recall, gemini_recall
    return [claude_recall.ADAPTER, codex_recall.ADAPTER, gemini_recall.ADAPTER]


def adapter(name: str) -> AgentAdapter:
    """Look up a single adapter by agent name (raises KeyError if unknown)."""
    for a in adapters():
        if a.name == name:
            return a
    raise KeyError(name)


def index_adapter(store, adapter: AgentAdapter, sessions_dir: Path | None = None, *,
                  embedder=None, reindex: bool = False) -> dict:
    """Index one agent's sessions incrementally via its adapter. Generic over all agents;
    yields {turns:0, files:0, skipped:0} if the directory is absent."""
    root = Path(sessions_dir or adapter.default_dir)
    if not root.exists():
        return {"turns": 0, "files": 0, "skipped": 0}
    return index_sessions(store, adapter.find_sessions(root), adapter.parse,
                          embedder=embedder, reindex=reindex,
                          agent=adapter.name, state_prefix=adapter.state_prefix)


def index_all(store, dirs: dict | None = None, *, embedder=None, reindex: bool = False) -> dict:
    """Index and sum conversations across every registered adapter.
    `dirs` optionally overrides a source's directory by agent name (e.g. {"codex": path});
    a missing/None entry uses the adapter's default. Called by `index --source agent`."""
    dirs = dirs or {}
    total = {"turns": 0, "files": 0, "skipped": 0}
    for a in adapters():
        r = index_adapter(store, a, dirs.get(a.name), embedder=embedder, reindex=reindex)
        for k in total:
            total[k] += r[k]
    return total
