"""F4 — Claude Code transcript (JSONL) parser.

Extraction spec:
  real question = type=="user" AND content is a string AND isMeta!=true AND isSidechain==false
  answer        = only text blocks in a type=="assistant" content[]
  context       = timestamp / cwd / gitBranch / tool_use (files)
  title         = type=="ai-title"
Location: `~/.claude/projects/<cwd>/*.jsonl` (config.CLAUDE_PROJECTS).

Turn / commit-matching / Document conversion / incremental engine are reused from
agent_recall (the shared base).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .. import config
from . import agent_recall
from .agent_recall import Turn

_NOISE_PREFIXES = (
    "<command-name>", "<command-message>", "<local-command-stdout>",
    "<bash-stdout>", "<bash-input>", "Caveat:",
    "<task-notification>",   # harness-injected background task notification — not a question
    "This session is being continued from a previous conversation",
)


def _is_real_question(rec: dict) -> bool:
    if rec.get("type") != "user" or rec.get("isMeta") is True or rec.get("isSidechain") is True:
        return False
    content = rec.get("message", {}).get("content")
    if not isinstance(content, str):
        return False
    s = content.lstrip()
    return bool(s) and not s.startswith(_NOISE_PREFIXES)


def _assistant_text(rec: dict):
    content = rec.get("message", {}).get("content")
    if not isinstance(content, list):
        return "", [], []
    texts, tools, files = [], [], []
    for block in content:
        bt = block.get("type")
        if bt == "text":
            texts.append(block.get("text", ""))
        elif bt == "tool_use":
            tools.append(block.get("name", "?"))
            inp = block.get("input", {}) or {}
            fp = inp.get("file_path") or inp.get("path") or inp.get("notebook_path")
            if fp:
                files.append(fp)
    return "\n".join(t for t in texts if t).strip(), tools, files


def _parse_ts(rec: dict):
    ts = rec.get("timestamp")
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_session(path: Path) -> list[Turn]:
    """Parse one session file into a list of Turn. File order = chronological order."""
    title, turns, cur = "", [], None
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("type") == "ai-title":
                title = rec.get("aiTitle") or title
                continue
            if _is_real_question(rec):
                if cur:
                    turns.append(cur)
                cwd = rec.get("cwd", "") or ""
                cur = Turn(
                    question=agent_recall._clean(rec["message"]["content"]), when=_parse_ts(rec),
                    cwd=cwd, repo=Path(cwd).name if cwd else "(unknown)",
                    branch=rec.get("gitBranch", "") or "",
                    session_id=rec.get("sessionId", "") or Path(path).stem,
                )
                continue
            if rec.get("type") == "assistant" and cur is not None:
                text, tools, files = _assistant_text(rec)
                if text:
                    cur.answer = (cur.answer + "\n" + text).strip() if cur.answer else text
                cur.tools.extend(tools)
                cur.files.extend(files)
    if cur:
        turns.append(cur)
    for t in turns:
        t.session_title = title
        t.tools = sorted(set(t.tools))
        t.files = sorted(set(t.files))
    return turns


def iter_turns(projects_dir: Path | None = None, link: bool = True) -> Iterator[Turn]:
    """All sessions' Turns in chronological order. If link=True, also match resulting commits."""
    root = Path(projects_dir or config.CLAUDE_PROJECTS)
    if not root.exists():
        return
    turns: list[Turn] = []
    for path in root.rglob("*.jsonl"):
        try:
            turns.extend(parse_session(path))
        except Exception:
            continue
    turns.sort(key=lambda t: t.when or datetime.min.replace(tzinfo=timezone.utc))
    for t in turns:
        if link:
            agent_recall.link_commits(t)
        yield t


def documents(projects_dir: Path | None = None):
    for turn in iter_turns(projects_dir):
        yield agent_recall.turn_to_document(turn, agent="claude")


def index_claude(store, projects_dir: Path | None = None, *,
                 embedder=None, reindex: bool = False) -> dict:
    """Index Claude Code conversations incrementally per session file (delegates to the shared engine).
    state key `agent:<path>` — compatible with existing indexes."""
    root = Path(projects_dir or config.CLAUDE_PROJECTS)
    if not root.exists():
        return {"turns": 0, "files": 0, "skipped": 0}
    return agent_recall.index_sessions(store, root.rglob("*.jsonl"), parse_session,
                                       embedder=embedder, reindex=reindex,
                                       agent="claude", state_prefix="agent")
