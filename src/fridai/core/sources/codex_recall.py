"""F4 extension — parse OpenAI Codex CLI rollout (JSONL) into Turns.

Location: `~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl` (config.CODEX_SESSIONS).
Format (verified against codex-cli 0.143 real data):
  each line is {"timestamp", "type", "payload"}. (other types: event_msg/turn_context/world_state ignored)
  - type=="session_meta": payload.cwd for the working dir. (no git branch here -> branch left empty)
  - type=="response_item":
      payload.type=="message": role=="user"->question, "assistant"->answer, "developer"->excluded.
        content[] is {type: input_text|output_text, text} -> concatenate text.
        NOTE: role=="user" also carries injected `<environment_context>` blocks -> filtered as noise.
      payload.type=="function_call": name (e.g. exec_command) + arguments. exec_command is a
        shell cmd with no structured file path, so no file extraction (only tools with a
        path-like key). -> commit matching relies on the time-proximity fallback.

Turn / commit-matching / Document conversion / incremental engine are shared from agent_recall.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .. import config
from . import agent_recall
from .agent_recall import Turn

_USER_ROLES = {"user"}          # actual user prompts. "developer" (instructions)/"system" excluded as noise.
# Injected blocks that arrive as role=="user" but aren't real questions (confirmed on real data).
_NOISE_PREFIXES = ("<environment_context>", "<user_instructions>")
_FILE_KEYS = ("path", "file_path", "filename", "notebook_path")


def _ts(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None


def _text(content) -> str:
    """Text only from message content. content is a string or a [{type,text}] array."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(b.get("text", "") for b in content
                     if isinstance(b, dict) and b.get("text")).strip()


def _files_from_args(arguments) -> list[str]:
    """Best-effort file-path extraction from function_call arguments (a JSON string)."""
    try:
        obj = json.loads(arguments) if isinstance(arguments, str) else (arguments or {})
    except (ValueError, TypeError):
        return []
    if not isinstance(obj, dict):
        return []
    return [obj[k] for k in _FILE_KEYS if isinstance(obj.get(k), str)]


def parse_session(path: Path) -> list[Turn]:
    """Parse one Codex rollout file into a list of Turn. File order = chronological order."""
    cwd = branch = ""
    turns: list[Turn] = []
    cur: Turn | None = None
    with Path(path).open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            typ = rec.get("type")
            payload = rec.get("payload") or {}
            when = _ts(rec.get("timestamp"))
            if typ == "session_meta":
                cwd = payload.get("cwd") or cwd
                git = payload.get("git")
                branch = (git.get("branch") if isinstance(git, dict) else None) \
                    or payload.get("git_branch") or branch
                continue
            if typ != "response_item":
                continue
            ptype = payload.get("type")
            if ptype == "message":
                role = payload.get("role")
                text = _text(payload.get("content"))
                if role in _USER_ROLES:
                    q = agent_recall._clean(text)
                    if not q or q.startswith(_NOISE_PREFIXES):
                        continue        # injected environment_context etc. isn't a real question
                    if cur:
                        turns.append(cur)
                    cur = Turn(question=q, when=when, cwd=cwd,
                               repo=Path(cwd).name if cwd else "(unknown)",
                               branch=branch, session_id=Path(path).stem)
                elif role == "assistant" and cur is not None and text:
                    cur.answer = (cur.answer + "\n" + text).strip() if cur.answer else text
            elif ptype == "function_call" and cur is not None:
                cur.tools.append(payload.get("name", "?"))
                cur.files.extend(_files_from_args(payload.get("arguments")))
    if cur:
        turns.append(cur)
    for t in turns:
        t.tools = sorted(set(t.tools))
        t.files = sorted(set(t.files))
    return turns


def index_codex(store, sessions_dir: Path | None = None, *,
                embedder=None, reindex: bool = False) -> dict:
    """Incrementally index Codex CLI sessions. 0 if the directory is absent. Reuses the shared engine."""
    root = Path(sessions_dir or config.CODEX_SESSIONS)
    if not root.exists():
        return {"turns": 0, "files": 0, "skipped": 0}
    return agent_recall.index_sessions(store, root.rglob("rollout-*.jsonl"), parse_session,
                                       embedder=embedder, reindex=reindex,
                                       agent="codex", state_prefix="codex")
