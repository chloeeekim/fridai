"""F4 extension — parse Gemini CLI sessions (chats JSONL) into Turns.

Location: `~/.gemini/tmp/<project>/chats/session-*.jsonl` (config.GEMINI_SESSIONS).
Format (verified against gemini-cli 0.50 real data):
  each project dir holds `.project_root` (absolute working path) + a `chats/` log.
  chats/session-*.jsonl: first line is a header {sessionId,projectHash,...}, later lines are
    message records {id, timestamp, type, content, [thoughts, tokens, model, toolCalls]}.
    type: "user" (question) / "gemini" (answer) / "info"/"error" (noise).
    content=[{text}]. toolCalls=[{name, args:{dir_path/file_path/…}, result}].
  NOTE: the first type=="user" message carries an injected `<session_context>` block -> filtered.
  Nested `chats/<sessionId>/<uuid>.jsonl` (checkpoints) are duplicates -> excluded (glob-limited).

Turn / commit-matching / Document conversion / incremental engine are shared from agent_recall.
"""
from __future__ import annotations

import json
from pathlib import Path

from .. import config
from . import agent_recall
from .agent_recall import Turn, _clean, _files_from_args, _text, _ts

_NOISE_PREFIXES = ("<session_context>", "<environment_context>")


def _project_root(chats_file: Path) -> str:
    """Recover the project working path from a chats file path (.project_root, else dir name)."""
    proj_dir = chats_file.parent.parent          # …/tmp/<project>/chats/x.jsonl → <project>
    try:
        return (proj_dir / ".project_root").read_text(encoding="utf-8").strip()
    except OSError:
        return str(proj_dir)


def parse_session(path: Path) -> list[Turn]:
    """Parse one Gemini chats/session-*.jsonl into a list of Turn."""
    cwd = _project_root(Path(path))
    repo = Path(cwd).name if cwd else "(unknown)"
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
            if typ is None or "content" not in rec:   # skip non-message lines (header/$set etc.)
                continue
            when = _ts(rec.get("timestamp"))
            text = _text(rec.get("content"))
            if typ == "user":
                q = _clean(text)
                if not q or q.startswith(_NOISE_PREFIXES):
                    continue
                if cur:
                    turns.append(cur)
                cur = Turn(question=q, when=when, cwd=cwd, repo=repo,
                           session_id=Path(path).stem)
            elif typ == "gemini" and cur is not None:
                if text:
                    cur.answer = (cur.answer + "\n" + text).strip() if cur.answer else text
                for tc in rec.get("toolCalls") or []:
                    if isinstance(tc, dict):
                        cur.tools.append(tc.get("name", "?"))
                        cur.files.extend(_files_from_args(tc.get("args")))
    if cur:
        turns.append(cur)
    for t in turns:
        t.tools = sorted(set(t.tools))
        t.files = sorted(set(t.files))
    return turns


def index_gemini(store, sessions_dir: Path | None = None, *,
                 embedder=None, reindex: bool = False) -> dict:
    """Incrementally index Gemini CLI sessions. 0 if the directory is absent. Reuses the shared engine.
    The glob `*/chats/*.jsonl` matches only per-project live sessions (nested checkpoints excluded)."""
    root = Path(sessions_dir or config.GEMINI_SESSIONS)
    if not root.exists():
        return {"turns": 0, "files": 0, "skipped": 0}
    return agent_recall.index_sessions(store, root.glob("*/chats/*.jsonl"), parse_session,
                                       embedder=embedder, reindex=reindex,
                                       agent="gemini", state_prefix="gemini")
