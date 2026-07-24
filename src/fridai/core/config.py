"""fridai global settings — paths/constants. No external dependencies."""
from __future__ import annotations

import os
from pathlib import Path

# Data home (kept separate from the working repo). Override with FRIDAI_HOME (for tests).
HOME = Path(os.environ.get("FRIDAI_HOME", Path.home() / ".fridai"))
DB_PATH = HOME / "index.db"

# F4 sources: coding-agent conversation locations. Silently skipped if absent.
CLAUDE_PROJECTS = Path(
    os.environ.get("FRIDAI_CLAUDE_PROJECTS", Path.home() / ".claude" / "projects")
)
CODEX_SESSIONS = Path(
    os.environ.get("FRIDAI_CODEX_SESSIONS", Path.home() / ".codex" / "sessions")
)
GEMINI_SESSIONS = Path(
    os.environ.get("FRIDAI_GEMINI_SESSIONS", Path.home() / ".gemini" / "tmp")
)

# High-entropy secret heuristic. Off by default (precise rules only) — long camelCase
# identifiers etc. produce too many false positives.
REDACT_ENTROPY = os.environ.get("FRIDAI_REDACT_ENTROPY", "").lower() in ("1", "true", "yes")

# How far to demote agent_turns with no work signal (pure question/recall turns) in ranking.
# Larger = solution artifacts (code/commits/work turns) rank first. 0 disables the rerank.
WORK_PENALTY = int(os.environ.get("FRIDAI_WORK_PENALTY", "8"))

# Time window (minutes) for matching a question to its resulting commit.
COMMIT_WINDOW_MIN = int(os.environ.get("FRIDAI_COMMIT_WINDOW_MIN", "180"))

# Embedding (fastembed) settings. FASTEMBED_MODEL must match between indexing and
# querying — changing it needs a full `fridai index --reindex --source all`.
# EMBED_BACKEND="none" disables embeddings (lexical-only search).
FASTEMBED_MODEL = os.environ.get("FRIDAI_FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")
EMBED_BACKEND = os.environ.get("FRIDAI_EMBED_BACKEND", "").lower()


def ensure_home() -> Path:
    """Ensure the data-home directory exists and return it."""
    HOME.mkdir(parents=True, exist_ok=True)
    return HOME
