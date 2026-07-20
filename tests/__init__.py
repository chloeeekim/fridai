"""Test isolation — keep tests off the local environment (home dir, real agent
history, network embedding models).

When this package is loaded (before any test import), FRIDAI_* defaults are pinned
to temporary, isolated paths. `setdefault` means an explicit value from CI or a
developer is respected. As a result the suite reproduces identically on any machine
(without touching real ~/.fridai/~/.codex etc., and without downloading models).
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="fridai_tests_")

os.environ.setdefault("FRIDAI_HOME", os.path.join(_TMP, "home"))
os.environ.setdefault("FRIDAI_CLAUDE_PROJECTS", os.path.join(_TMP, "claude"))
os.environ.setdefault("FRIDAI_CODEX_SESSIONS", os.path.join(_TMP, "codex"))
os.environ.setdefault("FRIDAI_GEMINI_SESSIONS", os.path.join(_TMP, "gemini"))
# Embedder off -> avoid downloading the fastembed model (deterministic lexical search).
os.environ.setdefault("FRIDAI_EMBED_BACKEND", "none")
