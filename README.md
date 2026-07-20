# fridai 🛠️

[![PyPI](https://img.shields.io/pypi/v/fridai.svg)](https://pypi.org/project/fridai/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*English | [한국어](README.ko.md)*

A lightweight **MCP server** that lets coding agents (Claude Code, etc.) recall your past
**code, commits, and AI conversations** via the `recall` tool — **100% local**.
Search & recall only — **no local LLM required.**

- **Search-only (read-only).** The calling agent (an LLM) does the reasoning; fridai
  just returns evidence with sources.
- **Local embeddings (fastembed, onnx).** No Ollama or external API. Semantic search works
  out of the box (falls back to lexical if fastembed isn't installed).
- **Multi-agent history.** Auto-indexes Claude Code · OpenAI Codex CLI · Gemini CLI conversations.
- **Private by design.** Your memory never leaves the machine; secrets are auto-masked at index time.

**Status:** early development. **Requires** Python 3.10+ and `git` (for code/commit indexing).

## How it works

1. **Index** (`fridai index`) builds a local database from three source types:
   - **AI conversations** — parses each agent's session logs into question→answer *turns*, and
     matches a question to its **resulting git commit** (time window ∩ touched files).
   - **Code** — chunks git-tracked files by function/class (line-window fallback), with line ranges.
   - **Commits** — indexes git commit history (subject + changed files).
2. Everything is stored in **sqlite + FTS5** (lexical) plus **float32 vectors** at `~/.fridai/index.db`.
3. **Recall** fuses lexical (BM25) and vector (cosine) results via **RRF**, then reranks so real work
   artifacts (code/commits/edited turns) outrank bare question turns, and dedupes repeated questions.

## Install

```bash
pipx install fridai        # isolated (recommended)
# or: pip install fridai   # in a venv
```

Pulls `numpy` + `fastembed` (onnx) + `mcp`. Registers the `fridai` command.

## Quickstart

```bash
fridai index --source all             # build the index (current repo + all agent conversations)
fridai stats                          # index overview
claude mcp add fridai -- fridai mcp   # register with Claude Code
```

After registering, the agent recalls via the `recall` tool. Re-run `fridai index` anytime to
refresh — it's **incremental** (only changed files/sessions/new commits are reprocessed), so it's cheap.
To keep it fresh automatically, run `fridai index --watch` (reindexes every 15s; `--interval` to change),
or `fridai install-hook` to reindex on every git commit (no running process needed).

## CLI reference

| Command | Description |
| :--- | :--- |
| `fridai index` | Build/update the index. |
| `fridai mcp` | Run the stdio MCP server. |
| `fridai stats` | Print document counts by source, a per-agent conversation breakdown, and when the index was last updated. |
| `fridai install-hook` | Install a git post-commit hook that reindexes on each commit. |

`index` flags:

| Flag | Meaning |
| :--- | :--- |
| `--source agent\|code\|commits\|all` | What to index (default `all`). `agent` = all AI conversations. |
| `--path DIR` | Target repo for code/commits (default: current directory). |
| `--reindex` | Ignore incremental state and rebuild everything. |
| `--no-embed` | Skip embeddings (lexical index only). |
| `--no-prune` | Keep chunks of files deleted from git (code). |
| `--no-redact` | Turn off secret masking (on by default). |
| `--watch` | Keep reindexing on an interval until Ctrl-C. |
| `--interval N` | `--watch` poll interval in seconds (default 15). |

## The `recall` tool (MCP)

The server exposes one tool, `recall(query, k=5, repo="", source_type="")`:

- `query` — search text (natural language and/or code identifiers).
- `k` — max results (default 5).
- `repo` — empty = current working repo (server detects cwd); `"all"` = every repo; `"<name>"` = a specific repo.
- `source_type` — empty = all; one of `agent_turn` · `code` · `commit` · `note`.

It returns text with numbered, cited hits for the agent to read — for example:

```
fridai recall — 2 memory item(s) for "docker mount" (all repos, with sources):

### [1] myrepo 2026-07-01 [codex] session:how did I add the docker mount?
Q: how did I add the docker mount?
A: added a bind mount via volumes.

### [2] myrepo/docker-compose.yml:1-20
volumes: ...
```

Non-Claude sources are tagged (`🤖 codex` / `🤖 gemini` in the CLI, `[codex]`/`[gemini]` in citations).

## Registering with other MCP clients

fridai is a standard **stdio MCP server** — the launch command is `fridai mcp`. Any MCP-capable
client can use it. For clients that read an `mcpServers` config:

```json
{
  "mcpServers": {
    "fridai": { "command": "fridai", "args": ["mcp"] }
  }
}
```

## Indexed agent sources

| Agent | Default data path | Override env var |
| :--- | :--- | :--- |
| Claude Code | `~/.claude/projects/` | `FRIDAI_CLAUDE_PROJECTS` |
| OpenAI Codex CLI | `~/.codex/sessions/` | `FRIDAI_CODEX_SESSIONS` |
| Gemini CLI | `~/.gemini/tmp/` | `FRIDAI_GEMINI_SESSIONS` |

A missing agent directory is silently skipped.

## Security

Secrets (AWS keys, GitHub/Slack tokens, PEM private keys, JWTs, `password=`…) are auto-masked
at index time (on by default). Nothing is sent off the machine.

```bash
fridai index --source code --no-redact   # disable masking
echo "*.env" >> .fridaiignore                 # exclude paths (repo root or ~/.fridai/)
```

The high-entropy heuristic is **off by default** (too many false positives on long identifiers);
enable it with `FRIDAI_REDACT_ENTROPY=1`.

## Environment variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `FRIDAI_HOME` | `~/.fridai` | Data home (index DB, etc.). |
| `FRIDAI_CLAUDE_PROJECTS` | `~/.claude/projects` | Claude Code transcript location. |
| `FRIDAI_CODEX_SESSIONS` | `~/.codex/sessions` | Codex CLI session location. |
| `FRIDAI_GEMINI_SESSIONS` | `~/.gemini/tmp` | Gemini CLI session location. |
| `FRIDAI_EMBED_BACKEND` | auto | `none` disables embeddings (lexical only). |
| `FRIDAI_FASTEMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | fastembed model name. |
| `FRIDAI_REDACT_ENTROPY` | off | `1` enables the high-entropy secret heuristic. |
| `FRIDAI_WORK_PENALTY` | `8` | How far to demote pure question turns in ranking. `0` disables. |
| `FRIDAI_COMMIT_WINDOW_MIN` | `180` | Minutes window for matching a question to its resulting commit. |

**Embedder consistency:** index and query with the same embedder. Mixing models makes vectors
incomparable even at the same dimension — the store then falls back to lexical search. Reindex
(`--reindex`) after switching models.

## Development

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

CI (GitHub Actions) runs the suite on Python 3.10–3.14 plus a fastembed smoke check on every push/PR.
Tests are hermetic — `tests/__init__.py` isolates all `FRIDAI_*` paths and disables the embedder,
so no real `~/.fridai`/`~/.codex`/etc. is touched and no model is downloaded.

## License

MIT.
