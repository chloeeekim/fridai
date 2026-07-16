# Building fridai from scratch — design & build order

A roadmap for how one would implement fridai from zero. The guiding idea:
**stand up the thinnest working spine first, then expand outward — validating the
uncertain parts against real data as early as possible.** Test isolation, CI,
redaction, and the source-adapter abstraction are treated as day-one concerns, not
afterthoughts.

## Phase 0 — Scaffold & decisions
- Repo, `pyproject` (deps: `numpy`, `mcp`, `fastembed`), src layout, entry point (`fridai`).
- **From the start:** hermetic tests (`tests/__init__.py` isolates all `FRIDAI_*` paths and
  disables the embedder) + a CI matrix (Python 3.10–3.14).
- Up-front decisions: fastembed-only, English throughout, the `~/.fridai` data convention,
  and a **source-adapter interface** for multi-agent parsing.

> Why: bolting CI/isolation on later lets the code drift into machine-specific assumptions.
> (The Python-3.10-only `datetime.fromisoformat("…Z")` bug was caught precisely because the
> matrix existed.)

## Phase 1 — Storage & search spine (walking skeleton)
1. `models` (Document / SearchHit / make_id).
2. `store` (sqlite + FTS5 — **lexical search first**, no vectors yet) + `redact` (on by default).
3. Hand-insert one document → confirm `search_lexical` works.

> Why: the smallest thing that "stores and finds" — no embeddings, no agents. Once this
> works, everything else just attaches to it.

## Phase 2 — One source, end to end (prove the pipeline)
- Start with the most **deterministic** source, `code` (git ls-files → function/class chunking →
  index), and run the full index→search path through it.

> Why: a source that needs no schema guessing lets you stabilize the indexing engine first
> (incremental mtime/hash state, prune, idempotent upsert).

## Phase 3 — Search quality
- `embeddings` (fastembed) + `store.search_vector` (numpy cosine, dim guard) →
  `search` hybrid (BM25 + vector via **RRF**) → work-signal rerank + dedup + citation.

> Why: confirm recall is actually *useful* against a real index, tuning by eye/measurement.

## Phase 4 — MCP server (validate the core value early)
- `server.recall_tool` + `serve` (FastMCP), `cli mcp`.
- **Register with Claude Code and confirm `recall` actually works live** — even with only the
  `code` source indexed.

> Why: "an agent recalls your memory" is the product's reason to exist. Prove it early.

## Phase 5 — Agent sources (uncertain → real data first)
1. `agent_recall` **shared base** (Turn · commit matching · `index_sessions` engine ·
   `turn_to_document`) — designed as an interface from the start.
2. `claude_recall` — inspect real `~/.claude` files, *then* write the parser.
3. `codex_recall` / `gemini_recall` — **install the CLI → generate a real session → dump the
   on-disk schema → only then write the parser.**

> Why: the most painful lesson. Codex's documented schema differed from reality in two places;
> Gemini's format had moved from `.json` to `.jsonl`. **Never write a parser from docs alone.**

## Phase 6 — CLI & operational finish
- `cli` (index / mcp / stats) flags, `.fridaiignore`, the `commits` source, complete incremental indexing.

## Phase 7 — Distribution hardening
- README (en/ko), environment-variable reference, security docs, redaction entropy opt-in,
  and a **fastembed real-run smoke** job in CI (`continue-on-error` so network/HF flakiness
  doesn't fail the whole run).

---

## Cross-cutting principles (as important as the order)
- **Tests/CI from day one.** Each phase advances while green across 3.10–3.14.
- **Confirm parser schemas against real data before writing them.** (Phase 5.)
- **Incremental & idempotent** indexing baked into the engine (cheap re-runs).
- **Redaction on by default** — security is part of the value proposition.
- **Adapter abstraction from the start**, not when the second agent appears.

## What I'd do differently vs. how this repo actually came to be
This repo was **extracted** from a larger project. Building fresh, I would:
- Design the source-adapter interface up front (avoid the later `agent_recall` ↔ `claude_recall` split).
- Start English-only, fastembed-only, with consistent `fridai` naming (things that were
  translated/cleaned up after the fact).
- Make the `recall` output branding consistent from the beginning.

## One-line summary
Spine (store + lexical) → one source end-to-end → search quality → MCP proven early →
uncertain agent parsers validated on real data → finish & ship — with tests, CI, redaction,
and the adapter abstraction in place from the start.
