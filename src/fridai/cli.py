"""fridai CLI (stdlib argparse) — just two things: indexing + running the MCP server.

  fridai index [--source all|code|commits|agent] [--path DIR]
  fridai mcp        # MCP server (agents recall via the recall tool)
  fridai stats
No local LLM / answer generation (search & recall only). Embedding via fastembed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import config, embeddings
from .core.store import Store
from .core.sources import agent_recall, code, commits


def _open_store(redact: bool = True) -> Store:
    config.ensure_home()
    fresh = not Path(config.DB_PATH).exists()
    store = Store(config.DB_PATH, redact=redact)
    if fresh:
        print(f"👋 fridai first run — creating the index ({config.HOME})")
        print("   embedding (fastembed):",
              "ON (semantic)" if embeddings.get_embedder() else "OFF (lexical only — `pip install fastembed`)")
    return store


def cmd_index(args) -> None:
    valid = ("agent", "code", "commits", "all")
    if args.source not in valid:
        raise SystemExit(f"--source must be one of {valid}")
    store = _open_store(redact=not args.no_redact)
    print("secret redaction:", "OFF (--no-redact)" if args.no_redact else "ON")
    embedder = None if args.no_embed else embeddings.get_embedder()
    print("semantic embedding:", "ON (fastembed)" if embedder else "OFF (lexical only)")

    if args.source in ("agent", "all"):
        r = agent_recall.index_all(store, embedder=embedder, reindex=args.reindex)
        print(f"  conversations (Claude+Codex+Gemini): {r['turns']} turns / {r['files']} sessions (skipped {r['skipped']})")

    path = args.path or "."
    if args.source in ("code", "all"):
        r = code.index_code(path, store, reindex=args.reindex, embedder=embedder,
                            prune=not args.no_prune)
        print(f"  code: {r['chunks']} chunks / {r['files']} files (skipped {r['skipped']}, pruned {r['pruned']})")
    if args.source in ("commits", "all"):
        r = commits.index_commits(path, store, reindex=args.reindex, embedder=embedder)
        print(f"  commits: {r['commits']}")

    store.close()
    print(f"Done → {config.DB_PATH}")


def cmd_mcp(args) -> None:
    from . import server
    server.serve()


def cmd_stats(args) -> None:
    store = _open_store()
    st = store.stats()
    store.close()
    print(f"\nTotal documents: {st['total']}")
    if st["by_type"]:
        print("By source:", ", ".join(f"{k}={v}" for k, v in st["by_type"].items()))
    print("Embedding (semantic):", "ON" if embeddings.get_embedder() else "OFF (lexical search)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="fridai",
                                 description="fridai MCP server — recall past code/commits/AI conversations (search-only)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ix = sub.add_parser("index", help="index sources (code/commits/AI conversations)")
    ix.add_argument("--source", default="all", help="agent | code | commits | all (default: all)")
    ix.add_argument("--path", help="target repo path for code/commits (default: current dir)")
    ix.add_argument("--reindex", action="store_true", help="ignore incremental state and reindex everything")
    ix.add_argument("--no-embed", action="store_true", help="skip embeddings (lexical only)")
    ix.add_argument("--no-prune", action="store_true", help="skip pruning deleted files (code)")
    ix.add_argument("--no-redact", action="store_true", help="turn off secret redaction (default ON)")
    ix.set_defaults(func=cmd_index)

    mp = sub.add_parser("mcp", help="run the MCP server — agents recall past memory via `recall`")
    mp.set_defaults(func=cmd_mcp)

    st = sub.add_parser("stats", help="index overview")
    st.set_defaults(func=cmd_stats)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
