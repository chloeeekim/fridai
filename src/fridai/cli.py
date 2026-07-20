"""fridai CLI (stdlib argparse) — just two things: indexing + running the MCP server.

  fridai index [--source all|code|commits|agent] [--path DIR]
  fridai mcp        # MCP server (agents recall via the recall tool)
  fridai stats
No local LLM / answer generation (search & recall only). Embedding via fastembed.
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
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


def _run_index(store, source, path, embedder, *, reindex=False, prune=True) -> dict:
    """Run one indexing pass for the selected source(s). Returns per-source result dicts.
    Incremental (mtime/hash state), so repeated passes are cheap — used by both one-shot and --watch."""
    out: dict = {}
    if source in ("agent", "all"):
        out["agent"] = agent_recall.index_all(store, embedder=embedder, reindex=reindex)
    if source in ("code", "all"):
        out["code"] = code.index_code(path, store, reindex=reindex, embedder=embedder, prune=prune)
    if source in ("commits", "all"):
        out["commits"] = commits.index_commits(path, store, reindex=reindex, embedder=embedder)
    return out


def _print_index_result(res: dict) -> None:
    if "agent" in res:
        r = res["agent"]
        print(f"  conversations (Claude+Codex+Gemini): {r['turns']} turns / {r['files']} sessions (skipped {r['skipped']})")
    if "code" in res:
        r = res["code"]
        print(f"  code: {r['chunks']} chunks / {r['files']} files (skipped {r['skipped']}, pruned {r['pruned']})")
    if "commits" in res:
        print(f"  commits: {res['commits']['commits']}")


def _reindexed_counts(res: dict) -> dict:
    """What got (re)processed this pass — a 'something changed' signal for --watch."""
    return {
        "turns": (res.get("agent") or {}).get("turns", 0),
        "chunks": (res.get("code") or {}).get("chunks", 0),
        "commits": (res.get("commits") or {}).get("commits", 0),
    }


def cmd_index(args) -> None:
    valid = ("agent", "code", "commits", "all")
    if args.source not in valid:
        raise SystemExit(f"--source must be one of {valid}")
    store = _open_store(redact=not args.no_redact)
    print("secret redaction:", "OFF (--no-redact)" if args.no_redact else "ON")
    embedder = None if args.no_embed else embeddings.get_embedder()
    print("semantic embedding:", "ON (fastembed)" if embedder else "OFF (lexical only)")
    path = args.path or "."
    prune = not args.no_prune
    try:
        _print_index_result(_run_index(store, args.source, path, embedder,
                                       reindex=args.reindex, prune=prune))
        print(f"Done → {config.DB_PATH}")
        if args.watch:
            print(f"👀 watching (interval {args.interval}s, Ctrl-C to stop)")
            while True:
                time.sleep(args.interval)
                n = _reindexed_counts(_run_index(store, args.source, path, embedder, prune=prune))
                if any(n.values()):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] reindexed: "
                          f"+{n['turns']} turns, +{n['chunks']} chunks, +{n['commits']} commits")
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        store.close()


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


_HOOK_MARKER = "# fridai-auto-reindex"


def cmd_install_hook(args) -> None:
    """Install a git post-commit hook that reindexes on each commit (no running process needed)."""
    repo = Path(args.path or ".").resolve()
    gitdir = repo / ".git"
    if not gitdir.is_dir():
        raise SystemExit(f"Not a git repo: {repo}")
    hooks = gitdir / "hooks"
    hooks.mkdir(exist_ok=True)
    hook = hooks / "post-commit"
    if hook.exists() and not args.force:
        if _HOOK_MARKER not in hook.read_text(encoding="utf-8", errors="replace"):
            raise SystemExit(
                f"A post-commit hook already exists (not fridai's). Use --force to overwrite: {hook}")
    hook.write_text(
        "#!/bin/sh\n"
        f"{_HOOK_MARKER} — installed by `fridai install-hook`\n"
        f'fridai index --source {args.source} --path "{repo}" >/dev/null 2>&1 || true\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)
    print(f"✅ post-commit hook installed: {hook}")
    print(f"   `fridai index --source {args.source}` will now run on every commit.")


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
    ix.add_argument("--watch", action="store_true", help="keep reindexing on an interval (Ctrl-C to stop)")
    ix.add_argument("--interval", type=int, default=15, help="--watch polling interval in seconds (default: 15)")
    ix.set_defaults(func=cmd_index)

    mp = sub.add_parser("mcp", help="run the MCP server — agents recall past memory via `recall`")
    mp.set_defaults(func=cmd_mcp)

    st = sub.add_parser("stats", help="index overview")
    st.set_defaults(func=cmd_stats)

    ih = sub.add_parser("install-hook", help="install a git post-commit hook that reindexes on each commit")
    ih.add_argument("--path", help="target repo (default: current dir)")
    ih.add_argument("--source", default="all", help="what the hook indexes (default: all)")
    ih.add_argument("--force", action="store_true", help="overwrite an existing post-commit hook")
    ih.set_defaults(func=cmd_install_hook)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
