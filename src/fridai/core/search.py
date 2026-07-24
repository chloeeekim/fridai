"""Pure retrieval (search). No LLM dependency.

Lexical (BM25) / vector (cosine) / hybrid (RRF) retrieval + source citation + context serialization.
"""
from __future__ import annotations

import sys

from . import config
from .models import Document, SearchHit
from .store import Store

_warned_mismatch: set = set()


def _embedder_matches(store: Store, embedder) -> bool:
    """True unless the index was built with a different embedder than the query one.

    Only the vector dimension is enforced deeper in the store; two same-dim models would
    otherwise return silently-wrong results. On mismatch we warn once (stderr — never stdout,
    which is the MCP protocol channel) and let the caller fall back to lexical search."""
    stored = store.get_embedder_id()
    current = getattr(embedder, "model_id", None)
    if not stored or not current or stored == current:
        return True
    key = (stored, current)
    if key not in _warned_mismatch:
        _warned_mismatch.add(key)
        print(f"fridai: index was built with embedder '{stored}' but the current embedder is "
              f"'{current}'; falling back to lexical search. Re-run `fridai index --reindex "
              f"--source all` or set FRIDAI_FASTEMBED_MODEL to match.", file=sys.stderr)
    return False


def citation(doc: Document) -> str:
    """Human-readable source string per document type."""
    when = doc.timestamp.astimezone().strftime("%Y-%m-%d") if doc.timestamp else "?"
    if doc.source_type == "code":
        return f"{doc.repo}/{doc.meta.get('path', doc.path)}:" \
               f"{doc.meta.get('start_line')}-{doc.meta.get('end_line')}"
    if doc.source_type == "commit":
        return f"{doc.repo}@{doc.meta.get('sha', doc.path)} \"{doc.title}\""
    if doc.source_type == "agent_turn":
        agent = doc.meta.get("agent")
        tag = f"[{agent}] " if agent and agent != "claude" else ""   # mark non-Claude agents
        return f"{doc.repo} {when} {tag}session:{doc.meta.get('session_title') or doc.title}"
    if doc.source_type == "note":
        return f"note {when}"
    return doc.path or doc.id


RRF_C = 60   # RRF constant (larger softens rank-gap impact). Conventional value 60.


def rrf_fuse(ranked_lists: list[list[SearchHit]], k: int, c: int = RRF_C) -> list[SearchHit]:
    """Reciprocal Rank Fusion: fuse multiple ranked results by rank.

    A document at rank r (0-based) in each list gets 1/(c+r+1) added to its score.
    Robust even when score scales differ (BM25 vs cosine).
    """
    scores: dict[str, float] = {}
    docs: dict[str, SearchHit] = {}
    for hits in ranked_lists:
        for rank, h in enumerate(hits):
            did = h.document.id
            scores[did] = scores.get(did, 0.0) + 1.0 / (c + rank + 1)
            docs.setdefault(did, h)
    fused = [SearchHit(docs[i].document, s) for i, s in scores.items()]
    fused.sort(key=lambda x: -x.score)
    return fused[:k]


def hybrid_retrieve(store: Store, query: str, k: int = 5, *, embedder,
                    repo=None, source_type=None, since=None) -> list[SearchHit]:
    """Fuse lexical (BM25) and vector (cosine) via RRF. Returns lexical only if no vectors."""
    pool = max(k * 2, 10)
    lex = store.search_lexical(query, k=pool, repo=repo, source_type=source_type, since=since)
    qv = embedder.embed(query)
    vec = store.search_vector(qv, k=pool, repo=repo, source_type=source_type,
                              since=since) if qv else []
    if not vec:
        return lex[:k]
    return rrf_fuse([lex, vec], k)


def work_signal(doc: Document) -> bool:
    """Is this document 'actual work output'? code/commit/note are inherently work output.
    An agent_turn counts as a work turn only if it has file edits (files) or a **file-matched
    resulting commit**. (Time-proximity 'guessed' commits also attach to question turns, so
    they don't count as a work signal.)"""
    if doc.source_type != "agent_turn":
        return True
    m = doc.meta
    if m.get("files"):
        return True
    return any(len(c) > 2 and c[2] == "file" for c in (m.get("commits") or []))


def rerank_work_signal(hits: list[SearchHit], penalty: int | None = None) -> list[SearchHit]:
    """Demote agent_turns with no work signal (pure question/recall turns) by `penalty` positions.

    Mitigates recursive self-pollution: 'how did I do this?' question turns pushing out the
    actual solution turns/commits. Position-based penalty (ties preserve original order).
    penalty=None uses config.WORK_PENALTY.
    """
    if penalty is None:
        penalty = config.WORK_PENALTY
    keyed = [(i + (0 if work_signal(h.document) else penalty), i, h)
             for i, h in enumerate(hits)]
    keyed.sort(key=lambda x: (x[0], x[1]))
    return [h for _, _, h in keyed]


import re as _re

DEDUP_JACCARD = 0.8   # question-token Jaccard at/above this = treated as the same question


def _q_tokens(doc: Document) -> set:
    text = doc.meta.get("question") or doc.title or doc.text or ""
    return set(_re.findall(r"\w+", text.lower()))


def dedup_results(hits: list[SearchHit], thr: float = DEDUP_JACCARD) -> list[SearchHit]:
    """Collapse near-duplicate agent_turns (repeated same/similar questions) to one representative.

    Preserves post-rerank order, keeping the first (= top-ranked / work-signal) item as the
    representative. code/commit/note are not affected.
    """
    seen: list[set] = []
    out = []
    for h in hits:
        if h.document.source_type == "agent_turn":
            toks = _q_tokens(h.document)
            if toks and any(len(toks & s) / len(toks | s) >= thr for s in seen):
                continue           # near-identical to a question already seen -> skip
            seen.append(toks)
        out.append(h)
    return out


def retrieve(store: Store, query: str, k: int = 5, *, embedder=None,
             repo=None, source_type=None, since=None) -> list[SearchHit]:
    """Hybrid (BM25+vector RRF) if embedder given, else lexical. Work-signal rerank + dedup, then top-k."""
    pool = max(k * 3, 15)           # retrieve generously, then trim after rerank/dedup
    if embedder is not None and not _embedder_matches(store, embedder):
        embedder = None             # index built with a different model -> avoid wrong-vector hits
    if embedder is not None:
        hits = hybrid_retrieve(store, query, pool, embedder=embedder,
                               repo=repo, source_type=source_type, since=since)
    else:
        hits = store.search_lexical(query, k=pool, repo=repo, source_type=source_type, since=since)
    return dedup_results(rerank_work_signal(hits))[:k]


def build_context(hits: list[SearchHit]) -> str:
    """Context for an LLM prompt. Numbers each block and attaches its source."""
    return "\n\n".join(
        f"[{i}] ({citation(h.document)})\n{h.document.text}"
        for i, h in enumerate(hits, 1)
    )
