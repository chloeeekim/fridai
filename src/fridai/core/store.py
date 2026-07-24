"""sqlite3 + FTS5 store.

- documents      : normalized document body/meta
- documents_fts  : FTS5 full-text (lexical) search — works without any embedder
- vectors        : embeddings as float32 BLOB. Search is numpy-vectorized (pure-python fallback)
"""
from __future__ import annotations

import array
import json
import math
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from . import redact as _redact

try:
    import numpy as _np
except Exception:                       # no numpy -> pure-python fallback
    _np = None

from .models import Document, SearchHit

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    source_type TEXT NOT NULL,
    repo        TEXT,
    path        TEXT,
    title       TEXT,
    text        TEXT,
    ts          TEXT,
    meta_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_documents_repo ON documents(repo);
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
    USING fts5(id UNINDEXED, title, text, tokenize='unicode61');
CREATE TABLE IF NOT EXISTS vectors (
    doc_id TEXT PRIMARY KEY,
    vec    BLOB NOT NULL          -- float32 packed
);
CREATE TABLE IF NOT EXISTS index_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Safely handle tokens that could break FTS5 query syntax
_FTS_TOKEN = re.compile(r"[^\w가-힣]+", re.UNICODE)


def _to_utc_iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _fts_match(query: str) -> str:
    """Turn a user query into an FTS5 MATCH expression. Per-token **prefix matching** + OR.

    Prefix matching (`"mount"*`) matters especially for Korean: the unicode61 tokenizer
    doesn't split particles, so "마운트는" is one token — prefix matching still catches it via "마운트".
    """
    toks = [t for t in _FTS_TOKEN.split(query) if t]
    return " OR ".join(f'"{t}"*' for t in toks)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def _pack(vec) -> bytes:
    return array.array("f", vec).tobytes()


def _unpack(blob: bytes) -> array.array:
    a = array.array("f")
    a.frombytes(blob)
    return a


class Store:
    def __init__(self, path: str | Path, redact: bool = True):
        self.path = str(path)
        self.redact = redact          # mask secrets at index time
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(_SCHEMA)
        self._migrate_vectors()

    def _migrate_vectors(self) -> None:
        """In-place convert old (vec_json TEXT) -> float32 BLOB (vec); no re-embedding needed."""
        cols = [r[1] for r in self.con.execute("PRAGMA table_info(vectors)")]
        if "vec" in cols or "vec_json" not in cols:
            return
        self.con.execute("ALTER TABLE vectors RENAME TO vectors_old")
        self.con.execute("CREATE TABLE vectors (doc_id TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        for doc_id, vj in self.con.execute("SELECT doc_id, vec_json FROM vectors_old"):
            try:
                self.con.execute("INSERT OR REPLACE INTO vectors(doc_id, vec) VALUES (?,?)",
                                 (doc_id, _pack(json.loads(vj))))
            except Exception:
                pass
        self.con.execute("DROP TABLE vectors_old")
        self.con.commit()
        self.con.execute("VACUUM")      # reclaim disk (JSON->BLOB shrinks it a lot)

    # ── writes ──
    def upsert(self, docs: list[Document]) -> int:
        n = 0
        for d in docs:
            if self.redact:
                _redact.redact_document(d)           # mask secrets
            self.con.execute(
                "INSERT OR REPLACE INTO documents"
                "(id, source_type, repo, path, title, text, ts, meta_json)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (d.id, d.source_type, d.repo, d.path, d.title, d.text,
                 _to_utc_iso(d.timestamp), json.dumps(d.meta, ensure_ascii=False)),
            )
            self.con.execute("DELETE FROM documents_fts WHERE id = ?", (d.id,))
            # if an enrichment summary exists, include it in FTS so the 'why' is searchable too
            summary = (d.meta or {}).get("summary") or ""
            fts_text = f"{summary}\n{d.text or ''}" if summary else (d.text or "")
            self.con.execute(
                "INSERT INTO documents_fts(id, title, text) VALUES (?,?,?)",
                (d.id, d.title or "", fts_text),
            )
            if d.embedding is not None:
                self.con.execute(
                    "INSERT OR REPLACE INTO vectors(doc_id, vec) VALUES (?,?)",
                    (d.id, _pack(d.embedding)),
                )
            n += 1
        self.con.commit()
        return n

    # ── reads ──
    def _row_to_doc(self, row: sqlite3.Row) -> Document:
        ts = None
        if row["ts"]:
            try:
                ts = datetime.fromisoformat(row["ts"])
            except ValueError:
                ts = None
        return Document(
            id=row["id"], source_type=row["source_type"], repo=row["repo"],
            path=row["path"], title=row["title"], text=row["text"], timestamp=ts,
            meta=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    def get(self, doc_id: str) -> Document | None:
        row = self.con.execute("SELECT * FROM documents WHERE id = ?", (doc_id,)).fetchone()
        return self._row_to_doc(row) if row else None

    def _filter_sql(self, repo, source_type, since, alias="d"):
        clauses, params = [], []
        if repo:
            clauses.append(f"{alias}.repo = ?"); params.append(repo)
        if source_type:
            clauses.append(f"{alias}.source_type = ?"); params.append(source_type)
        if since:
            clauses.append(f"{alias}.ts >= ?"); params.append(_to_utc_iso(since))
        return clauses, params

    def search_lexical(self, query, k=10, repo=None, source_type=None, since=None) -> list[SearchHit]:
        match = _fts_match(query)
        if not match:
            return []
        clauses, params = self._filter_sql(repo, source_type, since)
        where = " AND " + " AND ".join(clauses) if clauses else ""
        sql = (
            "SELECT d.*, bm25(documents_fts) AS rank "
            "FROM documents_fts JOIN documents d ON d.id = documents_fts.id "
            "WHERE documents_fts MATCH ?" + where +
            " ORDER BY rank LIMIT ?"
        )
        rows = self.con.execute(sql, [match, *params, k]).fetchall()
        # lower bm25 = more relevant -> negate to make a score
        return [SearchHit(self._row_to_doc(r), -float(r["rank"])) for r in rows]

    def search_vector(self, qvec, k=10, repo=None, source_type=None, since=None) -> list[SearchHit]:
        clauses, params = self._filter_sql(repo, source_type, since)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.con.execute(
            "SELECT d.*, v.vec FROM vectors v JOIN documents d ON d.id = v.doc_id" + where,
            params,
        ).fetchall()
        if not rows:
            return []
        stored_dim = len(rows[0]["vec"]) // 4          # float32 = 4 bytes
        if stored_dim != len(qvec):                    # embedder backend/model mismatch -> lexical fallback
            return []
        if _np is not None:
            # load all rows into one matrix, then vectorized cosine (tens of times faster at scale)
            mat = _np.frombuffer(b"".join(r["vec"] for r in rows), dtype=_np.float32)
            mat = mat.reshape(len(rows), -1)
            q = _np.asarray(qvec, dtype=_np.float32)
            sims = (mat @ q) / (_np.linalg.norm(mat, axis=1) * (_np.linalg.norm(q) or 1.0) + 1e-9)
            top = _np.argsort(-sims)[:k]
            return [SearchHit(self._row_to_doc(rows[i]), float(sims[i])) for i in top]
        # fallback: pure python
        scored = [(_cosine(qvec, _unpack(r["vec"])), r) for r in rows]
        scored.sort(key=lambda x: -x[0])
        return [SearchHit(self._row_to_doc(r), s) for s, r in scored[:k]]

    def recent(self, source_type=None, repo=None, since=None, limit=50) -> list[Document]:
        clauses, params = self._filter_sql(repo, source_type, since, alias="documents")
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self.con.execute(
            "SELECT * FROM documents" + where + " ORDER BY ts DESC LIMIT ?",
            [*params, limit],
        ).fetchall()
        return [self._row_to_doc(r) for r in reversed(rows)]  # oldest -> newest

    def rescan_redact(self) -> int:
        """Re-redact already-indexed documents (to clean an existing db). Returns count changed."""
        changed = 0
        rows = self.con.execute("SELECT id FROM documents").fetchall()
        for (doc_id,) in rows:
            doc = self.get(doc_id)
            if doc is None:
                continue
            if _redact.redact_document(doc) > 0:
                summary = (doc.meta or {}).get("summary") or ""
                fts_text = f"{summary}\n{doc.text or ''}" if summary else (doc.text or "")
                self.con.execute(
                    "UPDATE documents SET title=?, text=?, meta_json=? WHERE id=?",
                    (doc.title, doc.text, json.dumps(doc.meta, ensure_ascii=False), doc_id))
                self.con.execute("DELETE FROM documents_fts WHERE id=?", (doc_id,))
                self.con.execute("INSERT INTO documents_fts(id, title, text) VALUES (?,?,?)",
                                 (doc_id, doc.title or "", fts_text))
                changed += 1
        self.con.commit()
        return changed

    def paths(self, source_type: str, repo: str) -> set[str]:
        """Set of distinct paths indexed for that source/repo."""
        rows = self.con.execute(
            "SELECT DISTINCT path FROM documents WHERE source_type=? AND repo=?",
            (source_type, repo)).fetchall()
        return {r[0] for r in rows}

    def stats(self) -> dict:
        total = self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        by_type = dict(self.con.execute(
            "SELECT source_type, COUNT(*) FROM documents GROUP BY source_type").fetchall())
        by_repo = dict(self.con.execute(
            "SELECT repo, COUNT(*) FROM documents GROUP BY repo").fetchall())
        # agent_turn docs broken down by source agent (meta['agent']: claude/codex/gemini)
        by_agent = dict(self.con.execute(
            "SELECT json_extract(meta_json, '$.agent'), COUNT(*) FROM documents "
            "WHERE source_type='agent_turn' AND json_extract(meta_json, '$.agent') IS NOT NULL "
            "GROUP BY 1").fetchall())
        last_indexed = self.con.execute("SELECT MAX(ts) FROM documents").fetchone()[0]
        return {"total": total, "by_type": by_type, "by_repo": by_repo,
                "by_agent": by_agent, "last_indexed": last_indexed}

    def forget_repo(self, repo: str) -> dict:
        """Remove every document for a repo (+ its FTS/vector rows) and the repo-scoped
        incremental state (code/commits), so a later `fridai index` restores it cleanly.
        Agent-turn state is keyed per session file, not per repo — those turns come back
        only on `fridai index --reindex`. Returns {documents, states}."""
        ids = [r[0] for r in self.con.execute(
            "SELECT id FROM documents WHERE repo=?", (repo,)).fetchall()]
        for did in ids:
            self.con.execute("DELETE FROM documents WHERE id=?", (did,))
            self.con.execute("DELETE FROM documents_fts WHERE id=?", (did,))
            self.con.execute("DELETE FROM vectors WHERE doc_id=?", (did,))
        prefix, commits_key = f"code:{repo}:", f"commits:{repo}"
        keys = [r[0] for r in self.con.execute("SELECT key FROM index_state").fetchall()
                if r[0].startswith(prefix) or r[0] == commits_key]
        for k in keys:
            self.con.execute("DELETE FROM index_state WHERE key=?", (k,))
        self.con.commit()
        return {"documents": len(ids), "states": len(keys)}

    def reset(self) -> int:
        """Wipe the entire index — documents, FTS, vectors, and all incremental state.
        Fully re-buildable with `fridai index` (source data on disk is untouched).
        Returns the number of documents removed."""
        n = self.con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        self.con.executescript(
            "DELETE FROM documents; DELETE FROM documents_fts; "
            "DELETE FROM vectors; DELETE FROM index_state;")
        self.con.commit()
        return n

    def delete_by_path(self, source_type: str, repo: str, path: str) -> int:
        """Remove all existing documents for a given file/path (cleanup before re-chunking)."""
        ids = [r[0] for r in self.con.execute(
            "SELECT id FROM documents WHERE source_type=? AND repo=? AND path=?",
            (source_type, repo, path)).fetchall()]
        for did in ids:
            self.con.execute("DELETE FROM documents WHERE id=?", (did,))
            self.con.execute("DELETE FROM documents_fts WHERE id=?", (did,))
            self.con.execute("DELETE FROM vectors WHERE doc_id=?", (did,))
        self.con.commit()
        return len(ids)

    # ── incremental indexing state ──
    def get_state(self, key: str) -> str | None:
        row = self.con.execute("SELECT value FROM index_state WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        self.con.execute(
            "INSERT OR REPLACE INTO index_state(key, value) VALUES (?,?)", (key, value))
        self.con.commit()

    def delete_state(self, key: str) -> None:
        self.con.execute("DELETE FROM index_state WHERE key=?", (key,))
        self.con.commit()

    # ── embedder identity (index/query consistency) ──
    # Records which embedder built the vectors so queries can detect a mismatch
    # (e.g. FRIDAI_FASTEMBED_MODEL differs between indexing and querying processes).
    _EMBEDDER_KEY = "meta:embedder_model_id"

    def get_embedder_id(self) -> str | None:
        return self.get_state(self._EMBEDDER_KEY)

    def set_embedder_id(self, model_id: str) -> None:
        self.set_state(self._EMBEDDER_KEY, model_id)

    def close(self):
        self.con.close()
