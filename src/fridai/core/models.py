"""Unified data model. Normalizes every source (code / commit / note / AI conversation) into a Document."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Document:
    """Basic unit of indexing and search."""
    id: str
    source_type: str            # "code" | "commit" | "note" | "agent_turn"
    repo: str
    path: str                   # file path / commit hash / session id
    title: str
    text: str
    timestamp: datetime | None = None
    meta: dict = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class SearchHit:
    document: Document
    score: float                # higher = more relevant


def make_id(source_type: str, *parts: str) -> str:
    """Stable document id from source/path/location (identical across reindexing)."""
    h = hashlib.sha1()
    h.update(source_type.encode())
    for p in parts:
        h.update(b"\x00")
        h.update(str(p).encode())
    return f"{source_type}:{h.hexdigest()[:16]}"
