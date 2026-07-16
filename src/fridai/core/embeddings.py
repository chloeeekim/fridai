"""Embedding provider — fastembed (onnx) only + fallback.

fridai uses only fastembed so it runs 100% offline (no local LLM / remote API).
`get_embedder()`: FastEmbedEmbedder if fastembed is installed, else None (lexical fallback).
Disable with `FRIDAI_EMBED_BACKEND=none`. `.model_id` identifies index-query consistency.
"""
from __future__ import annotations

import os

# fastembed (onnxruntime, no torch). If absent -> None -> lexical (FTS) fallback.
try:
    from fastembed import TextEmbedding as _FastEmbed
except Exception:
    _FastEmbed = None

_FASTEMBED_MODEL = os.environ.get("FRIDAI_FASTEMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")


class FastEmbedEmbedder:
    """onnx-based local embedder (no external server). Model is lazy-loaded once."""
    _model = None

    def __init__(self, model: str | None = None):
        self.model = model or _FASTEMBED_MODEL

    @property
    def model_id(self) -> str:
        return f"fastembed:{self.model}"

    def available(self) -> bool:
        return _FastEmbed is not None

    def embed(self, text: str) -> list[float] | None:
        if _FastEmbed is None:
            return None
        try:
            if FastEmbedEmbedder._model is None:
                FastEmbedEmbedder._model = _FastEmbed(model_name=self.model)
            vecs = list(FastEmbedEmbedder._model.embed([text[:2000]]))
            return [float(x) for x in vecs[0]] if vecs else None
        except Exception:
            return None


def get_embedder(model: str | None = None):
    """Pick the available embedder. fastembed -> None. Disable with FRIDAI_EMBED_BACKEND=none."""
    if os.environ.get("FRIDAI_EMBED_BACKEND", "").lower() == "none":
        return None
    fe = FastEmbedEmbedder(model)
    return fe if fe.available() else None


def _embed_input(doc) -> str:
    """Embedding input. If an enrichment summary exists, prepend it so the 'why' is in the vector too."""
    summary = (doc.meta or {}).get("summary")
    return f"{summary}\n{doc.text}" if summary else doc.text


def embed_documents(docs, embedder) -> list:
    """Fill in embeddings for docs (skip those already set). Leaves None if the embedder fails."""
    for d in docs:
        if d.embedding is None:
            d.embedding = embedder.embed(_embed_input(d))
    return docs
