"""Embedding provider — fastembed (onnx) only + fallback.

fridai uses only fastembed so it runs 100% offline (no local LLM / remote API).
`get_embedder()`: FastEmbedEmbedder if fastembed is installed, else None (lexical fallback).
Disable with `FRIDAI_EMBED_BACKEND=none`. `.model_id` identifies index-query consistency.
"""
from __future__ import annotations

from . import config

# fastembed (onnxruntime, no torch). If absent -> None -> lexical (FTS) fallback.
try:
    from fastembed import TextEmbedding as _FastEmbed
except Exception:
    _FastEmbed = None


class FastEmbedEmbedder:
    """onnx-based local embedder (no external server). Model is lazy-loaded once."""
    _model = None

    def __init__(self, model: str | None = None):
        self.model = model or config.FASTEMBED_MODEL

    @property
    def model_id(self) -> str:
        return f"fastembed:{self.model}"

    def available(self) -> bool:
        return _FastEmbed is not None

    def _get_model(self):
        if FastEmbedEmbedder._model is None:
            FastEmbedEmbedder._model = _FastEmbed(model_name=self.model)
        return FastEmbedEmbedder._model

    def embed_many(self, texts: list[str]) -> list:
        """Embed a whole batch in ONE fastembed call (it batches internally at batch_size=256).

        Passing texts one at a time pays per-call onnx overhead for every chunk, which
        dominates indexing time; a single batched call is many times faster. Returns a list
        aligned with `texts`; all-None if fastembed is absent or the call fails."""
        if _FastEmbed is None:
            return [None] * len(texts)
        if not texts:
            return []
        try:
            vecs = list(self._get_model().embed([t[:2000] for t in texts]))
            return [[float(x) for x in v] for v in vecs]
        except Exception:
            return [None] * len(texts)

    def embed(self, text: str) -> list[float] | None:
        return self.embed_many([text])[0]


def get_embedder(model: str | None = None):
    """Pick the available embedder. fastembed -> None. Disable with FRIDAI_EMBED_BACKEND=none."""
    if config.EMBED_BACKEND == "none":
        return None
    fe = FastEmbedEmbedder(model)
    return fe if fe.available() else None


def _embed_input(doc) -> str:
    """Embedding input. If an enrichment summary exists, prepend it so the 'why' is in the vector too."""
    summary = (doc.meta or {}).get("summary")
    return f"{summary}\n{doc.text}" if summary else doc.text


def embed_documents(docs, embedder) -> list:
    """Fill in embeddings for docs (skip those already set). Leaves None if the embedder fails.

    Embeds all pending docs in a single batched call when the embedder exposes `embed_many`
    (much faster than one call per doc); otherwise falls back to per-doc `embed`."""
    pending = [d for d in docs if d.embedding is None]
    if not pending:
        return docs
    texts = [_embed_input(d) for d in pending]
    embed_many = getattr(embedder, "embed_many", None)
    if embed_many is not None:
        for d, vec in zip(pending, embed_many(texts)):
            d.embedding = vec
    else:
        for d, text in zip(pending, texts):
            d.embedding = embedder.embed(text)
    return docs
