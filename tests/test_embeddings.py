"""fastembed-only embedder tests (pass even without fastembed installed — fallback/identifier checks)."""
import unittest
from unittest import mock

from fridai.core import config, embeddings
from fridai.core.models import Document


class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class FakeBatchEmbedder:
    """Embedder exposing embed_many, to verify batched indexing."""
    def __init__(self):
        self.batches = []

    def embed_many(self, texts):
        self.batches.append(list(texts))
        return [[float(len(t))] for t in texts]


class TestEmbedder(unittest.TestCase):
    def test_backend_none_forces_no_embedder(self):
        with mock.patch.object(config, "EMBED_BACKEND", "none"):
            self.assertIsNone(embeddings.get_embedder())

    def test_model_id_is_fastembed(self):
        self.assertTrue(embeddings.FastEmbedEmbedder(model="m").model_id.startswith("fastembed:"))

    def test_no_openai_compat_embedder(self):
        # the local-LLM / HTTP embedder is excluded from fridai
        self.assertFalse(hasattr(embeddings, "OpenAICompatEmbedder"))

    def test_embed_documents_with_injected_embedder(self):
        docs = [Document(id="1", source_type="code", repo="r", path="p", title="t", text="x")]
        embeddings.embed_documents(docs, FakeEmbedder())
        self.assertEqual(docs[0].embedding, [0.1, 0.2, 0.3])

    def test_embed_documents_skips_existing(self):
        docs = [Document(id="1", source_type="code", repo="r", path="p", title="t",
                         text="x", embedding=[9.0])]
        embeddings.embed_documents(docs, FakeEmbedder())
        self.assertEqual(docs[0].embedding, [9.0])

    def test_embed_documents_batches_in_one_call(self):
        docs = [Document(id=str(i), source_type="code", repo="r", path="p", title="t",
                         text="x" * i) for i in (1, 2, 3)]
        fe = FakeBatchEmbedder()
        embeddings.embed_documents(docs, fe)
        self.assertEqual(len(fe.batches), 1)              # a single batched call, not one per doc
        self.assertEqual(len(fe.batches[0]), 3)           # all pending docs in that batch
        self.assertEqual([d.embedding for d in docs], [[1.0], [2.0], [3.0]])

    def test_embed_documents_batches_only_pending(self):
        docs = [Document(id="0", source_type="code", repo="r", path="p", title="t",
                         text="keep", embedding=[9.0]),
                Document(id="1", source_type="code", repo="r", path="p", title="t", text="new")]
        fe = FakeBatchEmbedder()
        embeddings.embed_documents(docs, fe)
        self.assertEqual(fe.batches, [["new"]])           # pre-embedded doc excluded from the batch
        self.assertEqual(docs[0].embedding, [9.0])


if __name__ == "__main__":
    unittest.main()
