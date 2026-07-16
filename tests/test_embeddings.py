"""fastembed 전용 임베더 테스트 (fastembed 미설치여도 통과 — 폴백/식별자 검증)."""
import os
import unittest

from fridai.core import embeddings
from fridai.core.models import Document


class FakeEmbedder:
    def embed(self, text):
        return [0.1, 0.2, 0.3]


class TestEmbedder(unittest.TestCase):
    def test_backend_none_forces_no_embedder(self):
        prev = os.environ.get("FRIDAI_EMBED_BACKEND")
        os.environ["FRIDAI_EMBED_BACKEND"] = "none"
        try:
            self.assertIsNone(embeddings.get_embedder())
        finally:                                   # 전역 격리값을 지우지 않도록 원복
            if prev is None:
                os.environ.pop("FRIDAI_EMBED_BACKEND", None)
            else:
                os.environ["FRIDAI_EMBED_BACKEND"] = prev

    def test_model_id_is_fastembed(self):
        self.assertTrue(embeddings.FastEmbedEmbedder(model="m").model_id.startswith("fastembed:"))

    def test_no_openai_compat_embedder(self):
        # 로컬 LLM/HTTP 임베더는 fridai에서 제외됨
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


if __name__ == "__main__":
    unittest.main()
