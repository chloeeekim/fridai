"""User-note source tests — add_note builds a note Document, defaults its repo to the
current git repo, embeds/redacts on save, and is recallable like any other source."""
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fridai.core.sources import notes
from fridai.core.store import Store

WHEN = datetime(2026, 7, 20, tzinfo=timezone.utc)


class _StubEmbedder:
    def embed(self, text):
        return [1.0, 0.0, 0.0]


class TestAddNote(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_builds_note_document(self):
        doc = notes.add_note(self.store, "결제 재시도는 3회로 제한.\n이유: 멱등키 TTL 5분.",
                             repo="pay", when=WHEN)
        self.assertEqual(doc.source_type, "note")
        self.assertEqual(doc.repo, "pay")
        self.assertEqual(doc.title, "결제 재시도는 3회로 제한.")        # first line
        self.assertIn("멱등키", doc.text)                              # full body kept
        self.assertEqual(doc.timestamp, WHEN)
        self.assertEqual(self.store.get(doc.id).source_type, "note")   # persisted

    def test_empty_text_saves_nothing(self):
        self.assertIsNone(notes.add_note(self.store, "   \n  ", repo="r"))
        self.assertEqual(self.store.stats()["total"], 0)

    def test_recallable_by_lexical_search(self):
        notes.add_note(self.store, "도커 볼륨 마운트는 compose로.", repo="r", when=WHEN)
        hits = self.store.search_lexical("마운트", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.source_type, "note")

    def test_embedding_attached_when_embedder_given(self):
        doc = notes.add_note(self.store, "임베딩 노트", repo="r", when=WHEN,
                             embedder=_StubEmbedder())
        self.assertEqual(doc.embedding, [1.0, 0.0, 0.0])

    def test_secret_is_redacted_on_save(self):
        doc = notes.add_note(
            self.store, "AWS 키는 AKIAIOSFODNN7EXAMPLE 였음", repo="r", when=WHEN)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", self.store.get(doc.id).text)

    def test_default_repo_is_detected_from_git(self):
        repo = Path(tempfile.mkdtemp()) / "myproj"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        # repo=None -> detect from cwd; point _cwd_repo at the temp repo via its path
        self.assertEqual(notes._cwd_repo(str(repo)), "myproj")

    def test_unique_ids_per_note(self):
        a = notes.add_note(self.store, "같은 내용", repo="r", when=WHEN)
        b = notes.add_note(self.store, "같은 내용", repo="r",
                           when=datetime(2026, 7, 21, tzinfo=timezone.utc))
        self.assertNotEqual(a.id, b.id)                                # timestamp keeps them distinct
        self.assertEqual(self.store.stats()["total"], 2)


if __name__ == "__main__":
    unittest.main()
