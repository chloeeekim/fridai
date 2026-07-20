"""F1 code indexing unit tests. Create a temporary git repo and verify."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from fridai.core.sources import code
from fridai.core.store import Store


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)


class TestChunking(unittest.TestCase):
    def test_chunk_lines_overlap_and_bounds(self):
        lines = [f"L{i}" for i in range(1, 151)]  # 150 lines
        chunks = list(code.chunk_lines(lines, window=60, overlap=15))
        self.assertEqual(chunks[0][0], 1)            # first chunk starts at 1
        self.assertEqual(chunks[0][1], 60)           # ends at 60
        self.assertEqual(chunks[-1][1], 150)         # last chunk ends at total line count
        self.assertEqual(chunks[1][0], 46)           # step=45 -> next starts at 46

    def test_empty_file_no_chunks(self):
        self.assertEqual(list(code.chunk_lines([])), [])


class TestSymbolChunking(unittest.TestCase):
    def test_python_splits_by_def_with_preamble(self):
        src = ("import os\nimport sys\n\n"
               "def foo():\n    return 1\n\n"
               "def bar():\n    return 2\n").splitlines()
        chunks = list(code.chunk_symbols(src, "python"))
        texts = [c[2] for c in chunks]
        self.assertEqual(len(chunks), 3)               # preamble + foo + bar
        self.assertIn("import os", texts[0])           # preamble
        self.assertTrue(any("def foo" in t and "def bar" not in t for t in texts))
        self.assertTrue(any("def bar" in t for t in texts))

    def test_python_class_boundary(self):
        src = "class A:\n    def m(self):\n        pass\nclass B:\n    pass".splitlines()
        chunks = list(code.chunk_symbols(src, "python"))
        # class A (+method) and class B are separated
        self.assertTrue(any("class A" in c[2] for c in chunks))
        self.assertTrue(any("class B" in c[2] and "class A" not in c[2] for c in chunks))

    def test_kotlin_fun_boundary(self):
        src = ("package x\n\n"
               "fun first() {\n    println(1)\n}\n"
               "private fun second() {\n    println(2)\n}").splitlines()
        chunks = list(code.chunk_symbols(src, "kotlin"))
        self.assertTrue(any("fun first" in c[2] and "fun second" not in c[2] for c in chunks))
        self.assertTrue(any("fun second" in c[2] for c in chunks))

    def test_line_numbers_absolute_and_contiguous(self):
        src = ["import a", "", "def f():", "    return 1"]
        chunks = list(code.chunk_symbols(src, "python"))
        self.assertEqual(chunks[0][0], 1)              # preamble starts at 1
        self.assertEqual(chunks[-1][1], 4)             # last ends at total line count

    def test_unsupported_lang_falls_back_to_windows(self):
        src = [f"line {i}" for i in range(150)]
        sym = list(code.chunk_symbols(src, "text"))
        win = list(code.chunk_lines(src))
        self.assertEqual(sym, win)                      # same as the fallback

    def test_large_symbol_subsplit(self):
        src = ["def big():"] + [f"    x{i} = {i}" for i in range(200)]
        chunks = list(code.chunk_symbols(src, "python"))
        self.assertGreater(len(chunks), 1)              # exceeds MAX_UNIT -> re-split


class TestIgnore(unittest.TestCase):
    def test_is_ignored_patterns(self):
        pats = ["*.env", "secrets/", "config/local.py"]
        self.assertTrue(code.is_ignored("app.env", pats))
        self.assertTrue(code.is_ignored("deep/dir/x.env", pats))      # *.env in a subpath
        self.assertTrue(code.is_ignored("secrets/key.txt", pats))     # directory
        self.assertTrue(code.is_ignored("config/local.py", pats))
        self.assertFalse(code.is_ignored("src/app.py", pats))
        self.assertFalse(code.is_ignored("config/prod.py", pats))

    def test_fridaiignore_excludes_file_from_index(self):
        import subprocess, tempfile
        repo = Path(tempfile.mkdtemp())
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        (repo / "app.py").write_text("x = 1")
        (repo / "secret.env").write_text("API_KEY=abc123")
        (repo / ".fridaiignore").write_text("*.env\n# comment\n")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        store = Store(":memory:")
        try:
            code.index_code(repo, store)
            paths = store.paths("code", repo.name)
            self.assertIn("app.py", paths)
            self.assertNotIn("secret.env", paths)     # excluded by .fridaiignore
        finally:
            store.close()


class TestRepoIndexing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Path(self.tmp)
        _git(self.repo, "init", "-q")
        (self.repo / "app.py").write_text(
            "\n".join(f"line {i} jwt token refresh" for i in range(150)), encoding="utf-8")
        (self.repo / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")  # to be excluded
        _git(self.repo, "add", "-A")
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_tracked_files_lists_added(self):
        files = code.tracked_files(self.repo)
        self.assertIn("app.py", files)
        self.assertIn("logo.png", files)

    def test_binary_excluded(self):
        self.assertFalse(code.is_indexable(self.repo / "logo.png"))
        self.assertTrue(code.is_indexable(self.repo / "app.py"))

    def test_index_creates_chunks_and_is_searchable(self):
        res = code.index_code(self.repo, self.store)
        self.assertEqual(res["files"], 1)          # png excluded, py only
        self.assertGreater(res["chunks"], 1)       # 150 lines -> multiple chunks
        hits = self.store.search_lexical("jwt")
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.source_type, "code")
        self.assertIn(":", hits[0].document.title)  # path:start-end

    def test_incremental_skips_unchanged(self):
        code.index_code(self.repo, self.store)
        res2 = code.index_code(self.repo, self.store)
        self.assertEqual(res2["files"], 0)
        self.assertEqual(res2["skipped"], 1)

    def test_reindex_after_change(self):
        code.index_code(self.repo, self.store)
        (self.repo / "app.py").write_text("changed content here", encoding="utf-8")
        res = code.index_code(self.repo, self.store)
        self.assertEqual(res["files"], 1)          # change detected -> reindexed

    def test_prune_removes_deleted_file_chunks(self):
        (self.repo / "b.py").write_text("\n".join(f"x{i} = {i}" for i in range(80)),
                                        encoding="utf-8")
        _git(self.repo, "add", "-A")
        code.index_code(self.repo, self.store)
        self.assertIn("b.py", self.store.paths("code", self.repo.name))
        # after deleting b.py and reindexing -> b.py chunks are pruned
        (self.repo / "b.py").unlink()
        _git(self.repo, "add", "-A")
        res = code.index_code(self.repo, self.store)
        self.assertEqual(res["pruned"], 1)
        self.assertNotIn("b.py", self.store.paths("code", self.repo.name))
        self.assertIn("app.py", self.store.paths("code", self.repo.name))   # remaining file kept
        self.assertIsNone(self.store.get_state(f"code:{self.repo.name}:b.py"))

    def test_no_prune_keeps_stale(self):
        (self.repo / "b.py").write_text("y = 1", encoding="utf-8")
        _git(self.repo, "add", "-A")
        code.index_code(self.repo, self.store)
        (self.repo / "b.py").unlink()
        _git(self.repo, "add", "-A")
        res = code.index_code(self.repo, self.store, prune=False)
        self.assertEqual(res["pruned"], 0)
        self.assertIn("b.py", self.store.paths("code", self.repo.name))      # not pruned

    def test_meta_stores_repo_root(self):
        code.index_code(self.repo, self.store)
        hit = self.store.search_lexical("jwt")[0]
        self.assertEqual(hit.document.meta["root"], str(self.repo.resolve()))

    def test_index_with_embedder_enables_vector_search(self):
        class FakeEmbedder:
            def embed(self, text):
                return [1.0, 0.0, 0.0]
        code.index_code(self.repo, self.store, embedder=FakeEmbedder())
        hits = self.store.search_vector([1.0, 0.0, 0.0], k=3)
        self.assertTrue(hits)                       # vector stored and searchable
        self.assertEqual(hits[0].document.source_type, "code")


if __name__ == "__main__":
    unittest.main()
