"""F1 code 인덱싱 단위 테스트. 임시 git 레포 생성 후 검증."""
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
        lines = [f"L{i}" for i in range(1, 151)]  # 150줄
        chunks = list(code.chunk_lines(lines, window=60, overlap=15))
        self.assertEqual(chunks[0][0], 1)            # 첫 청크 시작=1
        self.assertEqual(chunks[0][1], 60)           # 끝=60
        self.assertEqual(chunks[-1][1], 150)         # 마지막 청크 끝=총줄수
        self.assertEqual(chunks[1][0], 46)           # step=45 → 다음 시작 46

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
        # class A(+method) 와 class B 가 분리
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
        self.assertEqual(chunks[0][0], 1)              # preamble 시작=1
        self.assertEqual(chunks[-1][1], 4)             # 마지막 끝=총줄수

    def test_unsupported_lang_falls_back_to_windows(self):
        src = [f"line {i}" for i in range(150)]
        sym = list(code.chunk_symbols(src, "text"))
        win = list(code.chunk_lines(src))
        self.assertEqual(sym, win)                      # 폴백 동일

    def test_large_symbol_subsplit(self):
        src = ["def big():"] + [f"    x{i} = {i}" for i in range(200)]
        chunks = list(code.chunk_symbols(src, "python"))
        self.assertGreater(len(chunks), 1)              # MAX_UNIT 초과 → 재분할


class TestIgnore(unittest.TestCase):
    def test_is_ignored_patterns(self):
        pats = ["*.env", "secrets/", "config/local.py"]
        self.assertTrue(code.is_ignored("app.env", pats))
        self.assertTrue(code.is_ignored("deep/dir/x.env", pats))      # *.env 하위경로
        self.assertTrue(code.is_ignored("secrets/key.txt", pats))     # 디렉터리
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
            self.assertNotIn("secret.env", paths)     # .fridaiignore로 제외
        finally:
            store.close()


class TestRepoIndexing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.repo = Path(self.tmp)
        _git(self.repo, "init", "-q")
        (self.repo / "app.py").write_text(
            "\n".join(f"line {i} jwt token refresh" for i in range(150)), encoding="utf-8")
        (self.repo / "logo.png").write_bytes(b"\x89PNG\x00\x00binary")  # 제외 대상
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
        self.assertEqual(res["files"], 1)          # png 제외, py만
        self.assertGreater(res["chunks"], 1)       # 150줄 → 여러 청크
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
        self.assertEqual(res["files"], 1)          # 변경 감지 → 재인덱싱

    def test_prune_removes_deleted_file_chunks(self):
        (self.repo / "b.py").write_text("\n".join(f"x{i} = {i}" for i in range(80)),
                                        encoding="utf-8")
        _git(self.repo, "add", "-A")
        code.index_code(self.repo, self.store)
        self.assertIn("b.py", self.store.paths("code", self.repo.name))
        # b.py 삭제 후 재인덱싱 → b.py 청크가 정리됨
        (self.repo / "b.py").unlink()
        _git(self.repo, "add", "-A")
        res = code.index_code(self.repo, self.store)
        self.assertEqual(res["pruned"], 1)
        self.assertNotIn("b.py", self.store.paths("code", self.repo.name))
        self.assertIn("app.py", self.store.paths("code", self.repo.name))   # 남은 파일 유지
        self.assertIsNone(self.store.get_state(f"code:{self.repo.name}:b.py"))

    def test_no_prune_keeps_stale(self):
        (self.repo / "b.py").write_text("y = 1", encoding="utf-8")
        _git(self.repo, "add", "-A")
        code.index_code(self.repo, self.store)
        (self.repo / "b.py").unlink()
        _git(self.repo, "add", "-A")
        res = code.index_code(self.repo, self.store, prune=False)
        self.assertEqual(res["pruned"], 0)
        self.assertIn("b.py", self.store.paths("code", self.repo.name))      # 정리 안 함

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
        self.assertTrue(hits)                       # 벡터가 저장되어 검색됨
        self.assertEqual(hits[0].document.source_type, "code")


if __name__ == "__main__":
    unittest.main()
