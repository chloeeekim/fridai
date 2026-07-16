"""F1 commit 인덱싱 단위 테스트. 임시 git 레포에 실제 커밋 생성."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from fridai.core.sources import commits
from fridai.core.store import Store


def _git(repo, *args):
    # 임시 레포는 개발자의 전역 git 설정(예: commit.gpgsign=true, 회사 서명키)에
    # 의존하지 않도록 서명을 끄고 신원을 명시 → 어느 환경에서도 재현 가능.
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t.io",
                    "-c", "user.name=tester", "-c", "commit.gpgsign=false",
                    *args], capture_output=True, check=True)


class TestCommitIndexing(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _git(self.repo, "init", "-q")
        (self.repo / "a.txt").write_text("a", encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "feat: add token refresh")
        (self.repo / "b.txt").write_text("b", encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "fix: mount path bug")
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_indexes_all_commits(self):
        res = commits.index_commits(self.repo, self.store)
        self.assertEqual(res["commits"], 2)
        self.assertEqual(self.store.stats()["by_type"]["commit"], 2)

    def test_commit_searchable_with_files(self):
        commits.index_commits(self.repo, self.store)
        hits = self.store.search_lexical("mount")
        self.assertTrue(hits)
        doc = hits[0].document
        self.assertEqual(doc.source_type, "commit")
        self.assertEqual(doc.title, "fix: mount path bug")
        self.assertIn("b.txt", doc.text)        # 변경 파일이 본문에 포함

    def test_incremental_skips_when_head_unchanged(self):
        commits.index_commits(self.repo, self.store)
        self.assertEqual(commits.index_commits(self.repo, self.store)["commits"], 0)

    def test_incremental_indexes_new_commit(self):
        commits.index_commits(self.repo, self.store)
        (self.repo / "c.txt").write_text("c", encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "chore: third")
        self.assertEqual(commits.index_commits(self.repo, self.store)["commits"], 1)

    def test_empty_repo_no_commits(self):
        empty = Path(tempfile.mkdtemp())
        _git(empty, "init", "-q")
        self.assertEqual(commits.index_commits(empty, self.store)["commits"], 0)

    def test_changes_returns_stat_and_patch(self):
        sha = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        out = commits.changes(self.repo, sha)
        self.assertIsNotNone(out)
        self.assertIn("b.txt", out)        # 마지막 커밋이 건드린 파일

    def test_changes_truncates(self):
        sha = subprocess.run(["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        out = commits.changes(self.repo, sha, max_lines=1)
        self.assertIn("truncated", out)

    def test_changes_bad_sha_returns_none(self):
        self.assertIsNone(commits.changes(self.repo, "deadbeef"))


if __name__ == "__main__":
    unittest.main()
