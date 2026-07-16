"""공용 베이스(agent_recall) 테스트 — 파서 무관 기능: _clean·summarize·link_commits·
turn_to_document·index_sessions(엔진)·index_all(합산)."""
import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fridai.core.sources import agent_recall as ar
from fridai.core.store import Store

WHEN = datetime(2026, 6, 17, tzinfo=timezone.utc)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t.io",
                    "-c", "user.name=tester", "-c", "commit.gpgsign=false", *args],
                   capture_output=True, check=True)


class TestClean(unittest.TestCase):
    def test_strips_injected_blocks(self):
        self.assertNotIn("무시", ar._clean("<system-reminder>무시</system-reminder>본문"))
        out = ar._clean("앞 <task-notification>\nx\n</task-notification> 뒤")
        self.assertNotIn("task-notification", out)
        self.assertIn("앞", out)
        self.assertIn("뒤", out)


class TestSummarize(unittest.TestCase):
    def test_first_meaningful_sentence(self):
        self.assertEqual(ar.summarize("헤더를 추가했습니다. 그리고 더 많은 내용."),
                         "헤더를 추가했습니다.")

    def test_truncates_long(self):
        s = ar.summarize("x" * 300, n=50)
        self.assertLessEqual(len(s), 50)

    def test_empty(self):
        self.assertEqual(ar.summarize(""), "")


class TestLinkCommits(unittest.TestCase):
    def test_no_git_repo_yields_no_commits(self):
        t = ar.Turn(question="q", when=WHEN, cwd="/nonexistent/repo", files=["A.kt"])
        ar.link_commits(t)
        self.assertEqual(t.commits, [])

    def _repo_with_commit(self):
        repo = Path(tempfile.mkdtemp())
        _git(repo, "init", "-q")
        (repo / "Auth.kt").write_text("x", encoding="utf-8")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "fix: token refresh")
        iso = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%cI"],
                             capture_output=True, text=True).stdout.strip()
        return repo, datetime.fromisoformat(iso.replace("Z", "+00:00"))  # 3.10 호환

    def test_file_match_commit_linked(self):
        repo, ctime = self._repo_with_commit()
        t = ar.Turn(question="q", when=ctime, cwd=str(repo), files=["Auth.kt"])
        ar.link_commits(t)
        self.assertEqual(len(t.commits), 1)
        self.assertEqual(t.commits[0][2], "file")            # 파일 일치
        self.assertIn("token refresh", t.commits[0][1])

    def test_time_proximity_fallback(self):
        repo, ctime = self._repo_with_commit()
        t = ar.Turn(question="q", when=ctime, cwd=str(repo), files=["other.txt"])
        ar.link_commits(t)
        self.assertEqual(t.commits[0][2], "time")            # 파일 불일치 → 시간근접


class TestTurnToDocument(unittest.TestCase):
    def _turn(self):
        return ar.Turn(question="JWT 만료 어떻게?", answer="refresh로 재발급함", when=WHEN,
                       cwd="/x/repo", repo="repo", files=["a.py"], session_id="s1")

    def test_defaults_to_claude(self):
        d = ar.turn_to_document(self._turn())
        self.assertEqual(d.source_type, "agent_turn")
        self.assertEqual(d.meta["agent"], "claude")
        self.assertTrue(d.id.startswith("agent_turn:"))
        self.assertIn("refresh로 재발급", d.meta["answer_summary"])

    def test_explicit_agent_tag(self):
        self.assertEqual(ar.turn_to_document(self._turn(), agent="codex").meta["agent"], "codex")


class TestIndexSessions(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write("{}\n"); f.close()
        self.path = Path(f.name)

    def tearDown(self):
        self.store.close()
        self.path.unlink(missing_ok=True)

    def _parse(self, path):
        return [ar.Turn(question="질문1 도커", when=WHEN, repo="r", session_id="s"),
                ar.Turn(question="질문2 인증", when=WHEN, repo="r", session_id="s")]

    def test_engine_indexes_tags_and_skips(self):
        r1 = ar.index_sessions(self.store, [self.path], self._parse,
                               agent="codex", state_prefix="codex")
        self.assertEqual((r1["files"], r1["turns"]), (1, 2))
        # agent 태깅 + state_prefix 키
        self.assertEqual(self.store.search_lexical("도커")[0].document.meta["agent"], "codex")
        self.assertIsNotNone(self.store.get_state(f"codex:{self.path}"))
        # 2차: mtime 동일 → skip
        r2 = ar.index_sessions(self.store, [self.path], self._parse,
                               agent="codex", state_prefix="codex")
        self.assertEqual(r2["skipped"], 1)

    def test_reindex_forces(self):
        ar.index_sessions(self.store, [self.path], self._parse, state_prefix="codex")
        r = ar.index_sessions(self.store, [self.path], self._parse,
                              state_prefix="codex", reindex=True)
        self.assertEqual(r["files"], 1)


class TestIndexAll(unittest.TestCase):
    def test_empty_dirs_sum_to_zero(self):
        # 세 에이전트 모듈을 실제로 import·호출하고 합산하는지(디렉터리 없으면 0)
        store = Store(":memory:")
        empty = Path(tempfile.mkdtemp())
        try:
            r = ar.index_all(store, empty / "a", empty / "b", empty / "c")
            self.assertEqual(r, {"turns": 0, "files": 0, "skipped": 0})
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
