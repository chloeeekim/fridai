"""MCP recall tool logic tests (no SDK needed — pure logic only)."""
import unittest
from datetime import datetime, timedelta, timezone

from fridai import server
from fridai.core.models import Document
from fridai.core.store import Store


class TestRecallTool(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.store.upsert([
            Document(id="t1", source_type="agent_turn", repo="r", path="s", title="q",
                     text="JWT 만료 처리", timestamp=datetime(2026, 6, 1, tzinfo=timezone.utc),
                     meta={"question": "JWT 만료 어떻게?", "answer_summary": "refresh로 재발급",
                           "commits": [["abc1234", "fix: refresh", "file"]]}),
            Document(id="c1", source_type="code", repo="r", path="auth.py",
                     title="auth.py:1-9", text="def refresh_token(): ...",
                     meta={"path": "auth.py", "start_line": 1, "end_line": 9}),
        ])

    def tearDown(self):
        self.store.close()

    def test_returns_results_with_citations(self):
        out = server.recall_tool("JWT", store=self.store)
        self.assertIn("memory item", out)
        self.assertIn("Q: JWT 만료 어떻게?", out)         # the original question (data) is kept as-is
        self.assertIn("refresh로 재발급", out)
        self.assertIn("commit abc1234", out)

    def test_source_filter(self):
        out = server.recall_tool("refresh", source_type="code", store=self.store)
        self.assertIn("auth.py", out)
        self.assertNotIn("Q: JWT", out)

    def test_no_hits_message(self):
        out = server.recall_tool("존재하지않는키워드xyz", store=self.store)
        self.assertIn("no relevant memory", out)

    def test_resolve_repo_scope(self):
        self.assertEqual(server._resolve_repo(None, "flower-device"), "flower-device")
        self.assertIsNone(server._resolve_repo("all", "flower-device"))
        self.assertEqual(server._resolve_repo("other-repo", "flower-device"), "other-repo")
        self.assertIsNone(server._resolve_repo(None, None))

    def test_cwd_repo_scopes_recall(self):
        self.store.upsert([
            Document(id="x1", source_type="code", repo="repoA", path="a.py", title="t",
                     text="공통키워드 alpha"),
            Document(id="x2", source_type="code", repo="repoB", path="b.py", title="t",
                     text="공통키워드 beta"),
        ])
        out = server.recall_tool("공통키워드", store=self.store, cwd_repo="repoA")
        self.assertIn("repo=repoA", out)
        self.assertIn("alpha", out)
        self.assertNotIn("beta", out)
        out_all = server.recall_tool("공통키워드", store=self.store, cwd_repo="repoA", repo="all")
        self.assertIn("all repos", out_all)


class TestSince(unittest.TestCase):
    def test_parse_since(self):
        now = datetime.now(timezone.utc)
        self.assertIsNone(server._parse_since(None))
        self.assertIsNone(server._parse_since("nonsense"))
        d = server._parse_since("7d")
        self.assertAlmostEqual((now - d).total_seconds(), 7 * 86400, delta=60)

    def test_since_filters_old_out(self):
        s = Store(":memory:")
        now = datetime.now(timezone.utc)
        s.upsert([
            Document(id="old", source_type="code", repo="r", path="a.py", title="a",
                     text="docker mount old", timestamp=now - timedelta(days=30)),
            Document(id="new", source_type="code", repo="r", path="b.py", title="b",
                     text="docker mount new", timestamp=now - timedelta(days=1)),
        ])
        try:
            out = server.recall_tool("docker mount", store=s, repo="all", since="7d")
            self.assertIn("b.py", out)
            self.assertNotIn("a.py", out)        # a 30-day-old doc is excluded by since=7d
        finally:
            s.close()


class TestEmptyIndex(unittest.TestCase):
    def test_empty_index_message(self):
        s = Store(":memory:")
        try:
            self.assertIn("index is empty", server.recall_tool("anything", store=s))
        finally:
            s.close()

    def test_nonempty_no_match_is_not_empty_message(self):
        s = Store(":memory:")
        s.upsert([Document(id="c", source_type="code", repo="r", path="p", title="t", text="hello")])
        try:
            out = server.recall_tool("zzznomatchzzz", store=s, repo="all")
            self.assertIn("no relevant memory", out)   # not empty — just no match
            self.assertNotIn("index is empty", out)
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
