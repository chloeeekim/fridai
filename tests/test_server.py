"""MCP recall 툴 로직 테스트 (SDK 불필요 — 순수 로직만)."""
import unittest
from datetime import datetime, timezone

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
        self.assertIn("Q: JWT 만료 어떻게?", out)         # 질문 원문(데이터)은 그대로
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


if __name__ == "__main__":
    unittest.main()
