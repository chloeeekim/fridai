"""자립형 인덱싱 → 회수 e2e (LLM 없이). Codex·Gemini 세션을 인덱싱하고 recall로 회수."""
import json
import tempfile
import unittest
from pathlib import Path

from fridai import server
from fridai.core.sources import agent_recall
from fridai.core.store import Store


def _codex_dir() -> Path:
    d = Path(tempfile.mkdtemp()) / "2026" / "07" / "01"
    d.mkdir(parents=True)
    rows = [
        {"timestamp": "2026-07-01T10:00:00Z", "type": "session_meta",
         "payload": {"cwd": "/home/u/repoX", "cli_version": "0.143.0"}},
        {"timestamp": "2026-07-01T10:00:02Z", "type": "response_item",
         "payload": {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "코덱스 도커 마운트 질문"}]}},
        {"timestamp": "2026-07-01T10:00:06Z", "type": "response_item",
         "payload": {"type": "message", "role": "assistant",
                     "content": [{"type": "output_text", "text": "volumes 로 마운트."}]}},
    ]
    p = d / "rollout-2026-07-01T10-00-00-abc.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p.parents[2]        # sessions 루트


def _gemini_dir() -> Path:
    proj = Path(tempfile.mkdtemp()) / "repoY"
    (proj / "chats").mkdir(parents=True)
    (proj / ".project_root").write_text("/home/u/repoY", encoding="utf-8")
    rows = [
        {"sessionId": "s", "kind": "session"},
        {"id": "m1", "timestamp": "2026-07-02T09:00:00Z", "type": "user",
         "content": [{"text": "제미나이 인증 질문"}]},
        {"id": "m2", "timestamp": "2026-07-02T09:00:05Z", "type": "gemini",
         "content": [{"text": "OAuth 로 로그인."}]},
    ]
    (proj / "chats" / "session-2026-07-02T09-00-x.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return proj.parents[0]     # tmp 루트


class TestIndexAndRecall(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.no_claude = Path(tempfile.mkdtemp()) / "none"   # 존재하지 않는 Claude 경로

    def tearDown(self):
        self.store.close()

    def test_index_all_then_recall(self):
        r = agent_recall.index_all(self.store, self.no_claude, _codex_dir(), _gemini_dir())
        self.assertEqual(r["turns"], 2)           # codex 1 + gemini 1
        # 출처 에이전트 태깅
        cx = self.store.search_lexical("도커 마운트", k=3)
        self.assertEqual(cx[0].document.meta.get("agent"), "codex")
        gm = self.store.search_lexical("인증", k=3)
        self.assertEqual(gm[0].document.meta.get("agent"), "gemini")
        # retrieve via the recall tool
        out = server.recall_tool("도커 마운트", store=self.store, repo="all")
        self.assertIn("memory item", out)


if __name__ == "__main__":
    unittest.main()
