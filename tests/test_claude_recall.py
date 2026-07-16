"""Claude Code transcript 파서 테스트. 합성 transcript로 노이즈 필터/추출/증분 검증."""
import json
import tempfile
import unittest
from pathlib import Path

from fridai.core.sources import claude_recall as cr
from fridai.core.store import Store


def _write_jsonl(lines: list[dict]) -> Path:
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for obj in lines:
        f.write(json.dumps(obj) + "\n")
    f.close()
    return Path(f.name)


# 진짜질문1 + assistant + tool_result 노이즈 + 이어하기 노이즈 + 진짜질문2 + assistant
SESSION = [
    {"type": "ai-title", "aiTitle": "Fix auth header"},
    {"type": "user", "isMeta": False, "isSidechain": False,
     "timestamp": "2026-06-17T09:43:00.000Z", "cwd": "/x/flower-ad-server",
     "gitBranch": "release/1.0.31",
     "message": {"role": "user", "content": "<system-reminder>무시</system-reminder>d_ip 헤더 추가하고 싶어"}},
    {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "thinking", "thinking": "생각..."},
        {"type": "text", "text": "헤더를 추가했습니다. 동작합니다."},
        {"type": "tool_use", "name": "Edit", "input": {"file_path": "/x/DeviceService.kt"}},
    ]}},
    {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "content": "ok"}]}},
    {"type": "user", "isSidechain": False,
     "message": {"role": "user",
                 "content": "This session is being continued from a previous conversation..."}},
    {"type": "user", "isMeta": False, "isSidechain": False,
     "timestamp": "2026-06-17T10:00:00.000Z", "cwd": "/x/flower-ad-server",
     "gitBranch": "release/1.0.31",
     "message": {"role": "user", "content": "테스트도 추가해줘"}},
    {"type": "assistant", "message": {"role": "assistant", "content": [
        {"type": "text", "text": "테스트 추가 완료."}]}},
]


class TestParse(unittest.TestCase):
    def setUp(self):
        self.path = _write_jsonl(SESSION)
        self.turns = cr.parse_session(self.path)

    def tearDown(self):
        self.path.unlink(missing_ok=True)

    def test_only_real_questions_extracted(self):
        self.assertEqual([t.question for t in self.turns],
                         ["d_ip 헤더 추가하고 싶어", "테스트도 추가해줘"])   # system-reminder 제거·노이즈 제외

    def test_answer_and_context_captured(self):
        t = self.turns[0]
        self.assertIn("헤더를 추가했습니다", t.answer)
        self.assertNotIn("생각...", t.answer)              # thinking 제외
        self.assertEqual(t.repo, "flower-ad-server")
        self.assertEqual(t.branch, "release/1.0.31")
        self.assertIn("Edit", t.tools)
        self.assertIn("/x/DeviceService.kt", t.files)

    def test_session_title_applied(self):
        self.assertTrue(all(t.session_title == "Fix auth header" for t in self.turns))


class TestNoiseFilter(unittest.TestCase):
    def test_task_notification_not_a_question(self):
        path = _write_jsonl([
            {"type": "user", "isSidechain": False, "timestamp": "2026-06-22T00:00:00Z",
             "cwd": "/x/r", "message": {"role": "user",
              "content": "<task-notification>\n<task-id>abc</task-id>\n</task-notification>"}},
            {"type": "user", "isMeta": False, "isSidechain": False,
             "timestamp": "2026-06-22T00:01:00Z", "cwd": "/x/r",
             "message": {"role": "user", "content": "진짜 질문"}},
        ])
        try:
            self.assertEqual([t.question for t in cr.parse_session(path)], ["진짜 질문"])
        finally:
            path.unlink(missing_ok=True)


class TestIncrementalIndex(unittest.TestCase):
    def setUp(self):
        self.projects = Path(tempfile.mkdtemp())
        self.session = self.projects / "s1.jsonl"
        with self.session.open("w", encoding="utf-8") as f:
            for obj in SESSION:
                f.write(json.dumps(obj) + "\n")
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_first_run_indexes_then_skips(self):
        r1 = cr.index_claude(self.store, self.projects)
        self.assertEqual(r1["files"], 1)
        self.assertEqual(r1["turns"], 2)
        r2 = cr.index_claude(self.store, self.projects)
        self.assertEqual(r2["files"], 0)
        self.assertEqual(r2["skipped"], 1)

    def test_reindex_flag_forces(self):
        cr.index_claude(self.store, self.projects)
        self.assertEqual(cr.index_claude(self.store, self.projects, reindex=True)["files"], 1)

    def test_missing_dir(self):
        self.assertEqual(cr.index_claude(self.store, Path("/no/such/dir")),
                         {"turns": 0, "files": 0, "skipped": 0})

    def test_agent_tagged_claude(self):
        cr.index_claude(self.store, self.projects)
        hits = self.store.search_lexical("헤더", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.meta.get("agent"), "claude")


if __name__ == "__main__":
    unittest.main()
