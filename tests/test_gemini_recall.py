"""F4 extension (#85) — Gemini CLI chats JSONL parser + incremental indexing unit tests.

The fixture reproduces the gemini-cli 0.50 real schema (a header line + message
record lines + <session_context> noise).
"""
import json
import tempfile
import unittest
from pathlib import Path

from fridai.core.sources import agent_recall as ar
from fridai.core.sources import gemini_recall as gm
from fridai.core.store import Store

HEADER = {"sessionId": "s-uuid", "projectHash": "abc", "startTime": "2026-07-09T08:52:33Z",
          "lastUpdated": "2026-07-09T09:00:00Z", "kind": "session"}
MESSAGES = [
    {"id": "m0", "timestamp": "2026-07-09T08:52:33Z", "type": "user",
     "content": [{"text": "<session_context>\nThis is the Gemini CLI...\n</session_context>"}]},
    {"id": "m1", "timestamp": "2026-07-09T08:53:00Z", "type": "user",
     "content": [{"text": "이 레포 무슨 프로젝트야?"}]},
    {"id": "m2", "timestamp": "2026-07-09T08:53:10Z", "type": "info",
     "content": "Attempting to open authentication page..."},
    {"id": "m3", "timestamp": "2026-07-09T08:53:20Z", "type": "gemini", "model": "gemini-2.0",
     "thoughts": "...", "tokens": {"input": 10},
     "content": [{"text": "로컬 AI 코딩 기억 비서 프로젝트입니다."}],
     "toolCalls": [{"id": "t1", "name": "list_directory",
                    "args": {"dir_path": "/home/u/myrepo/src"}, "result": []}]},
    {"id": "m4", "timestamp": "2026-07-09T08:55:00Z", "type": "user",
     "content": [{"text": "README 고쳐줘"}]},
    {"id": "m5", "timestamp": "2026-07-09T08:55:30Z", "type": "gemini",
     "content": [{"text": "README를 수정했습니다."}],
     "toolCalls": [{"id": "t2", "name": "write_file",
                    "args": {"file_path": "/home/u/myrepo/README.md"}, "result": []}]},
    {"id": "m6", "timestamp": "2026-07-09T08:56:00Z", "type": "error",
     "content": "[API Error]"},
]


def _write_session(header, messages, cwd="/home/u/myrepo") -> Path:
    proj = Path(tempfile.mkdtemp()) / "myrepo"
    (proj / "chats").mkdir(parents=True)
    if cwd is not None:
        (proj / ".project_root").write_text(cwd, encoding="utf-8")
    p = proj / "chats" / "session-2026-07-09T08-52-abc.jsonl"
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        for m in messages:
            f.write(json.dumps(m) + "\n")
    return p


class TestParse(unittest.TestCase):
    def setUp(self):
        self.turns = gm.parse_session(_write_session(HEADER, MESSAGES))

    def test_only_real_user_questions(self):
        # injected <session_context>, info, error excluded; only 2 real questions
        self.assertEqual([t.question for t in self.turns],
                         ["이 레포 무슨 프로젝트야?", "README 고쳐줘"])

    def test_gemini_answer_attached(self):
        self.assertIn("기억 비서", self.turns[0].answer)
        self.assertIn("수정했습니다", self.turns[1].answer)

    def test_cwd_from_project_root(self):
        self.assertEqual(self.turns[0].repo, "myrepo")     # .project_root -> cwd -> repo
        self.assertEqual(self.turns[0].cwd, "/home/u/myrepo")

    def test_toolcalls_yield_tools_and_files(self):
        self.assertIn("list_directory", self.turns[0].tools)
        self.assertIn("/home/u/myrepo/src", self.turns[0].files)
        self.assertIn("/home/u/myrepo/README.md", self.turns[1].files)

    def test_header_and_noninformative_lines_skipped(self):
        # the header line (no content) does not create a turn
        self.assertEqual(len(self.turns), 2)


class TestProjectRootFallback(unittest.TestCase):
    def test_falls_back_to_dir_name_without_project_root(self):
        p = _write_session(HEADER, MESSAGES, cwd=None)     # no .project_root
        turns = gm.parse_session(p)
        self.assertEqual(turns[0].repo, "myrepo")          # falls back to the directory name


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.session = _write_session(HEADER, MESSAGES)
        self.root = self.session.parents[2]      # …/tmp root (= parent of myrepo)
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_indexes_then_skips(self):
        r1 = ar.index_adapter(self.store, gm.ADAPTER, self.root)
        self.assertEqual(r1["files"], 1)
        self.assertEqual(r1["turns"], 2)
        self.assertEqual(ar.index_adapter(self.store, gm.ADAPTER, self.root)["skipped"], 1)

    def test_agent_tagged_gemini(self):
        ar.index_adapter(self.store, gm.ADAPTER, self.root)
        hits = self.store.search_lexical("프로젝트", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.meta.get("agent"), "gemini")

    def test_nested_checkpoint_excluded(self):
        # chats/<sessionId>/<uuid>.jsonl (nested checkpoints) are excluded by the glob -> no duplicate indexing
        nested = self.session.parent / "s-uuid"
        nested.mkdir()
        (nested / "cp.jsonl").write_text(
            json.dumps(HEADER) + "\n" + json.dumps(MESSAGES[1]) + "\n", encoding="utf-8")
        r = ar.index_adapter(self.store, gm.ADAPTER, self.root)
        self.assertEqual(r["files"], 1)          # nested file not counted

    def test_missing_dir(self):
        self.assertEqual(ar.index_adapter(self.store, gm.ADAPTER, Path("/no/such/dir")),
                         {"turns": 0, "files": 0, "skipped": 0})


if __name__ == "__main__":
    unittest.main()
