"""F4 extension (#84) — Codex CLI rollout parser + incremental indexing unit tests.

The fixture reproduces the rollout schema learned from docs/reverse engineering
(validation against real Codex data is done separately).
"""
import json
import tempfile
import unittest
from pathlib import Path

from fridai.core.sources import codex_recall as cx
from fridai.core.store import Store

# one rollout-*.jsonl session (matching the codex-cli 0.143 real schema):
# session_meta(cwd) + developer(excluded) + user environment block(noise) + 2 real user questions
# + exec_command(shell, no file extraction) + apply_patch(path key -> file extracted) + assistant answers
SESSION = [
    {"timestamp": "2026-07-01T10:00:00Z", "type": "session_meta",
     "payload": {"session_id": "s1", "cwd": "/home/u/myrepo",
                 "model_provider": "openai", "cli_version": "0.143.0"}},
    {"timestamp": "2026-07-01T10:00:01Z", "type": "turn_context",
     "payload": {"turn_id": "t1", "cwd": "/home/u/myrepo"}},          # should be ignored
    {"timestamp": "2026-07-01T10:00:01Z", "type": "response_item",
     "payload": {"type": "message", "role": "developer",
                 "content": [{"type": "input_text", "text": "<permissions instructions>..."}]}},
    {"timestamp": "2026-07-01T10:00:02Z", "type": "response_item",
     "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text",
                              "text": "<environment_context>\n  <cwd>/home/u/myrepo</cwd>\n</environment_context>"}]}},
    {"timestamp": "2026-07-01T10:00:03Z", "type": "response_item",
     "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "왜 로그인이 401 나지?"}]}},
    {"timestamp": "2026-07-01T10:00:05Z", "type": "response_item",
     "payload": {"type": "function_call", "name": "exec_command",
                 "arguments": "{\"cmd\":\"grep -r 401 src\",\"workdir\":\"/home/u/myrepo\"}", "call_id": "c1"}},
    {"timestamp": "2026-07-01T10:00:06Z", "type": "event_msg",           # should be ignored
     "payload": {"type": "agent_message", "message": "..."}},
    {"timestamp": "2026-07-01T10:00:07Z", "type": "response_item",
     "payload": {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "토큰 만료가 원인. refresh로 재발급."}]}},
    {"timestamp": "2026-07-01T10:05:00Z", "type": "response_item",
     "payload": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": "그럼 그 직전엔?"}]}},
    {"timestamp": "2026-07-01T10:05:02Z", "type": "response_item",
     "payload": {"type": "function_call", "name": "apply_patch",
                 "arguments": "{\"file_path\": \"src/auth.py\"}", "call_id": "c2"}},
    {"timestamp": "2026-07-01T10:05:03Z", "type": "response_item",
     "payload": {"type": "message", "role": "assistant",
                 "content": [{"type": "output_text", "text": "직전엔 헤더 누락이었음."}]}},
]


def _write_session(rows) -> Path:
    d = Path(tempfile.mkdtemp()) / "2026" / "07" / "01"
    d.mkdir(parents=True)
    p = d / "rollout-2026-07-01T10-00-00-abc.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


class TestParse(unittest.TestCase):
    def setUp(self):
        self.turns = cx.parse_session(_write_session(SESSION))

    def test_extracts_only_real_user_turns(self):
        # developer(instructions) and injected <environment_context> blocks excluded; only 2 real questions
        self.assertEqual([t.question for t in self.turns],
                         ["왜 로그인이 401 나지?", "그럼 그 직전엔?"])

    def test_assistant_text_attached(self):
        self.assertIn("refresh로 재발급", self.turns[0].answer)
        self.assertIn("헤더 누락", self.turns[1].answer)

    def test_meta_cwd_from_session_meta(self):
        self.assertEqual(self.turns[0].repo, "myrepo")     # session_meta.cwd -> repo

    def test_exec_command_yields_no_file(self):
        # shell tools have no structured file path -> tool name only, no files
        self.assertIn("exec_command", self.turns[0].tools)
        self.assertEqual(self.turns[0].files, [])

    def test_path_keyed_tool_extracts_file(self):
        # tools with a path-like key (e.g. apply_patch) do extract files
        self.assertIn("apply_patch", self.turns[1].tools)
        self.assertIn("src/auth.py", self.turns[1].files)

    def test_timestamp_parsed(self):
        self.assertIsNotNone(self.turns[0].when)


class TestHelpers(unittest.TestCase):
    def test_text_from_string_or_array(self):
        self.assertEqual(cx._text("hi"), "hi")
        self.assertEqual(cx._text([{"type": "output_text", "text": "a"},
                                   {"type": "x", "text": "b"}]), "a\nb")
        self.assertEqual(cx._text(None), "")

    def test_files_from_args_bad_json(self):
        self.assertEqual(cx._files_from_args("not json"), [])
        self.assertEqual(cx._files_from_args('{"path": "a.py"}'), ["a.py"])
        self.assertEqual(cx._files_from_args('{"nope": 1}'), [])


class TestIndex(unittest.TestCase):
    def setUp(self):
        self.session = _write_session(SESSION)
        self.root = self.session.parents[2]      # parent of …/2026 = sessions root
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_indexes_then_skips(self):
        r1 = cx.index_codex(self.store, self.root)
        self.assertEqual(r1["files"], 1)
        self.assertEqual(r1["turns"], 2)
        r2 = cx.index_codex(self.store, self.root)
        self.assertEqual(r2["skipped"], 1)        # same mtime -> skip

    def test_agent_tagged_codex(self):
        cx.index_codex(self.store, self.root)
        hits = self.store.search_lexical("401", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.meta.get("agent"), "codex")

    def test_missing_dir(self):
        self.assertEqual(cx.index_codex(self.store, Path("/no/such/dir")),
                         {"turns": 0, "files": 0, "skipped": 0})


class TestIndexAll(unittest.TestCase):
    def test_aggregates_claude_and_codex(self):
        from fridai.core.sources import agent_recall as ar
        # one Claude session
        proj = Path(tempfile.mkdtemp())
        with (proj / "s.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "user", "isSidechain": False,
                                "timestamp": "2026-07-01T09:00:00Z", "cwd": "/x/r",
                                "message": {"role": "user", "content": "클로드 질문"}}) + "\n")
            f.write(json.dumps({"type": "assistant", "timestamp": "2026-07-01T09:00:01Z",
                                "message": {"content": [{"type": "text", "text": "답"}]}}) + "\n")
        codex_root = _write_session(SESSION).parents[2]
        gemini_root = Path(tempfile.mkdtemp())           # empty dir -> 0 gemini (real ~/.gemini isolated)
        store = Store(":memory:")
        try:
            r = ar.index_all(store, proj, codex_root, gemini_root)
            self.assertEqual(r["files"], 2)              # claude 1 + codex 1
            self.assertEqual(r["turns"], 3)              # claude 1 + codex 2
            agents = {h.document.meta.get("agent")
                      for h in store.search_lexical("질문", k=10) + store.search_lexical("401", k=10)}
            self.assertEqual(agents, {"claude", "codex"})
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
