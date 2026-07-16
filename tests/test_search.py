"""회수/랭킹(search.py) 단위 테스트 — RRF·작업신호·dedup·citation·retrieve."""
import unittest
from datetime import datetime, timezone

from fridai.core import search
from fridai.core.models import Document, SearchHit
from fridai.core.store import Store

WHEN = datetime(2026, 6, 17, tzinfo=timezone.utc)


def _hit(id, st="code", repo="r", text="x", meta=None):
    return SearchHit(Document(id=id, source_type=st, repo=repo, path="p", title=id,
                              text=text, timestamp=WHEN, meta=meta or {}), 0.0)


class TestCitation(unittest.TestCase):
    def test_code(self):
        d = Document(id="c", source_type="code", repo="r", path="a.py", title="t", text="x",
                     meta={"path": "a.py", "start_line": 1, "end_line": 9})
        self.assertEqual(search.citation(d), "r/a.py:1-9")

    def test_commit(self):
        d = Document(id="c", source_type="commit", repo="r", path="p", title="fix: x",
                     text="x", meta={"sha": "abc1234"})
        self.assertEqual(search.citation(d), 'r@abc1234 "fix: x"')

    def test_agent_turn_claude_has_no_tag(self):
        d = Document(id="t", source_type="agent_turn", repo="r", path="s", title="q",
                     text="t", timestamp=WHEN, meta={"agent": "claude", "session_title": "S"})
        cit = search.citation(d)
        self.assertIn("session:S", cit)
        self.assertNotIn("[claude]", cit)             # claude has no marker (default)

    def test_agent_turn_non_claude_shows_agent_tag(self):
        d = Document(id="t", source_type="agent_turn", repo="r", path="s", title="q",
                     text="t", timestamp=WHEN, meta={"agent": "codex", "session_title": "S"})
        self.assertIn("[codex]", search.citation(d))   # 비-Claude는 출처 표시


class TestRRF(unittest.TestCase):
    def test_rewards_agreement(self):
        lex = [_hit("B"), _hit("A"), _hit("C")]
        vec = [_hit("B"), _hit("A"), _hit("D")]       # B 양쪽 1위, A 양쪽 2위
        fused = search.rrf_fuse([lex, vec], k=4)
        self.assertEqual(fused[0].document.id, "B")
        ids = [h.document.id for h in fused]
        self.assertLess(ids.index("A"), ids.index("C"))  # 양쪽 등장 A > 한쪽 C


class TestWorkSignal(unittest.TestCase):
    def test_work_signal(self):
        code = _hit("c", st="code").document
        turn_work = _hit("t1", st="agent_turn", meta={"files": ["a.py"]}).document
        turn_q = _hit("t2", st="agent_turn", meta={"files": [], "commits": []}).document
        self.assertTrue(search.work_signal(code))          # code=작업물
        self.assertTrue(search.work_signal(turn_work))     # 파일 편집한 턴
        self.assertFalse(search.work_signal(turn_q))       # 순수 질문 턴

    def test_rerank_demotes_question_only_turn(self):
        def turn(i, files):
            return _hit(i, st="agent_turn", text=i, meta={"files": files})
        hits = [turn("질문", []), turn("해결", ["s3.cfg"])]
        out = search.rerank_work_signal(hits, penalty=3)
        self.assertEqual([h.document.id for h in out], ["해결", "질문"])


class TestDedup(unittest.TestCase):
    def test_merges_repeated_questions(self):
        def turn(i, q):
            return _hit(i, st="agent_turn", meta={"question": q})
        hits = [turn("a", "도커 마운트 어떻게 추가해"), turn("b", "도커 마운트 어떻게 추가해"),
                turn("c", "완전히 다른 질문 인증 토큰")]
        out = search.dedup_results(hits)
        ids = [h.document.id for h in out]
        self.assertIn("a", ids)
        self.assertNotIn("b", ids)      # a와 near-dup → 병합
        self.assertIn("c", ids)


class TestRetrieve(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")

    def tearDown(self):
        self.s.close()

    def test_lexical_retrieve(self):
        self.s.upsert([Document(id="c1", source_type="code", repo="r", path="a.py",
                                title="a.py:1-9", text="jwt token refresh")])
        hits = search.retrieve(self.s, "jwt", k=5)
        self.assertTrue(hits)
        self.assertEqual(hits[0].document.id, "c1")

    def test_build_context_has_citations(self):
        self.s.upsert([Document(id="c1", source_type="code", repo="r", path="a.py",
                                title="a.py:1-9", text="jwt",
                                meta={"path": "a.py", "start_line": 1, "end_line": 9})])
        ctx = search.build_context(search.retrieve(self.s, "jwt", k=5))
        self.assertIn("[1]", ctx)
        self.assertIn("r/a.py:1-9", ctx)

    def test_retrieve_uses_embedder_when_present(self):
        self.s.upsert([Document(id="v1", source_type="code", repo="r", path="p",
                                title="t", text="문서 내용", embedding=[1.0, 0.0])])

        class FakeEmbedder:
            def embed(self, q):
                return [1.0, 0.0]
        hits = search.retrieve(self.s, "무관한단어", k=3, embedder=FakeEmbedder())
        self.assertTrue(hits)                          # 어휘론 0건이나 벡터로 회수
        self.assertEqual(hits[0].document.id, "v1")


if __name__ == "__main__":
    unittest.main()
