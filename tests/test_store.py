"""Store(sqlite+FTS5) 단위 테스트. stdlib unittest."""
import unittest
from datetime import datetime, timezone

from fridai.core.models import Document, make_id
from fridai.core.store import Store, _cosine, _fts_match


def _doc(title, text, repo="repo1", st="agent_turn", emb=None, ts=None):
    return Document(
        id=make_id(st, title), source_type=st, repo=repo, path="p",
        title=title, text=text, timestamp=ts, meta={"k": "v"}, embedding=emb,
    )


class TestStore(unittest.TestCase):
    def setUp(self):
        self.s = Store(":memory:")

    def tearDown(self):
        self.s.close()

    def test_upsert_and_get(self):
        d = _doc("도커 마운트 추가", "s3files 마운트를 docker compose에 추가")
        self.assertEqual(self.s.upsert([d]), 1)
        got = self.s.get(d.id)
        self.assertIsNotNone(got)
        self.assertEqual(got.title, "도커 마운트 추가")
        self.assertEqual(got.meta["k"], "v")

    def test_upsert_is_idempotent(self):
        d = _doc("t", "x")
        self.s.upsert([d]); self.s.upsert([d])   # 같은 id 두 번
        self.assertEqual(self.s.stats()["total"], 1)

    def test_lexical_search_finds_and_filters(self):
        self.s.upsert([
            _doc("도커 마운트", "s3files mount 추가", repo="r1"),
            _doc("인증 헤더", "X-Forwarded-For 헤더 추가", repo="r2"),
        ])
        hits = self.s.search_lexical("마운트")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].document.title, "도커 마운트")
        # repo 필터
        self.assertEqual(len(self.s.search_lexical("추가", repo="r2")), 1)

    def test_lexical_empty_query(self):
        self.assertEqual(self.s.search_lexical("   "), [])

    def test_vector_search_dim_mismatch_returns_empty(self):
        # 저장 벡터(3dim)와 쿼리(2dim) 불일치 → 크래시 없이 빈 결과(어휘 폴백 유도)
        self.s.upsert([_doc("a", "x", emb=[1.0, 0.0, 0.0])])
        self.assertEqual(self.s.search_vector([1.0, 0.0], k=3), [])

    def test_vector_search_ranks_by_cosine(self):
        self.s.upsert([
            _doc("a", "x", emb=[1.0, 0.0, 0.0]),
            _doc("b", "y", emb=[0.0, 1.0, 0.0]),
        ])
        hits = self.s.search_vector([0.9, 0.1, 0.0], k=2)
        self.assertEqual(hits[0].document.title, "a")  # 첫 벡터에 더 가까움

    def test_recent_orders_oldest_to_newest(self):
        older = _doc("older", "x", ts=datetime(2026, 1, 1, tzinfo=timezone.utc))
        newer = _doc("newer", "y", ts=datetime(2026, 6, 1, tzinfo=timezone.utc))
        self.s.upsert([newer, older])
        docs = self.s.recent(limit=10)
        self.assertEqual([d.title for d in docs], ["older", "newer"])

    def test_stats(self):
        self.s.upsert([_doc("a", "x", st="agent_turn"), _doc("b", "y", st="commit")])
        st = self.s.stats()
        self.assertEqual(st["total"], 2)
        self.assertEqual(st["by_type"]["commit"], 1)

    def test_stats_by_agent_and_last_indexed(self):
        def turn(title, agent, ts):
            return Document(id=make_id("agent_turn", title), source_type="agent_turn",
                            repo="r", path="p", title=title, text="x", timestamp=ts,
                            meta={"agent": agent})
        self.s.upsert([
            turn("c1", "claude", datetime(2026, 1, 1, tzinfo=timezone.utc)),
            turn("c2", "claude", datetime(2026, 3, 1, tzinfo=timezone.utc)),
            turn("g1", "gemini", datetime(2026, 5, 1, tzinfo=timezone.utc)),
            _doc("code1", "y", st="code"),        # non-agent doc: absent from by_agent
        ])
        st = self.s.stats()
        self.assertEqual(st["by_agent"], {"claude": 2, "gemini": 1})
        self.assertTrue(st["last_indexed"].startswith("2026-05-01"))  # MAX(ts)

    def test_delete_by_path(self):
        d1 = Document(id="code:1", source_type="code", repo="r", path="a.py",
                      title="a", text="x", embedding=[1.0, 0.0])
        d2 = Document(id="code:2", source_type="code", repo="r", path="b.py",
                      title="b", text="y")
        self.s.upsert([d1, d2])
        removed = self.s.delete_by_path("code", "r", "a.py")
        self.assertEqual(removed, 1)
        self.assertIsNone(self.s.get("code:1"))
        self.assertIsNotNone(self.s.get("code:2"))
        # fts/vectors도 함께 정리되어 검색에 안 잡힘
        self.assertEqual(self.s.search_lexical("x"), [])

    def test_index_state_roundtrip(self):
        self.assertIsNone(self.s.get_state("k"))
        self.s.set_state("k", "v1")
        self.assertEqual(self.s.get_state("k"), "v1")
        self.s.set_state("k", "v2")
        self.assertEqual(self.s.get_state("k"), "v2")

    def test_delete_state(self):
        self.s.set_state("k", "v")
        self.s.delete_state("k")
        self.assertIsNone(self.s.get_state("k"))

    def test_paths_distinct_per_source_repo(self):
        self.s.upsert([
            Document(id="c1", source_type="code", repo="r", path="a.py", title="t", text="x"),
            Document(id="c2", source_type="code", repo="r", path="a.py", title="t", text="y"),
            Document(id="c3", source_type="code", repo="r", path="b.py", title="t", text="z"),
            Document(id="n1", source_type="note", repo="r", path="note", title="t", text="w"),
        ])
        self.assertEqual(self.s.paths("code", "r"), {"a.py", "b.py"})  # distinct, note 제외


class TestVectorMigration(unittest.TestCase):
    def test_migrates_vec_json_to_blob_without_reembed(self):
        import json, sqlite3, tempfile, os
        path = os.path.join(tempfile.mkdtemp(), "old.db")
        # 구버전 스키마(vec_json) 모사 + 문서/벡터 1건
        con = sqlite3.connect(path)
        con.executescript(
            "CREATE TABLE documents(id TEXT PRIMARY KEY, source_type TEXT, repo TEXT, path TEXT,"
            " title TEXT, text TEXT, ts TEXT, meta_json TEXT);"
            "CREATE TABLE vectors(doc_id TEXT PRIMARY KEY, vec_json TEXT NOT NULL);")
        con.execute("INSERT INTO documents VALUES('d1','code','r','p','t','x',NULL,'{}')")
        con.execute("INSERT INTO vectors VALUES('d1', ?)", (json.dumps([1.0, 0.0, 0.0]),))
        con.commit(); con.close()
        # Store 열면 자동 마이그레이션 → 벡터 검색 동작(재임베딩 없이)
        s = Store(path)
        try:
            cols = [r[1] for r in s.con.execute("PRAGMA table_info(vectors)")]
            self.assertIn("vec", cols)
            self.assertNotIn("vec_json", cols)
            hits = s.search_vector([1.0, 0.0, 0.0], k=1)
            self.assertEqual(hits[0].document.id, "d1")
        finally:
            s.close()


class TestRedactionIntegration(unittest.TestCase):
    def test_upsert_redacts_by_default(self):
        s = Store(":memory:")
        try:
            s.upsert([_doc("키", "AWS key AKIAIOSFODNN7EXAMPLE 사용")])
            hits = s.search_lexical("AWS")
            self.assertTrue(hits)
            self.assertNotIn("AKIAIOSFODNN7EXAMPLE", hits[0].document.text)
        finally:
            s.close()

    def test_no_redact_preserves(self):
        s = Store(":memory:", redact=False)
        try:
            s.upsert([_doc("키", "AKIAIOSFODNN7EXAMPLE")])
            self.assertTrue(any("AKIAIOSFODNN7EXAMPLE" in h.document.text
                                for h in s.search_lexical("AKIA")))
        finally:
            s.close()

    def test_rescan_redact_cleans_existing(self):
        s = Store(":memory:", redact=False)            # 원문으로 먼저 저장
        try:
            s.upsert([_doc("키", "secret AKIAIOSFODNN7EXAMPLE 노출")])
            changed = s.rescan_redact()
            self.assertEqual(changed, 1)
            self.assertFalse(any("AKIAIOSFODNN7EXAMPLE" in h.document.text
                                 for h in s.search_lexical("AKIA")))
        finally:
            s.close()


class TestHelpers(unittest.TestCase):
    def test_cosine(self):
        self.assertAlmostEqual(_cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(_cosine([1, 0], [0, 1]), 0.0)

    def test_fts_match_prefix_and_sanitizes(self):
        # 특수문자 제거 + 토큰별 접두 매칭(한국어 조사 대응)
        self.assertEqual(_fts_match("도커 마운트"), '"도커"* OR "마운트"*')
        self.assertEqual(_fts_match("a (b) c!"), '"a"* OR "b"* OR "c"*')

    def test_prefix_matches_korean_particle(self):
        s = Store(":memory:")
        try:
            s.upsert([_doc("t", "S3 마운트는 efs 설치로 해결")])
            self.assertTrue(s.search_lexical("마운트"))  # "마운트는"을 접두로 매칭
        finally:
            s.close()


if __name__ == "__main__":
    unittest.main()
