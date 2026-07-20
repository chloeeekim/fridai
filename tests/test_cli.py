"""CLI 단위 테스트 — _run_index 인덱싱 패스, watch 델타 카운트, 파서 기본값."""
import subprocess
import tempfile
import unittest
from pathlib import Path

from fridai import cli
from fridai.core.store import Store


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t.io",
                    "-c", "user.name=tester", "-c", "commit.gpgsign=false", *args],
                   capture_output=True, check=True)


class TestRunIndex(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _git(self.repo, "init", "-q")
        (self.repo / "app.py").write_text(
            "\n".join(f"line {i} jwt token" for i in range(80)), encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "feat: initial")
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_code_source_indexes_and_is_incremental(self):
        r1 = cli._run_index(self.store, "code", str(self.repo), None)
        self.assertIn("code", r1)
        self.assertGreater(r1["code"]["chunks"], 0)
        self.assertNotIn("commits", r1)                 # source="code" only
        # 2차 패스: 변경 없음 → 청크 0(전부 skip)
        r2 = cli._run_index(self.store, "code", str(self.repo), None)
        self.assertEqual(r2["code"]["chunks"], 0)

    def test_all_source_runs_every_indexer(self):
        r = cli._run_index(self.store, "all", str(self.repo), None)
        self.assertEqual(set(r), {"agent", "code", "commits"})
        self.assertEqual(r["commits"]["commits"], 1)     # 초기 커밋 1개

    def test_reindexed_counts_signal(self):
        r = cli._run_index(self.store, "all", str(self.repo), None)
        n = cli._reindexed_counts(r)
        self.assertGreater(n["chunks"], 0)
        self.assertEqual(n["commits"], 1)
        # 재실행 → 변화 없음 → 전부 0 (watch가 조용해짐)
        n2 = cli._reindexed_counts(cli._run_index(self.store, "all", str(self.repo), None))
        self.assertEqual(n2, {"turns": 0, "chunks": 0, "commits": 0})


class TestInstallHook(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp())
        _git(self.repo, "init", "-q")
        self.parser = cli.build_parser()

    def _run(self, *extra):
        args = self.parser.parse_args(["install-hook", "--path", str(self.repo), *extra])
        args.func(args)

    def test_installs_executable_hook_with_marker(self):
        self._run()
        hook = self.repo / ".git" / "hooks" / "post-commit"
        self.assertTrue(hook.exists())
        self.assertTrue(hook.stat().st_mode & 0o111)          # executable
        content = hook.read_text()
        self.assertIn(cli._HOOK_MARKER, content)
        self.assertIn(str(self.repo), content)
        self.assertIn("fridai index", content)

    def test_idempotent_on_own_hook(self):
        self._run(); self._run()                              # own marker present → OK to overwrite
        self.assertTrue((self.repo / ".git/hooks/post-commit").exists())

    def test_refuses_foreign_hook_without_force(self):
        hook = self.repo / ".git" / "hooks" / "post-commit"
        hook.write_text("#!/bin/sh\necho mine\n")
        with self.assertRaises(SystemExit):
            self._run()
        self._run("--force")                                  # --force overwrites
        self.assertIn(cli._HOOK_MARKER, hook.read_text())

    def test_not_a_repo_errors(self):
        args = self.parser.parse_args(["install-hook", "--path", tempfile.mkdtemp()])
        with self.assertRaises(SystemExit):
            args.func(args)


class TestParser(unittest.TestCase):
    def test_watch_defaults(self):
        a = cli.build_parser().parse_args(["index", "--watch"])
        self.assertTrue(a.watch)
        self.assertEqual(a.interval, 15)             # 기본 15초

    def test_interval_configurable(self):
        a = cli.build_parser().parse_args(["index", "--watch", "--interval", "30"])
        self.assertEqual(a.interval, 30)

    def test_no_watch_by_default(self):
        self.assertFalse(cli.build_parser().parse_args(["index"]).watch)


if __name__ == "__main__":
    unittest.main()
