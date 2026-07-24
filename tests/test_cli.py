"""CLI unit tests — _run_index indexing pass, watch delta counts, parser defaults."""
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
        # 2nd pass: nothing changed -> 0 chunks (all skipped)
        r2 = cli._run_index(self.store, "code", str(self.repo), None)
        self.assertEqual(r2["code"]["chunks"], 0)

    def test_all_source_runs_every_indexer(self):
        r = cli._run_index(self.store, "all", str(self.repo), None)
        self.assertEqual(set(r), {"agent", "code", "commits"})
        self.assertEqual(r["commits"]["commits"], 1)     # 1 initial commit

    def test_reindexed_counts_signal(self):
        r = cli._run_index(self.store, "all", str(self.repo), None)
        n = cli._reindexed_counts(r)
        self.assertGreater(n["chunks"], 0)
        self.assertEqual(n["commits"], 1)
        # rerun -> no change -> all 0 (watch stays quiet)
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
        self.assertIn("nohup", content)                        # detached
        self.assertTrue(content.rstrip().endswith("&"))        # backgrounded

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


class TestVersion(unittest.TestCase):
    def test_version_flag_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            cli.build_parser().parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)

    def test_pkg_version_is_str(self):
        self.assertIsInstance(cli._pkg_version(), str)


class TestMcpConfig(unittest.TestCase):
    def test_print_config_flag_parsed(self):
        self.assertTrue(cli.build_parser().parse_args(["mcp", "--print-config"]).print_config)
        self.assertFalse(cli.build_parser().parse_args(["mcp"]).print_config)

    def test_config_text_covers_every_client(self):
        txt = cli._mcp_config_text()
        self.assertIn("claude mcp add fridai", txt)      # Claude Code
        self.assertIn('"mcpServers"', txt)               # Gemini / generic JSON
        self.assertIn("[mcp_servers.fridai]", txt)       # Codex TOML
        self.assertIn('args = ["mcp"]', txt)

    def test_client_filter_shows_only_that_client(self):
        txt = cli._mcp_config_text("codex")
        self.assertIn("[mcp_servers.fridai]", txt)       # codex block present
        self.assertNotIn("claude mcp add", txt)          # others omitted
        self.assertNotIn('"mcpServers"', txt)

    def test_client_choice_is_validated(self):
        self.assertEqual(
            cli.build_parser().parse_args(["mcp", "--print-config", "--client", "gemini"]).client,
            "gemini")
        with self.assertRaises(SystemExit):              # unknown client rejected by argparse
            cli.build_parser().parse_args(["mcp", "--client", "cursor"])

    def test_print_config_prints_and_does_not_serve(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_mcp(cli.build_parser().parse_args(["mcp", "--print-config"]))
        self.assertIn("claude mcp add fridai", buf.getvalue())   # returned without starting stdio server


class TestNote(unittest.TestCase):
    def _parse(self, *argv):
        return cli.build_parser().parse_args(["note", *argv])

    def test_empty_note_errors(self):
        with self.assertRaises(SystemExit):
            cli.cmd_note(self._parse(""))

    def test_repo_and_global_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            cli.cmd_note(self._parse("hi", "--repo", "r", "--global"))

    def test_note_saved_and_recallable(self):
        import contextlib
        import io
        from fridai.core import config
        from fridai.core.store import Store
        Store(config.DB_PATH).reset()                      # isolated temp home (tests/__init__)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_note(self._parse("멱등키 TTL은 5분", "--repo", "pay", "--no-embed"))
        self.assertIn("note saved to repo 'pay'", buf.getvalue())
        s = Store(config.DB_PATH)
        try:
            hits = s.search_lexical("멱등키", k=5)
            self.assertTrue(hits)
            self.assertEqual(hits[0].document.source_type, "note")
        finally:
            s.reset()
            s.close()


class TestStats(unittest.TestCase):
    def test_stats_prints_by_repo_sorted_desc(self):
        import contextlib
        import io
        from fridai.core import config
        from fridai.core.models import Document
        from fridai.core.store import Store
        s = Store(config.DB_PATH)                          # isolated temp home (tests/__init__)
        s.reset()
        s.upsert([
            Document(id="c1", source_type="code", repo="big", path="a", title="t", text="x"),
            Document(id="c2", source_type="code", repo="big", path="b", title="t", text="y"),
            Document(id="c3", source_type="commit", repo="small", path="s", title="t", text="z"),
        ])
        s.close()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_stats(cli.build_parser().parse_args(["stats"]))
        out = buf.getvalue()
        self.assertIn("By repo:", out)
        self.assertIn("big=2", out)
        self.assertIn("small=1", out)
        self.assertLess(out.index("big=2"), out.index("small=1"))   # higher count first
        Store(config.DB_PATH).reset()                      # leave the shared temp db clean


class TestForget(unittest.TestCase):
    def _parse(self, *argv):
        return cli.build_parser().parse_args(["forget", *argv])

    def test_requires_exactly_one_target(self):
        with self.assertRaises(SystemExit):
            cli.cmd_forget(self._parse())                 # neither --repo nor --all
        with self.assertRaises(SystemExit):
            cli.cmd_forget(self._parse("--repo", "r", "--all"))   # both

    def test_forget_repo_reports_removal(self):
        import contextlib
        import io
        from fridai.core import config
        from fridai.core.models import Document
        from fridai.core.store import Store
        s = Store(config.DB_PATH)                          # isolated temp home (tests/__init__)
        s.reset()
        s.upsert([Document(id="code:x", source_type="code", repo="proj",
                           path="a.py", title="a", text="hello world")])
        s.close()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.cmd_forget(self._parse("--repo", "proj"))
        self.assertIn("forgot repo 'proj'", buf.getvalue())
        s = Store(config.DB_PATH)
        self.assertEqual(s.stats()["total"], 0)
        s.close()


class TestParser(unittest.TestCase):
    def test_watch_defaults(self):
        a = cli.build_parser().parse_args(["index", "--watch"])
        self.assertTrue(a.watch)
        self.assertEqual(a.interval, 15)             # default 15s

    def test_interval_configurable(self):
        a = cli.build_parser().parse_args(["index", "--watch", "--interval", "30"])
        self.assertEqual(a.interval, 30)

    def test_no_watch_by_default(self):
        self.assertFalse(cli.build_parser().parse_args(["index"]).watch)


if __name__ == "__main__":
    unittest.main()
