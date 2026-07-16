"""비밀정보 레다션 테스트 — 시크릿은 마스킹, 정상 코드/SHA는 보존."""
import unittest

from fridai.core import redact
from fridai.core.models import Document


class TestRedactText(unittest.TestCase):
    def _masked(self, text):
        out, n = redact.redact_text(text)
        return out, n

    def test_aws_access_key(self):
        out, n = self._masked("key = AKIAIOSFODNN7EXAMPLE")
        self.assertIn("REDACTED:AWS_ACCESS_KEY", out)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", out)
        self.assertEqual(n, 1)

    def test_github_token(self):
        out, _ = self._masked("token: ghp_" + "a" * 36)
        self.assertIn("REDACTED:GITHUB_TOKEN", out)

    def test_private_key_block(self):
        pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----"
        out, _ = self._masked(pem)
        self.assertIn("REDACTED:PRIVATE_KEY", out)
        self.assertNotIn("MIIabc", out)

    def test_password_kv_masks_value_keeps_key(self):
        out, _ = self._masked('password = "hunter2secret"')
        self.assertIn("password", out)              # 키는 유지
        self.assertNotIn("hunter2secret", out)      # 값만 마스킹

    def test_high_entropy_opt_in_only(self):
        secret = "Wja7c8KQ2pLZ9xVf3RtBn6MeYh1Ds0Uq4Gi5Ko7"   # base64풍 혼합
        out_off, _ = redact.redact_text(secret, entropy=False)
        self.assertEqual(out_off, secret)                     # 기본(OFF)은 안 가림
        out_on, n = redact.redact_text(secret, entropy=True)
        self.assertIn("REDACTED:HIGH_ENTROPY", out_on)        # opt-in 시 가림
        self.assertEqual(n, 1)

    def test_camelcase_identifier_not_redacted_by_default(self):
        # 실제 오탐 사례: 긴 camelCase 식별자(대/소/숫자 혼합 32자+)
        ident = "maxDownloadBandwidthDeviceSystemInfoV2"
        out, n = redact.redact_text(ident)                    # 기본 entropy OFF
        self.assertEqual(out, ident)
        self.assertEqual(n, 0)

    # ── 오탐 방지 ──
    def test_git_sha_not_redacted(self):
        out, n = self._masked("커밋 a3f9c2b1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9 에서 고침")
        self.assertIn("a3f9c2b1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9", out)  # 소문자 hex SHA 보존
        self.assertEqual(n, 0)

    def test_normal_code_not_redacted(self):
        code = "def refresh_token(old_token):\n    return mint(old_token.user_id)"
        out, n = self._masked(code)
        self.assertEqual(out, code)
        self.assertEqual(n, 0)

    def test_empty(self):
        self.assertEqual(redact.redact_text(""), ("", 0))


class TestRedactDocument(unittest.TestCase):
    def test_redacts_text_title_and_meta(self):
        d = Document(id="1", source_type="agent_turn", repo="r", path="s",
                     title="AKIAIOSFODNN7EXAMPLE 관련", text="key AKIAIOSFODNN7EXAMPLE",
                     meta={"question": "AKIAIOSFODNN7EXAMPLE 왜?", "answer": "secret",
                           "files": ["a.py"]})
        n = redact.redact_document(d)
        self.assertGreaterEqual(n, 3)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", d.text)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", d.title)
        self.assertNotIn("AKIAIOSFODNN7EXAMPLE", d.meta["question"])
        self.assertEqual(d.meta["files"], ["a.py"])     # 비문자열 meta 보존


if __name__ == "__main__":
    unittest.main()
