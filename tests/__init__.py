"""테스트 격리 — 로컬 환경(홈 디렉터리·실제 에이전트 기록·네트워크 임베딩 모델)과 분리.

이 패키지가 로드되는 시점(테스트 임포트 전)에 FRIDAI_* 기본값을 임시/격리 경로로 고정한다.
`setdefault`라 CI나 개발자가 명시적으로 지정하면 그 값을 존중한다. 덕분에 어느 머신에서도
(실제 ~/.fridai·~/.codex 등을 건드리지 않고, 모델 다운로드 없이) 동일하게 재현된다.
"""
import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="fridai_tests_")

os.environ.setdefault("FRIDAI_HOME", os.path.join(_TMP, "home"))
os.environ.setdefault("FRIDAI_CLAUDE_PROJECTS", os.path.join(_TMP, "claude"))
os.environ.setdefault("FRIDAI_CODEX_SESSIONS", os.path.join(_TMP, "codex"))
os.environ.setdefault("FRIDAI_GEMINI_SESSIONS", os.path.join(_TMP, "gemini"))
# 임베더 끔 → fastembed 모델 네트워크 다운로드 방지(어휘 검색으로 결정적 동작).
os.environ.setdefault("FRIDAI_EMBED_BACKEND", "none")
