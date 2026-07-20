# fridai 🛠️

[![PyPI](https://img.shields.io/pypi/v/fridai.svg)](https://pypi.org/project/fridai/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*[English](README.md) | 한국어*

코딩 에이전트(Claude Code 등)가 `recall` 툴로 당신의 과거 **코드·커밋·AI 대화 기록**을
**100% 로컬**에서 회수하게 해주는 경량 **MCP 서버**. 검색·회수 전용 — **로컬 LLM이 필요 없습니다.**

- **검색 전용(read-only).** 추론은 호출하는 에이전트(LLM)가 하고, fridai는 출처 포함 근거만 회수.
- **로컬 임베딩(fastembed, onnx).** Ollama·외부 API 불필요. 시맨틱 검색이 기본 동작(fastembed 없으면 어휘 검색 폴백).
- **멀티 에이전트 기록.** Claude Code · OpenAI Codex CLI · Gemini CLI 대화를 자동 인덱싱.
- **설계상 프라이버시.** 기억은 기기 밖으로 안 나가고, 시크릿은 인덱싱 시 자동 마스킹.

**상태:** 초기 개발. **요구사항:** Python 3.10+ 및 `git`(코드·커밋 인덱싱용).

## 동작 원리

1. **인덱싱**(`fridai index`)이 세 가지 소스로 로컬 DB를 만듭니다:
   - **AI 대화** — 각 에이전트의 세션 로그를 질문→답변 *턴*으로 파싱하고, 질문을 **결과 git 커밋**과
     매칭(시간창 ∩ 건드린 파일).
   - **코드** — git 추적 파일을 함수/클래스 단위로 청킹(폴백은 N줄 윈도우), 인용용 라인범위 보존.
   - **커밋** — git 커밋 히스토리(제목 + 변경 파일) 인덱싱.
2. 모두 `~/.fridai/index.db`의 **sqlite + FTS5**(어휘) + **float32 벡터**로 저장.
3. **회수**는 어휘(BM25)와 벡터(코사인)를 **RRF**로 융합한 뒤, 실제 작업 산물(코드/커밋/편집 턴)이
   단순 질문 턴보다 위로 오게 재랭킹하고 반복 질문을 중복 제거합니다.

## 설치

```bash
pipx install fridai        # 격리 설치(권장)
# 또는: pip install fridai  # venv에서
```

`numpy` + `fastembed`(onnx) + `mcp`가 딸려옵니다. `fridai` 명령이 등록됩니다.

## 빠른 시작

```bash
fridai index --source all             # 인덱스 생성(현재 레포 + 모든 에이전트 대화)
fridai stats                          # 인덱스 개요
claude mcp add fridai -- fridai mcp   # Claude Code에 등록
```

등록 후 에이전트가 `recall` 툴로 회수합니다. 언제든 `fridai index`를 다시 돌리면 갱신되는데,
**증분**(변경된 파일·세션·새 커밋만 재처리)이라 가볍습니다.
자동으로 최신 유지하려면 `fridai index --watch`(기본 15초마다 재인덱싱, `--interval`로 조정),
또는 `fridai install-hook`으로 커밋마다 재인덱싱(별도 프로세스 불필요).

## CLI 레퍼런스

| 명령 | 설명 |
| :--- | :--- |
| `fridai index` | 인덱스 생성/갱신. |
| `fridai mcp` | stdio MCP 서버 실행. |
| `fridai stats` | 소스별 문서 수, 에이전트별 대화 분해, 마지막 인덱싱 시각 출력. |
| `fridai install-hook` | 커밋마다 재인덱싱하는 git post-commit 훅 설치. |

`index` 플래그:

| 플래그 | 의미 |
| :--- | :--- |
| `--source agent\|code\|commits\|all` | 인덱싱 대상(기본 `all`). `agent` = 모든 AI 대화. |
| `--path DIR` | code/commits 대상 레포(기본: 현재 디렉터리). |
| `--reindex` | 증분 무시하고 전체 재구성. |
| `--no-embed` | 임베딩 생략(어휘 인덱스만). |
| `--no-prune` | git에서 삭제된 파일의 청크 유지(code). |
| `--no-redact` | 시크릿 마스킹 끄기(기본 ON). |
| `--watch` | Ctrl-C까지 주기적으로 재인덱싱. |
| `--interval N` | `--watch` 폴링 간격(초, 기본 15). |

## `recall` 툴 (MCP)

서버는 단일 툴 `recall(query, k=5, repo="", source_type="")`을 노출합니다:

- `query` — 검색어(자연어 및/또는 코드 식별자).
- `k` — 최대 결과 수(기본 5).
- `repo` — 빈값 = 현재 작업 레포(서버 cwd 감지); `"all"` = 전체; `"<이름>"` = 특정 레포.
- `source_type` — 빈값 = 전체; `agent_turn` · `code` · `commit` · `note` 중 하나.

에이전트가 읽을 수 있게 번호·출처가 달린 텍스트를 반환합니다. 예:

```
fridai recall — 2 memory item(s) for "docker mount" (all repos, with sources):

### [1] myrepo 2026-07-01 [codex] session:how did I add the docker mount?
Q: how did I add the docker mount?
A: added a bind mount via volumes.

### [2] myrepo/docker-compose.yml:1-20
volumes: ...
```

비-Claude 소스는 태깅됩니다(CLI에선 `🤖 codex`/`🤖 gemini`, 출처엔 `[codex]`/`[gemini]`).

## 다른 MCP 클라이언트에 등록

fridai는 표준 **stdio MCP 서버**이고 실행 명령은 `fridai mcp` 입니다. MCP를 지원하는 어떤
클라이언트에서도 쓸 수 있습니다. `mcpServers` 설정을 읽는 클라이언트라면:

```json
{
  "mcpServers": {
    "fridai": { "command": "fridai", "args": ["mcp"] }
  }
}
```

## 연동 에이전트 경로

| 에이전트 | 기본 데이터 경로 | 오버라이드 환경변수 |
| :--- | :--- | :--- |
| Claude Code | `~/.claude/projects/` | `FRIDAI_CLAUDE_PROJECTS` |
| OpenAI Codex CLI | `~/.codex/sessions/` | `FRIDAI_CODEX_SESSIONS` |
| Gemini CLI | `~/.gemini/tmp/` | `FRIDAI_GEMINI_SESSIONS` |

없는 에이전트 디렉터리는 조용히 건너뜁니다.

## 보안

시크릿(AWS 키, GitHub/Slack 토큰, PEM 개인키, JWT, `password=` 등)은 인덱싱 시 자동 마스킹됩니다(기본 ON).
기기 밖으로 아무것도 나가지 않습니다.

```bash
fridai index --source code --no-redact   # 마스킹 끄기
echo "*.env" >> .fridaiignore                 # 경로 제외(레포 루트 또는 ~/.fridai/)
```

고엔트로피 휴리스틱은 긴 식별자 오탐이 많아 **기본 OFF**입니다. `FRIDAI_REDACT_ENTROPY=1`로 켭니다.

## 환경변수

| 변수 | 기본값 | 설명 |
| :--- | :--- | :--- |
| `FRIDAI_HOME` | `~/.fridai` | 데이터 홈(인덱스 DB 등). |
| `FRIDAI_CLAUDE_PROJECTS` | `~/.claude/projects` | Claude Code transcript 위치. |
| `FRIDAI_CODEX_SESSIONS` | `~/.codex/sessions` | Codex CLI 세션 위치. |
| `FRIDAI_GEMINI_SESSIONS` | `~/.gemini/tmp` | Gemini CLI 세션 위치. |
| `FRIDAI_EMBED_BACKEND` | 자동 | `none`이면 임베딩 끔(어휘만). |
| `FRIDAI_FASTEMBED_MODEL` | `nomic-ai/nomic-embed-text-v1.5` | fastembed 모델명. |
| `FRIDAI_REDACT_ENTROPY` | off | `1`이면 고엔트로피 시크릿 휴리스틱 활성. |
| `FRIDAI_WORK_PENALTY` | `8` | 순수 질문 턴 강등 정도. `0`이면 끔. |
| `FRIDAI_COMMIT_WINDOW_MIN` | `180` | 질문↔결과 커밋 매칭 시간창(분). |

**임베더 일치:** 인덱싱과 쿼리는 같은 임베더로. 모델을 섞으면 차원이 같아도 벡터가 호환되지 않아
어휘 검색으로 폴백합니다. 모델을 바꿨으면 `--reindex`.

## 개발

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

CI(GitHub Actions)가 push/PR마다 Python 3.10–3.14에서 테스트 + fastembed 실동작 스모크를 돌립니다.
테스트는 격리돼 있어(`tests/__init__.py`가 모든 `FRIDAI_*` 경로를 임시로 두고 임베더를 끔) 실제
`~/.fridai`·`~/.codex` 등을 건드리지 않고 모델도 받지 않습니다.

## 라이선스

MIT.
