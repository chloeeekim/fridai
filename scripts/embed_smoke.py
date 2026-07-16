"""fastembed 실동작 스모크 — 실제 모델을 받아 임베딩이 되는지 1회 확인.

CI의 embed-smoke 잡에서만 실행(네트워크로 모델 다운로드 필요). 일반 unittest 스위트는
FRIDAI_EMBED_BACKEND=none 으로 이 경로를 타지 않는다. 여기서는 명시적으로 강제한다.
로컬에서 돌리려면 fastembed 가 설치돼 있어야 한다: `pip install fastembed`.
"""
import os
import sys

os.environ.pop("FRIDAI_EMBED_BACKEND", None)   # 강제 none 방지 → 실제 fastembed 사용

from fridai.core import embeddings   # noqa: E402


def main() -> int:
    e = embeddings.get_embedder()
    if e is None:
        print("FAIL: fastembed 임베더를 만들지 못함(미설치 또는 로드 실패)", file=sys.stderr)
        return 1
    if not e.model_id.startswith("fastembed:"):
        print(f"FAIL: 예상치 못한 임베더: {e.model_id}", file=sys.stderr)
        return 1
    vec = e.embed("fridai embedding smoke test — 임베딩 실동작 확인")
    if not (isinstance(vec, list) and vec and all(isinstance(x, float) for x in vec[:5])):
        print(f"FAIL: 유효한 벡터가 아님: {vec!r}", file=sys.stderr)
        return 1
    print(f"OK fastembed smoke — model={e.model_id} dim={len(vec)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
