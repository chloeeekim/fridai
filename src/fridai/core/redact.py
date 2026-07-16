"""Secret redaction. Masks secrets as an indexing pre-step.

Core to user trust — even locally, secrets shouldn't leak into a plaintext index.
A precise regex ruleset + a conservative high-entropy heuristic. Disable with `--no-redact`.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from . import config

_MASK = "«REDACTED:{}»"

# Precise patterns (almost no false positives)
_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                re.DOTALL), "PRIVATE_KEY"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS_ACCESS_KEY"),
    (re.compile(r"ghp_[0-9A-Za-z]{36}"), "GITHUB_TOKEN"),
    (re.compile(r"gh[oprsu]_[0-9A-Za-z]{36,}"), "GITHUB_TOKEN"),
    (re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"), "SLACK_TOKEN"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "API_KEY"),
    (re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), "JWT"),
    # key=value style secrets (mask the value only)
    # excludes the mask char («) from the value charset so already-masked tokens aren't re-matched
    (re.compile(r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|access[_-]?key|"
                r"client[_-]?secret)\b(\s*[=:]\s*)([\"']?)([^\s\"'«]{6,})\3"),
     "SECRET_KV"),
]

# High-entropy candidate token: 32+ chars, mixed upper/lower/digits (excludes git SHA (lowercase hex) & snake_case)
_HE_TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{32,}")


def _shannon(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _looks_secret(tok: str) -> bool:
    return (any(c.isupper() for c in tok) and any(c.islower() for c in tok)
            and any(c.isdigit() for c in tok) and _shannon(tok) > 4.0)


def redact_text(text: str, entropy: bool | None = None) -> tuple[str, int]:
    """Returns (masked text, mask count). entropy=None follows config.REDACT_ENTROPY (default OFF)."""
    if not text:
        return text, 0
    if entropy is None:
        entropy = config.REDACT_ENTROPY
    n = 0
    for rx, label in _RULES:
        if label == "SECRET_KV":
            text, c = rx.subn(lambda m: f"{m.group(1)}{m.group(2)}{_MASK.format('SECRET')}", text)
        else:
            text, c = rx.subn(_MASK.format(label), text)
        n += c

    if entropy:                     # off by default (many false positives); only when explicitly enabled
        def _he(m):
            nonlocal n
            if _looks_secret(m.group(0)):
                n += 1
                return _MASK.format("HIGH_ENTROPY")
            return m.group(0)
        text = _HE_TOKEN.sub(_he, text)
    return text, n


_META_KEYS = ("question", "answer", "answer_summary", "summary")


def redact_document(doc) -> int:
    """Mask the document's text/title and key meta strings in place. Returns total mask count."""
    total = 0
    doc.text, c = redact_text(doc.text); total += c
    doc.title, c = redact_text(doc.title); total += c
    for k in _META_KEYS:
        v = doc.meta.get(k)
        if isinstance(v, str):
            doc.meta[k], c = redact_text(v); total += c
    return total
