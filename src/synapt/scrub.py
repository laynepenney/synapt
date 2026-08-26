"""Stdlib-only secret scrubbing shared by normal and bounded paths."""

from __future__ import annotations

import hashlib
import re


def _redact(match: re.Match) -> str:
    digest = hashlib.sha256(match.group().encode()).hexdigest()[:8]
    return f"[REDACTED:{digest}]"


PATTERNS: list[re.Pattern] = [
    re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    re.compile(r"gh[pso]_[A-Za-z0-9]{36,}"),
    re.compile(r"(?:xox[bpeas]|xapp)-[A-Za-z0-9./-]{10,}"),
    re.compile(r"ak-[A-Za-z0-9_-]{20,}"),
    re.compile(r"pypi-[A-Za-z0-9_]{50,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    re.compile(
        r"Authorization:\s*(?:Bearer|Key|Basic|Token)\s+[A-Za-z0-9._/+:=-]{8,}",
        re.IGNORECASE,
    ),
    re.compile(
        r"-----BEGIN[A-Z \t]*PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END[A-Z \t]*PRIVATE KEY-----",
    ),
    re.compile(r"-----BEGIN[A-Z \t]*PRIVATE KEY-----"),
    re.compile(
        r"(?:postgres|postgresql|mysql|mongodb(?:\+srv)?|redis|amqp)://"
        r"[^@\s]{0,64}:[^@\s]{1,128}@[^\s]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:API_KEY|SECRET_KEY|_TOKEN|_SECRET|PASSWORD|PRIVATE_KEY|ACCESS_KEY"
        r"|_KEY|_CREDENTIAL|_CREDENTIALS|_AUTH|_PASS|_PASSPHRASE)"
        r"(?![A-Za-z_])"
        r"""[=:]\s*['"]?[A-Za-z0-9_/+.:=-]{8,}['"]?""",
        re.IGNORECASE,
    ),
]

_ARTIFACT_TAG_RE = re.compile(
    r"<(system-reminder|local-command-caveat|available-deferred-tools|env)"
    r"(?:\s[^>]*)?>.*?</\1>",
    re.DOTALL,
)
_ARTIFACT_OPEN_RE = re.compile(
    r"<(system-reminder|local-command-caveat|available-deferred-tools)"
    r"(?:\s[^>]*)?>.*",
    re.DOTALL,
)
_INTERRUPTED_LITERAL = "[Request interrupted by user for tool use]"
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MARKDOWN_BOLD_ITALIC_RE = re.compile(r"\*{1,2}([^*]+)\*{1,2}")


def strip_system_artifacts(text: str) -> str:
    """Remove runtime-injected artifacts from user-visible text."""
    if not text:
        return text
    text = _ARTIFACT_TAG_RE.sub("", text)
    text = _ARTIFACT_OPEN_RE.sub("", text)
    text = text.replace(_INTERRUPTED_LITERAL, "")
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def strip_markdown_formatting(text: str) -> str:
    """Remove bold and italic Markdown markers."""
    if not text:
        return text
    return _MARKDOWN_BOLD_ITALIC_RE.sub(r"\1", text)


def scrub_text(text: str) -> str:
    """Replace supported secret classes with deterministic placeholders."""
    if not text:
        return text
    for pattern in PATTERNS:
        text = pattern.sub(_redact, text)
    return text
