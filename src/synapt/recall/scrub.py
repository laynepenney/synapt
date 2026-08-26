"""Transcript-tree scrubbing built on shared stdlib-only primitives."""

from __future__ import annotations

import json
from pathlib import Path

from synapt.scrub import (
    PATTERNS,
    scrub_text,
    strip_markdown_formatting,
    strip_system_artifacts,
)

__all__ = [
    "PATTERNS",
    "scrub_jsonl",
    "scrub_text",
    "strip_markdown_formatting",
    "strip_system_artifacts",
]


def scrub_jsonl(src: Path, dst: Path | None = None) -> Path:
    """Scrub secrets from every supported text field in a JSONL transcript."""
    if dst is None:
        dst = src

    lines: list[str] = []
    with open(src, encoding="utf-8") as stream:
        for raw in stream:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                lines.append(scrub_text(raw))
                continue

            _scrub_entry(entry)
            lines.append(json.dumps(entry, ensure_ascii=False))

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dst


def _scrub_entry(entry: dict) -> None:
    """Mutate *entry* in place, scrubbing text fields."""
    msg = entry.get("message")
    if not isinstance(msg, dict):
        return

    content = msg.get("content")
    if isinstance(content, str):
        msg["content"] = scrub_text(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                block["text"] = scrub_text(block["text"])
            if isinstance(block.get("thinking"), str):
                block["thinking"] = scrub_text(block["thinking"])
            if isinstance(block.get("input"), dict):
                _scrub_dict_values(block["input"])
            if isinstance(block.get("content"), str):
                block["content"] = scrub_text(block["content"])
            if isinstance(block.get("content"), list):
                for sub in block["content"]:
                    if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                        sub["text"] = scrub_text(sub["text"])


def _scrub_dict_values(value: dict) -> None:
    """Recursively scrub string values in a dictionary."""
    for key, item in value.items():
        if isinstance(item, str):
            value[key] = scrub_text(item)
        elif isinstance(item, dict):
            _scrub_dict_values(item)
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                if isinstance(nested, str):
                    item[index] = scrub_text(nested)
                elif isinstance(nested, dict):
                    _scrub_dict_values(nested)
