"""synapt.recall — persistent conversational memory for Claude Code and ChatGPT sessions."""

__version__ = "0.17.0"

from synapt.recall.core import (
    TranscriptChunk,
    TranscriptIndex,
    build_index,
    parse_transcript,
    project_slug,
    project_index_dir,
    project_archive_dir,
    project_worktree_dir,
    project_transcript_dir,
    project_transcript_dirs,
    all_worktree_archive_dirs,
    _is_real_user_message,
    _extract_user_text,
    _extract_assistant_content,
)
from synapt.recall.chatgpt import (
    parse_chatgpt_archive,
    parse_chatgpt_conversation,
)

__all__ = [
    "TranscriptChunk",
    "TranscriptIndex",
    "build_index",
    "parse_transcript",
    "parse_chatgpt_archive",
    "parse_chatgpt_conversation",
    "_is_real_user_message",
    "_extract_user_text",
    "_extract_assistant_content",
    "project_slug",
    "project_index_dir",
    "project_archive_dir",
    "project_worktree_dir",
    "project_transcript_dir",
    "project_transcript_dirs",
    "all_worktree_archive_dirs",
]
