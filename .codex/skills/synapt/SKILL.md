---
name: synapt
description: synapt recall and dashboard workflow. Use this when operating the MCP memory/channel tools, debugging recall index or channel issues, or working on dashboard and mission-control flows.
---

# synapt Workflow

Use the synapt MCP and repo surfaces deliberately. Prefer the narrowest tool that matches the job, and keep channel/index/root-resolution bugs scoped to the real store in play.

## Core Rules

1. Treat `#dev` as shared operational memory. Read the live channel state before acting on stale assumptions.
2. Use `recall_quick` for cheap speculative lookup, `recall_search` for real retrieval, and `recall_context` only when you need the raw supporting detail.
3. Use `recall_channel(action="unread")` for loop polling first; fall back to a bounded `read` only when unread looks suspicious.
4. Distinguish local project recall state from shared/global channel or org state before debugging.
5. For dashboard or channel bugs, verify the actual runtime checkout and install path before assuming the local repo is what is serving traffic.

## Tool Choice

### Retrieval

```text
recall_quick
recall_search
recall_context
recall_sessions
recall_resume
recall_files
```

### Channel / Ops

```text
recall_channel(action="unread")
recall_channel(action="read", limit=5)
recall_channel(action="post")
recall_channel(action="who")
recall_channel(action="heartbeat")
```

### Persistent memory

```text
recall_journal(action="read" | "write" | "pending")
recall_save
recall_promote
recall_identity
recall_career
```

## Channel Debugging

When channel behavior looks wrong, check these explicitly:

- which project root resolved the channel DB path
- whether the session is in the correct channel store
- whether `unread`, `who`, and `join` agree on membership
- whether the bug is channel-log based, DB-based, or dashboard/API glue

Do not assume a dashboard send bug, unread bug, and store-routing bug are the same issue.

## Index / Recall Debugging

When a recall tool is slow or timing out:

1. isolate whether the cost is `_get_index()` / load time or the actual operation
2. check whether embeddings are being loaded unnecessarily
3. distinguish “no local index” from “shared index is huge/slow”
4. verify whether the current workspace is using the intended recall store

For session/history tools, prefer the no-embeddings path when embeddings are not required.

## Dashboard / Mission Control

For dashboard work, check all three paths:

- API shape in `dashboard/app.py`
- recall/channel read/write helpers underneath
- actual runtime install path serving the dashboard

Cross-project mission-control work should preserve the selected org/project/channel all the way through fetch, send, and live refresh.

## Review Bias

When reviewing synapt changes, look for:

- wrong store/root/channel path resolution
- hidden fallback from shared wrapper to local DB implementation
- tests that only cover the happy path, not cross-project or restart behavior
- performance fixes that accidentally pull embeddings or full index loads into cheap operations
