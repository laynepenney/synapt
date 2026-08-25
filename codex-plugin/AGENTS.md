# synapt Codex Plugin

Use this plugin when Codex should have direct access to synapt recall over MCP.

## What it provides

- local `synapt server` MCP access
- bounded recall context on every session start
- recall search, save, context, journal, and channel tools
- no identity or org behavior; this is OSS-only packaging of existing recall primitives

## Install shapes

Plugin form:

```bash
codex plugin add synapt-dev/recall-plugin
```

Direct MCP fallback:

```bash
codex mcp add synapt -- synapt server
```

## Usage guidance

- prefer `recall_quick` for cheap speculative lookup
- use `recall_search` for real retrieval
- use `recall_save` only for durable facts, conventions, or decisions
- treat `#dev` as shared operational memory when the session is part of team coordination

## Boundary

Premium boundary: this plugin is OSS because it only packages the public synapt MCP server and instructions. Identity, org routing, and premium policy remain outside this scaffold.
