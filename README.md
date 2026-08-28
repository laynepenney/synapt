<!-- mcp-name: io.github.synapt-dev/recall -->
<p align="center">
  <img src="assets/banner.png" alt="Synapt" width="100%">
</p>

<p align="center">
  <a href="https://pypi.org/project/synapt/"><img src="https://img.shields.io/pypi/v/synapt?color=7c5cbf" alt="PyPI"></a>
  <a href="https://pypi.org/project/synapt/"><img src="https://img.shields.io/pypi/pyversions/synapt?color=00e5cc" alt="Python"></a>
  <a href="https://github.com/synapt-dev/recall/blob/main/LICENSE"><img src="https://img.shields.io/github/license/synapt-dev/recall" alt="License"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Anthropic_Memory-ready-D97706?logo=anthropic&logoColor=white" alt="Anthropic Memory Tool">
  <img src="https://img.shields.io/badge/OpenAI_Agents-ready-412991?logo=openai&logoColor=white" alt="OpenAI Agents">
  <img src="https://img.shields.io/badge/Google_ADK-ready-4285F4?logo=google&logoColor=white" alt="Google ADK">
  <img src="https://img.shields.io/badge/LangChain-ready-1C3C3C?logo=langchain&logoColor=white" alt="LangChain">
  <img src="https://img.shields.io/badge/CrewAI-ready-FF6B35?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0id2hpdGUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iOCIgY3k9IjgiIHI9IjYiLz48L3N2Zz4=&logoColor=white" alt="CrewAI">
  <img src="https://img.shields.io/badge/Claude_Code-ready-D97706?logo=anthropic&logoColor=white" alt="Claude Code">
  <img src="https://img.shields.io/badge/Codex_CLI-ready-412991?logo=openai&logoColor=white" alt="Codex CLI">
</p>

<p align="center">
  Memory and coordination infrastructure for AI coding teams.<br>
  Search past sessions, preserve decisions, coordinate agents, and ship from one persistent system of record.
</p>

<p align="center">
  <a href="https://synapt.dev">Website</a> &middot;
  <a href="https://synapt.dev/guide.html">Guide</a> &middot;
  <a href="https://synapt.dev/blog/">Blog</a> &middot;
  <a href="https://x.com/synapt_dev">@synapt_dev</a>
</p>

---

**synapt** gives Claude Code, Codex CLI, OpenCode, and other MCP-compatible assistants persistent operational memory.

It closes the gap between a one-shot assistant and a real working team:
- recall prior sessions, file history, decisions, and unresolved work
- preserve context in journals, reminders, and knowledge nodes
- coordinate multiple agents through shared channels, directives, and task claims
- scale from solo recall on a laptop to multi-agent operational memory

Agent skill files for repository-native use live in:
- `.codex/skills/synapt/SKILL.md`
- `.claude/skills/synapt/SKILL.md`

## Three-command quickstart

For the default Claude Code path:

```bash
pip install synapt
claude mcp add synapt -- synapt server
synapt init
```

That gives you:
- a project-local `.synapt/` memory store
- indexed Claude Code and Codex transcripts
- Claude hooks for automatic archive/build flow
- the published Codex `dev-loop` skill installed into `${CODEX_HOME:-~/.codex}/skills/dev-loop/`

## Platform setup

### Claude Code

Recommended:

```bash
pip install synapt
claude mcp add synapt -- synapt server
synapt init
```

### Codex CLI

Install:

```bash
pip install synapt
```

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.synapt]
command = "synapt"
args = ["server"]
```

Then initialize the project:

```bash
synapt init
```

### Cursor

Add to project-local `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "synapt": {
      "command": "synapt",
      "args": ["server"]
    }
  }
}
```

Then run:

```bash
synapt init
```

### Windsurf

Add to `~/.codeium/windsurf/mcp_config.json`:

```json
{
  "mcpServers": {
    "synapt": {
      "command": "synapt",
      "args": ["server"]
    }
  }
}
```

Then run:

```bash
synapt init
```

### OpenCode

Add to `~/.config/opencode/opencode.json`:

```json
{
  "mcp": {
    "synapt": {
      "type": "local",
      "command": ["synapt", "server"],
      "enabled": true
    }
  }
}
```

Then run:

```bash
synapt init
```

### Manual MCP config

If your client accepts stdio MCP definitions directly, use:

```json
{
  "mcpServers": {
    "synapt": {
      "type": "stdio",
      "command": "synapt",
      "args": ["server"]
    }
  }
}
```

## Framework integrations

synapt plugs into popular agent frameworks as a drop-in memory backend. Each adapter wraps recall's search and save API for the framework's native session interface.

Install all integration dependencies at once:

```bash
pip install synapt[all-integrations]
```

Or install only the ones you need:

```bash
pip install synapt[anthropic]    # Anthropic Memory Tool
pip install synapt[openai]       # OpenAI Agents SDK
pip install synapt[google-adk]   # Google ADK
pip install synapt[langchain]    # LangChain
pip install synapt[crewai]       # CrewAI
```

### Anthropic Memory Tool

`SynaptMemoryTool` is a drop-in replacement for `BetaAbstractMemoryTool`. It presents recall's knowledge graph as a virtual filesystem that Claude can view, create, edit, and search. All writes are persisted as durable recall knowledge nodes.

```bash
pip install synapt[anthropic]
```

**Before** (default memory, no cross-session persistence):

```python
from anthropic import Anthropic

client = Anthropic()
response = client.beta.messages.run_tools(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Remember that we use blue-green deploys"}],
    tools=[{"type": "memory_20250818", "name": "memory"}],
).until_done()
```

**After** (recall-backed memory with hybrid search):

```python
from anthropic import Anthropic
from synapt.integrations.anthropic import SynaptMemoryTool

client = Anthropic()
memory = SynaptMemoryTool()

response = client.beta.messages.run_tools(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Remember that we use blue-green deploys"}],
    tools=[memory],
).until_done()
```

An async variant (`SynaptAsyncMemoryTool`) is available for `AsyncAnthropic` clients.

### OpenAI Agents SDK

`SynaptSession` is a session persistence adapter for the Agents SDK `Session` protocol. Items are stored in SQLite with non-blocking async DB access. Recall search and memory context are available as convenience methods.

```bash
pip install synapt[openai]
```

**Before** (default SQLiteSession, no cross-session search):

```python
from agents import Agent, Runner
from agents.memory import SQLiteSession

session = SQLiteSession(session_id="user-123", db_path="sessions.db")
runner = Runner(agent=agent, session=session)
```

**After** (recall-backed session with memory context):

```python
from agents import Agent, Runner
from synapt.integrations.openai_agents import SynaptSession

session = SynaptSession(session_id="user-123")
runner = Runner(agent=agent, session=session)

# Retrieve recall context for agent prompts
context = session.get_memory_context("deployment strategy")

# Persist durable knowledge
session.save_to_recall("Always use UTC timestamps", category="convention")
```

### Google ADK

`SynaptMemoryService` is a drop-in replacement for ADK's `InMemoryMemoryService`. Session events are indexed in recall's knowledge graph; `search_memory` uses hybrid retrieval instead of keyword matching.

```bash
pip install synapt[google-adk]
```

**Before** (default in-memory, keyword-only search):

```python
from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService

memory = InMemoryMemoryService()
agent = LlmAgent(model="gemini-2.5-flash", name="my_agent", memory_service=memory)
```

**After** (recall-backed, hybrid search):

```python
from google.adk.agents import LlmAgent
from synapt.integrations.google_adk import SynaptMemoryService

memory = SynaptMemoryService()
agent = LlmAgent(model="gemini-2.5-flash", name="my_agent", memory_service=memory)
```

### LangChain

`SynaptChatMessageHistory` implements LangChain's `BaseChatMessageHistory`. Messages are stored in SQLite with WAL mode; recall search and knowledge persistence are one method call away.

```bash
pip install synapt[langchain]
```

```python
from langchain_core.messages import HumanMessage
from synapt.integrations.langchain import SynaptChatMessageHistory

history = SynaptChatMessageHistory(session_id="user-123")
history.add_messages([HumanMessage(content="hello")])
print(history.messages)

# Semantic search across all indexed sessions
results = history.search("deployment config")

# Persist a decision as a durable knowledge node
history.save_to_recall("Always use UTC timestamps", category="convention")
```

Works with `RunnableWithMessageHistory`, `ConversationChain`, and any LangChain component that accepts a `BaseChatMessageHistory`.

### CrewAI

`SynaptMemory` provides long-term memory storage for CrewAI crews, backed by recall's hybrid search.

```bash
pip install synapt[crewai]
```

```python
from synapt.integrations.crewai import SynaptMemory

memory = SynaptMemory()
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    memory=memory,
)
crew.kickoff()
```

### Google ADK

`SynaptMemoryService` implements ADK's `BaseMemoryService` with tenant-scoped search and persistence.

```bash
pip install synapt[google-adk]
```

```python
from synapt.integrations.google_adk import SynaptMemoryService

memory = SynaptMemoryService()
# search_memory scopes results by app_name and user_id
# add_memory persists with tenant tags for isolation
```

### Claude Code (native memory backend)

Replace Claude Code's built-in memory with recall-backed persistent memory in one line:

```python
from synapt.integrations.anthropic import SynaptMemoryTool

# Drop-in replacement for BetaLocalFilesystemMemoryTool
tool = SynaptMemoryTool()
```

Every `create`, `view`, `str_replace`, and `search` call now goes through recall: files are persisted, content is enriched, and search returns semantically relevant results across sessions. No configuration needed; `pip install synapt` includes everything.

For MCP-based recall (search, journals, channels, 20+ tools):

```bash
pip install synapt
claude mcp add synapt -- synapt server
synapt init
```

`synapt init` installs session hooks for automatic transcript archiving.

The repository's `claude-plugin/` package also owns a bounded `SessionStart`
hook. Install it from this repository's marketplace:

```text
/plugin marketplace add synapt-dev/recall
/plugin install synapt-recall@synapt-plugins
```

Current recall context is then injected automatically without resuming an old
conversation. Do not also install the legacy global `SessionStart` hook because
Claude Code runs both registrations.

For shared workspaces, link `claude-plugin/project-settings.json` to
`.claude/settings.json`. After the folder is trusted, Claude Code prompts once
for the marketplace and plugin consent, then keeps the plugin updated.

### Codex CLI

Install synapt and register the MCP server:

```bash
pip install synapt
```

Add to `~/.codex/config.toml`:

```toml
[mcp_servers.synapt]
command = "synapt"
args = ["server"]
```

Then initialize the project:

```bash
synapt init
```

`synapt init` installs the `dev-loop` skill automatically, giving Codex recall search, channel coordination, and journal access.

The repository's `codex-plugin/` package owns the equivalent bounded
`SessionStart` hook and the bounded `SessionEnd` recovery checkpoint. A
gripspace can link that package's `hooks/hooks.json` to workspace
`.codex/hooks.json`, keeping the capability in recall while making continuity
automatic for every Codex session in the workspace.

### Session continuity policy

SessionStart recovery is intent-aware. A compaction-triggered start is always a
no-op because the runtime already carried the live conversation forward. The
default `automatic` policy also treats `/clear` as an intentional clean slate.

Set the policy globally in `~/.synapt/config.json` or per project in
`.synapt/recall/config.json`:

```json
{
  "session_start": {
    "continuity": "automatic"
  }
}
```

Available modes are `off`, `explicit`, `automatic`, and `always`. `explicit`
only injects recovery context for a runtime `resume`. `automatic` covers normal
startup, resume, and fork. `always` also crosses a clear boundary when that is
explicitly desired. `SYNAPT_SESSION_START_CONTINUITY` provides the equivalent
environment override.

### Recall data root

Every `recall_channel` response reports the absolute channel state store it
used. This makes identical calls from different worktrees distinguishable,
including empty results and failures after the store has been resolved.

Recall resolves its data root in this order:

1. an explicit project directory, or `SYNAPT_RECALL_ROOT` when no project
   directory was supplied
2. the repository's main worktree
3. the GitGrip workspace root
4. the current working directory

The reported path is operational provenance only. Public recall does not infer
or report the caller's identity from it.

## What `synapt init` does

Run from a project root:

```bash
synapt init
```

It will:
1. archive project-relevant Claude Code and Codex transcripts
2. build the `.synapt/recall/index/recall.db` search index
3. register the Synapt MCP server in Claude Code when the `claude` CLI is available
4. install Claude hooks for bounded `SessionStart`, bounded `SessionEnd`
   checkpoint capture, and `PreCompact`
5. deploy the packaged Codex `dev-loop` skill
6. add `.synapt/` to `.gitignore`

`synapt recall setup` remains available as the explicit recall-scoped equivalent.

## Product tiers

synapt is one memory system with a clear adoption ladder:

### 1. Solo recall

Search prior sessions, file history, timelines, and journals on one machine.

```bash
synapt recall search "how did we fix auth"
synapt recall files "src/auth.py"
synapt recall timeline
```

### 2. Multi-agent memory

Add channels, directives, reminders, and task claims for coordinated execution across worktrees and agents.

```python
recall_channel(action="join", channel="dev", name="Atlas")
recall_channel(action="intent", channel="dev", message="reviewing PR #403")
recall_channel(action="claim", channel="dev", message="m_abc123")
```

### 3. Dashboard

Expose the same shared operational memory in a browser-facing mission-control surface.

### 4. Spawn / orchestration

Use synapt as the memory and coordination substrate beneath higher-level agent orchestration.

## Core features

- **Hybrid search**: BM25 + embeddings + reciprocal rank fusion + reranking
- **File-aware recall**: find where a file, bug, issue, or decision was handled before
- **Journal + knowledge**: durable summaries, extracted facts, contradictions, and timeline arcs
- **Agent channels**: shared append-only coordination across sessions and worktrees
- **Cross-client memory**: Claude Code and Codex transcripts converge into one searchable system
- **Portable archive**: export/import `.synapt-archive` state between machines
- **Plugin system**: extend MCP tools and CLI commands through Python entry points

## Benchmarks

### LOCOMO

LOCOMO evaluates long conversational memory over 10 conversations and 1540 QA pairs.

*Selected published LOCOMO results as of 2026-05-21. Methodologies vary across systems including judge model, benchmark harness, and conversation count. This table does not present a comparable leaderboard. synapt's 72.4 is the March 2026 audited reproducible run under the gpt-4o-mini judge methodology matching the Mem0 paper. Rows marked `[methodology-caveat]` use different judge models or harnesses and are not directly comparable to synapt's audited run. Rows marked `[historical: ENGRAM paper]` reflect each system's score as published in the ENGRAM paper Table 1 with the shared gpt-4o-mini backbone, which may have been superseded by newer releases from those systems under different methodologies.*

| System | LOCOMO J-score | Multi-Hop | Temporal | Methodology / Source |
|--------|---------------:|----------:|---------:|----------------------|
| EverCore `[methodology-caveat]` | 93.05 | — | — | EverOS repo + EverMemOS paper ([arXiv:2601.02163](https://arxiv.org/abs/2601.02163)); judge/harness pending verification |
| Hindsight `[methodology-caveat]` | 92.0 (AMB) / 89.61 (paper) | — | — | Hindsight AMB benchmark page + paper ([arXiv:2512.12818](https://arxiv.org/abs/2512.12818)) |
| MemMachine `[methodology-caveat]` | 91.69 | — | — | [arXiv:2604.04853](https://arxiv.org/abs/2604.04853) + Apache-2.0 repo; gpt-4.1-mini agent mode |
| Mem0 managed `[methodology-caveat]` | 91.8 / 90.2 | — | — | Mem0 docs May 2026; managed-platform optimizations not in OSS SDK |
| Engram `[historical: ENGRAM paper]` | 77.55 ± 0.13 | 79.79 | 70.79 | ENGRAM paper Table 1 ([arXiv:2511.12960](https://arxiv.org/abs/2511.12960)); shared gpt-4o-mini backbone |
| Zep\* (Memobase-team corrected) `[methodology-caveat]` | 75.14 | 66.04 | 79.79 | Memobase LOCOMO docs; supersedes the ENGRAM Zep row |
| Memobase v0.0.37 `[methodology-caveat]` | 75.78 | 46.88 | **85.05** | Memobase LOCOMO docs; judge may be gpt-4o not gpt-4o-mini |
| memOS `[historical: ENGRAM paper]` | 72.99 ± 0.14 | 63.70 | 72.68 | ENGRAM paper Table 1 |
| Full-Context `[historical: ENGRAM paper]` | 72.60 ± 0.07 | — | — | ENGRAM paper Table 1; upper-bound control |
| **synapt (audited)** | **72.4** | **70.92** | 59.19 | March 2026 audited rerun; gpt-4o-mini judge; Ministral 8B cloud enrich |
| **synapt local-first** | **72.4** | 67.02 | 61.06 | March 2026 audited rerun; gpt-4o-mini judge; **local 3B on M2 Air** |
| Mem0 `[historical: ENGRAM paper]` | 64.73 ± 0.17 | 57.85 | 53.34 | ENGRAM paper Table 1; Mem0's newer releases use different methodologies (see Mem0 managed row) |
| Zep `[historical: ENGRAM paper]` | 42.29 ± 0.18 | 42.10 | 19.47 | ENGRAM paper Table 1; see Zep\* row for the Memobase-team corrected number |

What matters for the pitch:
- synapt's 72.4 is reproducible end-to-end against published methodology including the gpt-4o-mini judge, 1540 questions, and 10 conversations from the Mem0 paper
- the local-first 3B path achieves the same audited overall score on commodity hardware
- LOCOMO rankings shift with judge model, harness version, and OSS-versus-managed-platform scope. This table does not assert a leaderboard position

Sources:
- [LOCOMO benchmark](https://snap-research.github.io/locomo/)
- [Mem0 paper](https://arxiv.org/abs/2504.19413)
- [ENGRAM paper](https://arxiv.org/abs/2511.12960)
- [Hindsight paper](https://arxiv.org/abs/2512.12818)
- [MemMachine paper](https://arxiv.org/abs/2604.04853)
- [EverMemOS paper](https://arxiv.org/abs/2601.02163)
- [Memobase LOCOMO benchmark](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)

### CodeMemo

CodeMemo evaluates coding-memory tasks across factual recall, debugging context, architecture, temporal ordering, conventions, and cross-session continuity.

*Comparison as of 2026-05-21. Single-evaluation snapshot under matched methodology.*

| System | Overall |
|--------|---------|
| **synapt v0.6.2** | **90.51** |
| Mem0 | 76.00 |

Source:
- [CodeMemo benchmark](evaluation/codememo/README.md)

## Security and compliance

synapt is built for teams that care where memory lives and how it is inspected.

- **Local-first by default**: transcripts, channels, journals, and indexes live on disk under `.synapt/`
- **No mandatory cloud memory backend**: core recall works locally
- **Inspectable storage**: JSONL transcripts plus SQLite/FTS5 state
- **Portable backup path**: export/import via `.synapt-archive`
- **Optional remote behavior is explicit**: sync and plugin integrations are opt-in

For disclosure and reporting policy, see [SECURITY.md](SECURITY.md).

## Example workflows

Search a prior fix:

```bash
synapt recall search "why did we disable snippets in retrieval-only mode"
```

Find a file’s prior context:

```bash
synapt recall files "src/synapt/recall/channel.py"
```

Run Codex on a timed review loop:

```bash
./scripts/codex-loop.sh \
  --interval 60 \
  --prompt "check #dev, review fresh PRs, or pick up the next unowned task. Post what you're doing in #dev." \
  -- --full-auto
```

## Why teams adopt it

Without memory, every new assistant session starts as a stranger.

With synapt, teams can:
- recover prior decisions instead of re-deriving them
- hand off work without losing context
- coordinate multiple agents without duplicating effort
- keep operational memory local, inspectable, and portable

That is the difference between an assistant demo and an operational system.
