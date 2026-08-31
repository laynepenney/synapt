# Changelog

All notable changes to synapt are documented here.

## [Unreleased]

### Fixed
- **A session that ended without a handoff is named as such at the next wake.**
  A host crash, kill, or forced shutdown runs no SessionEnd, so no checkpoint
  is written, and nobody journals a session they did not know was ending. The
  session-start hook then rendered whatever `checkpoint.json` held, which can be
  another session's clean SessionEnd (measured after a crash: a subagent's tail
  under "LAST CHECKPOINT", twelve hours of the main session's work with no
  record, and nothing saying so). `synapt.recall.resume.detect_unclean_end`
  judges the newest previous caller transcript against the authored journals
  and a checkpoint bound to that session's id. A handoff is session-bound
  evidence: a journal bound to the session covers it when written within the
  fifteen-minute grace before its last activity or any time after; a legacy
  sessionless journal covers only inside the symmetric window; a journal bound
  to another session never covers, so later work by other sessions cannot erase
  the verdict. Exclusion of the starting
  session is by identity alone: recency is not liveness evidence, since a
  crash followed by a fast restart sits inside any recency window;
  when neither covers it, the wake leads with an `UNCLEAN END` block that names
  the session, the gap, the foreign checkpoint if any, and the session's own
  bounded recovered tail, and `synapt resume` carries the same verdict in its
  header. The starting session is excluded by the hook payload's `session_id`
  (or `CLAUDE_CODE_SESSION_ID` / `CODEX_THREAD_ID` on the generic
  `synapt recall startup` path); a caller that cannot name its own session does
  not publish the verdict, so a wake never reports itself.
- **`LAST CHECKPOINT` names the session it belongs to**, so a checkpoint from
  one session is not read as the bridge for another.
- **The recovered tail skips runtime-authored user lines.** `<task-notification>`,
  `<system-reminder>` and the slash-command echo family are written with the
  user role but are not the operator's words; the checkpoint reader now keeps
  the last human line instead (the crashed session's real last request, not the
  background-task notice that followed it). Prose that merely quotes a tag keeps
  its author.
- **A #dev backlog is read one line per message, up to thirty.** Five messages
  at full detail is the right read after a quiet night and the wrong one after a
  gap: with eleven unread, three rendered inside the channel byte cap and the
  two that mattered were withheld. Above five unread the wake switches to the
  one-line form so it covers what happened rather than the three newest posts.
- **Cross-gripspace direct messages no longer silo.** `send_message` reported
  "Delivered" on a cross-org send while the recipient saw nothing: both the JSONL
  inbox and the SQLite delivery-state store resolve through the gripspace-local
  `_channels_dir(project_dir)`, so a send from gripspace A landed in A's store
  while a read from gripspace B queried B's store and found nothing. The fix adds
  a project-independent, org-canonical cross-org root
  (`~/.synapt/channels/<org>/_cross-org/direct/<recipient>.jsonl`, honoring
  `SYNAPT_SHARED_CHANNELS_DIR`): `send_message` dual-writes the local inbox and
  the canonical inbox, and `read_inbox` unions the local SQLite delivered set
  with the canonical messages, deduped by `message_id`, urgent-then-oldest, with
  a local tracking row written for first-read cross-org messages so re-reads
  dedup and `ack` transitions them. Forward-fix (dual-write + union-read), no
  destructive migration of existing inboxes.
- **Messages the operator queued mid-turn are indexed.** Typing while a turn is
  running does not write a `type: "user"` line: Claude Code emits a
  `queue-operation` line, which the parser skips as unstructured noise, plus an
  `attachment` line whose `attachment.prompt` carries the text. Neither was read,
  so the operator's own words never reached `user_text` and a query for something
  they said mid-turn could return the assistant's later paraphrase and never the
  message itself.

  `origin.kind` is the safety boundary and there are at least three cohorts, so
  it also defines the measurement cohort: `human` (the operator, the only user
  turn), `peer` (another agent's queued message), and entries with no `origin`
  key at all (`commandMode == "task-notification"`, machine background events).
  Presence of an `origin` cannot distinguish authorship, since both `human` and
  `peer` have one; only `origin.kind == "human"` is accepted.

  Measured on one workspace at 2026-08-30T13:44Z, and every count is a property
  of that moment because these files are appended to continuously: of 510
  `queued_command` attachments, 352 were `human` against 1075 ordinary user
  turns — about a quarter of what the operator said — and 349 of the 352 appear
  nowhere else. The remaining 158 are task-notifications, not empty text.

  `prompt` arrives as a string or as a list of content blocks; both are read,
  and only text blocks within a list are the operator's words.

  **A rebuild is required for the fix to reach existing chunks.**
  `--incremental` decides what to re-parse from an mtime-and-size manifest, so
  transcripts that have not changed since their last index are skipped no matter
  what the parser now understands. Run a full build, or invalidate the manifest.

## [0.21.0] — 2026-08-26

**Scope.** This release promotes `v0.20.0..b8f22c4`: 7 commits and 2 first-parent
units, measured with `git rev-list --count` and `git rev-list --first-parent --count`
over that exact range.

### Added
- **Bounded SessionEnd recovery checkpoints.** Claude Code and Codex hooks capture a
  scrubbed continuity checkpoint inside the three-second SessionEnd ceiling without
  importing SQLite, subprocess, or the recall indexing stack.
- **Runtime compaction handoff indexing.** Recall locates trustworthy compaction
  summaries while indexing sessions and can surface the newest handoff on a later
  fresh start.
- **Standalone release binaries.** Release automation now builds hermetic macOS arm64
  and Linux x86_64 executables with checksums for package-manager distribution.

### Changed
- **SessionStart continuity is configurable.** `off`, `explicit`, `automatic`, and
  `always` modes let users choose how much startup recovery they want. Automatic is
  the default.
- **Compaction-triggered starts do no resume work.** Plugin and global SessionStart
  matchers exclude `compact`, with an early CLI guard as a backstop. The active
  conversation already carries its context across compaction.
- **Agent plugin versions advance to 0.3.0.** Both the Claude Code and Codex bundles
  include the SessionEnd checkpoint and compact-start exclusion.

## [0.20.0] — 2026-08-25

**Scope.** This release promotes `v0.19.1..5b891d0`: 7 commits and 2 first-parent
units, measured with `git rev-list --count` and `git rev-list --first-parent --count`
over that exact range.

### Added
- **Automatic bounded session continuity for Codex and Claude Code.** The repository
  plugin bundles now own SessionStart hooks that inject a time-budgeted wake
  without making transcript resume the normal launch path. The bundles are distributed
  from the repository rather than inside the Python sdist or wheel, and their versions
  advance from 0.1.0 to 0.2.0.
- **`synapt recall catchup` runs deferred session maintenance explicitly.** The command
  archives newly available transcripts and rebuilds stale recall state outside the
  synchronous SessionStart hook.

### Changed
- **Session listing and resume hydrate only selected sessions.** Routing metadata is
  read first, then full event payloads are loaded only for the bounded result set.
- **SessionStart defers transcript catchup.** The hook budgets its wake and records the
  outcome. Journal compaction and startup-context generation remain synchronous.

### Fixed
- **Windows build locking.** Windows can read the build-lock stamp while the lock is held.

## [0.19.0] — 2026-08-21

**Counting rule, stated so it can be checked.** Range `ca62880..bc594f3` — from the 0.18.0
promotion merge to the `dev` tip at this release. Over that range: **35 commits total, 17 of
them merge commits and 18 direct. 17 first-parent commits.**

Composition of those 17, by path: **12 touch `src/` when each unit is diffed against its first
parent**; 5 touch only tests, docs, or fixtures; and one of those 5 (`116ffcc`) is a
post-promotion topology-repair merge carrying no content by its own message. "Reviewed units
landed" is therefore 16 if you mean units that changed something, and 17 if you mean merges in
the range.

**The diff method is part of the claim, not pedantry.** A merge commit has no canonical diff.
Against first parent — or `git log --first-parent --diff-merges=first-parent` — the answer is
12. But `git show <merge> --name-only`, which is the command most readers will reach for first,
returns **0**: a combined diff omits everything that was not a conflict resolution, and it does
so silently rather than as an error. A count offered for checking is worth only as much as the
method that reproduces it.

**Why 0.19.0 and not 0.18.1.** 0.18.0 was published to PyPI at `2026-08-07T18:01:21Z`. **Every
one of the 17 units above landed after that instant** — measured against each commit's date,
not inferred. The published 0.18.0 artifact contains none of this work. PyPI forbids
re-uploading a version, so this is a new minor rather than a patch to the existing one.

An earlier draft of this section said "11 of the units above landed after that date." **The
number 11 was real but attached to the wrong category** — it counts units carrying product
changes, which the prose never named, while the sentence was making a claim about *timing*,
where the true answer is all 17. Caught in review before this shipped. Worth recording rather than silently
correcting, because this section's heading invites a reader to check the numbers, and a reader
who finds one that fails has no way to know the other four are exact.

### Fixed
- **Windows test collection, broken on every pull request since 2026-08-07.**
  `tests/recall_store_isolation.py` imported the POSIX-only `pwd` module at module scope, and
  `conftest.py` imports that module at *root* scope. The `ModuleNotFoundError` therefore fired
  before any collection at all — earlier than skip markers exist, so no marker could have
  applied — taking down the entire Windows run rather than one module. The import is now made
  at call time. POSIX behaviour is byte-identical.

  **What is recorded here is the fact, without a theory attached.** Across all 19 pull-request
  runs since 0.18.0 published, three Windows jobs failed in **every single one**, without
  exception. That is measured, not sampled.

  Why it went unaddressed for two weeks is **not something this entry can answer honestly.** An
  earlier draft offered an explanation — that the rest of the matrix stayed green, so a
  partially-red result read as a flaky lane rather than a broken one. Measurement does not
  support it: that pattern holds in 8 of the 19 runs and is false in the other 11, where the
  entire matrix was red for reasons this fix does not address and this entry does not diagnose.
  The explanation was removed rather than softened.

  Deliberately **not** replaced with `Path.home()`: that function reads `$HOME`, which is
  precisely the value a test fixture can move, and the protected boundary must not be
  derivable from the value under test or the guarantee becomes circular. On Windows the
  function still raises, loudly and at the point of use, which is the honest outcome for a
  POSIX-only guarantee.

### Added
- **Incremental builds by default**, with a new `maintain` command and change-detection
  idempotence.

### Changed
- **Summary work moved out of `build` and into `maintain`.** This is a user-visible behaviour
  change, not only an internal one: a `build` that previously produced summaries no longer
  does, and `maintain` is where that work now happens.

### Improved
- **Store and data-root isolation**, including a resolution fix so that membership takes
  precedence over locality.
- **Journal correctness**, and a root-resolution fix across the archive verbs — export and
  import, and also the archive, CLI, and server paths.
- **`code_git` hardening** and a session-start prompt fix.

### Documentation
- Fixture provenance is now declared for the identify test fixtures.

### Note on the changelog gap
**0.15.2, 0.15.3, 0.16.0, 0.17.0, and 0.18.0 shipped without changelog entries** — the gap is
wider than the three versions an earlier draft named. Rather than reconstruct them after the
fact from commit archaeology, it is recorded here honestly.

Their content **is** recoverable from the git history between the corresponding tags. **All five
were checked on the public remote** with `ls-remote` — `v0.15.2`, `v0.15.3`, `v0.16.0`,
`v0.17.0`, and `v0.18.0` — not merely the three an earlier draft vouched for while naming five.
A recovery instruction is worth only as much as the refs it names.

## [0.15.1] — 2026-05-12

### Fixed
- Fixed `recall_search` falsely reporting a missing index after the cached
  embedding-enabled index was loaded. `_get_index()` now correctly updates the
  embedding-cache flag instead of swallowing an `UnboundLocalError` and
  returning `None`.
- Fixed `recall_reload()` for editable installs so it restarts the MCP server
  even when the package version string has not changed.

## [0.11.0] — 2026-04-09

### Added
- **Agent-attributed recall** — `TranscriptChunk` now carries an optional `agent_id` field, auto-populated from `SYNAPT_AGENT_ID` env var. Scoped search via `lookup(agent_id="opus")` returns only that agent's transcripts plus legacy chunks and shared knowledge nodes. Wildcard `agent_id="*"` searches all. (#618)
- **ActionRegistry for plugin-aware channel dispatch** — new `synapt.recall.actions` module replaces the monolithic if/elif dispatcher in `recall_channel()`. OSS registers 13 base actions; premium plugin can register additional actions or override existing ones at import time. Three-tier status model (available/locked/unknown) for action discovery. (#621, #622)
- **Structured channel message types** — messages can carry a `msg_type` field (status, claim, pr, code, message) for filtering on read. (#444)

### Improved
- **Channel dispatch** — `recall_channel()` now routes through the shared ActionRegistry instead of a 150-line hard-coded switch. Net -128 lines in server.py.
- **SQLite schema migration** — `agent_id TEXT` column added transparently to existing databases via `_migrate_chunks_table()`.

## [0.6.1] — 2026-03-13

### Added
- **Secret scrubbing** — API keys, tokens, passwords, JWTs, and connection strings are scrubbed from transcripts at index time with deterministic `[REDACTED:hash]` placeholders. New `synapt recall rescrub` CLI command retroactively cleans existing archives (#65)
- **X/Twitter MCP plugin** — read timelines, search, post, reply, and thread via `synapt.plugins` entry point with prompt injection safeguards (#62)
- **MCP server setup guide** — step-by-step configuration docs for Claude Code, Cursor, and Windsurf (#66)
- **Advanced search/config reference** — documentation for all search parameters, intent types, and configuration options (#66)
- **Windows & cross-platform support** — platform-aware paths, optional MLX/ONNX, graceful fallbacks (#45)

### Improved
- **Content-aware adaptive filtering** — conversations classified as code/personal/mixed; personal content gets relaxed consolidation but `max_knowledge=0` in retrieval (#54)
- **Proactive recall** — `recall_quick` tool for fast, speculative memory checks; improved MCP server instructions (#54)
- **Blog & website** — index page, mobile responsive, cross-links, SEO, OG tags, token efficiency section (#55, #56, #57)

### Benchmarks
- **LOCOMO J-Score: 73.38%** (v0.5.1, Ministral 3B local) — unchanged from v0.6.0; now documented with Full-Context comparison (#56)
- **CodeMemo: 94% J-Score** — new benchmark for code-project memory (50 questions, 1 project)

## [0.6.0] — 2026-03-12

### Added
- **Aggregation-aware entity search** — reduced tier discounts and wider search limits for aggregation queries, plus entity-only knowledge FTS to surface scattered facts about a person across sessions
- **Category-intent alignment** — knowledge nodes whose category matches the query intent get a 1.5x boost (e.g., decision nodes for decision queries)
- **Inferential multi-hop patterns** — aggregation classifier now handles "would X enjoy", "based on the conversation", "who is [Name]" style queries
- **Decision intent** — new intent category for surfacing past decisions, with dedicated patterns and journal decision boost
- **MCP server --dev mode** — auto-reload on source changes via `synapt server --dev` (requires `watchfiles`)
- **Plugin backend registry** — extensible model routing via `synapt.backends` entry points
- **CLI subcommand discovery** — plugins can register CLI commands via `synapt.commands` entry points
- **Tool result enrichment** — enrichment summaries now include tool output content (config values, URLs, command outputs)
- **Entity-anchored FTS** — supplementary FTS search using extracted entities for better multi-hop retrieval
- **Source session IDs** — knowledge nodes display their source session for provenance tracking

### Improved
- **Embedding-based inline dedup** — cosine similarity (≥0.80) fallback after Jaccard for knowledge node deduplication
- **Generic knowledge filter** — tool-tautology patterns and specificity signals remove low-value knowledge nodes
- **Cluster summary hallucination detection** — novel entity check prevents fabricated summaries
- **Word-aware truncation** — prevents mid-word corruption in knowledge node content
- **Garbled knowledge rejection** — filters corrupt nodes with section prefixes or malformed content
- **3B model robustness** — improved handling of smaller model outputs (JSON repair, truncated dict repair, text fallback parser)
- **Intent classification expanded** — broader factual, decision, and aggregation pattern coverage with tightened patterns to reduce false positives
- **Aggregation knowledge_boost** — increased from 1.5 to 2.5 after A/B testing showed higher boost improves retrieval
- **Default decoder model** — switched from Llama-3.2-3B to Ministral-3-3B-Instruct-2512-4bit

### Fixed
- **Intent threading** — intent parameter now threaded through both global and progressive lookup paths
- **Chunk ID collision** — fixed for short session IDs
- **Consolidation** — fixed producing 0 knowledge nodes for non-code data
- **Timestamp truncation** — preserve full timestamp for free-text date formats
- **Knowledge interleaving** — interleave by relevance instead of always prepending

### Benchmarks
- **LOCOMO J-Score: 73.38%** (Ministral 3B local enrichment) — beats Full-Context upper bound (72.90%), Mem0+Graph (68.44%), Mem0 (66.88%), Zep (65.99%)
- Open-domain 80.14% — best of all systems tested
- Multi-hop 70.21% — best of all systems tested

## [0.5.0] — 2026-03-10

### Added
- **Hybrid RRF search** — reciprocal rank fusion combining FTS5, BM25, and semantic embeddings
- **Intent classification** — routes queries to adjust embedding weight, recency decay, and knowledge boost
- **Knowledge graph** — LLM-powered enrichment and consolidation pipeline
- **Cross-encoder reranking** — ms-marco-MiniLM-L-6-v2 for result re-ranking
- **ONNX Runtime inference** — 6.6x faster T5 enrichment on CPU
- **Configurable model selection** — global and project-level config with env var overrides
- **Result deduplication** — Jaccard-based near-duplicate filtering
- **Confidence-weighted knowledge boost** — higher-confidence nodes rank higher
- **Query result cache** — repeated lookups return cached results
- **Temporal date extraction** — parse date ranges from search queries
- **Source expansion** — knowledge nodes include source_turns for chunk-level provenance
- **Proactive MCP instructions** — server instructions tell Claude to search before answering
- **GitHub Pages site** — landing page at synapt.dev
- **CLA bot** — contributor license agreement enforcement

### Fixed
- **Consolidation model routing** — fixed T5 being used instead of decoder-only
- **Markdown response parser** — handle markdown-wrapped JSON from LLMs
- **Non-project path filtering** — filter irrelevant paths from journal file lists
- **Enrichment prompt** — generalized for non-code sessions

## [0.3.0] — 2026-03-08

Initial public release with core recall functionality:
- Session transcript indexing and search
- BM25 + semantic embedding retrieval
- Journal entries and cross-session reminders
- MCP server with 13 tools
- Timeline and session listing
- Plugin architecture
