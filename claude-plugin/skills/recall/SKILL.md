---
name: recall
description: Persistent memory across Claude Code sessions. Search before you act.
---

# synapt recall

Persistent memory for Claude Code. The plugin loads bounded current context at
session start. Search past sessions for deeper context, save durable knowledge,
and maintain context across conversations.

## When to use (do this automatically, without being asked)

- **Before making a design decision**: `recall_search` for prior discussion
- **When debugging an error**: `recall_search` for past fixes
- **When user references past work**: `recall_search` immediately
- **Starting a session**: use the injected current context first, then `recall_journal` when more detail is needed
- **When unsure if something was discussed**: `recall_quick` (fast, cheap)
- **When you need file history**: `recall_files` for who changed what and why

## Tool selection

| Tool | Use when | Cost |
|------|----------|------|
| `recall_quick` | Speculative check, unsure if relevant history exists | ~500 tokens |
| `recall_search` | Need full transcript chunks with detail | ~2000 tokens |
| `recall_files` | Need file change history and rationale | ~1500 tokens |
| `recall_journal` | Reading/writing session notes | ~500 tokens |
| `recall_save` | Persisting a durable fact, decision, or convention | ~200 tokens |
| `recall_remind` | Setting cross-session reminders | ~200 tokens |

## Key principle

When in doubt, search. A `recall_quick` check costs ~500 tokens and takes <100ms. Missing relevant past context costs far more in wasted work and repeated mistakes.

## Do NOT search for

- General programming questions or syntax help
- API docs or library documentation
- Anything not specific to this project's history
