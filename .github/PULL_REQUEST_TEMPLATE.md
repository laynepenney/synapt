## Summary

<!-- What does this PR do? 1-3 sentences. -->

## Closes

<!-- Link issues this PR resolves. Use "Closes #N" for auto-close on merge. -->
<!-- Example: Closes #123, Closes #456 -->


## Premium Boundary

**Premium boundary**: <!-- e.g. "recall is OSS. This PR adds channel infrastructure, not identity or quality improvement." -->

### Boundary Checklist

- [ ] **Boundary declaration** present above (one-line: what repo, OSS or premium, why)
- [ ] **Identity test**: does this PR answer "who is this agent?" or "what workspace is this?" If yes, it must go in premium (`synapt-dev/premium`; local worktree `synapt-private/`).
- [ ] **Plugin seam**: if this extends OSS for a premium feature, is the extension seam built in OSS first, with the implementation in premium (`synapt-dev/premium`; local worktree `synapt-private/`)?
- [ ] **Boundary-adjacent check**: does this PR do any of the following? If yes to any, it requires premium migration or explicit justification.
  - Parse `workspace.toml`, `agent.toml`, or `repo.toml`
  - Derive agent identity from filesystem layout
  - Add `g2_` prefixed identifiers or other premium-only formats
  - Persist workspace or agent context into shared state

## Test Plan

<!-- How was this tested? What should reviewers verify? -->
