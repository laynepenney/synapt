#!/usr/bin/env bash
# Create an isolated eval environment: git worktree + dedicated venv.
#
# Usage:
#   ./scripts/eval-worktree.sh              # worktree from HEAD
#   ./scripts/eval-worktree.sh abc1234      # worktree from specific commit
#   ./scripts/eval-worktree.sh v0.6.2       # worktree from tag
#
# Output:
#   /tmp/synapt-eval-<ref>/          — isolated worktree
#   /tmp/synapt-eval-<ref>/.venv/    — dedicated venv with synapt installed
#
# To run evals:
#   source /tmp/synapt-eval-<ref>/.venv/bin/activate
#   cd /tmp/synapt-eval-<ref>
#   python -m evaluation.codememo.eval --recalldb --model gpt-4o-mini
#
# Cleanup:
#   ./scripts/eval-worktree.sh --cleanup <ref>

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Cleanup mode ---
if [[ "${1:-}" == "--cleanup" ]]; then
    REF="${2:?Usage: $0 --cleanup <ref>}"
    WORKTREE="/tmp/synapt-eval-${REF}"
    if [[ -d "$WORKTREE" ]]; then
        echo "Removing worktree: $WORKTREE"
        git -C "$REPO_ROOT" worktree remove "$WORKTREE" --force 2>/dev/null || rm -rf "$WORKTREE"
        echo "Done."
    else
        echo "No worktree at $WORKTREE"
    fi
    exit 0
fi

# --- Create mode ---
TARGET="${1:-HEAD}"
REF=$(git -C "$REPO_ROOT" rev-parse --short "$TARGET" 2>/dev/null || echo "$TARGET")
WORKTREE="/tmp/synapt-eval-${REF}"

if [[ -d "$WORKTREE" ]]; then
    echo "Worktree already exists: $WORKTREE"
    echo "  To reuse: source $WORKTREE/.venv/bin/activate"
    echo "  To remove: $0 --cleanup $REF"
    exit 1
fi

echo "=== Creating eval worktree ==="
echo "  Repo:     $REPO_ROOT"
echo "  Target:   $TARGET"
echo "  Ref:      $REF"
echo "  Worktree: $WORKTREE"
echo

# Create detached worktree at the target commit
git -C "$REPO_ROOT" worktree add --detach "$WORKTREE" "$TARGET"

# Create dedicated venv (isolated from dev environment)
echo
echo "=== Creating venv ==="
python3 -m venv "$WORKTREE/.venv"
source "$WORKTREE/.venv/bin/activate"

# Install synapt (non-editable — frozen at this commit)
echo
echo "=== Installing synapt ==="
pip install --quiet "$WORKTREE"

# Verify
echo
echo "=== Verification ==="
INSTALLED_VERSION=$(python -c "import synapt.recall; print(synapt.recall.__version__)")
CODE_REF=$(git -C "$WORKTREE" rev-parse --short HEAD)
CODE_DIRTY=$(git -C "$WORKTREE" diff --quiet && echo "clean" || echo "DIRTY")

echo "  synapt version: $INSTALLED_VERSION"
echo "  code ref:       $CODE_REF ($CODE_DIRTY)"
echo "  python:         $(which python)"
echo "  venv:           $WORKTREE/.venv"
echo

echo "=== Ready ==="
echo "  source $WORKTREE/.venv/bin/activate"
echo "  cd $WORKTREE"
echo "  python -m evaluation.codememo.eval --recalldb --model gpt-4o-mini"
echo
echo "  Cleanup: $0 --cleanup $REF"
