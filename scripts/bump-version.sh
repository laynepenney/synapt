#!/bin/bash
# Bump synapt version across all files, commit, and tag.
#
# Usage:
#   ./scripts/bump-version.sh 0.7.0
#   ./scripts/bump-version.sh patch     # 0.6.2 → 0.6.3
#   ./scripts/bump-version.sh minor     # 0.6.2 → 0.7.0
#   ./scripts/bump-version.sh major     # 0.6.2 → 1.0.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYPROJECT="$REPO_ROOT/pyproject.toml"
INIT_PY="$REPO_ROOT/src/synapt/recall/__init__.py"

# --- Read current version ---
CURRENT=$(grep '^version = ' "$PYPROJECT" | head -1 | sed 's/version = "\(.*\)"/\1/')
if [ -z "$CURRENT" ]; then
    echo "Error: could not read version from $PYPROJECT" >&2
    exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

# --- Compute new version ---
ARG="${1:-}"
if [ -z "$ARG" ]; then
    echo "Current version: $CURRENT"
    echo ""
    echo "Usage: $0 <version|patch|minor|major>"
    echo "  $0 patch   →  $MAJOR.$MINOR.$((PATCH + 1))"
    echo "  $0 minor   →  $MAJOR.$((MINOR + 1)).0"
    echo "  $0 major   →  $((MAJOR + 1)).0.0"
    echo "  $0 0.7.0   →  0.7.0"
    exit 0
fi

case "$ARG" in
    patch) NEW="$MAJOR.$MINOR.$((PATCH + 1))" ;;
    minor) NEW="$MAJOR.$((MINOR + 1)).0" ;;
    major) NEW="$((MAJOR + 1)).0.0" ;;
    [0-9]*.[0-9]*.[0-9]*) NEW="$ARG" ;;
    *)
        echo "Error: invalid version '$ARG'. Use patch, minor, major, or X.Y.Z" >&2
        exit 1
        ;;
esac

if [ "$NEW" = "$CURRENT" ]; then
    echo "Already at version $CURRENT"
    exit 0
fi

echo "Bumping: $CURRENT → $NEW"
echo ""

# --- Update files ---
# pyproject.toml
sed -i '' "s/^version = \"$CURRENT\"/version = \"$NEW\"/" "$PYPROJECT"
echo "  ✓ pyproject.toml"

# src/synapt/recall/__init__.py
sed -i '' "s/__version__ = \"$CURRENT\"/__version__ = \"$NEW\"/" "$INIT_PY"
echo "  ✓ src/synapt/recall/__init__.py"

# --- Verify ---
VERIFY_PYPROJECT=$(grep '^version = ' "$PYPROJECT" | head -1 | sed 's/version = "\(.*\)"/\1/')
VERIFY_INIT=$(grep '__version__' "$INIT_PY" | sed 's/__version__ = "\(.*\)"/\1/' | tr -d ' ')

if [ "$VERIFY_PYPROJECT" != "$NEW" ] || [ "$VERIFY_INIT" != "$NEW" ]; then
    echo ""
    echo "Error: verification failed!" >&2
    echo "  pyproject.toml: $VERIFY_PYPROJECT" >&2
    echo "  __init__.py:    $VERIFY_INIT" >&2
    exit 1
fi

echo ""
echo "Version bumped to $NEW"
echo ""
echo "Next steps:"
echo "  gr add pyproject.toml src/synapt/__init__.py"
echo "  gr commit -m \"chore: bump version to $NEW\""
echo "  git tag v$NEW"
echo "  gr push -u && git push --tags"
