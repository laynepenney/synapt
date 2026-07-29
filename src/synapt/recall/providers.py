"""Provider protocols for the OSS/premium seam.

OSS defines the protocols; implementations may be supplied by a
downstream layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Org seam
# ---------------------------------------------------------------------------


@dataclass
class OrgInfo:
    """Resolved org identity."""

    org_id: str
    source: str  # where the value was resolved from
    name: str | None = None
    metadata: dict | None = None


@runtime_checkable
class OrgProvider(Protocol):
    """Protocol for org identity resolution."""

    def resolve_org(self) -> OrgInfo | None:
        """Resolve the current org. Returns None if no org context."""
        ...

    def org_metadata(self, org_id: str) -> dict:
        """Return org-level settings/metadata."""
        ...

    def validate_org(self, org_id: str) -> bool:
        """Check whether the given org_id is valid for this session."""
        ...


# ---------------------------------------------------------------------------
# Entitlements seam
# ---------------------------------------------------------------------------


@runtime_checkable
class EntitlementProvider(Protocol):
    """Protocol for capability/entitlement queries."""

    def has_feature(self, feature: str) -> bool:
        """Return whether a named capability is enabled."""
        ...

    def tier(self) -> str:
        """Return the current entitlement tier (free, team, enterprise)."""
        ...

    def is_degraded(self) -> bool:
        """Return True if entitlements have expired or are degraded."""
        ...

    def summary(self) -> dict[str, bool]:
        """Return a dict of feature name -> enabled status."""
        ...


class FreeEntitlementProvider:
    """Default provider when no other implementation is installed. Everything is free tier."""

    def has_feature(self, feature: str) -> bool:
        return False

    def tier(self) -> str:
        return "free"

    def is_degraded(self) -> bool:
        return False

    def summary(self) -> dict[str, bool]:
        return {}
