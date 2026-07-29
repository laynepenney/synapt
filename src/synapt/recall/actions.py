"""Plugin-aware action registry for recall_channel dispatch.

Replaces the monolithic if/elif dispatcher in server.py with a pluggable
action registry that a downstream layer may extend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# Premium action names — known to OSS even without premium installed.
# These show as "locked" in status() and return upgrade messages on dispatch.
PREMIUM_ACTION_NAMES = frozenset(
    {
        "directive",
        "claim",
        "unclaim",
        "intent",
        "board",
        "mute",
        "unmute",
        "kick",
        "broadcast",
    }
)


@dataclass
class _ActionEntry:
    handler: Callable[..., str]
    tier: str
    description: str | None = None


class ActionRegistry:
    """Registry of channel actions with tier-aware dispatch."""

    def __init__(self) -> None:
        self._entries: dict[str, _ActionEntry] = {}
        self._known_premium: set[str] = set()

    def register(
        self,
        name: str,
        handler: Callable[..., str],
        tier: str = "oss",
        description: str | None = None,
    ) -> None:
        """Register an action handler. Later registrations override earlier ones."""
        self._entries[name] = _ActionEntry(
            handler=handler, tier=tier, description=description
        )

    def dispatch(self, action_name: str, **kwargs: Any) -> str:
        """Dispatch an action by name. Returns a string result or error message."""
        entry = self._entries.get(action_name)
        if entry is not None:
            return entry.handler(**kwargs)
        if action_name in self._known_premium:
            return (
                f"Action '{action_name}' requires premium plugin. "
                f"See synapt.dev/premium for details."
            )
        return f"Unknown action: '{action_name}'."

    @property
    def actions(self) -> set[str]:
        """Set of registered (dispatchable) action names."""
        return set(self._entries)

    @property
    def known_actions(self) -> set[str]:
        """All known action names: registered + known-but-locked premium stubs."""
        return self.actions | self._known_premium

    def tier(self, name: str) -> str:
        """Return the tier of a registered action, or 'unknown'."""
        entry = self._entries.get(name)
        return entry.tier if entry else "unknown"

    def actions_by_tier(self, tier: str) -> set[str]:
        """Return set of registered action names for a given tier."""
        return {n for n, e in self._entries.items() if e.tier == tier}

    def description(self, name: str) -> str | None:
        """Return the description of a registered action."""
        entry = self._entries.get(name)
        return entry.description if entry else None

    def status(self, name: str) -> str:
        """Return 'available', 'locked', or 'unknown' for an action name."""
        if name in self._entries:
            return "available"
        if name in self._known_premium:
            return "locked"
        return "unknown"


# ---------------------------------------------------------------------------
# OSS handler wrappers — each accepts **kwargs and calls the channel function
# ---------------------------------------------------------------------------


def _handle_join(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_join

    return channel_join(
        channel=kwargs.get("channel", "dev"),
        display_name=kwargs.get("name"),
    )


def _handle_leave(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_leave

    return channel_leave(channel=kwargs.get("channel", "dev"))


def _handle_post(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_post

    message = kwargs.get("message")
    if not message:
        return "Error: message is required for 'post' action."
    attachments = kwargs.get("attachments")
    attachment_list = (
        [p.strip() for p in attachments.split(";")] if attachments else None
    )
    return channel_post(
        channel=kwargs.get("channel", "dev"),
        message=message,
        pin=kwargs.get("pin", False),
        attachment_paths=attachment_list,
        display_name=kwargs.get("name"),
        msg_type=kwargs.get("msg_type") or "message",
    )


def _handle_read(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_read

    return channel_read(
        channel=kwargs.get("channel", "dev"),
        limit=kwargs.get("limit", 20),
        show_pins=kwargs.get("show_pins", True),
        detail=kwargs.get("detail", "medium"),
        msg_type=kwargs.get("msg_type"),
    )


def _handle_read_message(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_read_message

    message = kwargs.get("message")
    if not message:
        return "Error: message_id is required for 'read_message' action."
    return channel_read_message(
        channel=kwargs.get("channel", "dev"),
        message_id=message,
    )


def _handle_who(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_who

    return channel_who()


def _handle_heartbeat(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_heartbeat

    return channel_heartbeat()


def _handle_unread(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_unread_read

    detail = kwargs.get("detail", "medium")
    show_pins = kwargs.get("show_pins", True)
    unread_pins = show_pins if detail in ("high", "max") else False
    return channel_unread_read(
        limit=kwargs.get("limit", 20),
        show_pins=unread_pins,
        detail=detail,
    )


def _handle_pin(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_pin

    message = kwargs.get("message")
    if not message:
        return "Error: message_id is required for 'pin' action."
    return channel_pin(
        channel=kwargs.get("channel", "dev"),
        message_id=message,
    )


def _handle_unpin(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_unpin

    message = kwargs.get("message")
    if not message:
        return "Error: message_id is required for 'unpin' action."
    return channel_unpin(
        channel=kwargs.get("channel", "dev"),
        message_id=message,
    )


def _handle_list(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_list_channels

    channels = channel_list_channels()
    if not channels:
        return "No channels yet."
    return "Channels: " + ", ".join(f"#{c}" for c in channels)


def _handle_search(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_search

    message = kwargs.get("message")
    if not message:
        return "Error: query is required for 'search' action."
    results = channel_search(message, agent_id=kwargs.get("name"))
    if not results:
        return "No matching channel messages."
    lines = ["## Channel search results"]
    for r in results:
        ts = r["timestamp"][:16]
        lines.append(
            f"  [{r['message_id']}] #{r['channel']} {ts}  {r['from']}: {r['body']}"
        )
    return "\n".join(lines)


def _handle_rename(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_rename

    message = kwargs.get("message")
    if not message:
        return "Error: message is required for 'rename' action (the new display name)."
    return channel_rename(new_name=message)


def _handle_directive(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_directive

    message = kwargs.get("message")
    to = kwargs.get("to")
    if not message:
        return "Error: message is required for 'directive' action."
    if not to:
        return "Error: 'to' (target agent) is required for 'directive' action."
    return channel_directive(
        channel=kwargs.get("channel", "dev"),
        message=message,
        to=to,
        display_name=kwargs.get("name"),
    )


def _handle_mute(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_mute

    target = kwargs.get("target")
    if not target:
        return "Error: target agent is required for 'mute' action."
    return channel_mute(target=target, channel=kwargs.get("channel", "dev"))


def _handle_unmute(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_unmute

    target = kwargs.get("target")
    if not target:
        return "Error: target agent is required for 'unmute' action."
    return channel_unmute(target=target, channel=kwargs.get("channel", "dev"))


def _handle_kick(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_kick

    target = kwargs.get("target")
    if not target:
        return "Error: target agent is required for 'kick' action."
    return channel_kick(target=target, channel=kwargs.get("channel", "dev"))


def _handle_broadcast(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_broadcast

    message = kwargs.get("message")
    if not message:
        return "Error: message is required for 'broadcast' action."
    return channel_broadcast(message=message, display_name=kwargs.get("name"))


def _handle_board(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_board

    return channel_board(
        channel=kwargs.get("channel", "dev"),
        message=kwargs.get("message"),
    )


def _handle_claim(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_claim

    message = kwargs.get("message")
    if not message:
        return "Error: message is required for 'claim' action (the message_id to claim)."
    return channel_claim(message_id=message, channel=kwargs.get("channel", "dev"))


def _handle_unclaim_runtime(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_unclaim

    message = kwargs.get("message")
    if not message:
        return "Error: message is required for 'unclaim' action (the message_id to release)."
    return channel_unclaim(message_id=message)


def _handle_intent(**kwargs: Any) -> str:
    from synapt.recall.channel import channel_claim_intent

    message = kwargs.get("message")
    if not message:
        return "Error: message is required for 'intent' action (describe what you're about to create)."
    ok, result = channel_claim_intent(intent=message, channel=kwargs.get("channel", "dev"))
    return result


# ---------------------------------------------------------------------------
# Default registry factory
# ---------------------------------------------------------------------------

_OSS_HANDLERS: dict[str, tuple[Callable[..., str], str | None]] = {
    "join": (_handle_join, "Join a channel"),
    "leave": (_handle_leave, "Leave a channel"),
    "post": (_handle_post, "Post a message to a channel"),
    "read": (_handle_read, "Read recent messages"),
    "read_message": (_handle_read_message, "Read a specific message by ID"),
    "who": (_handle_who, "Show who is online"),
    "heartbeat": (_handle_heartbeat, "Send a heartbeat"),
    "unread": (_handle_unread, "Check for unread messages"),
    "pin": (_handle_pin, "Pin a message"),
    "unpin": (_handle_unpin, "Unpin a message"),
    "list": (_handle_list, "List all channels"),
    "search": (_handle_search, "Search channel history"),
    "rename": (_handle_rename, "Rename your display name"),
}

_RUNTIME_COORDINATION_HANDLERS: dict[str, tuple[Callable[..., str], str | None]] = {
    "directive": (_handle_directive, "Send a targeted directive to an agent"),
    "mute": (_handle_mute, "Mute an agent in a channel"),
    "unmute": (_handle_unmute, "Unmute an agent in a channel"),
    "kick": (_handle_kick, "Remove an agent from a channel"),
    "broadcast": (_handle_broadcast, "Broadcast a message across channels"),
    "board": (_handle_board, "Read or update the channel status board"),
    "claim": (_handle_claim, "Claim a task message"),
    "unclaim": (_handle_unclaim_runtime, "Release a task claim"),
    "intent": (_handle_intent, "Claim an intent before creating an issue or PR"),
}


def get_default_registry() -> ActionRegistry:
    """Create the default OSS action registry with all base actions registered."""
    reg = ActionRegistry()
    for name, (handler, desc) in _OSS_HANDLERS.items():
        reg.register(name, handler, tier="oss", description=desc)
    reg._known_premium = set(PREMIUM_ACTION_NAMES)
    return reg


_DEFAULT_REGISTRY: ActionRegistry | None = None


def get_action_registry() -> ActionRegistry:
    """Return the process-wide channel action registry.

    OSS installs the base registry once. A downstream layer may register
    additional coordination handlers (directive, claim, board, etc.) via
    the plugin system.  Without premium, those actions show as "locked".
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = get_default_registry()
    return _DEFAULT_REGISTRY


def register_coordination_handlers(registry: ActionRegistry | None = None) -> None:
    """Register the runtime coordination handlers on a registry.

    Public extension point for a downstream layer to register additional
    handlers.
    """
    reg = registry or get_action_registry()
    for name, (handler, desc) in _RUNTIME_COORDINATION_HANDLERS.items():
        reg.register(name, handler, tier="premium", description=desc)


def reset_action_registry() -> None:
    """Reset the shared registry for tests."""
    global _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = None
