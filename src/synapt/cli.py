"""synapt -- unified CLI for recall and MCP server.

Usage:
    synapt init      # one-command project setup
    synapt recall search "query"
    synapt recall build --incremental
    synapt recall stats
    synapt recall setup

    synapt server   # starts unified MCP server

Plugins can register additional subcommands via the ``synapt.commands``
entry-point group (e.g. ``repair``, ``watch``).
"""

from __future__ import annotations

import importlib.metadata
import sys


def _discover_commands() -> dict[str, callable]:
    """Discover plugin subcommands via ``synapt.commands`` entry points.

    Each entry point should reference a callable ``main()`` function.
    """
    commands: dict[str, callable] = {}
    for ep in importlib.metadata.entry_points(group="synapt.commands"):
        commands[ep.name] = ep
    return commands


def _dispatch_recall(*args: str) -> None:
    from synapt.recall.cli import main as recall_main

    sys.argv = ["synapt recall"] + list(args)
    recall_main()


def main():
    extra_commands = _discover_commands()

    if len(sys.argv) < 2:
        _print_help(extra_commands)
        sys.exit(1)

    subcmd = sys.argv[1]

    if subcmd in ("-h", "--help", "help"):
        _print_help(extra_commands)
        return

    if subcmd == "--version":
        from synapt.recall import __version__

        print(f"synapt {__version__}")
        return

    if subcmd == "recall":
        _dispatch_recall(*sys.argv[2:])
    elif subcmd == "init":
        _dispatch_recall("setup", *sys.argv[2:])
    elif subcmd == "server":
        # Remove the subcommand from argv so the sub-CLI sees correct args.
        sys.argv = [f"synapt {subcmd}"] + sys.argv[2:]
        from synapt.server import main as server_main

        server_main()
    elif subcmd == "dashboard":
        sys.argv = [f"synapt {subcmd}"] + sys.argv[2:]
        try:
            from synapt.dashboard.app import main as dashboard_main
        except ImportError:
            print(
                "Dashboard requires extra dependencies.\n"
                "Install with: pip install synapt[dashboard]",
                file=sys.stderr,
            )
            sys.exit(1)
        dashboard_main()
    elif subcmd in extra_commands:
        # Remove the subcommand from argv so the plugin sees correct args.
        sys.argv = [f"synapt {subcmd}"] + sys.argv[2:]
        ep = extra_commands[subcmd]
        cli_main = ep.load()
        cli_main()
    else:
        print(f"Unknown command: {subcmd}", file=sys.stderr)
        _print_help(extra_commands)
        sys.exit(1)


def _print_help(extra_commands: dict | None = None):
    lines = [
        "synapt -- persistent conversational memory for AI coding assistants",
        "",
        "Commands:",
        "  init      One-command project setup",
        "  recall    Search and manage past session transcripts",
        "  dashboard Launch mission control UI",
        "  server    Start the unified MCP server (--dev for auto-reload)",
    ]
    if extra_commands:
        for name in sorted(extra_commands):
            lines.append(f"  {name:<9} (from synapt-private)")
    lines.append("")
    lines.append("Run 'synapt <command> --help' for details on each command.")
    lines.append("")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
