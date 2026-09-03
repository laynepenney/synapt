"""synapt.server — unified MCP server composing recall + discovered plugins.

Recall is built-in (core OSS).  Repair, watch, and any third-party tools
are discovered at startup via ``importlib.metadata`` entry points in the
``synapt.plugins`` group.  See ``synapt.plugins`` for the plugin protocol.

Run with:
    synapt server          # normal mode
    synapt server --dev    # auto-reload on source changes

Add to Claude Code (.mcp.json):
    {"mcpServers": {"synapt": {"type": "stdio", "command": "synapt", "args": ["server"]}}}

Dev mode (--dev) watches src/synapt/ for .py changes and restarts the
server process automatically, so MCP tools always run the latest code.
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import sys


def _handle_sigterm(signum, frame):
    """Convert SIGTERM into a clean exit so atexit handlers run.

    sys.exit(0) doesn't work reliably inside asyncio event loops, so
    we manually run atexit callbacks and os._exit(0) to ensure a zero
    exit code.
    """
    atexit._run_exitfuncs()
    os._exit(0)


# Install before anything else — ensures clean exit code when Claude
# Code sends SIGTERM on session end. Without this, SIGTERM kills the
# process with exit code -15, which Claude reports as "MCP tool failed."
signal.signal(signal.SIGTERM, _handle_sigterm)

logger = logging.getLogger("synapt.server")


def _serve():
    """Start the MCP server (no reload)."""
    from synapt.plugins import register_plugins
    from synapt.recall.server import (
        MCP_INSTRUCTIONS,
        ValidatingFastMCP,
        _resolved_provenance_line,
        register_tools as recall_register,
    )

    # Provenance disclosure: stdout is the MCP
    # protocol channel, so this goes to stderr, same as the --dev logger
    # above. A version match alone does not prove the same code is running
    # -- an editable install's linked worktree can be repointed while the
    # declared version stays put, which is exactly how four live agent
    # servers ran unmerged code with nothing anywhere saying so.
    print(f"[synapt] {_resolved_provenance_line()}", file=sys.stderr)

    mcp = ValidatingFastMCP(
        "synapt",
        instructions=MCP_INSTRUCTIONS,
    )

    # Core: always available
    recall_register(mcp)

    # Plugins: discovered via entry points, gracefully degraded
    registered = register_plugins(mcp)
    for p in registered:
        logger.debug("Loaded plugin: %s %s", p.name, p.version or "(no version)")

    mcp.run()


def _find_watch_paths() -> list[str]:
    """Find source directories to watch for --dev mode.

    Discovers the synapt package's source directory by walking up from
    the package's __file__ location. Also watches any editable-install
    downstream plugin sources.
    """
    import synapt

    paths: list[str] = []

    # Main package source(s) — synapt is a PEP 420 namespace package, so iterate __path__
    # to watch every distribution contributing to the synapt namespace (recall, runner, etc.)
    for ns_dir in synapt.__path__:
        paths.append(os.path.abspath(ns_dir))

    # Also watch downstream plugin packages if installed editable
    try:
        import importlib.metadata
        for ep in importlib.metadata.entry_points(group="synapt.plugins"):
            mod = ep.value.split(":")[0]  # e.g., "your_package.plugins"
            top_pkg = mod.split(".")[0]   # e.g., "your_package"
            try:
                pkg = __import__(top_pkg)
                plugin_dir = os.path.dirname(os.path.abspath(pkg.__file__))
                if plugin_dir not in paths:
                    paths.append(plugin_dir)
            except ImportError:
                pass
    except Exception:
        pass

    return paths


def _dev_serve():
    """Start the MCP server with auto-reload on source changes.

    Spawns the actual server as a child process and proxies stdio.
    When a .py file changes in the watched directories, the child
    is killed and restarted with fresh imports.

    Uses threading to proxy stdin/stdout between parent and child
    without blocking the file watcher.

    Note: MCP protocol state is lost on reload — the client must
    re-initialize after a restart. This is acceptable for dev mode
    where the developer manually re-invokes tools.
    """
    import subprocess
    import threading
    import time

    try:
        from watchfiles import watch
    except ImportError:
        print(
            "watchfiles is required for --dev mode: pip install watchfiles",
            file=sys.stderr,
        )
        sys.exit(1)

    watch_paths = _find_watch_paths()

    # Log to stderr (stdout is for MCP protocol)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [dev] %(message)s",
        stream=sys.stderr,
    )
    dev_log = logging.getLogger("synapt.dev")

    dev_log.info("Dev mode: watching %s", ", ".join(watch_paths))

    child: subprocess.Popen | None = None
    stop_event = threading.Event()
    # Signals proxy threads to stop for the current child cycle
    child_stop = threading.Event()

    def start_child():
        nonlocal child
        child_stop.clear()
        # Spawn ourselves without --dev to run the actual server
        cmd = [sys.executable, "-m", "synapt.server"]
        child = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
        )
        dev_log.info("Server started (pid %d)", child.pid)
        return child

    def proxy_stdin(proc):
        """Forward parent stdin -> child stdin."""
        try:
            while not child_stop.is_set() and proc.poll() is None:
                data = sys.stdin.buffer.read1(8192)
                if not data:
                    break
                proc.stdin.write(data)
                proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass

    def proxy_stdout(proc):
        """Forward child stdout -> parent stdout."""
        try:
            while not child_stop.is_set() and proc.poll() is None:
                data = proc.stdout.read1(8192)
                if not data:
                    break
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
        except (BrokenPipeError, OSError):
            pass

    def monitor_child(proc, crash_event, watch_stop_event):
        """Watch for unexpected child death and break out of watch()."""
        proc.wait()
        if not child_stop.is_set():
            dev_log.warning("Server crashed (pid %d, exit %d)", proc.pid, proc.returncode)
            crash_event.set()
            watch_stop_event.set()  # Unblock watch() iterator

    def kill_child(stdin_thread=None, stdout_thread=None):
        nonlocal child
        child_stop.set()
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            dev_log.info("Server stopped (pid %d)", child.pid)
        # Wait for proxy threads to drain before starting new ones
        if stdin_thread and stdin_thread.is_alive():
            stdin_thread.join(timeout=2)
        if stdout_thread and stdout_thread.is_alive():
            stdout_thread.join(timeout=2)
        child = None

    def handle_sigterm(signum, frame):
        stop_event.set()
        child_stop.set()
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
        os._exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    try:
        while not stop_event.is_set():
            proc = start_child()

            # Per-cycle events
            crash_event = threading.Event()
            # Controls watch() lifetime — set on crash or global stop
            watch_stop = threading.Event()

            # Start proxy threads
            stdin_thread = threading.Thread(
                target=proxy_stdin, args=(proc,), daemon=True,
            )
            stdout_thread = threading.Thread(
                target=proxy_stdout, args=(proc,), daemon=True,
            )
            monitor_thread = threading.Thread(
                target=monitor_child, args=(proc, crash_event, watch_stop),
                daemon=True,
            )
            stdin_thread.start()
            stdout_thread.start()
            monitor_thread.start()

            # Watch for changes — blocks until a .py file changes,
            # the child crashes (via watch_stop), or global stop
            restarted = False
            for changes in watch(
                *watch_paths,
                watch_filter=lambda change, path: path.endswith(".py"),
                stop_event=watch_stop,
                raise_interrupt=False,
            ):
                if crash_event.is_set():
                    restarted = True
                    break
                changed_files = [
                    os.path.basename(p) for _, p in changes
                ]
                dev_log.info(
                    "Detected changes: %s — restarting server",
                    ", ".join(changed_files[:5]),
                )
                restarted = True
                break  # Exit watch loop to restart

            if crash_event.is_set():
                restarted = True

            kill_child(stdin_thread, stdout_thread)

            if not restarted:
                # Watch ended without restart (stop_event or error)
                break

            # Brief pause for filesystem to settle
            time.sleep(0.3)

    finally:
        child_stop.set()
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()


def main():
    """Entry point for the unified synapt MCP server.

    Supports --dev flag for auto-reload during development.
    """
    if "--dev" in sys.argv:
        sys.argv.remove("--dev")
        _dev_serve()
    else:
        _serve()


if __name__ == "__main__":
    main()
