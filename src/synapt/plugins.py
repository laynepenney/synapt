"""synapt.plugins — entry-point plugin discovery and loading.

Plugins register via ``[project.entry-points."synapt.plugins"]`` in their
pyproject.toml.  Each entry point module must provide at least ONE of:

- ``register_tools(mcp: FastMCP) -> None`` — register MCP tools on the server
- ``register_scoring(scoring_module) -> None`` — register/activate chunk
  scoring strategies via the ``synapt.recall.scoring`` registry (PR4f-B
  plugin-time activation per config#339 Q3 ratification)

A plugin may provide both. Plugins with neither callable are logged and skipped.

Two-phase design: ``discover_plugins()`` loads modules and validates them,
``register_plugins()`` invokes the appropriate registration callables.  The
separation creates a clean seam for future license checks.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Any

logger = logging.getLogger("synapt.plugins")

ENTRY_POINT_GROUP = "synapt.plugins"


class LoadedPlugin:
    """Metadata about a successfully loaded plugin."""

    __slots__ = ("name", "version", "module", "entry_point_name")

    def __init__(self, name: str, version: str, module: Any, entry_point_name: str):
        self.name = name
        self.version = version
        self.module = module
        self.entry_point_name = entry_point_name

    def __repr__(self) -> str:
        return f"LoadedPlugin({self.name!r}, {self.version!r})"


def discover_plugins() -> list[LoadedPlugin]:
    """Discover and load all plugins registered under the 'synapt.plugins' group.

    Returns a list of LoadedPlugin for each successfully loaded plugin.
    Plugins that fail to import or lack register_tools are logged and skipped.
    """
    plugins: list[LoadedPlugin] = []
    eps = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)

    for ep in eps:
        try:
            module = ep.load()
        except Exception:
            logger.warning("Plugin %r failed to import", ep.name, exc_info=True)
            continue

        has_tools = callable(getattr(module, "register_tools", None))
        has_scoring = callable(getattr(module, "register_scoring", None))
        if not has_tools and not has_scoring:
            logger.warning(
                "Plugin %r has neither register_tools nor register_scoring "
                "callable, skipping",
                ep.name,
            )
            continue

        name = getattr(module, "PLUGIN_NAME", ep.name)
        version = getattr(module, "PLUGIN_VERSION", "")
        plugins.append(LoadedPlugin(name, version, module, ep.name))
        logger.debug("Discovered plugin: %s %s", name, version)

    return plugins


def register_plugins(
    mcp: Any, plugins: list[LoadedPlugin] | None = None
) -> list[LoadedPlugin]:
    """Discover plugins (if not provided) and register their tools + scoring.

    For each plugin, invokes whichever of `register_tools(mcp)` and
    `register_scoring(scoring_module)` the plugin provides. Failures in one
    callable do not block the other; failures across both result in the plugin
    being skipped from the registered list.

    Returns the list of plugins where at least one registration callable
    succeeded.
    """
    if plugins is None:
        plugins = discover_plugins()

    registered: list[LoadedPlugin] = []
    for plugin in plugins:
        succeeded_any = False

        if callable(getattr(plugin.module, "register_tools", None)):
            try:
                plugin.module.register_tools(mcp)
                succeeded_any = True
                logger.debug("Registered tools for plugin: %s", plugin.name)
            except Exception:
                logger.warning(
                    "Plugin %r register_tools() failed",
                    plugin.name,
                    exc_info=True,
                )

        if callable(getattr(plugin.module, "register_scoring", None)):
            try:
                from synapt.recall import scoring

                plugin.module.register_scoring(scoring)
                succeeded_any = True
                logger.debug("Registered scoring for plugin: %s", plugin.name)
            except Exception:
                logger.warning(
                    "Plugin %r register_scoring() failed",
                    plugin.name,
                    exc_info=True,
                )

        if succeeded_any:
            registered.append(plugin)

    return registered
