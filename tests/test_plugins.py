"""Tests for synapt.plugins — plugin discovery and registration."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from synapt.plugins import (
    ENTRY_POINT_GROUP,
    LoadedPlugin,
    discover_plugins,
    register_plugins,
)


def _make_entry_point(
    name: str,
    module: types.ModuleType | None = None,
    load_error: Exception | None = None,
):
    """Create a mock EntryPoint that loads the given module."""
    ep = MagicMock()
    ep.name = name
    ep.group = ENTRY_POINT_GROUP
    if load_error:
        ep.load.side_effect = load_error
    else:
        ep.load.return_value = module
    return ep


def _make_plugin_module(
    has_register: bool = True,
    plugin_name: str | None = None,
    plugin_version: str | None = None,
    register_error: Exception | None = None,
    has_scoring: bool = False,
    scoring_error: Exception | None = None,
):
    """Create a fake plugin module.

    PR4f-B adds optional `register_scoring(scoring_module)` callable alongside
    the existing `register_tools(mcp)` callable. A plugin must provide at least
    one to be discoverable.
    """
    mod = types.ModuleType("fake_plugin")
    if has_register:
        if register_error:
            def register_tools(mcp):
                raise register_error
        else:
            def register_tools(mcp):
                mcp.tool()(lambda: "test")
        mod.register_tools = register_tools
    if has_scoring:
        if scoring_error:
            def register_scoring(scoring):
                raise scoring_error
        else:
            def register_scoring(scoring):
                # Side-effect to verify invocation: record the scoring module on
                # the plugin module so tests can assert it was called with the
                # canonical synapt.recall.scoring module.
                mod._scoring_module_received = scoring
        mod.register_scoring = register_scoring
    if plugin_name is not None:
        mod.PLUGIN_NAME = plugin_name
    if plugin_version is not None:
        mod.PLUGIN_VERSION = plugin_version
    return mod


_PATCH_TARGET = "synapt.plugins.importlib.metadata.entry_points"


class TestDiscoverPlugins:
    def test_discovers_valid_plugin(self):
        mod = _make_plugin_module(plugin_name="test-repair", plugin_version="1.0")
        ep = _make_entry_point("repair", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "test-repair"
        assert plugins[0].version == "1.0"
        assert plugins[0].entry_point_name == "repair"

    def test_falls_back_to_entry_point_name(self):
        mod = _make_plugin_module()  # no PLUGIN_NAME
        ep = _make_entry_point("watch", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert plugins[0].name == "watch"
        assert plugins[0].version == ""

    def test_skips_import_failure(self):
        ep = _make_entry_point("broken", load_error=ImportError("no such module"))
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 0

    def test_skips_missing_register_tools(self):
        """Backward-compat: plugin with no register_tools AND no register_scoring
        is skipped (existing contract preserved). PR4f-B widens the discovery
        contract — either callable suffices."""
        mod = _make_plugin_module(has_register=False, has_scoring=False)
        ep = _make_entry_point("incomplete", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 0

    def test_discovers_plugin_with_only_register_scoring(self):
        """PR4f-B: plugin with only register_scoring (no register_tools) is
        discoverable. Enables scoring-only premium plugins."""
        mod = _make_plugin_module(has_register=False, has_scoring=True)
        ep = _make_entry_point("scoring-only", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].entry_point_name == "scoring-only"

    def test_discovers_plugin_with_both_callables(self):
        """PR4f-B: plugin with BOTH register_tools and register_scoring is
        discoverable. Enables full-stack premium plugins."""
        mod = _make_plugin_module(has_register=True, has_scoring=True)
        ep = _make_entry_point("full-stack", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 1

    def test_discovers_multiple_plugins(self):
        mod1 = _make_plugin_module(plugin_name="repair")
        mod2 = _make_plugin_module(plugin_name="watch")
        eps = [_make_entry_point("repair", mod1), _make_entry_point("watch", mod2)]
        with patch(_PATCH_TARGET, return_value=eps):
            plugins = discover_plugins()
        assert len(plugins) == 2
        names = {p.name for p in plugins}
        assert names == {"repair", "watch"}

    def test_empty_when_no_plugins_installed(self):
        with patch(_PATCH_TARGET, return_value=[]):
            plugins = discover_plugins()
        assert plugins == []


class TestRegisterPlugins:
    def test_registers_tools_on_mcp(self):
        mod = _make_plugin_module()
        mcp = MagicMock()
        plugin = LoadedPlugin("test", "1.0", mod, "test")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 1
        assert mcp.tool.called

    def test_skips_registration_failure(self):
        mod = _make_plugin_module(register_error=RuntimeError("boom"))
        mcp = MagicMock()
        plugin = LoadedPlugin("broken", "1.0", mod, "broken")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 0

    def test_auto_discovers_when_no_plugins_passed(self):
        mod = _make_plugin_module()
        ep = _make_entry_point("auto", mod)
        mcp = MagicMock()
        with patch(_PATCH_TARGET, return_value=[ep]):
            registered = register_plugins(mcp)
        assert len(registered) == 1

    def test_partial_failure_registers_remaining(self):
        good_mod = _make_plugin_module(plugin_name="good")
        bad_mod = _make_plugin_module(plugin_name="bad", register_error=RuntimeError("fail"))
        mcp = MagicMock()
        plugins = [
            LoadedPlugin("good", "", good_mod, "good"),
            LoadedPlugin("bad", "", bad_mod, "bad"),
        ]
        registered = register_plugins(mcp, plugins)
        assert len(registered) == 1
        assert registered[0].name == "good"


class TestLoadedPlugin:
    def test_repr(self):
        p = LoadedPlugin("repair", "1.0", None, "repair")
        assert repr(p) == "LoadedPlugin('repair', '1.0')"

    def test_slots(self):
        p = LoadedPlugin("repair", "1.0", None, "repair")
        assert not hasattr(p, "__dict__")


# === PR4f-B scoring-registration tests ===


class TestRegisterScoring:
    """PR4f-B: register_plugins also invokes register_scoring with the canonical
    synapt.recall.scoring module on plugins that provide it."""

    def test_invokes_register_scoring_with_scoring_module(self):
        """register_scoring receives synapt.recall.scoring module."""
        mod = _make_plugin_module(has_register=False, has_scoring=True)
        mcp = MagicMock()
        plugin = LoadedPlugin("scoring-only", "1.0", mod, "scoring-only")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 1

        # Verify the scoring module passed to register_scoring is synapt.recall.scoring
        from synapt.recall import scoring as canonical_scoring
        assert mod._scoring_module_received is canonical_scoring

    def test_scoring_only_plugin_does_not_call_mcp(self):
        """Plugin without register_tools must not have MCP.tool() invoked."""
        mod = _make_plugin_module(has_register=False, has_scoring=True)
        mcp = MagicMock()
        plugin = LoadedPlugin("scoring-only", "1.0", mod, "scoring-only")
        register_plugins(mcp, [plugin])
        assert not mcp.tool.called

    def test_tools_only_plugin_does_not_attempt_scoring(self):
        """Backward-compat: plugin without register_scoring still registers tools."""
        mod = _make_plugin_module(has_register=True, has_scoring=False)
        mcp = MagicMock()
        plugin = LoadedPlugin("tools-only", "1.0", mod, "tools-only")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 1
        assert mcp.tool.called

    def test_both_callables_invoked_independently(self):
        """Plugin with BOTH register_tools and register_scoring sees both invoked."""
        mod = _make_plugin_module(has_register=True, has_scoring=True)
        mcp = MagicMock()
        plugin = LoadedPlugin("full-stack", "1.0", mod, "full-stack")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 1
        assert mcp.tool.called
        assert hasattr(mod, "_scoring_module_received")

    def test_register_scoring_failure_does_not_block_register_tools(self):
        """register_tools succeeds even if register_scoring raises."""
        mod = _make_plugin_module(
            has_register=True,
            has_scoring=True,
            scoring_error=RuntimeError("scoring boom"),
        )
        mcp = MagicMock()
        plugin = LoadedPlugin("partial-fail", "1.0", mod, "partial-fail")
        registered = register_plugins(mcp, [plugin])
        # tools succeeded → plugin still in registered list
        assert len(registered) == 1
        assert mcp.tool.called

    def test_register_tools_failure_does_not_block_register_scoring(self):
        """register_scoring succeeds even if register_tools raises."""
        mod = _make_plugin_module(
            has_register=True,
            has_scoring=True,
            register_error=RuntimeError("tools boom"),
        )
        mcp = MagicMock()
        plugin = LoadedPlugin("partial-fail", "1.0", mod, "partial-fail")
        registered = register_plugins(mcp, [plugin])
        # scoring succeeded → plugin still in registered list
        assert len(registered) == 1
        assert hasattr(mod, "_scoring_module_received")

    def test_both_failures_skip_plugin(self):
        """Plugin is skipped from registered list when BOTH callables fail."""
        mod = _make_plugin_module(
            has_register=True,
            has_scoring=True,
            register_error=RuntimeError("tools boom"),
            scoring_error=RuntimeError("scoring boom"),
        )
        mcp = MagicMock()
        plugin = LoadedPlugin("total-fail", "1.0", mod, "total-fail")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 0


# === PR4f-B DI-threading-seam reject/accept E2E ===


class TestPluginScoringDIThreadingSeam:
    """Paired reject/accept E2E threading tests for the plugin-time scoring
    activation seam (per feedback_xfail_removal_doc_sweep.md DI-threading-seam
    rule).
    """

    def test_accept_scoring_plugin_strategy_activates(self):
        """Accept path: a plugin's register_scoring CAN register + activate
        a strategy that subsequently shows up in get_active_strategy()."""
        from synapt.recall import scoring as canonical_scoring
        canonical_scoring.reset_registry()
        try:
            class _FakeStrategy:
                name = "plugin-fake"
                window = 16

                def score(self, inputs):
                    return [
                        canonical_scoring.ScoredChunk(
                            input=i, score=1.0, strategy_name=self.name
                        )
                        for i in inputs
                    ]

            def register_scoring(scoring):
                scoring.register_scoring_strategy("plugin-fake", _FakeStrategy())
                scoring.activate_scoring_strategy("plugin-fake")

            mod = types.ModuleType("fake_premium_plugin")
            mod.register_scoring = register_scoring
            mcp = MagicMock()
            plugin = LoadedPlugin("fake-premium", "1.0", mod, "fake-premium")

            registered = register_plugins(mcp, [plugin])
            assert len(registered) == 1

            active = canonical_scoring.get_active_strategy()
            assert active.name == "plugin-fake"
        finally:
            canonical_scoring.reset_registry()

    def test_reject_scoring_plugin_failure_does_not_corrupt_registry(self):
        """Reject path: plugin scoring registration failure leaves registry
        usable for subsequent valid registrations.

        Atlas review-1 (recall#824) Blocker 2: previous version of this test
        used RecencyScoring as the recovery strategy. RecencyScoring.name ==
        "recency" matches the default fallback strategy returned by
        get_active_strategy() when no strategy is activated. Asserting
        `get_active_strategy().name == "recency"` therefore passed whether
        recovery activation succeeded OR fallback intermediated — false-positive.

        Fix: use a custom recovery strategy with a unique name AND assert
        get_active_strategy() returns the specific instance registered (identity
        check), so the test fails closed if activation did not actually occur.
        """
        from synapt.recall import scoring as canonical_scoring
        canonical_scoring.reset_registry()
        try:
            def register_scoring(scoring):
                raise RuntimeError("plugin scoring registration failed")

            mod = types.ModuleType("broken_premium_plugin")
            mod.register_scoring = register_scoring
            mcp = MagicMock()
            plugin = LoadedPlugin("broken-premium", "1.0", mod, "broken-premium")

            registered = register_plugins(mcp, [plugin])
            assert len(registered) == 0

            # Recovery uses a custom strategy with a unique name so the
            # assertion below distinguishes "recovery activation succeeded"
            # from "fallback to default RecencyScoring intermediated".
            class _RecoveryStrategy:
                name = "recovery-distinct-strategy"
                window = 8

                def score(self, inputs):
                    return [
                        canonical_scoring.ScoredChunk(
                            input=i, score=0.5, strategy_name=self.name
                        )
                        for i in inputs
                    ]

            recovery_instance = _RecoveryStrategy()
            canonical_scoring.register_scoring_strategy(
                "recovery", recovery_instance
            )
            canonical_scoring.activate_scoring_strategy("recovery")
            active = canonical_scoring.get_active_strategy()
            # Identity check + distinct-name check both required so the test
            # fails closed if the activation path was bypassed.
            assert active is recovery_instance
            assert active.name == "recovery-distinct-strategy"
            assert active.name != "recency"  # not the default fallback
        finally:
            canonical_scoring.reset_registry()
