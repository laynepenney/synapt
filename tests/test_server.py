"""Tests for synapt.server module."""

import os
import sys
import unittest
from unittest.mock import patch


class TestServerDevMode(unittest.TestCase):
    """Test --dev flag handling and helper functions."""

    def test_main_without_dev(self):
        """main() without --dev calls _serve()."""
        with patch("synapt.server._serve") as mock_serve:
            with patch.object(sys, "argv", ["synapt server"]):
                from synapt.server import main
                main()
                mock_serve.assert_called_once()

    def test_main_with_dev(self):
        """main() with --dev calls _dev_serve()."""
        with patch("synapt.server._dev_serve") as mock_dev:
            with patch.object(sys, "argv", ["synapt server", "--dev"]):
                from synapt.server import main
                main()
                mock_dev.assert_called_once()

    def test_main_with_help_flag_prints_help_and_does_not_serve(self):
        """`synapt server --help` must print usage, not start the server.

        The top-level help text (`synapt --help`) promises "Run 'synapt
        <command> --help' for details on each command" -- until now
        `server` was the one subcommand that broke that promise by
        silently starting the MCP server on stdio instead of printing
        anything."""
        import io

        out = io.StringIO()
        with patch("synapt.server._serve") as mock_serve, \
             patch("synapt.server._dev_serve") as mock_dev, \
             patch.object(sys, "argv", ["synapt server", "--help"]), \
             patch.object(sys, "stdout", out):
            from synapt.server import main
            main()
        mock_serve.assert_not_called()
        mock_dev.assert_not_called()
        self.assertIn("server", out.getvalue().lower())

    def test_main_with_h_flag_prints_help_and_does_not_serve(self):
        """-h is the short form and must behave identically to --help."""
        import io

        out = io.StringIO()
        with patch("synapt.server._serve") as mock_serve, \
             patch("synapt.server._dev_serve") as mock_dev, \
             patch.object(sys, "argv", ["synapt server", "-h"]), \
             patch.object(sys, "stdout", out):
            from synapt.server import main
            main()
        mock_serve.assert_not_called()
        mock_dev.assert_not_called()
        self.assertIn("server", out.getvalue().lower())

    def test_dev_flag_removed_from_argv(self):
        """--dev flag should be removed from sys.argv."""
        captured_argv = []

        def capture_dev_serve():
            captured_argv.extend(sys.argv)

        with patch("synapt.server._dev_serve", side_effect=capture_dev_serve):
            with patch.object(sys, "argv", ["synapt server", "--dev"]):
                from synapt.server import main
                main()
                self.assertNotIn("--dev", captured_argv)

    def test_serve_prints_provenance_to_stderr_before_running(self):
        """`synapt server` must disclose which code it is
        running, not just its declared version -- stdout is the MCP
        protocol channel, so this has to land on stderr,
        matching the --dev logger's own comment two lines above this."""
        import io

        order: list[str] = []
        fake_mcp = unittest.mock.Mock()
        fake_mcp.run.side_effect = lambda: order.append("run")

        err = io.StringIO()
        with patch("synapt.recall.server.ValidatingFastMCP", return_value=fake_mcp), \
             patch("synapt.plugins.register_plugins", return_value=[]), \
             patch("synapt.recall.server.register_tools"), \
             patch(
                 "synapt.recall.server._resolved_provenance_line",
                 return_value="synapt vTEST — running from /fixture (editable install)",
             ), \
             patch.object(sys, "stderr", err):
            from synapt.server import _serve
            _serve()

        printed = err.getvalue()
        self.assertIn(
            "synapt vTEST — running from /fixture (editable install)", printed,
        )
        fake_mcp.run.assert_called_once()

    def test_find_watch_paths(self):
        """_find_watch_paths returns at least the synapt package directory."""
        from synapt.server import _find_watch_paths
        paths = _find_watch_paths()
        self.assertTrue(len(paths) >= 1)
        # First path should be the synapt package dir
        self.assertTrue(
            paths[0].endswith("synapt"),
            f"Expected synapt dir, got {paths[0]}",
        )
        # Should be an actual directory
        self.assertTrue(os.path.isdir(paths[0]))


class TestNovelEntitiesInServer(unittest.TestCase):
    """Verify that _novel_entities from clustering is importable."""

    def test_import(self):
        from synapt.recall.clustering import _novel_entities
        self.assertIsNotNone(_novel_entities)


if __name__ == "__main__":
    unittest.main()
