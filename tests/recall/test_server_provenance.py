"""Tests for the startup provenance disclosure line.

A declared version string does not pin which code ran: an editable install
can keep reporting the same version while the linked worktree underneath it
is silently repointed -- a real recurring production incident (agent MCP
servers running unmerged code from an unrelated worktree with nothing
anywhere saying so), not a hypothetical. These tests hold the resolved-path and
editable-detection logic to a real fixture rather than trusting that
`importlib.metadata` behaves the way the docstring assumes.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from synapt.recall import server


class _FakeDistribution:
    def __init__(self, direct_url_text: str | None):
        self._text = direct_url_text

    def read_text(self, filename: str):
        assert filename == "direct_url.json"
        return self._text


def test_resolved_install_kind_editable_when_dir_info_editable_true():
    dist = _FakeDistribution(json.dumps({"url": "file:///x", "dir_info": {"editable": True}}))
    with patch("importlib.metadata.distribution", return_value=dist):
        assert server._resolved_install_kind() == "editable"


def test_resolved_install_kind_non_editable_when_dir_info_editable_false():
    dist = _FakeDistribution(json.dumps({"url": "file:///x", "dir_info": {"editable": False}}))
    with patch("importlib.metadata.distribution", return_value=dist):
        assert server._resolved_install_kind() == "non-editable"


def test_resolved_install_kind_non_editable_when_dir_info_absent():
    # A normal (non-editable) sdist/wheel install has a direct_url.json with
    # no dir_info.editable key at all -- the absence, not a False, is what a
    # real non-editable install actually produces.
    dist = _FakeDistribution(json.dumps({"url": "file:///x"}))
    with patch("importlib.metadata.distribution", return_value=dist):
        assert server._resolved_install_kind() == "non-editable"


def test_resolved_install_kind_non_editable_when_direct_url_missing():
    # PyPI-installed wheels frequently carry no direct_url.json at all.
    dist = _FakeDistribution(None)
    with patch("importlib.metadata.distribution", return_value=dist):
        assert server._resolved_install_kind() == "non-editable"


def test_resolved_install_kind_unknown_when_distribution_lookup_raises():
    with patch("importlib.metadata.distribution", side_effect=Exception("not found")):
        assert server._resolved_install_kind() == "unknown"


def test_resolved_provenance_line_carries_version_path_and_kind():
    with patch.object(server, "_STARTUP_VERSION", "9.9.9"), \
         patch.object(server, "_resolved_install_kind", return_value="editable"):
        line = server._resolved_provenance_line()
    assert "v9.9.9" in line
    assert "editable install" in line
    # The path is derived from the real imported module, not hardcoded --
    # asserting it names the actual synapt.recall package directory proves
    # the line tracks the real import rather than a fixed string.
    import synapt.recall as real_pkg
    from pathlib import Path
    assert str(Path(real_pkg.__file__).resolve().parent) in line


def test_resolved_provenance_line_degrades_to_unknown_location_on_file_error():
    with patch.object(server, "_synapt_pkg") as fake_pkg, \
         patch.object(server, "_resolved_install_kind", return_value="unknown"):
        # __file__ access raising is the shape a frozen/zipapp import can
        # take; the line must still return, never raise.
        type(fake_pkg).__file__ = property(lambda self: (_ for _ in ()).throw(AttributeError()))
        line = server._resolved_provenance_line()
    assert "unknown" in line


def test_resolved_provenance_line_names_the_root_source():
    with patch.object(server, "_STARTUP_VERSION", "9.9.9"), \
         patch.object(server, "_resolved_install_kind", return_value="editable"), \
         patch.object(server, "describe_root_source", return_value="marker:/some/shared/root"):
        line = server._resolved_provenance_line()
    assert "marker:/some/shared/root" in line


def test_with_provenance_appends_without_mutating_original_text():
    with patch.object(server, "_resolved_provenance_line", return_value="synapt vX — running from /nowhere (unknown install)"):
        result = server._with_provenance("original body text")
    assert result.startswith("original body text")
    assert "Provenance: synapt vX — running from /nowhere (unknown install)" in result
