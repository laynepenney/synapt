"""Metadata contract for the installable MCP server surface."""

from __future__ import annotations

import tomllib
from pathlib import Path


_PROJECT = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
_BASE = _PROJECT["project"]["dependencies"]
_EXTRAS = _PROJECT["project"]["optional-dependencies"]


def _names(requirements: list[str]) -> set[str]:
    return {requirement.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].strip() for requirement in requirements}


def test_server_closure_stays_installable_without_model_or_provider_stacks() -> None:
    assert _BASE == ["mcp[cli]==1.28.1"]
    assert _EXTRAS["server"] == ["mcp[cli]==1.28.1"]
    assert set(_EXTRAS["full-runtime"]) == {
        "sentence-transformers>=2.0",
        "mlx-lm>=0.10; sys_platform == 'darwin' and platform_machine == 'arm64'",
        "anthropic>=0.70",
        "openai-agents>=0.10",
    }


def test_full_install_keeps_every_previously_base_runtime_dependency() -> None:
    full = set(_EXTRAS["full"])
    assert set(_EXTRAS["full-runtime"]) <= full
    assert _names(_EXTRAS["full-runtime"]) <= _names(_EXTRAS["all"])
