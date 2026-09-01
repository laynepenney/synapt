"""Metadata contract for the installable MCP server surface."""

from __future__ import annotations

import tomllib
from pathlib import Path


_PROJECT = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
_BASE = _PROJECT["project"]["dependencies"]
_EXTRAS = _PROJECT["project"]["optional-dependencies"]

# Exact strings from the pre-split `all` extra.  Package-name comparison is not
# enough: lowering a bound keeps the name while weakening an existing full
# install promise.
_FORMER_ALL = {
    "fastapi>=0.100",
    "uvicorn[standard]>=0.20",
    "sse-starlette>=1.0",
    "markdown>=3.4",
    "nh3>=0.2.14",
    "anthropic>=0.77.0",
    "openai-agents>=0.10.0",
    "google-adk>=1.0.0",
    "langchain-core>=0.2",
    "crewai>=1.4.0",
    "huggingface_hub>=0.20",
    "onnxruntime>=1.16",
    "optimum[onnxruntime]>=1.16",
    "transformers>=4.30",
    "peft>=0.7",
    "synapt-extract>=0.6.0",
    "tree-sitter>=0.23",
    "tree-sitter-language-pack>=0.7",
}


def _names(requirements: list[str]) -> set[str]:
    return {requirement.split("[", 1)[0].split("=", 1)[0].split(">", 1)[0].strip() for requirement in requirements}


def _assert_former_all(full: set[str]) -> None:
    missing = _FORMER_ALL - full
    assert not missing, f"full weakened or dropped former all requirements: {sorted(missing)}"


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


def test_full_preserves_every_exact_pre_split_all_requirement() -> None:
    _assert_former_all(set(_EXTRAS["full"]))


def test_exact_pin_guard_rejects_deletion_and_weaker_substitution() -> None:
    full = set(_EXTRAS["full"])
    for removed in ("anthropic>=0.77.0", "openai-agents>=0.10.0"):
        mutated = full - {removed}
        try:
            _assert_former_all(mutated)
        except AssertionError:
            continue
        raise AssertionError(f"deleting {removed} did not make the exact-pin guard fail")

    weakened = (full - {"anthropic>=0.77.0"}) | {"anthropic>=0.70"}
    try:
        _assert_former_all(weakened)
    except AssertionError:
        return
    raise AssertionError("replacing the former exact anthropic pin with its weaker runtime bound passed")
