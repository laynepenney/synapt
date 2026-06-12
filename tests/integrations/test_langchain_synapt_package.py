"""Packaging tests for the standalone ``langchain-synapt`` distribution."""

from __future__ import annotations

import subprocess
import sys
import venv
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = REPO_ROOT / "langchain-synapt"


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
        text=True,
        capture_output=True,
    )


def _rebuild_dist(target: Path) -> Path:
    dist_dir = target / "dist"
    shutil.rmtree(dist_dir, ignore_errors=True)
    _run([sys.executable, "-m", "build", str(target)])
    wheels = sorted(
        dist_dir.glob("*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    assert wheels, f"expected wheel build output in {dist_dir}"
    return wheels[0]


def test_langchain_synapt_package_has_required_files() -> None:
    assert (PACKAGE_DIR / "pyproject.toml").exists()
    assert (PACKAGE_DIR / "README.md").exists()
    assert (PACKAGE_DIR / "src" / "langchain_synapt" / "__init__.py").exists()


def test_langchain_synapt_builds_wheel() -> None:
    wheel = _rebuild_dist(PACKAGE_DIR)
    assert wheel.name.startswith("langchain_synapt-")


def test_langchain_synapt_installs_and_exports_history_class(tmp_path: Path) -> None:
    synapt_wheel = _rebuild_dist(REPO_ROOT)
    langchain_wheel = _rebuild_dist(PACKAGE_DIR)

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True)
    python = _venv_python(venv_dir)
    pip = [str(python), "-m", "pip"]

    _run(pip + ["install", "langchain-core>=0.2"])
    _run(pip + ["install", "--no-deps", str(synapt_wheel)])
    _run(pip + ["install", "--no-deps", str(langchain_wheel)])

    result = _run(
        [
            str(python),
            "-c",
            (
                "from pathlib import Path; "
                "from langchain_core.messages import HumanMessage; "
                "from langchain_synapt import SynaptChatMessageHistory; "
                "db = Path('history.db'); "
                "history = SynaptChatMessageHistory(session_id='pkg-test', db_path=db); "
                "history.add_messages([HumanMessage(content='hello from wheel')]); "
                "print(history.messages[0].content)"
            ),
        ]
    )
    assert result.stdout.strip() == "hello from wheel"
