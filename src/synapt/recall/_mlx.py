"""Shared MLX availability guard.

Centralizes the backend availability check so enrich.py and consolidate.py don't
duplicate the try/except ImportError pattern.

WHAT THIS PREDICATE MUST ANSWER, and what it used to answer instead
-------------------------------------------------------------------
Callers ask one question: *can MLX inference actually run here?* The predicate used to
answer a different one -- *does the MLX wrapper import?* -- and the two come apart on
every non-Apple-Silicon machine.

`synapt._models.mlx_client` imports `mlx_lm` and `mlx` FUNCTION-LOCALLY (`_load`,
`_load_fused`, `chat`), never at module scope. So the wrapper imports cleanly even where
the backend was never installed -- and `mlx-lm` is declared in pyproject only for
`sys_platform == 'darwin' and platform_machine == 'arm64'`, so it is absent by design on
the Linux and Windows CI runners and on every Linux/Windows install.

The wrapper-import predicate therefore reported AVAILABLE with no backend present, which
broke both of its jobs:

* every `skipUnless(MLX_AVAILABLE, ...)` guard silently stopped firing, so real-model
  tests ran with nothing to run against and reported FAILED where the honest answer is
  SKIPPED
* the production consumers (enrich, server, consolidate, cli) took the MLX path on
  platforms that cannot execute it, failing at call time instead of returning INSTALL_MSG

So the check resolves the BACKEND modules. A guard whose predicate cannot report absence
is not a guard.
"""

from __future__ import annotations

import importlib.util

# The modules the inference path needs AT CALL TIME. Importing MLXClient proves none of
# them are reachable, because it does not import them until a method runs.
_BACKEND_MODULES = ("mlx_lm", "mlx")

# The installable distribution that provides them. Named in SKIP_REASON so a skip tells
# the reader what to do about it rather than only that something was missing.
_BACKEND_PACKAGE = "mlx-lm"


def _missing_backend_modules() -> tuple[str, ...]:
    """Return the backend modules that cannot be resolved in this environment.

    Resolves SPECS rather than importing. The distinction is deliberate and is the same
    one that governs `find_spec` over `pytest.importorskip`: a module that EXISTS but
    raises on import must leave the predicate reporting available, so the real ImportError
    surfaces at the call site instead of being silently downgraded to "absent."

    Absent and broken are different facts, and only absence is expected here.
    """
    missing: list[str] = []
    for name in _BACKEND_MODULES:
        try:
            resolved = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # ImportError: a parent package is itself missing.
            # ValueError: the entry in sys.modules has no __spec__.
            resolved = False
        if not resolved:
            missing.append(name)
    return tuple(missing)


MLX_AVAILABLE = False
_MISSING_BACKENDS: tuple[str, ...] = _BACKEND_MODULES

try:
    from synapt._models.mlx_client import MLXClient, MLXOptions  # noqa: F401
    from synapt._models.base import Message  # noqa: F401
except ImportError:
    # The wrapper itself is unreachable, so the backend question is moot: every module
    # stays listed as missing and the predicate remains False.
    pass
else:
    _MISSING_BACKENDS = _missing_backend_modules()
    MLX_AVAILABLE = not _MISSING_BACKENDS


SKIP_REASON = (
    ""
    if MLX_AVAILABLE
    else (
        "requires the MLX backend, missing: "
        + ", ".join(_MISSING_BACKENDS)
        + f" -- install with `pip install {_BACKEND_PACKAGE}`. recall declares it only for "
        "sys_platform=='darwin' and platform_machine=='arm64', so it is absent by design "
        "on Linux and Windows."
    )
)

INSTALL_MSG = (
    "MLX is required for this feature.\n"
    "Install with: pip install mlx-lm\n"
    "Then re-run this command."
)
