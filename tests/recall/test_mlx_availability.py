"""The MLX availability predicate must answer about the BACKEND, not the wrapper.

`synapt._models.mlx_client` imports `mlx_lm` and `mlx` FUNCTION-LOCALLY (`_load`,
`_load_fused`, `chat`), never at module scope. So the wrapper imports cleanly on every
platform -- including the Linux and Windows CI runners, where `mlx-lm` is never installed
because pyproject declares it only for
`sys_platform == 'darwin' and platform_machine == 'arm64'`.

The predicate used to be "does the wrapper import?", which therefore reported AVAILABLE on
runners with no backend at all. Two consequences, and the second is the one that matters:

1. Every `skipUnless(_MLX_AVAILABLE, ...)` guard silently stopped firing. The real-judge
   tests ran with no backend, inference returned None, and the suite reported FAILED where
   the honest answer is SKIPPED. Six of twelve unit-test jobs were red on every branch.
2. The nine PRODUCTION consumers (enrich, server, consolidate, cli) took the MLX path on
   platforms that cannot run it, failing at call time instead of returning INSTALL_MSG.

A guard whose predicate cannot report absence is not a guard. These witnesses pin the
predicate to the question its callers actually ask: can inference RUN here?
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall import _mlx


# The modules the MLX inference path actually needs at call time. `mlx_lm` is imported by
# MLXClient.chat/_load; `mlx` by _load_fused. Neither is imported at module scope.
BACKEND_MODULES = ("mlx_lm", "mlx")


def _synthetic_module(name: str) -> types.ModuleType:
    """A module object that `importlib.util.find_spec` will report as PRESENT.

    A bare ModuleType has `__spec__ = None`, which makes find_spec raise rather than
    report presence, so the spec is set explicitly.
    """
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


class TestMLXAvailabilityPredicate(unittest.TestCase):
    """Absence must be reportable, presence must be reportable, and the simulation of
    absence must itself be proven to take effect."""

    def setUp(self):
        # Whatever these tests do to the module, restore the real state afterwards so no
        # later test in the session inherits a simulated predicate.
        self.addCleanup(importlib.reload, _mlx)

    # ---------------------------------------------------------------- the witness

    def test_predicate_reports_unavailable_when_the_backend_is_absent(self):
        """THE WITNESS. Blocking the backend must make the predicate say unavailable.

        `sys.modules[name] = None` is the established absence simulation: it makes
        `import name` raise ImportError AND `importlib.util.find_spec(name)` return None,
        so it covers both the module-scope face and the call-time face.
        """
        with patch.dict(sys.modules, {name: None for name in BACKEND_MODULES}):
            reloaded = importlib.reload(_mlx)
            self.assertFalse(
                reloaded.MLX_AVAILABLE,
                "predicate reported MLX AVAILABLE while the backend modules "
                f"{BACKEND_MODULES} were absent -- this is the false positive that stops "
                "every skipUnless guard from firing on non-Apple-Silicon runners",
            )

    def test_skip_reason_names_the_missing_dependency(self):
        """A skip nobody can act on is only marginally better than a failure.

        The reason must name the installable dependency, not merely a platform.
        """
        with patch.dict(sys.modules, {name: None for name in BACKEND_MODULES}):
            reloaded = importlib.reload(_mlx)
            reason = reloaded.SKIP_REASON
            self.assertTrue(reason, "SKIP_REASON must be non-empty when MLX is absent")
            self.assertIn(
                "mlx-lm",
                reason,
                f"skip reason must name the installable package `mlx-lm` -- got {reason!r}",
            )
            self.assertIn(
                "mlx_lm",
                reason,
                f"skip reason must name the missing module -- got {reason!r}",
            )

    # ---------------------------------------------------------------- the controls

    def test_control_the_absence_simulation_actually_takes_effect(self):
        """CONTROL on the instrument, not on the code under test.

        If the block silently failed, the witness above would be asserting nothing. This
        proves both faces of absence are really simulated.
        """
        with patch.dict(sys.modules, {name: None for name in BACKEND_MODULES}):
            for name in BACKEND_MODULES:
                self.assertIsNone(
                    importlib.util.find_spec(name),
                    f"find_spec({name!r}) still reports a spec -- the block did not take",
                )
            with self.assertRaises(ImportError):
                importlib.import_module("mlx_lm")

    def test_control_predicate_reports_available_when_the_backend_is_present(self):
        """CONTROL proving the witness is not satisfied by an always-False predicate.

        Presence is SYNTHESIZED rather than read from the host, so this control is
        meaningful on Linux and Windows too -- where the real backend is absent by design
        and a host-dependent control would simply skip, leaving the witness unguarded.
        """
        injected = {name: _synthetic_module(name) for name in BACKEND_MODULES}
        with patch.dict(sys.modules, injected):
            reloaded = importlib.reload(_mlx)
            self.assertTrue(
                reloaded.MLX_AVAILABLE,
                "predicate reported unavailable while both backend modules were present "
                "-- an always-False predicate would pass the witness and fail here",
            )
            self.assertEqual(
                reloaded.SKIP_REASON,
                "",
                "SKIP_REASON must be empty when MLX is available",
            )

    def test_benchmarks_module_guard_tracks_real_availability(self):
        """THE SECOND WITNESS -- the same defect class with the opposite sign.

        `test_benchmarks_llm.py` read MLX availability out of `synapt.recall.clustering`,
        which never defined it. The import raised, a broad `except ImportError` set the
        flag False, and the module skipped on EVERY platform -- 16 tests providing no
        signal anywhere, on Apple Silicon included.

        A stuck-True predicate fails loudly; a stuck-False one disappears, because a skip
        reads as fine. This pins the guard to the real predicate so neither can drift.
        """
        def load_benchmarks_module():
            path = Path(__file__).with_name("test_benchmarks_llm.py")
            spec = importlib.util.spec_from_file_location("_probe_benchmarks_llm", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        # Backend present AND opted in -> the module must NOT skip. This is the assertion
        # the old code could never satisfy on any platform.
        injected = {name: _synthetic_module(name) for name in BACKEND_MODULES}
        with patch.dict(sys.modules, injected):
            importlib.reload(_mlx)
            with patch.dict(os.environ, {"SYNAPT_RUN_LLM_BENCHMARKS": "1"}):
                module = load_benchmarks_module()
                self.assertFalse(
                    module.pytestmark.args[0],
                    "the benchmarks module still skips with the backend present and the "
                    "opt-in set -- that is the stuck-False guard, which runs nowhere and "
                    f"says so nowhere. reason={module.pytestmark.kwargs.get('reason')!r}",
                )

            # Opt-in cleared -> it skips, but for the OPT-IN reason, not a backend reason.
            # A skip is only honest if it names which condition was unmet.
            with patch.dict(os.environ, {"SYNAPT_RUN_LLM_BENCHMARKS": ""}):
                module = load_benchmarks_module()
                reason = module.pytestmark.kwargs.get("reason", "")
                self.assertTrue(module.pytestmark.args[0])
                self.assertIn(
                    "SYNAPT_RUN_LLM_BENCHMARKS",
                    reason,
                    f"skip must name the opt-in that was unmet -- got {reason!r}",
                )

        # Backend absent -> it skips naming the BACKEND, not the opt-in.
        with patch.dict(sys.modules, {name: None for name in BACKEND_MODULES}):
            importlib.reload(_mlx)
            module = load_benchmarks_module()
            self.assertTrue(module.pytestmark.args[0])
            self.assertIn(
                "mlx-lm",
                module.pytestmark.kwargs.get("reason", ""),
                "with the backend absent the skip must name the missing dependency",
            )

    def test_backend_neutral_names_survive_the_backend_being_absent(self):
        """THE THIRD WITNESS -- a latent defect that only a WORKING guard exposes.

        `enrich` imported `Message` inside `if _MLX_AVAILABLE:`. That survived only while
        the predicate was stuck True everywhere. `Message` is backend-neutral: it comes
        from `synapt._models.base` and `_enrich_single_window` builds one for whatever
        client it is handed -- including the Modal and Ollama clients the router returns on
        hosts with no MLX, which never pass the MLX guard in `enrich_session`.

        So correcting the predicate without this would have swapped a red matrix for a
        NameError on the enrichment path of every non-Apple-Silicon install. Fixing a guard
        can expose defects the broken guard was hiding, and they ship together or not at
        all.
        """
        # Registered BEFORE the simulation, so restoration runs even when an assertion
        # below fails. A restore placed after the assertions is cleanup conditional on its
        # own success: one failure would leave both modules pinned in simulated absence for
        # the remainder of the session and redden unrelated tests, turning a single honest
        # failure into a cascade whose origin is no longer visible.
        #
        # addCleanup is LIFO, and the order matters: _mlx must be restored BEFORE enrich is
        # reloaded, or enrich re-reads a predicate that is still simulated and keeps a stale
        # False. Registering enrich first means it runs second.
        enrich_module = importlib.import_module("synapt.recall.enrich")
        self.addCleanup(importlib.reload, enrich_module)
        self.addCleanup(importlib.reload, _mlx)

        with patch.dict(sys.modules, {name: None for name in BACKEND_MODULES}):
            importlib.reload(_mlx)
            enrich = importlib.reload(enrich_module)

            self.assertFalse(
                enrich._MLX_AVAILABLE,
                "precondition: the simulated runner must report MLX unavailable",
            )
            # Message: backend-neutral, used by every client the router can return.
            self.assertTrue(
                hasattr(enrich, "Message"),
                "Message is unbound with the MLX backend absent -- every non-MLX client "
                "path through _enrich_single_window raises NameError on this platform",
            )
            # MLXClient / MLXOptions: bound even where the backend is absent, because the
            # predicate governs USE, not binding. Anything addressing them by name on this
            # module -- including a test that patches them to force the MLX path -- gets
            # AttributeError instead of the guarded path when binding is conditional.
            for name in ("MLXClient", "MLXOptions"):
                self.assertTrue(
                    hasattr(enrich, name),
                    f"{name} is unbound with the MLX backend absent. Binding is always "
                    "safe here: mlx_client imports its backend function-locally, so the "
                    "module imports on every platform -- which is the very property this "
                    "change is about.",
                )

    def test_control_a_broken_backend_is_not_disguised_as_an_absent_one(self):
        """Absent and broken are different facts, and only absence is expected here.

        A module that EXISTS but raises on import must leave the predicate reporting
        available, so the real ImportError surfaces at the call site instead of being
        silently downgraded to a skip. This is why the predicate resolves specs rather
        than wrapping the backend import in try/except ImportError.
        """
        broken = _synthetic_module("mlx_lm")
        injected = {"mlx_lm": broken, "mlx": _synthetic_module("mlx")}
        with patch.dict(sys.modules, injected):
            reloaded = importlib.reload(_mlx)
            self.assertTrue(
                reloaded.MLX_AVAILABLE,
                "a backend that is present-but-broken was reported as unavailable; that "
                "hides a real ImportError behind a skip",
            )


if __name__ == "__main__":
    unittest.main()
