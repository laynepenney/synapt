"""Witnesses for the recall test-store isolation guard.

Ref #955 — closes at promotion.

The governing claim is narrow: a test cannot silently write to the real
home-level channel store, and an implicit recall data path cannot silently
resolve into a real checkout. These tests pin that claim.

Two things are deliberately separated here:

* the *boundary derivation* — that the protected root comes from the operating
  system account and cannot be moved by a fixture — is pinned once, against
  the real root, by reading only.
* the *refusal mechanics* — that a resolved candidate under a protected root
  is refused before the write — are pinned against a decoy root registered for
  the duration of a single test.

That split exists so no witness has to attempt a write at the real store to
prove the guard prevents writes at the real store. A prevention witness that
contaminates when it fails is not a safety net; it is the defect with a test
wrapped around it. Registering a decoy *adds* a protected root and can never
remove the real one, so the split does not open a bypass.
"""

from __future__ import annotations

import os
import pathlib
import sys
from pathlib import Path

import pytest

from synapt.recall import channel as channel_mod
from synapt.recall import core as core_mod
from synapt.recall import direct as direct_mod
from synapt.recall import journal as journal_mod

# NOTE on off-POSIX: only the witnesses that derive the boundary from the passwd
# home are POSIX-only, and they are exactly the ones that request the
# ``protected_channel_root`` / ``rederive_protected_root`` fixtures — those
# fixtures skip off-POSIX (see conftest). The portable witnesses here — the
# data-root guarantees and the decoy-root refusal mechanics, which use
# ``register_protected_root`` and never touch ``pwd``/``os.getuid`` — run on
# every platform. A module-level skip would collapse those two scopes and turn
# one honest POSIX-only absence into thirty absent witnesses (Atlas, r2).


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _entries(root: Path) -> set[str]:
    """Every path under *root*, as a comparable set.

    Enumerates the whole tree, not ``*.jsonl``. The issue already has evidence
    of fixture material below the attachments tree, so a JSONL-only audit
    reports success by construction while missing the artifact.
    """
    if not root.exists():
        return set()
    return {str(p) for p in root.rglob("*")}


def _message(body: str) -> "channel_mod.ChannelMessage":
    """A minimal ChannelMessage; timestamp and type have no defaults."""
    return channel_mod.ChannelMessage(
        timestamp="2026-08-07T00:00:00Z",
        channel="dev",
        type="message",
        body=body,
        from_agent="s_witness",
    )


def _make_gripspace(tmp: Path, org: str = "decoy-org", repo: str = "decoy-repo") -> Path:
    """Build a gripspace whose manifest resolves to <org>/<repo>.

    Mirrors the real layout closely enough that ``_global_channels_dir``
    takes its tier-2 branch rather than falling through to tier 3.
    """
    root = tmp / "gripspace"
    manifest = root / ".gitgrip" / "spaces" / "main"
    manifest.mkdir(parents=True)
    (root / ".gitgrip" / "griptrees.json").write_text("{}", encoding="utf-8")
    (manifest / "gripspace.yml").write_text(
        f"manifest:\n  url: git@github.com:{org}/{repo}.git\n",
        encoding="utf-8",
    )
    core_mod._gripspace_cache.clear()
    return root


# ---------------------------------------------------------------------------
# W0 — the boundary derivation itself
# ---------------------------------------------------------------------------

def test_protected_root_is_derived_from_the_os_account_not_from_patched_home(
    protected_channel_root, rederive_protected_root, monkeypatch, tmp_path
):
    """The protected boundary survives a fixture moving HOME and Path.home().

    This is the one witness that must speak about the real root, and it does so
    by reading only. If the boundary were derived from the value a test is
    currently asking the code to use, every other witness here would be
    circular: the test would move the boundary and then prove nothing crosses
    it.
    """
    import pwd

    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
    assert protected_channel_root == account_home / ".synapt" / "channels"

    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: tmp_path / "fake-home"))

    # Re-derived under a fully patched home, the boundary must not move.
    assert rederive_protected_root() == account_home / ".synapt" / "channels"


# ---------------------------------------------------------------------------
# W1 — the ordinary path is isolated, and the real store is untouched
# ---------------------------------------------------------------------------

def test_ordinary_channel_post_lands_in_a_pytest_owned_directory(
    protected_channel_root, tmp_path, monkeypatch
):
    """An ordinary post writes under the session root and never the real store."""
    baseline = _entries(protected_channel_root)

    channels = tmp_path / "channels"
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(channels))

    resolved = channel_mod._channels_dir()
    assert resolved == channels

    channel_mod.channel_post("dev", "isolation witness", project_dir=tmp_path)

    written = channels / "dev.jsonl"
    assert written.exists()
    assert "isolation witness" in written.read_text(encoding="utf-8")

    assert _entries(protected_channel_root) == baseline


def test_the_guard_is_armed_with_no_per_test_fixture_at_all(
    protected_channel_root, store_isolation_error
):
    """A test that sets nothing up is still protected.

    This is the property that matters, and it is deliberately *not* "the
    environment override is set." An earlier draft asserted exactly that, and
    it was measuring the wrong thing: leaving a session-wide override in place
    made tier 1 outrank the tier-3 resolution some tests deliberately exercise,
    and turned independent tests into shared-state ones. The override is a
    semantic change to path resolution; the guard is not. So what a
    fixture-less test inherits is the guard, not a redirect.
    """
    assert channel_mod._store_path_policy is not None

    with pytest.raises(store_isolation_error):
        # Pointed at the protected root explicitly rather than relying on this
        # machine happening to resolve there.
        channel_mod._guard_store_path("witness", protected_channel_root / "dev.jsonl")


def test_the_data_root_guard_is_armed_by_default(
    request, store_isolation_error, monkeypatch
):
    """The data-root guard is installed with no flag passed.

    This is the witness the flip rests on, and it is deliberately a DIRECT
    observation of the policy rather than a comparison between two run modes.
    Before the burn-down, running with the flag differed from running without
    it, and that difference was the proof the flag was wired. Now the two agree
    — which is both the success criterion and the exact signature of a guard
    that has stopped working. Modal agreement cannot tell those apart any more,
    so the evidence has to come from here.
    """
    if not request.config.getoption("strict_recall_data_root"):
        # Discriminate on the ARGV signal, not on the resolved option. Those
        # two agree today and come apart in exactly the case that matters: if
        # the default is ever silently reverted to off, the resolved option
        # reads False with no flag passed, and skipping here would announce
        # "deliberately disarmed" about a disarm nobody requested. That is the
        # skip LYING rather than merely going quiet — a false reason is worse
        # than no reason, because it explains away the very thing it should
        # surface.
        assert "--no-strict-recall-data-root" in sys.argv, (
            "the data-root guard is disarmed but nobody passed the flag — the "
            "default has been reverted; this must fail rather than skip"
        )
        pytest.skip("deliberately disarmed via --no-strict-recall-data-root")

    assert core_mod._data_root_policy is not None, (
        "the data-root guard must be armed without passing any flag"
    )

    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    outside = Path(__file__).resolve().parents[2]

    with pytest.raises(store_isolation_error):
        core_mod.project_data_dir(outside)


def test_the_debug_escape_hatch_is_wired_end_to_end(request, install_policy):
    """``--no-strict-recall-data-root`` reaches the install decision.

    Two links, asserted separately, because either one alone is satisfiable
    without the other: the option must resolve to the same destination the
    installer reads, and the installer must actually decline to arm when told
    not to. An escape hatch nobody has proven works is worse than none — it
    invites the belief that the guard can be stepped around cleanly, and the
    first person to need it finds out otherwise under pressure.

    Deliberately not a nested pytest run. ``pytester`` executes in-process and
    shares module globals, which is how the guard silently disarmed itself
    once already; a second nested run to test disarming would be the same
    hazard aimed at the thing it broke before.
    """
    # Link 1: the option exists and resolves to the destination the installer
    # reads. Under --no-strict-recall-data-root that destination is False,
    # which is itself the proof the flag reaches it.
    armed = request.config.getoption("strict_recall_data_root")
    assert armed is ("--no-strict-recall-data-root" not in sys.argv)

    # Link 2: told not to arm, the installer does not arm.
    previous = core_mod.set_data_root_policy(None)
    try:
        install_policy(False)
        assert core_mod._data_root_policy is None
        # Link 3: the hatch is SCOPED. It disarms the data-root guard and
        # nothing else. Without this, the escape could widen into a full
        # disarm — taking the channel guard, which is #955's contract and was
        # never opt-in — and every other witness here would still pass, because
        # they arm their own policy or assert against a decoy. An escape hatch
        # nobody has bounded is a hatch that grows.
        assert channel_mod._store_path_policy is not None, (
            "the data-root escape must not disarm the channel guard"
        )
        install_policy(True)
        assert core_mod._data_root_policy is not None
    finally:
        core_mod.set_data_root_policy(previous)


def test_passing_both_flags_resolves_last_wins(pytester_precedence):
    """With both flags given, the LAST one decides — pinned, not inherited.

    Found in review. The behaviour was already last-wins, but by argparse
    accident rather than by decision: undocumented, untested, and therefore
    free to change under a dependency bump with nothing going red. That is the
    same shape as the rest of this file — a behaviour nobody chose, invisible
    until it decides something.

    The concrete case is a CI base config carrying one flag and a job appending
    the other, where the outcome is settled by argument order nobody wrote
    down. Last-wins is the conventional answer and a good one; this makes it a
    decision by pinning it, and the docstring on the option now says so.
    """
    armed_last, disarmed_last = pytester_precedence
    assert armed_last is True, "--strict last must arm"
    assert disarmed_last is False, "--no-strict last must disarm"


def test_layer_one_covers_the_import_window(isolation_policy):
    """The pre-collection default exists, even though tests do not run under it.

    ``pytest_configure`` runs before any test module is imported, which is the
    window no fixture can reach — module-level code and collection-time helpers
    resolve there. The per-test fixture then hands resolution back. Both halves
    are load-bearing and neither is visible from the other's vantage point.
    """
    assert isolation_policy.session_channel_root is not None
    assert isolation_policy.session_channel_root.exists()
    assert isolation_policy.session_data_root is not None


# ---------------------------------------------------------------------------
# W2 — manifest-resolving gripspace with the override absent
# ---------------------------------------------------------------------------

def test_gripspace_resolution_without_override_fails_before_creating_anything(
    register_protected_root, store_isolation_error, tmp_path, monkeypatch
):
    """Tier-2 resolution is refused before its own mkdir.

    ``_channels_dir`` mkdirs the global directory *as part of resolving it*, so
    a guard installed only at the append seam would already be too late — the
    directory would exist under the real store before anything tried to write a
    message into it.
    """
    decoy_home = tmp_path / "home"
    decoy_root = decoy_home / ".synapt" / "channels"
    register_protected_root(decoy_root)

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: decoy_home))
    monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)

    gripspace = _make_gripspace(tmp_path)
    before = _entries(decoy_root)

    with pytest.raises(store_isolation_error) as excinfo:
        channel_mod._channels_dir(project_dir=gripspace)

    assert _entries(decoy_root) == before
    assert not decoy_root.exists()

    message = str(excinfo.value)
    assert "decoy-org" in message and "decoy-repo" in message
    assert str(decoy_root) in message
    assert "test_gripspace_resolution_without_override_fails" in message


def test_the_public_post_api_is_refused_end_to_end(
    register_protected_root, store_isolation_error, tmp_path, monkeypatch
):
    """The refusal survives the real entry point, not just the internal seam.

    ``channel.py`` contains seven broad ``except Exception`` handlers, and the
    refusal is an ``AssertionError`` — which one of them would happily swallow,
    leaving a test that posts into a protected store and reports success. The
    other witnesses call the seam directly and so cannot detect that. This one
    goes through ``channel_post``, the function real callers use.
    """
    decoy_home = tmp_path / "home"
    decoy_root = decoy_home / ".synapt" / "channels"
    register_protected_root(decoy_root)

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: decoy_home))
    monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)

    gripspace = _make_gripspace(tmp_path)
    before = _entries(decoy_root)

    with pytest.raises(store_isolation_error):
        channel_mod.channel_post("dev", "must not reach the store", project_dir=gripspace)

    assert _entries(decoy_root) == before
    assert not decoy_root.exists()


def test_unsetting_the_override_mid_test_does_not_buy_a_bypass(
    register_protected_root, store_isolation_error, tmp_path, monkeypatch
):
    """A test that clears the environment still cannot reach the store.

    An env-var-only harness is bypassable by exactly this move, which is why
    the refusal lives at the resolver rather than in the fixture that sets the
    default.
    """
    decoy_home = tmp_path / "home"
    register_protected_root(decoy_home / ".synapt" / "channels")
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: decoy_home))

    gripspace = _make_gripspace(tmp_path)
    os.environ.pop("SYNAPT_SHARED_CHANNELS_DIR", None)

    with pytest.raises(store_isolation_error):
        channel_mod._channels_dir(project_dir=gripspace)


# ---------------------------------------------------------------------------
# W3 — an explicit channels_dir argument
# ---------------------------------------------------------------------------

def test_explicit_channels_dir_at_the_store_fails_before_the_append(
    register_protected_root, store_isolation_error, tmp_path
):
    """The explicit-argument form bypasses resolution, so it is guarded separately."""
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    register_protected_root(decoy_root)

    target = decoy_root / "decoy-org" / "decoy-repo"
    before = _entries(decoy_root)

    msg = _message("must not be appended")

    with pytest.raises(store_isolation_error):
        channel_mod._append_message(msg, channels_dir=target)

    assert _entries(decoy_root) == before
    assert not (target / "dev.jsonl").exists()


# ---------------------------------------------------------------------------
# W4 — attachments
# ---------------------------------------------------------------------------

def test_attachment_copy_into_the_store_fails_before_the_copy(
    register_protected_root, store_isolation_error, tmp_path, monkeypatch
):
    """Attachments are a channel-owned write surface and go through the same policy.

    A cleanup that enumerates only ``*.jsonl`` misses this path by
    construction, which is precisely why it needs its own witness rather than
    inheriting confidence from the JSONL one.
    """
    decoy_home = tmp_path / "home"
    decoy_root = decoy_home / ".synapt" / "channels"
    register_protected_root(decoy_root)

    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: decoy_home))
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(decoy_root))

    source = tmp_path / "payload.txt"
    source.write_text("attachment body", encoding="utf-8")
    before = _entries(decoy_root)

    with pytest.raises(store_isolation_error):
        channel_mod._copy_attachments("msg-witness", [str(source)])

    assert _entries(decoy_root) == before


# ---------------------------------------------------------------------------
# W5 — channel-owned child paths other than <name>.jsonl
# ---------------------------------------------------------------------------

def test_direct_inbox_child_path_is_covered(
    register_protected_root, store_isolation_error, tmp_path, monkeypatch
):
    """The direct-message tree resolves through the same seam and is covered.

    Pinning only ``dev.jsonl`` would prove the guard for the file class the
    defect was first noticed in and nothing else.
    """
    decoy_home = tmp_path / "home"
    register_protected_root(decoy_home / ".synapt" / "channels")
    monkeypatch.setattr(pathlib.Path, "home", staticmethod(lambda: decoy_home))
    monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)

    gripspace = _make_gripspace(tmp_path)

    with pytest.raises(store_isolation_error):
        direct_mod._direct_dir(project_dir=gripspace)


def test_local_to_global_migration_is_covered(
    register_protected_root, store_isolation_error, tmp_path
):
    """``migrate_channels_to_global`` composes the store path itself.

    It never calls ``_channels_dir`` and never takes a ``channels_dir``
    argument, so it sits outside both forms the design note enumerated. Killing
    the named surfaces proves the named set; this is the pristine case that sat
    outside the enumeration.
    """
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    register_protected_root(decoy_root)

    local = tmp_path / "local-channels"
    local.mkdir()
    (local / "dev.jsonl").write_text(
        '{"channel":"dev","from_agent":"s_witness","body":"migrating","timestamp":"2026-08-07T00:00:00Z"}\n',
        encoding="utf-8",
    )
    before = _entries(decoy_root)

    with pytest.raises(store_isolation_error):
        channel_mod.migrate_channels_to_global(local, decoy_root, "decoy-org", "decoy-repo")

    assert _entries(decoy_root) == before


# ---------------------------------------------------------------------------
# The SQLite surfaces — and why exactly one of them is guarded
# ---------------------------------------------------------------------------

def test_state_db_at_the_store_is_refused_before_the_file_is_created(
    register_protected_root, store_isolation_error, tmp_path
):
    """``_open_state_db`` takes its path from the caller and can reach the store.

    Same shape as an explicit ``channels_dir``: it reaches around resolution
    entirely, so the resolver guard cannot see it. Found in review — the JSONL
    surface was covered and this one was not, and nothing said so.
    """
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    register_protected_root(decoy_root)

    target = decoy_root / "decoy-org" / "decoy-repo" / "_state.db"
    before = _entries(decoy_root)

    with pytest.raises(store_isolation_error):
        channel_mod._open_state_db(target)

    assert _entries(decoy_root) == before
    assert not target.exists()


def test_channels_db_is_data_root_surface_not_channel_store_surface(
    protected_channel_root, tmp_path
):
    """``channels.db`` is deliberately outside the channel guard, and must stay so.

    It is composed from ``project_data_dir``, so it can never resolve inside the
    global channel store — meaning a channel-store check here could not refuse
    anything. Adding one would look like tightening coverage while installing a
    check that cannot fail, which is the exact defect this harness exists to
    prevent.

    Pinned because the asymmetry looks like an oversight to anyone reading the
    seam list, and the obvious "fix" is the wrong move. The real coverage comes
    from the data-root policy, which ``project_data_dir`` consults.
    """
    resolved = channel_mod._db_path(project_dir=tmp_path).resolve()

    assert protected_channel_root not in resolved.parents
    assert resolved != protected_channel_root

    consulted = []
    previous = core_mod.set_data_root_policy(lambda op, p: consulted.append(op))
    try:
        channel_mod._db_path(project_dir=tmp_path)
    finally:
        core_mod.set_data_root_policy(previous)

    assert consulted, "channels.db must be reachable by the data-root policy"


# ---------------------------------------------------------------------------
# The guard must survive anything that clears the module global
# ---------------------------------------------------------------------------

def test_guard_is_rearmed_after_a_module_reload_disarms_it(rearm_guard):
    """``importlib.reload`` rebinds the module dict and silently clears the policy.

    It cannot be fixed where it happens — a reload replaces the module wholesale
    — so the harness re-arms before each test instead. That closes the whole
    class rather than the two mechanisms currently known, because the next one
    will arrive with the same signature: green suite, absent protection.
    """
    import importlib

    importlib.reload(channel_mod)
    assert channel_mod._store_path_policy is None, "reload should clear the policy"

    rearm_guard()
    assert channel_mod._store_path_policy is not None


# ---------------------------------------------------------------------------
# W6 / W7 — containment is a path property, not a string property
# ---------------------------------------------------------------------------

def test_symlink_resolving_into_the_store_is_refused(
    register_protected_root, store_isolation_error, tmp_path
):
    """A candidate is resolved before it is judged."""
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    decoy_root.mkdir(parents=True)
    register_protected_root(decoy_root)

    link = tmp_path / "innocent-looking"
    link.symlink_to(decoy_root, target_is_directory=True)

    msg = _message("via symlink")
    before = _entries(decoy_root)

    with pytest.raises(store_isolation_error):
        channel_mod._append_message(msg, channels_dir=link)

    assert _entries(decoy_root) == before


def test_sibling_with_a_textual_prefix_is_accepted(
    register_protected_root, tmp_path, monkeypatch
):
    """``channels-backup`` is not a descendant of ``channels``.

    A ``startswith`` check refuses this path and would be indistinguishable
    from a correct guard on every other witness in this file. This is the case
    that separates containment from string prefixing.
    """
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    register_protected_root(decoy_root)

    sibling = tmp_path / "home" / ".synapt" / "channels-backup"
    assert str(sibling).startswith(str(decoy_root))

    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(sibling))

    resolved = channel_mod._channels_dir()
    assert resolved == sibling

    channel_mod.channel_post("dev", "sibling is fine", project_dir=tmp_path)
    assert (sibling / "dev.jsonl").exists()


def test_the_protected_root_itself_is_refused_not_only_its_children(
    register_protected_root, store_isolation_error, tmp_path
):
    """The boundary is closed, not half-open."""
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    register_protected_root(decoy_root)

    msg = _message("at the root itself")
    with pytest.raises(store_isolation_error):
        channel_mod._append_message(msg, channels_dir=decoy_root)


# ---------------------------------------------------------------------------
# W8 — the deliberate live-store opt-in
# ---------------------------------------------------------------------------

def test_the_mark_alone_is_not_permission(pytester_isolated):
    """A marked test without the command-line option still fails.

    The mark records intent; the option records authorization. Collapsing them
    would let a single decorator re-open the store.
    """
    result = pytester_isolated.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*recall test isolation violation*"])


def test_the_option_alone_is_not_permission(pytester_isolated_unmarked):
    """An unmarked test does not become authorized because the flag was passed."""
    result = pytester_isolated_unmarked.runpytest(
        "--allow-live-channel-store-tests", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(failed=1)


def test_mark_plus_option_authorizes_and_says_so_loudly(pytester_isolated):
    """Both present: the write proceeds and the authorization is reported."""
    result = pytester_isolated.runpytest(
        "--allow-live-channel-store-tests", "-p", "no:cacheprovider"
    )
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*authorized to touch the live channel store*"])


def test_the_default_failure_does_not_advertise_the_opt_in(
    register_protected_root, store_isolation_error, tmp_path
):
    """The first remedy offered must be isolation, never authorization.

    A failure message that leads with the bypass converts a safety guard into a
    documented workaround, and the next person to hit it will reach for the
    flag rather than fix the fixture.
    """
    decoy_root = tmp_path / "home" / ".synapt" / "channels"
    register_protected_root(decoy_root)

    msg = _message("x")
    with pytest.raises(store_isolation_error) as excinfo:
        channel_mod._append_message(msg, channels_dir=decoy_root)

    message = str(excinfo.value)
    assert "SYNAPT_SHARED_CHANNELS_DIR" in message
    assert "allow-live-channel-store-tests" not in message


# ---------------------------------------------------------------------------
# W9 — the recall data root (journal, index, archive, knowledge)
# ---------------------------------------------------------------------------

def test_implicit_journal_resolution_cannot_silently_reach_a_real_checkout(
    strict_data_root, store_isolation_error,
):
    """``_journal_path()`` takes no project dir, so it follows cwd.

    This is the mechanism behind the live specimen: a CLI path that calls
    ``_journal_path()`` without the test project reads whichever journal the
    process working directory happens to supply. A real checkout supplies the
    operator's journal; a fresh worktree supplies an empty one, and a negative
    assertion silently changes result with no test-code change.

    Note the interaction with ``tests/recall/conftest.py``, which deliberately
    strips ``SYNAPT_RECALL_ROOT`` so these tests measure path *inference*
    rather than the override. That intent is right and is preserved here — but
    inference must not be allowed to land somewhere real. So the contract under
    the stripped-override regime is not "resolves pytest-owned"; it is
    "refuses, loudly." Silence is the only outcome ruled out.
    """
    with pytest.raises(store_isolation_error) as excinfo:
        journal_mod._journal_path()

    message = str(excinfo.value)
    assert "operation=project_data_dir" in message
    assert "expected under=" in message


def test_journal_resolution_is_independent_of_the_working_directory(
    strict_data_root, tmp_path, monkeypatch
):
    """Identical bytes, two working directories, one resolution.

    The specimen's defect was not a wrong path — it was a path that *varied*
    with where the process happened to be standing, which makes a passing run
    and a failing run indistinguishable at the source level. With a
    pytest-owned root supplied, the variance is gone.
    """
    root = tmp_path / "owned"
    root.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(root))
    monkeypatch.setenv("SYNAPT_RECALL_WORKTREE", "pytest-isolated")
    core_mod._gripspace_cache.clear()

    from_repo = journal_mod._journal_path().resolve()

    elsewhere = tmp_path / "fresh-worktree"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    core_mod._gripspace_cache.clear()

    from_elsewhere = journal_mod._journal_path().resolve()
    assert from_elsewhere == from_repo
    assert root.resolve() in from_repo.parents


def test_explicit_project_dir_outside_the_test_root_is_refused(
    strict_data_root, store_isolation_error, monkeypatch
):
    """An explicit project dir must still be proven test-owned.

    Otherwise a test can sidestep the channel guard entirely and still read or
    write the real journal, making its assertions depend on operator history.
    The repository checkout is used as the outside path because it is a real
    one — a scratch directory would not exercise the case that matters.
    """
    monkeypatch.delenv("SYNAPT_RECALL_ROOT", raising=False)
    repo_checkout = Path(__file__).resolve().parents[2]

    with pytest.raises(store_isolation_error):
        core_mod.project_data_dir(repo_checkout)


# ---------------------------------------------------------------------------
# The guard must not be disarmable by a nested run
# ---------------------------------------------------------------------------

def test_guard_survives_a_nested_pytest_run(pytester_isolated, tmp_path):
    """A nested pytest session must not leave the outer guard uninstalled.

    ``pytester`` runs in-process, so a nested conftest shares module globals
    with this session. A nested ``pytest_unconfigure`` that cleared the policy
    outright would disarm the guard for every test that ran afterwards, and
    nothing anywhere would report it — the suite would stay green while the
    protection it advertises was gone. Found exactly this way: three later
    witnesses started failing with the wrong error after the first nested run
    landed.
    """
    pytester_isolated.runpytest("-p", "no:cacheprovider")

    from synapt.recall import channel as channel_mod

    assert channel_mod._store_path_policy is not None, (
        "the nested run left the channel-store guard uninstalled"
    )


def test_recall_root_override_redirects_the_data_dir(tmp_path, monkeypatch):
    """The override exists and takes priority over cwd/worktree/gripspace resolution.

    Pinned on its own because Layer 1 depends on it: the design note specified
    this override as the mechanism, but no such override existed in the
    resolver, so the harness half it prescribed could not have worked.
    """
    root = tmp_path / "pytest-owned-root"
    root.mkdir()  # the override refuses a root that does not exist, by design
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(root))
    core_mod._gripspace_cache.clear()

    assert core_mod.project_data_dir() == root / ".synapt" / "recall"


def test_recall_root_override_refuses_a_root_that_does_not_exist(tmp_path, monkeypatch):
    """A mistyped root must not be minted as a fresh empty store.

    An empty history presents exactly like a real answer, so this failure has
    to be loud. Pinned because Layer 1 depends on the override, and a silent
    mint would make the harness look like it isolated when it had actually
    invented a new store.
    """
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(tmp_path / "typo-never-created"))
    core_mod._gripspace_cache.clear()

    with pytest.raises(ValueError, match="does not exist"):
        core_mod.project_data_dir()
