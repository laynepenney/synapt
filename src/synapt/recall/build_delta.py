"""Whether a build has anything to do, across every input class.

``incremental`` today gates exactly one call site, so a build with nothing to
do still walks channels and journals and still pays for the walk.  A real
no-op needs one signal that covers EVERY input the build reads: archived
transcripts, channel logs, and journals.

*** THE DANGEROUS DIRECTION IS "UP TO DATE", NOT "SLOW". ***

A signature that watches only transcripts would let a new channel message or a
fresh journal entry go unindexed while the build reports success.  That is
worse than a slow build, because a slow build is visible and a silently
skipped one is not: the operator sees "nothing to do", and the thing they just
wrote is missing from search with no error anywhere.  So every input class is
in the signature, and each has its own negative-control witness.

WHAT THE SIGNATURE IS, AND ITS ONE HONEST LIMIT.  Each input file contributes
its path, size, and modification time.  Content is not hashed: on a store with
tens of thousands of archived turns, hashing every byte on every build costs
more than the build this module exists to skip.

The residual, stated rather than left for someone to discover: an edit that
preserves BOTH size and mtime is invisible here.  In practice a write moves
mtime, and the archive layer refreshes on a newer mtime at equal size for
exactly this reason.  A deliberate same-size mtime-preserving rewrite is the
one case this signal misses, and a caller that needs to defeat it can pass
``--full``.  It is written down because an unstated limit is indistinguishable
from a bug once somebody hits it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import hashlib
import re

#: Manifest key. Absent means "no previous build to compare against", which is
#: NOT the same as "nothing changed" -- see `signature_from_manifest`.
MANIFEST_KEY = "input_signature"

SIGNATURE_VERSION = 1

_TRANSCRIPT_GLOB = "*.jsonl"

#: What `hashlib.sha256().hexdigest()` produces, and nothing else. Anchored at
#: both ends so a 64-hex substring inside a longer string does not qualify.
_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")


def _is_plain_int(value: Any) -> bool:
    """`type(...) is int`, deliberately NOT `isinstance`.

    ``bool`` SUBCLASSES ``int`` in Python, so ``isinstance(True, int)`` is True
    and ``True == 1``.  An ``isinstance`` guard therefore admits ``True`` and
    ``False`` into a field the dataclass declares as ``int``, and the same
    equality makes ``version: True`` compare equal to ``SIGNATURE_VERSION``.
    ``1.0 == 1`` gets in the same way.

    This is the whole class, not the one field it was first noticed on: any
    numeric guard written with ``isinstance`` or bare ``==`` in this module has
    the same hole.
    """
    return type(value) is int


@dataclass(frozen=True)
class InputSignature:
    """A fingerprint of every file a build would read.

    ``digest`` is what comparison uses.  ``file_count`` rides along because a
    signature that reports how much it saw makes an empty-inputs run legible
    instead of looking identical to an unchanged one.
    """

    digest: str
    file_count: int

    def to_manifest(self) -> dict[str, Any]:
        """Wrapped under `MANIFEST_KEY`, so the reader can tell absent from empty.

        The wrapper is what makes `signature_from_manifest` able to answer
        "this manifest predates signatures" distinctly from "this manifest has
        a signature that happens to be malformed". A bare payload would make
        every signature-less manifest indistinguishable from a corrupt one.
        """
        return {
            MANIFEST_KEY: {
                "version": SIGNATURE_VERSION,
                "digest": self.digest,
                "files": self.file_count,
            }
        }


def _iter_input_files(
    source_dirs: Sequence[Path] | None,
    channels_dir: Path | None,
    journal_paths: Iterable[Path] | None,
) -> list[Path]:
    """Every file the build would read, from all three input classes.

    Directories are GLOBBED rather than read from a remembered list, so a newly
    arrived file changes the signature.  A signature built only from files it
    already knew about cannot notice an arrival, which is the failure mode the
    new-file witness pins.
    """
    files: list[Path] = []

    for directory in source_dirs or ():
        directory = Path(directory)
        if directory.is_dir():
            files.extend(p for p in directory.rglob(_TRANSCRIPT_GLOB) if p.is_file())

    if channels_dir is not None:
        channels_dir = Path(channels_dir)
        if channels_dir.is_dir():
            files.extend(p for p in channels_dir.rglob(_TRANSCRIPT_GLOB) if p.is_file())

    for journal in journal_paths or ():
        journal = Path(journal)
        if journal.is_file():
            files.append(journal)

    return files


def compute_input_signature(
    source_dirs: Sequence[Path] | None = None,
    channels_dir: Path | None = None,
    journal_paths: Iterable[Path] | None = None,
) -> InputSignature:
    """Fingerprint the build's inputs across transcripts, channels and journals.

    Sorted before hashing so the digest depends on the input set and not on
    filesystem iteration order, which varies between runs and would otherwise
    make two identical states compare as different -- a no-op signal that
    reports "changed" on an unchanged store is merely useless, but one that
    does so *intermittently* is worse, because it trains people to distrust it.
    """
    entries: list[str] = []
    for path in _iter_input_files(source_dirs, channels_dir, journal_paths):
        try:
            stat = path.stat()
        except OSError:
            # Unreadable now, readable later, or vice versa: either way the
            # input set is not what it was. Record the path so the change is
            # visible rather than silently dropping the file from the digest.
            entries.append(f"{path}\x00unreadable")
            continue
        entries.append(f"{path}\x00{stat.st_size}\x00{stat.st_mtime_ns}")

    entries.sort()
    digest = hashlib.sha256("\x01".join(entries).encode("utf-8")).hexdigest()
    return InputSignature(digest=digest, file_count=len(entries))


def is_noop(previous: InputSignature | None, current: InputSignature) -> bool:
    """True only when a prior signature exists and matches the current one.

    ABSENCE IS NEVER A NO-OP.  A first build, or a build whose manifest was
    lost or corrupt, has nothing to compare against -- and "I cannot tell"
    must resolve to doing the work, not to skipping it.  Treating an absent
    previous as a match would make the very first build of a store, the one
    that has the most to do, the one that does nothing.

    THE DIGEST IS THE ONLY THING COMPARED, and `file_count` is deliberately
    not.  The digest is computed over every entry, so the count is derived
    from the same input rather than independent evidence about it: any change
    that alters the count necessarily alters the digest.  Comparing it too
    would look like defence in depth and provide none, because a second check
    over the same input cannot fail when the first one passes.

    That makes this a real contract rather than an implementation detail.  If
    the digest ever stops covering the full entry set, this function silently
    weakens, and nothing here would notice -- so the assumption is pinned by
    a witness rather than left to this comment.
    """
    if previous is None:
        return False
    return previous.digest == current.digest


def signature_to_manifest(signature: InputSignature) -> dict[str, Any]:
    """Render a signature for the manifest, JSON-round-trippable by construction."""
    return signature.to_manifest()


def signature_from_manifest(manifest: dict[str, Any] | None) -> InputSignature | None:
    """Recover a signature, or None when there is nothing trustworthy to recover.

    Returns None for absent, malformed, or version-mismatched payloads.  All
    three mean the same thing to the caller -- no usable prior state -- and
    `is_noop` turns that into "do the work".  Failing toward work is the only
    safe direction: the cost of a wrong None is one unnecessary build, and the
    cost of a wrong signature is a build that never happens.

    THIS DOCSTRING USED TO OVERSTATE WHAT THE CODE DID, which is the reason the
    validation below is shape-exact rather than approximate.  The first version
    accepted six malformed payloads while promising to reject them: `files` as
    ``True``/``False``/``-1`` (``bool`` subclasses ``int``, and nothing bounded
    the sign), `digest` as any non-empty string, and -- found while closing the
    class rather than the reported instances -- `version` as ``True`` or ``1.0``,
    both of which compare equal to ``1``.

    Stated exactly, because the honest scope matters more than the scary one: no
    WRONG NO-OP was reachable through those fields, since `is_noop` compares
    digests and a malformed digest cannot equal a real one.  What was reachable
    is worse in a quieter way -- a validator reporting "well-formed" over
    garbage, and a dataclass whose declared ``int`` could hold ``True``.  A gate
    that certifies what it did not check is the defect this module exists to
    argue against, so it does not get to have one.
    """
    if not isinstance(manifest, dict):
        return None
    payload = manifest.get(MANIFEST_KEY)
    if not isinstance(payload, dict):
        return None
    if not _is_plain_int(payload.get("version")):
        return None
    if payload.get("version") != SIGNATURE_VERSION:
        return None
    digest = payload.get("digest")
    files = payload.get("files")
    if not isinstance(digest, str) or _HEX64.match(digest) is None:
        return None
    if not _is_plain_int(files) or files < 0:
        return None
    return InputSignature(digest=digest, file_count=files)
