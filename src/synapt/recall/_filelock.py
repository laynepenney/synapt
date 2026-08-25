"""Cross-platform file locking.

Replaces direct fcntl.flock() calls with wrappers that work on
macOS, Linux, AND Windows. On Unix, uses fcntl.flock(). On Windows,
uses msvcrt.locking().

Three operations:
- lock_exclusive(f): Block until exclusive lock acquired on file object
- lock_exclusive_nb(fd): Non-blocking exclusive lock on file descriptor (int)
- unlock(f_or_fd): Release lock on file object or file descriptor
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    import msvcrt
    import os

    # msvcrt.locking() takes a MANDATORY byte-range lock (unlike POSIX
    # fcntl.flock, which is advisory): while it is held, OTHER handles cannot
    # even READ the locked bytes. Some of our lock files double as readable
    # stamps — ``_acquire_build_lock`` writes "pid … since …" at offset 0 so a
    # waiter can read WHO holds the lock — and locking those same bytes made the
    # read fail with PermissionError on Windows. So lock a single sentinel byte
    # far past any real content: the range need not exist in the file, every
    # caller locks the SAME byte so mutual exclusion is unchanged, and the
    # stamp at offset 0 stays readable. The current file position is saved and
    # restored so callers that write at offset 0 after locking are unaffected.
    # (Found on windows-latest CI, 2026-08-25.)
    _LOCK_OFFSET = 1 << 30

    def _lock_sentinel(fd: int, mode: int) -> None:
        prev = os.lseek(fd, 0, os.SEEK_CUR)
        os.lseek(fd, _LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(fd, mode, 1)
        finally:
            os.lseek(fd, prev, os.SEEK_SET)

    def lock_exclusive(f) -> None:
        """Acquire an exclusive lock on an open file (blocking)."""
        _lock_sentinel(f.fileno(), msvcrt.LK_LOCK)

    def lock_exclusive_nb(fd: int) -> None:
        """Acquire an exclusive lock on a file descriptor (non-blocking).

        Raises OSError (errno.EACCES or EDEADLOCK) if the lock cannot
        be acquired immediately.
        """
        _lock_sentinel(fd, msvcrt.LK_NBLCK)

    def unlock(f_or_fd) -> None:
        """Release a lock on a file object or file descriptor."""
        fd = f_or_fd if isinstance(f_or_fd, int) else f_or_fd.fileno()
        try:
            _lock_sentinel(fd, msvcrt.LK_UNLCK)
        except OSError:
            pass  # Already unlocked or fd closing

else:
    import fcntl

    def lock_exclusive(f) -> None:  # type: ignore[misc]
        """Acquire an exclusive lock on an open file (blocking)."""
        fcntl.flock(f, fcntl.LOCK_EX)

    def lock_exclusive_nb(fd: int) -> None:  # type: ignore[misc]
        """Acquire an exclusive lock on a file descriptor (non-blocking).

        Raises OSError (errno.EAGAIN/EWOULDBLOCK) if the lock cannot
        be acquired immediately.
        """
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(f_or_fd) -> None:  # type: ignore[misc]
        """Release a lock on a file object or file descriptor."""
        fd = f_or_fd if isinstance(f_or_fd, int) else f_or_fd.fileno()
        fcntl.flock(fd, fcntl.LOCK_UN)
