"""Encrypt-before-transport client over recall's shipped archive boundary."""

from __future__ import annotations

import json
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from synapt.recall.archive import export_recall_archive, import_recall_archive

from .crypto import decrypt_archive, encrypt_archive


@dataclass(frozen=True)
class SyncReceipt:
    object_id: str
    logical_clock: int


def upload_ciphertext(
    relay_url: str, payload: bytes, logical_clock: int
) -> dict[str, object]:
    query = urllib.parse.urlencode({"logical_clock": logical_clock})
    request = urllib.request.Request(
        f"{relay_url.rstrip('/')}/objects?{query}",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.HTTPError) as exc:
        raise RuntimeError("encrypted archive upload failed") from exc


def download_latest_ciphertext(
    relay_url: str,
) -> tuple[bytes, dict[str, object]]:
    request = urllib.request.Request(
        f"{relay_url.rstrip('/')}/objects/latest",
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            metadata: dict[str, object] = {
                "object_id": response.headers["X-Synapt-Object-Id"],
                "logical_clock": int(response.headers["X-Synapt-Logical-Clock"]),
            }
            return response.read(), metadata
    except (OSError, TypeError, ValueError, urllib.error.HTTPError) as exc:
        raise RuntimeError("encrypted archive download failed") from exc


def push_project_archive(
    project_dir: Path,
    *,
    relay_url: str,
    recipient: str,
    logical_clock: int,
    archive_path: Path | None = None,
) -> SyncReceipt:
    """Export, encrypt, and upload one portable recall archive."""
    project_dir = project_dir.resolve()
    if archive_path is not None:
        exported_path, _manifest = export_recall_archive(project_dir, archive_path)
        ciphertext = encrypt_archive(exported_path.read_bytes(), recipient)
        raw = upload_ciphertext(relay_url, ciphertext, logical_clock)
    else:
        with tempfile.TemporaryDirectory(prefix="synapt-encrypted-export-") as tmp:
            path = Path(tmp) / "recall.synapt-archive"
            exported_path, _manifest = export_recall_archive(project_dir, path)
            ciphertext = encrypt_archive(exported_path.read_bytes(), recipient)
            raw = upload_ciphertext(relay_url, ciphertext, logical_clock)
    return SyncReceipt(str(raw["object_id"]), int(raw["logical_clock"]))


def pull_project_archive(
    project_dir: Path,
    *,
    relay_url: str,
    identity: str,
) -> SyncReceipt:
    """Download, decrypt, and merge-import the newest portable archive."""
    project_dir = project_dir.resolve()
    ciphertext, metadata = download_latest_ciphertext(relay_url)
    plaintext = decrypt_archive(ciphertext, identity)
    with tempfile.TemporaryDirectory(prefix="synapt-encrypted-import-") as tmp:
        archive_path = Path(tmp) / "recall.synapt-archive"
        archive_path.write_bytes(plaintext)
        import_recall_archive(project_dir, archive_path, mode="merge")
    return SyncReceipt(
        str(metadata["object_id"]),
        int(metadata["logical_clock"]),
    )
