"""Blind HTTP relay for opaque encrypted recall archives.

The relay stores bytes by content digest and logical clock.  It has no key,
does not import the archive or crypto modules, and cannot inspect memory
semantics.  Conflict handling is deliberately last-write-wins by clock for
Spike A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


MAX_OBJECT_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class StoredObject:
    object_id: str
    logical_clock: int
    payload: bytes


class RelayStore:
    """Append opaque objects and expose the newest logical clock."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.root, 0o700)
        self._latest_path = self.root / "latest.json"
        self._write_lock = threading.Lock()

    def _latest_metadata(self) -> dict | None:
        if not self._latest_path.exists():
            return None
        return json.loads(self._latest_path.read_text(encoding="utf-8"))

    def put(self, payload: bytes, *, logical_clock: int) -> StoredObject:
        if not payload:
            raise ValueError("ciphertext payload must not be empty")
        if logical_clock < 1:
            raise ValueError("logical clock must be positive")
        with self._write_lock:
            current = self._latest_metadata()
            if current is not None and logical_clock <= int(current["logical_clock"]):
                raise ValueError("logical clock must advance beyond the latest object")

            digest = hashlib.sha256(payload).hexdigest()
            object_id = f"{logical_clock:020d}-{digest}"
            object_path = self.root / f"{object_id}.age"
            self._atomic_write(object_path, payload)
            metadata = {
                "object_id": object_id,
                "logical_clock": logical_clock,
                "file": object_path.name,
            }
            self._atomic_write(
                self._latest_path,
                json.dumps(metadata, sort_keys=True).encode("utf-8"),
            )
            return StoredObject(object_id, logical_clock, payload)

    def latest(self) -> StoredObject:
        metadata = self._latest_metadata()
        if metadata is None:
            raise FileNotFoundError("relay has no objects")
        path = self.root / metadata["file"]
        return StoredObject(
            str(metadata["object_id"]),
            int(metadata["logical_clock"]),
            path.read_bytes(),
        )

    def _atomic_write(self, path: Path, payload: bytes) -> None:
        fd, temporary = tempfile.mkstemp(prefix=".relay-", dir=self.root)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise


class RelayHandler(BaseHTTPRequestHandler):
    server: "RelayHTTPServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/health" and not parsed.query and not parsed.fragment:
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path != "/objects/latest" or parsed.query or parsed.fragment:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            item = self.server.store.latest()
        except FileNotFoundError:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "no objects"})
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(item.payload)))
        self.send_header("X-Synapt-Object-Id", item.object_id)
        self.send_header("X-Synapt-Logical-Clock", str(item.logical_clock))
        self.end_headers()
        self.wfile.write(item.payload)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path != "/objects" or parsed.fragment:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        try:
            query = parse_qs(parsed.query, strict_parsing=True)
            raw_clocks = query["logical_clock"]
            if len(raw_clocks) != 1:
                raise ValueError
            logical_clock = int(raw_clocks[0])
            length = int(self.headers.get("Content-Length", "-1"))
            if length < 1 or length > MAX_OBJECT_BYTES:
                raise ValueError
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError
            item = self.server.store.put(payload, logical_clock=logical_clock)
        except (KeyError, ValueError):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid object"})
            return
        self._send_json(
            HTTPStatus.CREATED,
            {"object_id": item.object_id, "logical_clock": item.logical_clock},
        )

    def log_message(self, format: str, *args: object) -> None:
        # Avoid logging object paths or payload-adjacent request details.
        return

    def _send_json(self, status: HTTPStatus, payload: dict) -> None:
        raw = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class RelayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], store: RelayStore):
        super().__init__(address, RelayHandler)
        self.store = store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Blind encrypted archive relay")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--store", type=Path, default=Path("/relay-data"))
    args = parser.parse_args(argv)
    server = RelayHTTPServer((args.host, args.port), RelayStore(args.store))
    server.serve_forever()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
