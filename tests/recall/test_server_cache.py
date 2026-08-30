from __future__ import annotations

from types import SimpleNamespace

from synapt.recall import server


def test_get_index_reuses_embedding_cache_without_false_missing_index(tmp_path, monkeypatch):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "recall.db").write_bytes(b"placeholder")

    fake_index = SimpleNamespace(chunks=[], _db=None)
    calls: list[tuple[object, bool]] = []

    def fake_load(directory, use_embeddings=False):
        calls.append((directory, use_embeddings))
        return fake_index

    monkeypatch.setattr(server, "_cached_index", None)
    monkeypatch.setattr(server, "_cached_mtime", 0.0)
    monkeypatch.setattr(server, "_cached_dir", None)
    monkeypatch.setattr(server, "_cached_has_embeddings", False)
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    monkeypatch.setattr(server.TranscriptIndex, "load", fake_load)

    first = server._get_index(use_embeddings=True)
    second = server._get_index(use_embeddings=True)

    assert first is fake_index
    assert second is fake_index
    assert calls == [(index_dir, True)]
    assert server._cached_has_embeddings is True


def test_get_index_reloads_when_another_writer_changes_the_wal(
    tmp_path, monkeypatch
):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "recall.db").write_bytes(b"base")
    wal = index_dir / "recall.db-wal"
    wal.write_bytes(b"first")
    loaded = [SimpleNamespace(chunks=[], _db=None) for _ in range(2)]
    calls = 0

    def fake_load(directory, use_embeddings=False):
        nonlocal calls
        value = loaded[calls]
        calls += 1
        return value

    monkeypatch.setattr(server, "_cached_index", None)
    monkeypatch.setattr(server, "_cached_mtime", ())
    monkeypatch.setattr(server, "_cached_dir", None)
    monkeypatch.setattr(server, "_cached_has_embeddings", False)
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    monkeypatch.setattr(server.TranscriptIndex, "load", fake_load)

    first = server._get_index(use_embeddings=False)
    wal.write_bytes(b"second-write")
    second = server._get_index(use_embeddings=False)

    assert first is loaded[0]
    assert second is loaded[1]
    assert calls == 2


def test_get_index_recognizes_a_sharded_index_without_legacy_files(
    tmp_path, monkeypatch
):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    (index_dir / "index.db").write_bytes(b"sharded-index")
    fake_index = SimpleNamespace(chunks=[], _db=None)
    monkeypatch.setattr(server, "_cached_index", None)
    monkeypatch.setattr(server, "_cached_mtime", ())
    monkeypatch.setattr(server, "_cached_dir", None)
    monkeypatch.setattr(server, "_cached_has_embeddings", False)
    monkeypatch.setattr(server, "project_index_dir", lambda: index_dir)
    monkeypatch.setattr(
        server.TranscriptIndex,
        "load",
        lambda directory, use_embeddings=False: fake_index,
    )

    assert server._get_index(use_embeddings=False) is fake_index
