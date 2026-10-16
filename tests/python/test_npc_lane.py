"""Lane ownership and transaction helpers; these are not real-game evidence."""

import json
import os
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from npc_lane import atomic_json, heavy_lock, read_json


def test_atomic_writer_retries_transient_windows_sharing_violation(
    tmp_path, monkeypatch
):
    original = os.replace
    calls = []

    def transient(source, target):
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("transient sharing violation")
        return original(source, target)

    monkeypatch.setattr(os, "replace", transient)
    monkeypatch.setattr("npc_lane.time.sleep", lambda _: None)
    target = tmp_path / "request.json"
    atomic_json(target, {"id": "new-transaction"})
    assert json.loads(target.read_text())["id"] == "new-transaction"
    assert len(calls) == 3 and not target.with_suffix(".tmp").exists()


def test_heavy_lock_releases_only_its_own_transaction(tmp_path):
    lock = tmp_path / "heavy.lock"
    with heavy_lock(lock, "owner-one"):
        assert json.loads((lock / "owner.json").read_text())["owner"] == "owner-one"
        with pytest.raises(RuntimeError):
            with heavy_lock(lock, "owner-two"):
                raise AssertionError("another owner entered the lane")
        assert lock.is_dir()
    assert not lock.exists()


def test_changed_lock_owner_is_not_removed(tmp_path):
    lock = tmp_path / "heavy.lock"
    with pytest.raises(RuntimeError, match="ownership changed"):
        with heavy_lock(lock, "owner-one"):
            (lock / "owner.json").write_text(json.dumps({"owner": "another-owner"}))
    assert lock.is_dir() and (lock / "owner.json").is_file()


def test_response_reader_retries_a_transient_sharing_violation(tmp_path, monkeypatch):
    target = tmp_path / "response.json"
    target.write_text('{"id":"ack"}')
    original = Path.read_text
    calls = []

    def transient(path, *args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise PermissionError("native rename handle still settling")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", transient)
    monkeypatch.setattr("npc_lane.time.sleep", lambda _: None)
    assert read_json(target)["id"] == "ack" and len(calls) == 2
