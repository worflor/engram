import bz2
import json
from pathlib import Path
import subprocess
import sys
import pytest
from engram_corpus import documents
from engram_network import WellNetwork

ROOT = Path(__file__).resolve().parents[2]


def test_text_and_jsonl_readers_are_ordered_and_bounded(tmp_path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    items = list(documents(tmp_path))
    assert [item.key for item in items] == ["a.txt", "b.txt"]
    assert all(item.sequence_start for item in items)
    with pytest.raises(ValueError):
        list(documents(tmp_path, max_chars=2))
    source = tmp_path / "items.jsonl"
    source.write_text(
        json.dumps({"id": "hello", "text": "world"}) + "\n\n", encoding="utf-8"
    )
    assert list(documents(source, format="jsonl"))[0].key == "hello"
    source.write_text('{"id":"missing"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        list(documents(source, format="jsonl"))


@pytest.mark.parametrize("compressed", [False, True])
def test_wiki_stream_ignores_redirects_and_namespaces(tmp_path, compressed):
    content = b"""<mediawiki xmlns="urn:test"><page><title>A</title><revision><text>alpha</text></revision></page><page><title>B</title><redirect title="A"/><revision><text>redirect</text></revision></page><page><title>C</title><revision><text>gamma</text></revision></page></mediawiki>"""
    path = tmp_path / ("wiki.xml.bz2" if compressed else "wiki.xml")
    path.write_bytes(bz2.compress(content) if compressed else content)
    items = list(documents(path, format="wiki"))
    assert [(item.key, item.payload) for item in items] == [
        ("A", "alpha"),
        ("C", "gamma"),
    ]


def test_actual_training_cli_resume_and_wrong_identity(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        "a river flows over a rock " * 12,
        "for key in values: print(key)\n" * 12,
        "a window opens above the garden " * 12,
    ]
    path.write_text(
        "\n".join(
            json.dumps({"id": str(i), "text": text}) for i, text in enumerate(rows)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "trained.engram"
    command = [
        sys.executable,
        str(ROOT / "tools/train_memory.py"),
        "--input",
        str(path),
        "--format",
        "jsonl",
        "--output",
        str(output),
        "--corpus-id",
        "authored-v1",
    ]
    subprocess.run(
        command + ["--dim", "8", "--limit", "2", "--checkpoint-every", "1"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    first = WellNetwork.load(output)
    assert first.memory._total_absorbed == first.memory.metadata["training_cursor"] == 2
    subprocess.run(
        command + ["--resume"], cwd=tmp_path, check=True, capture_output=True
    )
    resumed = WellNetwork.load(output)
    assert (
        resumed.memory._total_absorbed
        == resumed.memory.metadata["training_cursor"]
        == 3
    )
    assert not resumed.memory._well_transitions  # independent documents
    bad = subprocess.run(
        command + ["--resume", "--corpus-id", "wrong"],
        cwd=tmp_path,
        capture_output=True,
    )
    assert bad.returncode == 2
    assert b"identity/format differs" in bad.stderr
    assert WellNetwork.load(output).memory._total_absorbed == 3
