"""Local streaming corpus readers. No download policy or machine-specific paths."""

import bz2
import json
from pathlib import Path
from xml.etree import ElementTree
from engram_training import TrainingItem


def documents(path, *, format="text", max_chars=1000000):
    """Yield independent documents from UTF-8 text, JSONL or Wikipedia XML/bz2.

    Oversize documents raise; no silent truncation, network fetching or HTML
    interpretation. JSONL rows require text and may supply a string id. Wiki
    readers clear each completed page; decompression itself is not sandboxed.
    """
    path = Path(path)
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")

    def item(key, text):
        if (
            not isinstance(key, str)
            or not isinstance(text, str)
            or len(text) > max_chars
        ):
            raise ValueError("document requires a string ID and bounded text")
        return TrainingItem(key, text)

    if format == "text":
        paths = sorted(path.rglob("*.txt")) if path.is_dir() else [path]
        for source in paths:
            # Bound bytes before allocating the decoded document.
            with source.open("rb") as stream:
                data = stream.read(max_chars * 4 + 1)
            if len(data) > max_chars * 4:
                raise ValueError("text document exceeds size bound")
            key = source.relative_to(path).as_posix() if path.is_dir() else source.name
            yield item(key, data.decode("utf-8"))
    elif format == "jsonl":
        with path.open(encoding="utf-8") as stream:
            index = 0
            while line := stream.readline(max_chars * 6 + 1024):
                index += 1
                if not line.endswith("\n") and len(line) >= max_chars * 6 + 1024:
                    raise ValueError("JSONL row exceeds size bound")
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or "text" not in row:
                    raise ValueError("JSONL document needs a text field")
                yield item(row.get("id", str(index)), row["text"])
    elif format == "wiki":
        opener = bz2.open if path.suffix == ".bz2" else open
        with opener(path, "rb") as stream:
            context = ElementTree.iterparse(stream, events=("start", "end"))
            _, root = next(context)
            for event, element in context:
                if event == "end" and element.tag.rsplit("}", 1)[-1] == "page":
                    title, text, redirect = "", "", False
                    for child in element.iter():
                        name = child.tag.rsplit("}", 1)[-1]
                        if name == "title":
                            title = child.text or ""
                        elif name == "text":
                            text = child.text or ""
                        elif name == "redirect":
                            redirect = True
                    if text and not redirect:
                        yield item(title, text)
                    root.clear()
    else:
        raise ValueError("format must be text, jsonl or wiki")
