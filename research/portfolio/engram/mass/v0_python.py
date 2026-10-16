"""
v0_python.py — pure-Python MASS state machine. the reference
implementation. defines semantics; subsequent versions (numba,
fused) must match bytewise on prose regions.

this is a byte-level pushdown automaton. no tree, no allocation
per-byte, just state transitions.
"""

from __future__ import annotations


def mass_strip(text: str) -> str:
    """
    strip wikitext markup using a byte-level state machine.

    semantics:
      {{...}}   nested, skip entirely
      [[X|Y]]   emit Y (or X if no pipe)
      [[X]]     emit X
      <!-- -->  skip
      <tag...>  skip (including content for <ref>, <script>, <style>)
      &amp;     decode entity
    """
    out: list[str] = []
    i = 0
    n = len(text)
    brace_depth = 0

    # skip-to-end regions (when we see <ref>, we skip until </ref>)
    SKIP_TO_END = [
        ("<ref", "</ref>"),
        ("<script", "</script>"),
        ("<style", "</style>"),
        ("<!--", "-->"),
    ]
    ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
                "&apos;": "'", "&nbsp;": " "}

    while i < n:
        # template depth — highest priority
        if brace_depth > 0:
            if i + 1 < n and text[i] == "}" and text[i + 1] == "}":
                brace_depth -= 1
                i += 2
                continue
            if i + 1 < n and text[i] == "{" and text[i + 1] == "{":
                brace_depth += 1
                i += 2
                continue
            i += 1  # inside template, skip byte
            continue

        # template open
        if i + 1 < n and text[i] == "{" and text[i + 1] == "{":
            brace_depth = 1
            i += 2
            continue

        # link [[...]]
        if i + 1 < n and text[i] == "[" and text[i + 1] == "[":
            end = text.find("]]", i + 2)
            if end < 0:
                # unterminated; emit literally
                out.append(text[i])
                i += 1
                continue
            inner = text[i + 2:end]
            # strip off file/category prefixes that have no display text
            if inner.startswith(("File:", "Image:", "Category:")):
                i = end + 2
                continue
            # if piped, emit the last piece
            pipe = inner.rfind("|")
            if pipe >= 0:
                out.append(inner[pipe + 1:])
            else:
                out.append(inner)
            out.append(" ")
            i = end + 2
            continue

        # skip-to-end regions
        if text[i] == "<":
            matched = False
            for open_tag, close_tag in SKIP_TO_END:
                if text.startswith(open_tag, i):
                    end = text.find(close_tag, i + len(open_tag))
                    if end < 0:
                        # unterminated — skip to end of input
                        i = n
                    else:
                        i = end + len(close_tag)
                    matched = True
                    break
            if matched:
                continue
            # generic HTML tag: skip to next '>'
            end = text.find(">", i + 1)
            if end < 0:
                out.append(text[i])
                i += 1
            else:
                i = end + 1
            continue

        # entity decode
        if text[i] == "&":
            matched = False
            for entity, replacement in ENTITIES.items():
                if text.startswith(entity, i):
                    out.append(replacement)
                    i += len(entity)
                    matched = True
                    break
            if not matched:
                out.append(text[i])
                i += 1
            continue

        # table markers — skip line-leading pipes
        if text[i] == "|" and (i == 0 or text[i - 1] == "\n"):
            # skip to newline
            end = text.find("\n", i)
            i = end + 1 if end >= 0 else n
            continue

        # header markers '== foo =='
        if text[i] == "=" and (i == 0 or text[i - 1] == "\n"):
            # skip entire line
            end = text.find("\n", i)
            i = end + 1 if end >= 0 else n
            continue

        # default: emit byte
        out.append(text[i])
        i += 1

    return "".join(out)
