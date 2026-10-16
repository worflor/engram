"""
v4_tables.py — v3 + proper wiki table handling + edge cases found by
differential analysis on real enwiki.

additions over v3:
  - `{| ... |}` table state: enter TABLE mode on `{|`, exit on `|}`.
    inside TABLE: skip attribute lines, extract cell content.
  - `||` and `!!` mid-line cell separators also skip attribute prefixes
  - `<nowiki>...</nowiki>` skip (parser suppression)
  - `'''bold'''` and `''italic''` tick removal
  - self-closing tags `<ref .../>`, `<br/>` etc
  - `__MAGIC__` word skipping

this is no longer "just a state machine" — it's a real grammar. but
it's still a single allocation-free numba pass.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*a, **kw):
        def deco(f): return f
        return deco if a and callable(a[0]) else deco

from v2_tokenize import (
    LBRACE, RBRACE, LBRACK, RBRACK, LT, GT, AMP, PIPE, EQ, NL, SP,
    _match_at, _find_at, _is_wordchar, _lower,
)
from v3_fused import (
    _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
    _FILE, _IMAGE, _CATEGORY,
    _lookup, build_vocab_table,
)


BANG = 0x21       # '!'
TICK = 0x27       # "'"
UNDERSCORE = 0x5F # '_'
SLASH = 0x2F      # '/'


def _pat(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)

_NOWIKI_O = _pat("<nowiki>")
_NOWIKI_C = _pat("</nowiki>")
_PRE_O    = _pat("<pre>")
_PRE_C    = _pat("</pre>")


@njit(cache=True, boundscheck=False, nogil=True)
def mass_fused_v4(
    buf: np.ndarray,
    word_buf: np.ndarray,
    out_idx: np.ndarray,
    vocab_buf, vocab_off, vocab_pos, table,
    ref_o, ref_c, scr_o, scr_c, sty_o, sty_c, cmt_o, cmt_c,
    file_p, image_p, category_p,
    nowiki_o, nowiki_c, pre_o, pre_c,
) -> int:
    """
    full fused MASS v4. state:
      brace_depth        — nested `{{...}}` template depth
      table_depth        — nested `{|...|}` table depth (usually 0 or 1)
      ws, wl             — current word's start in word_buf and length

    output: vocab indices emitted when words close, with in-vocab
    lookup via open-addressing hash probe.
    """
    n = buf.shape[0]
    i = 0
    brace_depth = 0
    table_depth = 0

    n_out = 0
    wb_cursor = 0
    ws = -1
    wl = 0
    max_out = out_idx.shape[0]

    while i < n:
        b = buf[i]

        # inside a template: skip everything, track nesting
        if brace_depth > 0:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            if b == RBRACE and i + 1 < n and buf[i + 1] == RBRACE:
                brace_depth -= 1; i += 2; continue
            if b == LBRACE and i + 1 < n and buf[i + 1] == LBRACE:
                brace_depth += 1; i += 2; continue
            i += 1
            continue

        # template open
        if b == LBRACE and i + 1 < n and buf[i + 1] == LBRACE:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            brace_depth = 1
            i += 2
            continue

        # table open: `{|` at line start OR anywhere
        if b == LBRACE and i + 1 < n and buf[i + 1] == PIPE:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            table_depth += 1
            # skip to next newline (the rest of the table-open line is
            # attributes, no content).
            j = i
            while j < n and buf[j] != NL:
                j += 1
            i = j + 1 if j < n else n
            continue

        # table close: `|}` at line start
        if (b == PIPE and i + 1 < n and buf[i + 1] == RBRACE
                and (i == 0 or buf[i - 1] == NL)):
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            if table_depth > 0:
                table_depth -= 1
            # skip rest of line
            j = i
            while j < n and buf[j] != NL:
                j += 1
            i = j + 1 if j < n else n
            continue

        # single-bracket external link: [http://... display text]
        # mwparser strips these entirely. we do the same — skip the
        # url but extract any display text after the first space.
        if b == LBRACK and i + 1 < n and buf[i + 1] != LBRACK:
            # check if it looks like a URL: 'http', 'https', 'ftp', '//'
            is_url = False
            if i + 5 < n:
                if (buf[i+1] == 0x68 and buf[i+2] == 0x74
                        and buf[i+3] == 0x74 and buf[i+4] == 0x70):  # 'http'
                    is_url = True
                elif buf[i+1] == SLASH and buf[i+2] == SLASH:  # '//'
                    is_url = True
                elif (i + 4 < n and buf[i+1] == 0x66 and buf[i+2] == 0x74
                      and buf[i+3] == 0x70):  # 'ftp'
                    is_url = True
            if is_url:
                if ws >= 0:
                    if wl > 1:
                        idx = _lookup(word_buf, ws, wl, vocab_buf,
                                      vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx; n_out += 1
                    ws = -1; wl = 0
                # find closing ]
                close = -1
                j = i + 1
                while j < n:
                    if buf[j] == RBRACK:
                        close = j; break
                    j += 1
                if close < 0:
                    i += 1; continue
                # extract words after first space (display text)
                space = -1
                for k in range(i + 1, close):
                    if buf[k] == SP:
                        space = k; break
                if space >= 0:
                    iws = -1; iwl = 0
                    for k in range(space + 1, close):
                        bb = buf[k]
                        if _is_wordchar(bb):
                            if iws < 0:
                                iws = wb_cursor
                            word_buf[wb_cursor] = _lower(bb)
                            wb_cursor += 1; iwl += 1
                        else:
                            if iws >= 0 and iwl > 1:
                                idx = _lookup(word_buf, iws, iwl, vocab_buf,
                                              vocab_off, vocab_pos, table)
                                if idx >= 0 and n_out < max_out:
                                    out_idx[n_out] = idx; n_out += 1
                            iws = -1; iwl = 0
                    if iws >= 0 and iwl > 1:
                        idx = _lookup(word_buf, iws, iwl, vocab_buf,
                                      vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx; n_out += 1
                i = close + 1; continue

        # link [[...]] — emit the display text's words
        if b == LBRACK and i + 1 < n and buf[i + 1] == LBRACK:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            close = -1
            j = i + 2
            while j < n - 1:
                if buf[j] == RBRACK and buf[j + 1] == RBRACK:
                    close = j; break
                j += 1
            if close < 0:
                i += 1; continue
            inner_s = i + 2; inner_e = close
            # for File/Image/Category links, mwparser keeps the display
            # text after the last pipe (for files: the caption; for
            # categories: the sort key or category name). extract that.
            is_special = (_match_at(buf, inner_s, file_p)
                          or _match_at(buf, inner_s, image_p)
                          or _match_at(buf, inner_s, category_p))
            if is_special:
                # find colon (prefix separator), start extraction after it
                colon = -1
                for k in range(inner_s, min(inner_s + 16, inner_e)):
                    if buf[k] == 0x3A:  # ':'
                        colon = k; break
                extract_start = colon + 1 if colon >= 0 else inner_s
                # tokenize from [extract_start, inner_e)
                iws = -1; iwl = 0
                for k in range(extract_start, inner_e):
                    bb = buf[k]
                    # skip content inside | segments for files
                    # (thumb, alt=, size=) — heuristic: after first |,
                    # every token segment separated by | could be noise
                    # or caption. extract all for now; the vocab filter
                    # handles nonsense.
                    if _is_wordchar(bb):
                        if iws < 0:
                            iws = wb_cursor
                        word_buf[wb_cursor] = _lower(bb)
                        wb_cursor += 1; iwl += 1
                    else:
                        if iws >= 0 and iwl > 1:
                            idx = _lookup(word_buf, iws, iwl, vocab_buf,
                                          vocab_off, vocab_pos, table)
                            if idx >= 0 and n_out < max_out:
                                out_idx[n_out] = idx; n_out += 1
                        iws = -1; iwl = 0
                if iws >= 0 and iwl > 1:
                    idx = _lookup(word_buf, iws, iwl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                i = close + 2; continue
            pipe = -1
            for k in range(inner_e - 1, inner_s - 1, -1):
                if buf[k] == PIPE:
                    pipe = k; break
            ss = pipe + 1 if pipe >= 0 else inner_s
            ee = inner_e
            iws = -1; iwl = 0
            for k in range(ss, ee):
                bb = buf[k]
                if _is_wordchar(bb):
                    if iws < 0:
                        iws = wb_cursor
                    word_buf[wb_cursor] = _lower(bb)
                    wb_cursor += 1; iwl += 1
                else:
                    if iws >= 0 and iwl > 1:
                        idx = _lookup(word_buf, iws, iwl, vocab_buf,
                                      vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx; n_out += 1
                    iws = -1; iwl = 0
            if iws >= 0 and iwl > 1:
                idx = _lookup(word_buf, iws, iwl, vocab_buf,
                              vocab_off, vocab_pos, table)
                if idx >= 0 and n_out < max_out:
                    out_idx[n_out] = idx; n_out += 1
            i = close + 2
            continue

        # HTML-style tags
        if b == LT:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            if _match_at(buf, i, ref_o):
                # ref tags: mwparser's strip_code KEEPS the ref content
                # (just removes the tag markers). so we do the same —
                # skip just the opening tag, let content be tokenized
                # normally, let the generic `<...>` handler eat the
                # closing `</ref>`. this recovers author names and
                # citation prose that the previous skip-to-end version
                # was throwing away.
                j = i + 1
                # detect self-close first: scan to > or />
                self_close = False
                while j < n and buf[j] != GT:
                    if buf[j] == SLASH and j + 1 < n and buf[j + 1] == GT:
                        self_close = True
                        break
                    j += 1
                if self_close:
                    i = j + 2  # past />
                else:
                    i = j + 1 if j < n else n  # past >
                continue
            if _match_at(buf, i, scr_o):
                e = _find_at(buf, i + scr_o.shape[0], scr_c)
                i = e + scr_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, sty_o):
                e = _find_at(buf, i + sty_o.shape[0], sty_c)
                i = e + sty_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, cmt_o):
                e = _find_at(buf, i + cmt_o.shape[0], cmt_c)
                i = e + cmt_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, nowiki_o):
                e = _find_at(buf, i + nowiki_o.shape[0], nowiki_c)
                i = e + nowiki_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, pre_o):
                e = _find_at(buf, i + pre_o.shape[0], pre_c)
                i = e + pre_c.shape[0] if e >= 0 else n; continue
            # generic tag: scan to '>'
            j = i + 1
            while j < n and buf[j] != GT:
                j += 1
            i = j + 1 if j < n else n
            continue

        # line-leading handlers
        is_line_start = (i == 0 or buf[i - 1] == NL)

        if is_line_start:
            # `!` at line start: table header. skip until next `|` on line.
            if b == BANG:
                if ws >= 0:
                    if wl > 1:
                        idx = _lookup(word_buf, ws, wl, vocab_buf,
                                      vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx; n_out += 1
                    ws = -1; wl = 0
                # find `|` or NL
                j = i + 1
                while j < n and buf[j] != NL and buf[j] != PIPE:
                    j += 1
                # if found `|`, skip past attributes
                if j < n and buf[j] == PIPE:
                    i = j + 1
                else:
                    i = j + 1 if j < n else n
                continue

            # `|` at line start (but not `|-`, `|}`, `|+`): cell. skip attrs.
            if b == PIPE:
                if ws >= 0:
                    if wl > 1:
                        idx = _lookup(word_buf, ws, wl, vocab_buf,
                                      vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx; n_out += 1
                    ws = -1; wl = 0
                # `|-`, `|+`, `|}` → skip whole line
                if i + 1 < n and (buf[i + 1] == 0x2D      # '-'
                                  or buf[i + 1] == 0x2B  # '+'
                                  or buf[i + 1] == RBRACE):
                    j = i
                    while j < n and buf[j] != NL:
                        j += 1
                    i = j + 1 if j < n else n
                    continue
                # `|...|content` — skip attributes up to next `|`
                j = i + 1
                while j < n and buf[j] != NL and buf[j] != PIPE:
                    j += 1
                if j < n and buf[j] == PIPE:
                    i = j + 1
                else:
                    # no `|` on this line — whole line was cell content,
                    # just skip the leading `|`
                    i += 1
                continue

            # `=` at line start: header line, skip
            if b == EQ:
                if ws >= 0:
                    if wl > 1:
                        idx = _lookup(word_buf, ws, wl, vocab_buf,
                                      vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx; n_out += 1
                    ws = -1; wl = 0
                j = i
                while j < n and buf[j] != NL:
                    j += 1
                i = j + 1 if j < n else n
                continue

            # `__MAGIC__` words like __TOC__, __NOTOC__
            if b == UNDERSCORE and i + 1 < n and buf[i + 1] == UNDERSCORE:
                j = i + 2
                # scan uppercase letters/underscores until `__`
                while j < n and buf[j] != NL:
                    if buf[j] == UNDERSCORE and j + 1 < n and buf[j + 1] == UNDERSCORE:
                        j += 2
                        break
                    j += 1
                i = j
                continue

        # mid-line `||` cell separator (inside table): skip attributes
        if (b == PIPE and i + 1 < n and buf[i + 1] == PIPE):
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            # find next `|` or NL
            j = i + 2
            while j < n and buf[j] != NL and buf[j] != PIPE:
                j += 1
            if j < n and buf[j] == PIPE:
                i = j + 1
            else:
                i = i + 2
            continue

        # mid-line `!!` (table header cell separator)
        if (b == BANG and i + 1 < n and buf[i + 1] == BANG):
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            j = i + 2
            while j < n and buf[j] != NL and buf[j] != PIPE:
                j += 1
            if j < n and buf[j] == PIPE:
                i = j + 1
            else:
                i = i + 2
            continue

        # `'''` bold / `''` italic — just skip the ticks
        if b == TICK:
            # peek: 3 ticks = bold, 2 = italic, 1 = apostrophe (keep)
            if i + 2 < n and buf[i + 1] == TICK and buf[i + 2] == TICK:
                i += 3
                continue
            if i + 1 < n and buf[i + 1] == TICK:
                i += 2
                continue
            # single tick: part of a word (e.g., "it's")? treat as separator
            # to match how regex [a-z]+ would split "it's" → "it", "s"
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
            i += 1
            continue

        # normal prose: wordchar or separator
        if _is_wordchar(b):
            if ws < 0:
                ws = wb_cursor
                wl = 0
            word_buf[wb_cursor] = _lower(b)
            wb_cursor += 1
            wl += 1
        else:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx; n_out += 1
                ws = -1; wl = 0
        i += 1

    if ws >= 0 and wl > 1:
        idx = _lookup(word_buf, ws, wl, vocab_buf,
                      vocab_off, vocab_pos, table)
        if idx >= 0 and n_out < max_out:
            out_idx[n_out] = idx; n_out += 1

    return n_out


class FusedIngestV4:
    def __init__(self, word_to_idx: dict[str, int]):
        (self.vocab_buf, self.vocab_off,
         self.vocab_pos, self.table) = build_vocab_table(word_to_idx)

    def extract(self, text: str, max_out: int = 50000) -> np.ndarray:
        buf = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        word_buf = np.empty(buf.shape[0] + 16, dtype=np.uint8)
        out_idx = np.empty(max_out, dtype=np.int32)
        n = mass_fused_v4(
            buf, word_buf, out_idx,
            self.vocab_buf, self.vocab_off, self.vocab_pos, self.table,
            _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
            _FILE, _IMAGE, _CATEGORY,
            _NOWIKI_O, _NOWIKI_C, _PRE_O, _PRE_C,
        )
        return out_idx[:n].copy()
