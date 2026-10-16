"""
v2_tokenize.py — MASS + inline tokenization.

emits (word_start, word_length) pairs in a single pass. no plaintext
intermediate. one state machine, two concurrent coordinates:
  - markup state (template depth, skip regions)
  - word accumulator (start offset + current length)

the output is a stream of word spans pointing back into the input
buffer. a downstream stage translates spans → vocab indices.

this is the first version that fuses two previously-separate stages
(strip + tokenize) into the same loop.
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

# same byte constants as v1
LBRACE = 0x7B; RBRACE = 0x7D
LBRACK = 0x5B; RBRACK = 0x5D
LT     = 0x3C; GT     = 0x3E
AMP    = 0x26; PIPE   = 0x7C
EQ     = 0x3D; NL     = 0x0A
SP     = 0x20
COLON  = 0x3A

# alphabet classifier: in prose, these bytes START or CONTINUE a word.
# everything else is a word separator. we treat tokens as [a-z]+
# (matching tokenize_simple in engram_glove_test).


@njit(cache=True, inline="always")
def _is_wordchar(b: int) -> bool:
    # a-z, A-Z, 0-9: counted as word char, we lowercase on the fly.
    # we allow digits so "gpu2" stays one token; the downstream vocab
    # trie can reject it if the GloVe vocab doesn't contain it.
    return (0x61 <= b <= 0x7A) or (0x41 <= b <= 0x5A) or (0x30 <= b <= 0x39)


@njit(cache=True, inline="always")
def _lower(b: int) -> int:
    if 0x41 <= b <= 0x5A:
        return b + 0x20
    return b


def _pat(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)

_REF_O = _pat("<ref");    _REF_C = _pat("</ref>")
_SCR_O = _pat("<script"); _SCR_C = _pat("</script>")
_STY_O = _pat("<style");  _STY_C = _pat("</style>")
_CMT_O = _pat("<!--");    _CMT_C = _pat("-->")
_FILE  = _pat("File:"); _IMAGE = _pat("Image:"); _CATEGORY = _pat("Category:")


@njit(cache=True, boundscheck=False)
def _match_at(buf, i, pat):
    n = pat.shape[0]
    if i + n > buf.shape[0]:
        return False
    for k in range(n):
        if buf[i + k] != pat[k]:
            return False
    return True


@njit(cache=True, boundscheck=False)
def _find_at(buf, start, pat):
    n = buf.shape[0]; m = pat.shape[0]
    if m == 0: return start
    end = n - m + 1
    if end <= start: return -1
    first = pat[0]
    for i in range(start, end):
        if buf[i] == first:
            hit = True
            for k in range(1, m):
                if buf[i + k] != pat[k]:
                    hit = False; break
            if hit: return i
    return -1


@njit(cache=True, boundscheck=False)
def mass_tokenize_bytes(
    buf: np.ndarray,
    word_buf: np.ndarray,      # pre-allocated lowered-byte storage for words
    word_off: np.ndarray,      # int32[N] — start offset of each word in word_buf
    word_len: np.ndarray,      # int32[N] — length of each word
    ref_o, ref_c, scr_o, scr_c, sty_o, sty_c, cmt_o, cmt_c,
    file_p, image_p, category_p,
) -> tuple:
    """
    strip + tokenize in one pass. returns (n_words, word_buf_used).

    each discovered word lives at word_buf[word_off[k] : word_off[k]+word_len[k]]
    and is already lowercased. no python allocations inside.
    """
    n = buf.shape[0]
    i = 0
    brace_depth = 0

    n_words = 0
    wb_cursor = 0  # current end of word_buf
    cur_word_start = -1  # if >= 0, we're accumulating a word
    cur_word_len = 0

    max_words = word_off.shape[0]

    while i < n:
        b = buf[i]

        # inside a template: skip markup, but any word in progress
        # terminates because template break means context break.
        if brace_depth > 0:
            if cur_word_start >= 0:
                # commit word
                if n_words < max_words and cur_word_len > 1:
                    word_off[n_words] = cur_word_start
                    word_len[n_words] = cur_word_len
                    n_words += 1
                cur_word_start = -1
                cur_word_len = 0
            if b == RBRACE and i + 1 < n and buf[i + 1] == RBRACE:
                brace_depth -= 1
                i += 2
                continue
            if b == LBRACE and i + 1 < n and buf[i + 1] == LBRACE:
                brace_depth += 1
                i += 2
                continue
            i += 1
            continue

        # template open
        if b == LBRACE and i + 1 < n and buf[i + 1] == LBRACE:
            if cur_word_start >= 0:
                if n_words < max_words and cur_word_len > 1:
                    word_off[n_words] = cur_word_start
                    word_len[n_words] = cur_word_len
                    n_words += 1
                cur_word_start = -1
                cur_word_len = 0
            brace_depth = 1
            i += 2
            continue

        # link [[...]] — emit the display text through the tokenizer
        if b == LBRACK and i + 1 < n and buf[i + 1] == LBRACK:
            if cur_word_start >= 0:
                if n_words < max_words and cur_word_len > 1:
                    word_off[n_words] = cur_word_start
                    word_len[n_words] = cur_word_len
                    n_words += 1
                cur_word_start = -1
                cur_word_len = 0
            # find ]]
            close = -1
            j = i + 2
            while j < n - 1:
                if buf[j] == RBRACK and buf[j + 1] == RBRACK:
                    close = j; break
                j += 1
            if close < 0:
                i += 1; continue
            inner_s = i + 2; inner_e = close
            if (_match_at(buf, inner_s, file_p)
                    or _match_at(buf, inner_s, image_p)
                    or _match_at(buf, inner_s, category_p)):
                i = close + 2
                continue
            # last pipe segment
            pipe = -1
            for k in range(inner_e - 1, inner_s - 1, -1):
                if buf[k] == PIPE:
                    pipe = k; break
            s = pipe + 1 if pipe >= 0 else inner_s
            e = inner_e
            # run a mini-tokenize over [s, e)
            ws = -1; wl = 0
            for k in range(s, e):
                bb = buf[k]
                if _is_wordchar(bb):
                    if ws < 0:
                        ws = wb_cursor
                    word_buf[wb_cursor] = _lower(bb)
                    wb_cursor += 1
                    wl += 1
                else:
                    if ws >= 0 and wl > 1 and n_words < max_words:
                        word_off[n_words] = ws
                        word_len[n_words] = wl
                        n_words += 1
                    ws = -1; wl = 0
            if ws >= 0 and wl > 1 and n_words < max_words:
                word_off[n_words] = ws
                word_len[n_words] = wl
                n_words += 1
            i = close + 2
            continue

        # <... handling
        if b == LT:
            if cur_word_start >= 0:
                if n_words < max_words and cur_word_len > 1:
                    word_off[n_words] = cur_word_start
                    word_len[n_words] = cur_word_len
                    n_words += 1
                cur_word_start = -1
                cur_word_len = 0
            if _match_at(buf, i, ref_o):
                e = _find_at(buf, i + ref_o.shape[0], ref_c)
                i = e + ref_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, scr_o):
                e = _find_at(buf, i + scr_o.shape[0], scr_c)
                i = e + scr_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, sty_o):
                e = _find_at(buf, i + sty_o.shape[0], sty_c)
                i = e + sty_c.shape[0] if e >= 0 else n; continue
            if _match_at(buf, i, cmt_o):
                e = _find_at(buf, i + cmt_o.shape[0], cmt_c)
                i = e + cmt_c.shape[0] if e >= 0 else n; continue
            # generic tag
            j = i + 1
            while j < n and buf[j] != GT:
                j += 1
            i = j + 1 if j < n else n
            continue

        # line-leading pipe/equals: skip line
        if (b == PIPE or b == EQ) and (i == 0 or buf[i - 1] == NL):
            if cur_word_start >= 0:
                if n_words < max_words and cur_word_len > 1:
                    word_off[n_words] = cur_word_start
                    word_len[n_words] = cur_word_len
                    n_words += 1
                cur_word_start = -1
                cur_word_len = 0
            j = i
            while j < n and buf[j] != NL:
                j += 1
            i = j + 1 if j < n else n
            continue

        # normal prose byte: either continue or break a word
        if _is_wordchar(b):
            if cur_word_start < 0:
                cur_word_start = wb_cursor
                cur_word_len = 0
            word_buf[wb_cursor] = _lower(b)
            wb_cursor += 1
            cur_word_len += 1
        else:
            if cur_word_start >= 0:
                if n_words < max_words and cur_word_len > 1:
                    word_off[n_words] = cur_word_start
                    word_len[n_words] = cur_word_len
                    n_words += 1
                cur_word_start = -1
                cur_word_len = 0
        i += 1

    # flush trailing word
    if cur_word_start >= 0 and cur_word_len > 1 and n_words < max_words:
        word_off[n_words] = cur_word_start
        word_len[n_words] = cur_word_len
        n_words += 1

    return n_words, wb_cursor


def mass_tokenize(text: str, max_words: int = 30000):
    """python-friendly wrapper returning a list of word strings."""
    buf = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    word_buf = np.empty(buf.shape[0] + 16, dtype=np.uint8)
    word_off = np.empty(max_words, dtype=np.int32)
    word_len = np.empty(max_words, dtype=np.int32)
    n_words, _ = mass_tokenize_bytes(
        buf, word_buf, word_off, word_len,
        _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
        _FILE, _IMAGE, _CATEGORY,
    )
    words = []
    for k in range(n_words):
        s = word_off[k]; e = s + word_len[k]
        words.append(word_buf[s:e].tobytes().decode("utf-8", errors="replace"))
    return words
