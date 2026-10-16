"""
v1_numba.py — byte-array MASS state machine, numba-JIT'd.

differences from v0:
  - input is np.uint8 array, not string (no python object overhead)
  - pre-allocated np.uint8 output buffer (no list-append allocation)
  - all string ops replaced with byte comparisons / manual scans
  - compiled once; subsequent calls run at native speed, GIL-free

semantic target: match v0 output to within ±1% length on average,
bytewise-identical on prose regions that neither v0 nor v1 touch.
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit, types
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    def njit(*a, **kw):
        def deco(f): return f
        return deco if a and callable(a[0]) else deco


# byte constants (so njit can fold them)
LBRACE = 0x7B       # {
RBRACE = 0x7D       # }
LBRACK = 0x5B       # [
RBRACK = 0x5D       # ]
LT     = 0x3C       # <
GT     = 0x3E       # >
AMP    = 0x26       # &
PIPE   = 0x7C       # |
EQ     = 0x3D       # =
NL     = 0x0A       # \n
SP     = 0x20       # space
COLON  = 0x3A       # :
SLASH  = 0x2F       # /


# pre-encoded skip-to-end markers as uint8 arrays (numba-friendly)
def _b(s: str) -> np.ndarray:
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)

_REF_OPEN    = _b("<ref")
_REF_CLOSE   = _b("</ref>")
_SCRIPT_OPEN = _b("<script")
_SCRIPT_CLOSE= _b("</script>")
_STYLE_OPEN  = _b("<style")
_STYLE_CLOSE = _b("</style>")
_COMMENT_O   = _b("<!--")
_COMMENT_C   = _b("-->")
_FILE_P      = _b("File:")
_IMAGE_P     = _b("Image:")
_CATEGORY_P  = _b("Category:")

# entities (short list; covers >99% of wikitext)
_AMP_AMP  = _b("&amp;")
_AMP_LT   = _b("&lt;")
_AMP_GT   = _b("&gt;")
_AMP_QUOT = _b("&quot;")
_AMP_APOS = _b("&apos;")
_AMP_NBSP = _b("&nbsp;")


@njit(cache=True, boundscheck=False)
def _match_at(buf: np.ndarray, i: int, pat: np.ndarray) -> bool:
    """true if `pat` is a prefix of `buf[i:]`."""
    n = pat.shape[0]
    if i + n > buf.shape[0]:
        return False
    for k in range(n):
        if buf[i + k] != pat[k]:
            return False
    return True


@njit(cache=True, boundscheck=False)
def _find_at(buf: np.ndarray, start: int, pat: np.ndarray) -> int:
    """first index ≥ start where `pat` occurs in buf, else -1.
    simple scan; not KMP — pat is always tiny (≤ 10 bytes) so it
    doesn't matter."""
    n = buf.shape[0]
    m = pat.shape[0]
    if m == 0:
        return start
    end = n - m + 1
    if end <= start:
        return -1
    first = pat[0]
    for i in range(start, end):
        if buf[i] == first:
            hit = True
            for k in range(1, m):
                if buf[i + k] != pat[k]:
                    hit = False
                    break
            if hit:
                return i
    return -1


@njit(cache=True, boundscheck=False)
def mass_strip_bytes(
    buf: np.ndarray,
    out: np.ndarray,
    ref_open:   np.ndarray, ref_close:   np.ndarray,
    script_open:np.ndarray, script_close:np.ndarray,
    style_open: np.ndarray, style_close: np.ndarray,
    comment_o:  np.ndarray, comment_c:   np.ndarray,
    file_p:     np.ndarray, image_p:     np.ndarray, category_p: np.ndarray,
    amp_amp:    np.ndarray, amp_lt:      np.ndarray, amp_gt:     np.ndarray,
    amp_quot:   np.ndarray, amp_apos:    np.ndarray, amp_nbsp:   np.ndarray,
) -> int:
    """
    strip wikitext markup from `buf` into `out`. returns number of
    bytes written. out must be at least len(buf) bytes (upper bound).
    """
    n = buf.shape[0]
    i = 0
    o = 0
    brace_depth = 0

    while i < n:
        b = buf[i]

        # inside a template: skip, track nesting
        if brace_depth > 0:
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
            brace_depth = 1
            i += 2
            continue

        # link [[...]]
        if b == LBRACK and i + 1 < n and buf[i + 1] == LBRACK:
            # scan for ]]
            close = -1
            j = i + 2
            while j < n - 1:
                if buf[j] == RBRACK and buf[j + 1] == RBRACK:
                    close = j
                    break
                j += 1
            if close < 0:
                out[o] = b; o += 1
                i += 1
                continue
            inner_start = i + 2
            inner_end = close
            # skip file/image/category entirely
            if (_match_at(buf, inner_start, file_p)
                    or _match_at(buf, inner_start, image_p)
                    or _match_at(buf, inner_start, category_p)):
                i = close + 2
                continue
            # find last pipe in inner
            pipe = -1
            for k in range(inner_end - 1, inner_start - 1, -1):
                if buf[k] == PIPE:
                    pipe = k
                    break
            if pipe >= 0:
                for k in range(pipe + 1, inner_end):
                    out[o] = buf[k]; o += 1
            else:
                for k in range(inner_start, inner_end):
                    out[o] = buf[k]; o += 1
            out[o] = SP; o += 1
            i = close + 2
            continue

        # <... tag handling
        if b == LT:
            if _match_at(buf, i, ref_open):
                e = _find_at(buf, i + ref_open.shape[0], ref_close)
                i = e + ref_close.shape[0] if e >= 0 else n
                continue
            if _match_at(buf, i, script_open):
                e = _find_at(buf, i + script_open.shape[0], script_close)
                i = e + script_close.shape[0] if e >= 0 else n
                continue
            if _match_at(buf, i, style_open):
                e = _find_at(buf, i + style_open.shape[0], style_close)
                i = e + style_close.shape[0] if e >= 0 else n
                continue
            if _match_at(buf, i, comment_o):
                e = _find_at(buf, i + comment_o.shape[0], comment_c)
                i = e + comment_c.shape[0] if e >= 0 else n
                continue
            # generic tag: scan to >
            j = i + 1
            while j < n and buf[j] != GT:
                j += 1
            i = j + 1 if j < n else n
            continue

        # entity decode
        if b == AMP:
            if _match_at(buf, i, amp_amp):
                out[o] = AMP; o += 1;            i += amp_amp.shape[0];  continue
            if _match_at(buf, i, amp_lt):
                out[o] = LT;  o += 1;            i += amp_lt.shape[0];   continue
            if _match_at(buf, i, amp_gt):
                out[o] = GT;  o += 1;            i += amp_gt.shape[0];   continue
            if _match_at(buf, i, amp_quot):
                out[o] = 0x22; o += 1;           i += amp_quot.shape[0]; continue
            if _match_at(buf, i, amp_apos):
                out[o] = 0x27; o += 1;           i += amp_apos.shape[0]; continue
            if _match_at(buf, i, amp_nbsp):
                out[o] = SP;  o += 1;            i += amp_nbsp.shape[0]; continue
            out[o] = b; o += 1
            i += 1
            continue

        # line-leading '|' (table row) — skip to end-of-line
        if b == PIPE and (i == 0 or buf[i - 1] == NL):
            j = i
            while j < n and buf[j] != NL:
                j += 1
            i = j + 1 if j < n else n
            continue

        # line-leading '=' (header) — skip to end-of-line
        if b == EQ and (i == 0 or buf[i - 1] == NL):
            j = i
            while j < n and buf[j] != NL:
                j += 1
            i = j + 1 if j < n else n
            continue

        # default: emit
        out[o] = b; o += 1
        i += 1

    return o


def mass_strip(text: str) -> str:
    """python-friendly wrapper: takes str, returns str."""
    buf = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
    out = np.empty(buf.shape[0] + 16, dtype=np.uint8)  # upper bound
    n_out = mass_strip_bytes(
        buf, out,
        _REF_OPEN, _REF_CLOSE,
        _SCRIPT_OPEN, _SCRIPT_CLOSE,
        _STYLE_OPEN, _STYLE_CLOSE,
        _COMMENT_O, _COMMENT_C,
        _FILE_P, _IMAGE_P, _CATEGORY_P,
        _AMP_AMP, _AMP_LT, _AMP_GT, _AMP_QUOT, _AMP_APOS, _AMP_NBSP,
    )
    return out[:n_out].tobytes().decode("utf-8", errors="replace")
