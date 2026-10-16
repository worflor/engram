"""
v3_fused.py — MASS + tokenize + vocab lookup, all in one numba pass.

the TRUE north star: bytes → int32 GloVe indices, in a single
allocation-free call.

architecture:
  - vocab stored as flat uint8 buffer + int32 offsets (sorted by word)
  - open-addressing hash table maps FNV-1a(bytes) → vocab index
  - MASS state machine emits lowercased word bytes into a scratch
    buffer, then inline-hashes + probes
  - output: int32 vocab indices directly. no Python strings. no
    dict lookups. no GIL.
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
    _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
    _FILE, _IMAGE, _CATEGORY,
    LBRACE, RBRACE, LBRACK, RBRACK, LT, GT, AMP, PIPE, EQ, NL,
    _match_at, _find_at, _is_wordchar, _lower,
)


# ── vocab table builder ──────────────────────────────────────────────

def build_vocab_table(word_to_idx: dict[str, int]):
    """
    pack a python dict into three numba-friendly arrays:
      vocab_buf:  uint8  — concatenated lowercase word bytes
      vocab_off:  int32  — cumulative offsets, shape [V+1]
      vocab_pos:  int32  — the index value that was in the dict (what
                           we return on hit)
      table:      int32  — open-addressing hash table, shape [M]
                           -1 = empty slot, else an index into vocab_*
    """
    words = list(word_to_idx.keys())
    V = len(words)

    lens = np.fromiter((len(w.encode("utf-8")) for w in words),
                       dtype=np.int32, count=V)
    total_bytes = int(lens.sum())
    vocab_buf = np.empty(total_bytes, dtype=np.uint8)
    vocab_off = np.empty(V + 1, dtype=np.int32)
    vocab_pos = np.empty(V, dtype=np.int32)

    cursor = 0
    for i, w in enumerate(words):
        vocab_off[i] = cursor
        b = w.encode("utf-8")
        # lowercase for lookup parity with MASS's _lower()
        b_lower = b.lower()
        vocab_buf[cursor:cursor + len(b_lower)] = np.frombuffer(
            b_lower, dtype=np.uint8)
        vocab_pos[i] = word_to_idx[w]
        cursor += len(b_lower)
    vocab_off[V] = cursor

    # hash table — open addressing, load factor ≤ 50%
    M = 1
    while M < V * 2:
        M <<= 1
    table = np.full(M, -1, dtype=np.int32)
    mask = M - 1
    for i in range(V):
        s = int(vocab_off[i]); e = int(vocab_off[i + 1])
        h = _fnv1a_py(vocab_buf[s:e])
        slot = h & mask
        while table[slot] != -1:
            slot = (slot + 1) & mask
        table[slot] = i
    return vocab_buf, vocab_off, vocab_pos, table


def _fnv1a_py(arr: np.ndarray) -> int:
    """python reference hash (must match _fnv1a_numba byte-for-byte)."""
    h = 0xcbf29ce484222325
    prime = 0x100000001b3
    mod = 1 << 64
    for b in arr:
        h ^= int(b)
        h = (h * prime) % mod
    return h


# ── numba hash + probe ───────────────────────────────────────────────

@njit(cache=True, inline="always", boundscheck=False)
def _fnv1a_numba(buf: np.ndarray, start: int, length: int) -> np.uint64:
    h = np.uint64(0xcbf29ce484222325)
    prime = np.uint64(0x100000001b3)
    for k in range(length):
        h = h ^ np.uint64(buf[start + k])
        h = h * prime
    return h


@njit(cache=True, boundscheck=False, nogil=True)
def _lookup(word_buf: np.ndarray, ws: int, wl: int,
            vocab_buf: np.ndarray, vocab_off: np.ndarray,
            vocab_pos: np.ndarray, table: np.ndarray) -> int:
    """
    return the vocab index for word_buf[ws:ws+wl], or -1 if missing.
    """
    M = table.shape[0]
    mask = M - 1
    h = _fnv1a_numba(word_buf, ws, wl)
    slot = int(np.int64(h) & np.int64(mask))
    while True:
        idx = table[slot]
        if idx == -1:
            return -1
        vs = vocab_off[idx]
        ve = vocab_off[idx + 1]
        if ve - vs == wl:
            ok = True
            for k in range(wl):
                if word_buf[ws + k] != vocab_buf[vs + k]:
                    ok = False
                    break
            if ok:
                return vocab_pos[idx]
        slot = (slot + 1) & mask


# ── the fused pass ───────────────────────────────────────────────────

@njit(cache=True, boundscheck=False, nogil=True)
def mass_fused(
    buf: np.ndarray,
    word_buf: np.ndarray,
    out_idx: np.ndarray,       # int32[max_out] — emitted vocab indices
    # vocab
    vocab_buf, vocab_off, vocab_pos, table,
    # grammar patterns
    ref_o, ref_c, scr_o, scr_c, sty_o, sty_c, cmt_o, cmt_c,
    file_p, image_p, category_p,
) -> int:
    """
    strip + tokenize + vocab-lookup in one pass. returns number of
    indices written to out_idx.
    """
    n = buf.shape[0]
    i = 0
    brace_depth = 0

    n_out = 0
    wb_cursor = 0
    ws = -1
    wl = 0
    max_out = out_idx.shape[0]

    while i < n:
        b = buf[i]

        if brace_depth > 0:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx
                        n_out += 1
                ws = -1; wl = 0
            if b == RBRACE and i + 1 < n and buf[i + 1] == RBRACE:
                brace_depth -= 1; i += 2; continue
            if b == LBRACE and i + 1 < n and buf[i + 1] == LBRACE:
                brace_depth += 1; i += 2; continue
            i += 1; continue

        if b == LBRACE and i + 1 < n and buf[i + 1] == LBRACE:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx
                        n_out += 1
                ws = -1; wl = 0
            brace_depth = 1
            i += 2; continue

        if b == LBRACK and i + 1 < n and buf[i + 1] == LBRACK:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx
                        n_out += 1
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
            if (_match_at(buf, inner_s, file_p)
                    or _match_at(buf, inner_s, image_p)
                    or _match_at(buf, inner_s, category_p)):
                i = close + 2; continue
            pipe = -1
            for k in range(inner_e - 1, inner_s - 1, -1):
                if buf[k] == PIPE:
                    pipe = k; break
            ss = pipe + 1 if pipe >= 0 else inner_s
            ee = inner_e
            inner_ws = -1
            inner_wl = 0
            for k in range(ss, ee):
                bb = buf[k]
                if _is_wordchar(bb):
                    if inner_ws < 0:
                        inner_ws = wb_cursor
                    word_buf[wb_cursor] = _lower(bb)
                    wb_cursor += 1
                    inner_wl += 1
                else:
                    if inner_ws >= 0 and inner_wl > 1:
                        idx = _lookup(word_buf, inner_ws, inner_wl,
                                      vocab_buf, vocab_off, vocab_pos, table)
                        if idx >= 0 and n_out < max_out:
                            out_idx[n_out] = idx
                            n_out += 1
                    inner_ws = -1; inner_wl = 0
            if inner_ws >= 0 and inner_wl > 1:
                idx = _lookup(word_buf, inner_ws, inner_wl,
                              vocab_buf, vocab_off, vocab_pos, table)
                if idx >= 0 and n_out < max_out:
                    out_idx[n_out] = idx
                    n_out += 1
            i = close + 2; continue

        if b == LT:
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx
                        n_out += 1
                ws = -1; wl = 0
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
            j = i + 1
            while j < n and buf[j] != GT:
                j += 1
            i = j + 1 if j < n else n
            continue

        if (b == PIPE or b == EQ) and (i == 0 or buf[i - 1] == NL):
            if ws >= 0:
                if wl > 1:
                    idx = _lookup(word_buf, ws, wl, vocab_buf,
                                  vocab_off, vocab_pos, table)
                    if idx >= 0 and n_out < max_out:
                        out_idx[n_out] = idx
                        n_out += 1
                ws = -1; wl = 0
            j = i
            while j < n and buf[j] != NL:
                j += 1
            i = j + 1 if j < n else n
            continue

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
                        out_idx[n_out] = idx
                        n_out += 1
                ws = -1; wl = 0
        i += 1

    if ws >= 0 and wl > 1:
        idx = _lookup(word_buf, ws, wl, vocab_buf,
                      vocab_off, vocab_pos, table)
        if idx >= 0 and n_out < max_out:
            out_idx[n_out] = idx
            n_out += 1

    return n_out


# ── python-friendly wrapper ──────────────────────────────────────────

class FusedIngest:
    """stateful wrapper: build vocab table once, then call .extract()."""

    def __init__(self, word_to_idx: dict[str, int]):
        (self.vocab_buf, self.vocab_off,
         self.vocab_pos, self.table) = build_vocab_table(word_to_idx)

    def extract(self, text: str, max_out: int = 30000) -> np.ndarray:
        buf = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        word_buf = np.empty(buf.shape[0] + 16, dtype=np.uint8)
        out_idx = np.empty(max_out, dtype=np.int32)
        n = mass_fused(
            buf, word_buf, out_idx,
            self.vocab_buf, self.vocab_off, self.vocab_pos, self.table,
            _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
            _FILE, _IMAGE, _CATEGORY,
        )
        return out_idx[:n].copy()
