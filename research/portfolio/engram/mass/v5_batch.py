"""
v5_batch.py — process N articles in a SINGLE numba call.

given:
  texts_buf:   concatenated uint8 of all N articles
  offsets:     int32[N+1] — cumulative byte offsets
one numba call iterates the N articles internally, producing:
  out_idx:     int32[max_total] — all emitted indices
  out_starts:  int32[N+1] — where article k's indices begin/end

per-article python/numba boundary cost: ~1-3 μs. at 40 articles, that
was ~80 μs. batch mode collapses to ONE crossing. saves 40 calls of
overhead, plus also collapses all the np.empty / np.frombuffer Python
allocations into one prep pass.
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

from v4_tables import mass_fused_v4, _NOWIKI_O, _NOWIKI_C, _PRE_O, _PRE_C
from v3_fused import (
    _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
    _FILE, _IMAGE, _CATEGORY, build_vocab_table,
)


@njit(cache=True, boundscheck=False, nogil=True)
def mass_fused_batch_v4(
    texts_buf: np.ndarray,      # concatenated input bytes
    offsets: np.ndarray,         # int32[N+1]
    word_buf: np.ndarray,        # scratch per article (reused)
    out_idx: np.ndarray,         # int32[max_total]
    out_starts: np.ndarray,      # int32[N+1]
    vocab_buf, vocab_off, vocab_pos, table,
    ref_o, ref_c, scr_o, scr_c, sty_o, sty_c, cmt_o, cmt_c,
    file_p, image_p, category_p,
    nowiki_o, nowiki_c, pre_o, pre_c,
) -> int:
    """
    run mass_fused_v4 across N articles. returns total indices written.

    each article's output occupies out_idx[out_starts[k] : out_starts[k+1]].
    """
    n_articles = offsets.shape[0] - 1
    cursor = 0
    out_starts[0] = 0
    total_max = out_idx.shape[0]

    for k in range(n_articles):
        art_start = offsets[k]
        art_end = offsets[k + 1]
        art_buf = texts_buf[art_start:art_end]

        remaining = total_max - cursor
        # slice out_idx view for this article
        art_out = out_idx[cursor:cursor + remaining]

        n_emit = mass_fused_v4(
            art_buf, word_buf, art_out,
            vocab_buf, vocab_off, vocab_pos, table,
            ref_o, ref_c, scr_o, scr_c, sty_o, sty_c, cmt_o, cmt_c,
            file_p, image_p, category_p,
            nowiki_o, nowiki_c, pre_o, pre_c,
        )
        cursor += n_emit
        out_starts[k + 1] = cursor

    return cursor


class FusedIngestBatch:
    def __init__(self, word_to_idx: dict[str, int]):
        (self.vocab_buf, self.vocab_off,
         self.vocab_pos, self.table) = build_vocab_table(word_to_idx)

    def extract_batch(self, texts: list[str], max_per_article: int = 20_000):
        """
        process a list of article texts in ONE numba call. returns a
        list of int32 arrays (one per article).
        """
        N = len(texts)
        encoded = [t.encode("utf-8") for t in texts]
        lens = np.fromiter((len(e) for e in encoded), dtype=np.int32, count=N)
        total_bytes = int(lens.sum())

        texts_buf = np.empty(total_bytes, dtype=np.uint8)
        offsets = np.empty(N + 1, dtype=np.int32)
        offsets[0] = 0
        cursor = 0
        for k, e in enumerate(encoded):
            texts_buf[cursor:cursor + len(e)] = np.frombuffer(e, dtype=np.uint8)
            cursor += len(e)
            offsets[k + 1] = cursor

        # scratch buffers
        max_article = int(lens.max()) + 16 if N else 16
        word_buf = np.empty(max_article, dtype=np.uint8)
        out_idx = np.empty(N * max_per_article, dtype=np.int32)
        out_starts = np.empty(N + 1, dtype=np.int32)

        total = mass_fused_batch_v4(
            texts_buf, offsets, word_buf, out_idx, out_starts,
            self.vocab_buf, self.vocab_off, self.vocab_pos, self.table,
            _REF_O, _REF_C, _SCR_O, _SCR_C, _STY_O, _STY_C, _CMT_O, _CMT_C,
            _FILE, _IMAGE, _CATEGORY,
            _NOWIKI_O, _NOWIKI_C, _PRE_O, _PRE_C,
        )
        # slice per article
        results = []
        for k in range(N):
            s = out_starts[k]; e = out_starts[k + 1]
            results.append(out_idx[s:e].copy())
        return results
