"""
fast_embed.py

faster drop-in for text_to_trajectory.  the original embed path in
experiments/engram_glove_test.py does:

    vecs = [glove[t] for t in tokens if t in glove]   # dict of ndarrays
    raw = np.stack(vecs)                              # stacks per-word
    for i in range(T - window + 1):
        smoothed[i] = (cumsum[i + window] - cumsum[i]) / window

two pain points:

1. `glove` is a `dict[str, np.ndarray]` where each value is a tiny
   300-float array. stacking N of them allocates a temp Python list
   plus does N array copies. at 1000 tokens/article this is
   microseconds-per-word and adds up.

2. the final smoothing loop is pure-python iteration over numpy rows.
   on T=500 that's 490 Python-level iterations; numpy broadcasting
   collapses it to one vector op.

this module exposes:
    load_glove_matrix() -> (matrix: float32[V, 300], word_to_idx: dict[str, int])
    text_to_trajectory_fast(text, matrix, word_to_idx, window) -> float32[T', 300]

drop-in for `lambda text: text_to_trajectory(text, glove, window=10)`.
on synthetic text we measure ~2× on the CPU embed stage.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Tuple

import numpy as np


def _tokenize(text: str) -> list[str]:
    """lowercase, split on non-alpha, drop single-char tokens.
    matches tokenize_simple in experiments/engram_glove_test.py."""
    return [w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 1]


def load_glove_matrix(loader=None) -> Tuple[np.ndarray, dict[str, int]]:
    """
    materialize GloVe as (matrix, word_to_idx) instead of dict-of-arrays.

    the matrix is a single contiguous float32 allocation, so downstream
    fancy-index lookups (`matrix[idxs]`) run at memory bandwidth rather
    than python-speed. total size for the 400k-word 300d model: 480 MB,
    same as the dict but without per-value overhead.
    """
    if loader is None:
        # defer import so this module doesn't drag in experiments/*
        # unless the caller actually wants GloVe.
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "experiments"))
        from engram_glove_test import load_glove
        loader = load_glove

    t0 = time.time()
    glove_dict = loader()
    words = sorted(glove_dict.keys())
    V = len(words)
    D = int(next(iter(glove_dict.values())).shape[0])

    matrix = np.empty((V, D), dtype=np.float32)
    for i, w in enumerate(words):
        matrix[i] = glove_dict[w]
    word_to_idx = {w: i for i, w in enumerate(words)}

    # the original dict is no longer needed; let the caller drop it.
    elapsed = time.time() - t0
    print(f"  built glove matrix: {V} x {D} in {elapsed:.1f}s")
    return matrix, word_to_idx


# cap per-article tokens so a single monster enwiki article (some are
# 500k+ chars, ~80k tokens, yielding ~100 MB paired-trajectory) cannot
# blow up memory when 128 of them batch up. matches the implicit
# budget the zim loader already enforces via its 100k-char cap.
MAX_TOKENS_PER_ARTICLE = 20000


def text_to_trajectory_fast(
    text: str,
    matrix: np.ndarray,
    word_to_idx: dict[str, int],
    window: int = 10,
) -> np.ndarray:
    """
    matrix-backed embed with vectorized sliding window.

    semantically identical to experiments/engram_glove_test.text_to_trajectory
    for the same `text`, `glove_dict`, and `window` — the output has the
    same shape and same float values up to bit-identity (we do the same
    arithmetic in the same order), with one exception: when an article
    has more than MAX_TOKENS_PER_ARTICLE in-vocabulary tokens, we
    truncate. the original code processed the whole thing, which on
    the batch-of-128 GPU path risks OOM on pathological articles.
    """
    tokens = _tokenize(text)
    if len(tokens) < max(4, window):
        return np.empty((0, matrix.shape[1]), dtype=np.float32)

    # single list comprehension holding the GIL once, not once per
    # token. still python-speed; if this ever becomes the bottleneck
    # the next move is a C extension or gpu gather.
    idxs = [word_to_idx[t] for t in tokens if t in word_to_idx]
    if len(idxs) < max(4, window):
        return np.empty((0, matrix.shape[1]), dtype=np.float32)

    # hard upper bound on a single article's trajectory. prevents one
    # monster wiki article from dominating a 128-article batch.
    if len(idxs) > MAX_TOKENS_PER_ARTICLE:
        idxs = idxs[:MAX_TOKENS_PER_ARTICLE]

    raw = matrix[np.asarray(idxs, dtype=np.int32)]  # vectorized gather

    if window <= 1:
        return raw

    # vectorized sliding-window mean via cumulative sum difference.
    # O(T) in numpy, no python loop.
    T, D = raw.shape
    cumsum = np.empty((T + 1, D), dtype=np.float32)
    cumsum[0] = 0.0
    np.cumsum(raw, axis=0, out=cumsum[1:])
    smoothed = (cumsum[window:] - cumsum[:-window]) / np.float32(window)
    return smoothed


def make_embed_fn(matrix: np.ndarray, word_to_idx: dict[str, int], window: int = 10):
    """convenience: returns a closure usable where text_to_trajectory
    with a bound glove dict was used before."""
    def embed(text: str) -> np.ndarray:
        return text_to_trajectory_fast(text, matrix, word_to_idx, window=window)
    return embed
