"""
mass_embed.py — MASS v4 + matrix gather + sliding window, as a single
embed function callable from the hetero pipeline.

this is the final text→trajectory step when you want maximum CPU
speed. replaces mwparserfromhell + re.findall + dict lookup with:
    raw wikitext → FusedIngestV4.extract → matrix[idxs] → cumsum smooth
about 140× faster than mwparser on real enwiki, 96% jaccard.

the output trajectory is bit-compatible with what fast_embed.py
produced when fed the same (post-strip) text, so downstream codec
expects the same dtype/shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "mass"))

import numpy as np

from v4_tables import FusedIngestV4


def make_mass_embed_fn(
    matrix: np.ndarray,
    word_to_idx: dict[str, int],
    window: int = 10,
) -> Callable[[str], np.ndarray]:
    """
    returns a closure `embed(wikitext) → trajectory[T, D] float32`.

    takes RAW wikitext (not stripped). MASS handles all the markup
    navigation internally, emits vocab indices, which are then
    gathered from `matrix` and sliding-window-smoothed.
    """
    fused = FusedIngestV4(word_to_idx)
    D = matrix.shape[1]

    def embed(wikitext: str) -> np.ndarray:
        idxs = fused.extract(wikitext)
        if len(idxs) < max(4, window):
            return np.empty((0, D), dtype=np.float32)
        raw = matrix[idxs]
        T, _ = raw.shape
        if T < window:
            return raw
        cs = np.empty((T + 1, D), dtype=np.float32)
        cs[0] = 0
        np.cumsum(raw, axis=0, out=cs[1:])
        smoothed = (cs[window:] - cs[:-window]) / np.float32(window)
        return smoothed

    return embed


def make_mass_idx_fn(
    word_to_idx: dict[str, int],
) -> Callable[[str], np.ndarray]:
    """
    returns a closure `idx(wikitext) → int32[N]`.

    for the hetero_v3 pipeline: CPU workers run only MASS extraction
    and emit token indices. the heavy matrix gather + smooth + pair
    steps happen on the GPU via ChainEngine.gather_paired_batch.

    an index of -1 would mean "missing vocab"; FusedIngestV4 already
    drops those during extraction, so only valid idx ≥ 0 come out.
    """
    fused = FusedIngestV4(word_to_idx)

    def idx(wikitext: str) -> np.ndarray:
        return fused.extract(wikitext)

    return idx
