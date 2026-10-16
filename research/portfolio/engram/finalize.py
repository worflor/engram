"""finalize.py -- end-of-ingest polish pass for an EngramFile.

three cheap principled operations run once at the end of ingestion,
before the brain is saved:

  1. crystallize: fit the dream meta-oscillator from the dream buffer
     so the saved brain ships with its temporal-prediction layer ready.
     normally this fit is lazy (first measure() call) — doing it here
     means a freshly-loaded brain can predict on the first observation.

  2. consolidate: merge wells whose K-centroids are closer than
     consolidation_factor × typical_gap. cleans up the long tail of
     near-duplicate singleton wells that form on noisy boundary
     articles without affecting real discriminability. threshold is
     derived from the brain's own typical_gap (median pairwise well
     distance), not a hand-picked number.

  3. name: populate ef.annotations['well_labels'] via name_wells() so
     the saved brain has physics + composition + topology descriptions
     for every well, and round-tripping through .engram preserves them.

cost is O(W^2) for the pairwise distance matrix, which vectorizes
trivially at W ~ 10k. on typical brains (W < 2000) it runs in
milliseconds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from engram_codec import SEED_COUNT, WELL_ESTABLISHED, fit_pair


def finalize_brain(
    ef,
    *,
    consolidate: bool = True,
    consolidation_factor: float = 0.5,
    name: bool = True,
    crystallize: bool = True,
) -> dict:
    """polish an EngramFile at end-of-ingest.

    returns a report dict:
        wells_before, wells_after, wells_merged, wells_named, crystallized

    no arguments hard-coded: consolidation_factor is relative to the
    brain's own typical_gap; no absolute distance thresholds.
    """
    report = {
        "wells_before": len(ef._well_sum_K),
        "crystallized": False,
        "wells_merged": 0,
        "wells_named": 0,
    }

    if crystallize and ef.dream_meta_K is None and len(ef._dream_Ks) >= SEED_COUNT + 1:
        _crystallize_inline(ef)
        report["crystallized"] = True

    if consolidate:
        report["wells_merged"] = _consolidate_wells(
            ef, factor=consolidation_factor)

    # per-well meta-oscillators: fit AFTER consolidate so merged wells
    # aren't wasted on a fit that'll be discarded. each well with enough
    # dream history gets its own (meta_K, meta_G) so the brain can
    # predict each well's NEXT K-centroid from its own trajectory. this
    # is the true per-well predictive layer — global dream_meta is one
    # oscillator averaging every well; these are one oscillator PER
    # well, each tuned to that well's own dynamics.
    if crystallize:
        report["wells_meta_fit"] = _crystallize_per_well_meta(ef)

    if name:
        from engram.well_naming import name_wells
        labels = name_wells(ef)
        report["wells_named"] = len(labels)

    report["wells_after"] = len(ef._well_sum_K)
    return report


def _crystallize_inline(ef):
    """fit the meta-oscillator from the dream buffer. EngramFile.
    crystallize_dream takes a codec arg it doesn't use; inlining the
    fit_pair loop avoids the dependency and matches measure()'s
    identical lazy path.
    """
    K_hist = np.stack(ef._dream_Ks)
    P = K_hist.shape[1]
    ef.dream_meta_K = np.empty(P, dtype=np.complex128)
    ef.dream_meta_G = np.empty(P, dtype=np.complex128)
    for p in range(P):
        ef.dream_meta_K[p], ef.dream_meta_G[p], _ = fit_pair(K_hist[:, p])


def _crystallize_per_well_meta(ef) -> int:
    """fit a per-well AR(2) meta-oscillator from each well's own
    dream buffer. returns the number of wells that got a meta fit.

    each well's dream buffer is the time-ordered list of K-centroids
    the well absorbed. wells with >= SEED_COUNT+1 entries have enough
    history to fit their own (meta_K, meta_G) — a predictive layer
    specific to THAT well. the global ef.dream_meta_K answers "what
    does the whole brain do next?"; these answer "what does THIS well
    do next?"

    wells with fewer entries are skipped (fit is undefined at shorter
    lengths and we'd rather leave them absent than populate them with
    degenerate [1.0, 0.0] placeholders the router would have to special-
    case everywhere). per_well_dream_predict falls back to no-op on
    those wells, which is the honest behavior.
    """
    fit_count = 0
    for name, buf in ef._well_dream_Ks.items():
        if buf is None or len(buf) < SEED_COUNT + 1:
            continue
        K_hist = np.stack(buf)
        P = K_hist.shape[1]
        meta_K = np.empty(P, dtype=np.complex128)
        meta_G = np.empty(P, dtype=np.complex128)
        for p in range(P):
            meta_K[p], meta_G[p], _ = fit_pair(K_hist[:, p])
        ef._well_meta_K[name] = meta_K
        ef._well_meta_G[name] = meta_G
        fit_count += 1
    return fit_count


def _pairwise_well_distances(centroids: np.ndarray) -> np.ndarray:
    """vectorized pairwise RMS distance between rows of centroids.
    uses the matmul identity
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 Re(<a, b*>)
    so the output is O(W^2) memory, never O(W^2 * P). matches the
    metric absorb_gap.wgsl uses on GPU, so the threshold we derive
    from these distances is directly comparable to the one the brain
    saw during ingest.
    """
    W, P = centroids.shape
    abs_sq = (centroids.real ** 2 + centroids.imag ** 2).sum(axis=1)  # [W]
    inner = centroids @ centroids.conj().T  # [W, W] complex
    dists_sq = (abs_sq[:, None] + abs_sq[None, :] - 2 * inner.real) / P
    # floating-point noise can give small negatives on the diagonal
    np.clip(dists_sq, 0.0, None, out=dists_sq)
    return np.sqrt(dists_sq)


def _consolidate_wells(ef, factor: float = 0.5) -> int:
    """greedy merge wells with K-centroid distance < factor × typical_gap.

    returns the number of merges performed. `typical_gap` is the
    median pairwise distance between ESTABLISHED wells (count >=
    WELL_ESTABLISHED) — same definition the brain's router uses. a
    well-pair closer than half that gap is treated as duplicate:
    they'd both attract the same articles on the next route, so
    merging them costs no discriminability.
    """
    names = list(ef._well_sum_K.keys())
    W = len(names)
    if W < 2:
        return 0

    # centroids in complex P-dim
    P = ef.dim // 2
    centroids = np.empty((W, P), dtype=np.complex128)
    counts = np.empty(W, dtype=np.int64)
    for i, n in enumerate(names):
        c = int(ef._well_count.get(n, 0))
        counts[i] = c
        centroids[i] = ef._well_sum_K[n] / max(c, 1)

    dists = _pairwise_well_distances(centroids)

    # typical_gap from ESTABLISHED pairs only. if the brain hasn't
    # formed >= 2 established wells yet, we have no scale -> skip.
    est = counts >= WELL_ESTABLISHED
    if est.sum() < 2:
        return 0
    est_idx = np.where(est)[0]
    est_pairs = dists[est_idx][:, est_idx]
    # take the strict upper triangle (exclude diagonal zeros)
    iu = np.triu_indices(len(est_idx), k=1)
    typical_gap = float(np.median(est_pairs[iu]))

    threshold = typical_gap * factor
    if threshold <= 0:
        return 0

    # greedy: find the closest pair still below threshold, merge, repeat.
    # represent "alive" wells by a boolean mask; mask a merged well as
    # dead and update its distance row/column to +inf so it's ignored.
    alive = np.ones(W, dtype=bool)
    # blank the diagonal so self-pairs don't win
    np.fill_diagonal(dists, np.inf)

    merged = 0
    while True:
        # find closest pair among alive wells
        alive_idx = np.where(alive)[0]
        if len(alive_idx) < 2:
            break
        sub = dists[alive_idx][:, alive_idx]
        flat_min = sub.argmin()
        i_rel, j_rel = divmod(flat_min, sub.shape[1])
        best_d = float(sub[i_rel, j_rel])
        if best_d >= threshold or not np.isfinite(best_d):
            break
        i = int(alive_idx[i_rel])
        j = int(alive_idx[j_rel])
        if i == j:
            break  # defensive

        # merge j INTO i in ef's state
        a = names[i]
        b = names[j]
        ef._well_sum_K[a] = ef._well_sum_K[a] + ef._well_sum_K[b]
        ef._well_count[a] = int(ef._well_count[a]) + int(ef._well_count[b])
        # byte histogram if present
        if b in ef._well_sum_raw:
            if a in ef._well_sum_raw:
                ef._well_sum_raw[a] = ef._well_sum_raw[a] + ef._well_sum_raw[b]
            else:
                ef._well_sum_raw[a] = ef._well_sum_raw[b].copy()
            del ef._well_sum_raw[b]
        # G/S sums if present (v5+ format) — fold b into a before drop
        if hasattr(ef, "_well_sum_G") and b in ef._well_sum_G:
            if a in ef._well_sum_G:
                ef._well_sum_G[a] = ef._well_sum_G[a] + ef._well_sum_G[b]
            else:
                ef._well_sum_G[a] = ef._well_sum_G[b].copy()
            del ef._well_sum_G[b]
        if hasattr(ef, "_well_sum_S") and b in ef._well_sum_S:
            if a in ef._well_sum_S:
                ef._well_sum_S[a] = ef._well_sum_S[a] + ef._well_sum_S[b]
            else:
                ef._well_sum_S[a] = ef._well_sum_S[b].copy()
            del ef._well_sum_S[b]
        # per-well dream buffer (v6) — concatenate b's trajectory
        # after a's, then trim to the cap. time order within each
        # well is preserved; across the merge we can't recover the
        # true interleaving, so we just append.
        if hasattr(ef, "_well_dream_Ks") and b in ef._well_dream_Ks:
            buf_b = ef._well_dream_Ks[b]
            if a in ef._well_dream_Ks:
                merged_buf = ef._well_dream_Ks[a] + buf_b
            else:
                merged_buf = list(buf_b)
            cap = getattr(ef, "WELL_DREAM_CAP", len(merged_buf))
            if len(merged_buf) > cap:
                merged_buf = merged_buf[-cap:]
            ef._well_dream_Ks[a] = merged_buf
            del ef._well_dream_Ks[b]
        # per-well meta-oscillators (v6) — b's fit becomes stale on
        # merge; drop it so the next crystallize re-fits from the
        # merged buffer. a's fit is also stale but crystallize runs
        # after consolidate, so a fresh fit will replace it anyway.
        if hasattr(ef, "_well_meta_K") and b in ef._well_meta_K:
            del ef._well_meta_K[b]
        if hasattr(ef, "_well_meta_G") and b in ef._well_meta_G:
            del ef._well_meta_G[b]
        if hasattr(ef, "_well_meta_K") and a in ef._well_meta_K:
            del ef._well_meta_K[a]
        if hasattr(ef, "_well_meta_G") and a in ef._well_meta_G:
            del ef._well_meta_G[a]
        # transition graph (v6) — fold b's outgoing edges into a's,
        # and rewrite any edges pointing to b as edges pointing to a.
        if hasattr(ef, "_well_transitions"):
            # outgoing: b -> * becomes a -> *
            b_out = ef._well_transitions.pop(b, None)
            if b_out:
                a_out = ef._well_transitions.setdefault(a, {})
                for tgt, cnt in b_out.items():
                    # rewrite tgt==b (self-loop) to tgt==a
                    real_tgt = a if tgt == b else tgt
                    a_out[real_tgt] = a_out.get(real_tgt, 0) + cnt
            # incoming: * -> b becomes * -> a
            for src, targets in ef._well_transitions.items():
                if b in targets:
                    cnt = targets.pop(b)
                    targets[a] = targets.get(a, 0) + cnt
        del ef._well_sum_K[b]
        del ef._well_count[b]

        # update the cached centroid/count and recompute distances FROM i
        # to every other alive well.
        counts[i] = ef._well_count[a]
        centroids[i] = ef._well_sum_K[a] / max(counts[i], 1)
        # recompute row/col i against all wells (vectorized)
        abs_i = float((centroids[i].real ** 2 + centroids[i].imag ** 2).sum())
        abs_all = (centroids.real ** 2 + centroids.imag ** 2).sum(axis=1)
        inner = (centroids @ np.conj(centroids[i])).real  # [W]
        dists_i = np.sqrt(np.clip((abs_i + abs_all - 2 * inner) / P, 0, None))
        dists_i[i] = np.inf  # diagonal
        dists[i, :] = dists_i
        dists[:, i] = dists_i

        # kill j
        alive[j] = False
        dists[j, :] = np.inf
        dists[:, j] = np.inf
        merged += 1

    if merged > 0:
        ef._invalidate_caches()
    return merged
