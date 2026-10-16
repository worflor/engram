"""well_naming.py -- derive human-readable labels for the wells in an
EngramFile from the well's own intrinsic structure.

three independent axes go into the label:

  physics     -- from the well's K-centroid. |K| gives the damping
                 regime (decay/damped/stable/persist) and arg(K) gives
                 the frequency regime (slow/mid/fast). a well where
                 most pairs have |K| near 1 and low phase is a slowly-
                 drifting long-memory attractor; a well with high
                 phase variance is a chaotic one.

  composition -- byte TF-IDF against the global byte histogram. bytes
                 over-represented in this well vs the brain-wide
                 average become its compositional signature. a code
                 well surfaces `{ } ( ) ; =`; a prose well surfaces
                 letters and spaces. this requires the byte-level
                 histogram per well, which hetero_v3._absorb_worker
                 accumulates into ef._well_sum_raw when the Source is
                 byte-level.

  topology    -- each well's two nearest K-space neighbors among the
                 other wells. gives the well a relational anchor —
                 'near well_5' — independent of data semantics.

the label is physics:composition:topology by default, or a subset
when data for some axis isn't available (e.g. non-byte Source leaves
composition empty).

writes results into ef.annotations['well_labels']. annotations is the
existing freeform dict in the .engram spec; no file format change.

NOT a classifier, NOT a fine-tuned LLM labeller. purely intrinsic:
same data in -> same labels out. stable across re-ingestion of the
same corpus.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

# eigenvalues_all is the standard way the codec computes oscillator
# roots from (K, G) pairs. we use it to turn a well's AR(2) state
# into its actual physics — |λ_max| is the well's damping (closer to
# 1 = less damped, > 1 = unstable), arg(λ_max) is its dominant
# angular frequency. this is the real quantity that K alone proxied.
try:
    from engram_codec import eigenvalues_all as _eigenvalues_all
except ImportError:
    _eigenvalues_all = None


def _physics_tag(centroid_K: np.ndarray,
                 centroid_G: Optional[np.ndarray] = None) -> dict:
    """describe a well's attractor character from its full AR(2)
    state — K_centroid AND G_centroid together.

    if G is provided (v5 brains), we compute actual oscillator
    eigenvalues per pair:
        λ² - K*λ + G = 0   →   λ = (K ± √(K² - 4G)) / 2
    |λ_max| is the well's damping (how persistent the mode is — closer
    to 1 = less damped, = 1 = neutrally stable, > 1 = unstable).
    arg(λ_max) is its dominant angular frequency. these are the REAL
    physics quantities — |K| alone is a proxy that bounds them but
    doesn't pin them.

    if G is None (v4 brains without G, or unavailable), fall back to
    the K-only label. the system reports which mode it's in via the
    'physics' field in the result so readers can tell whether they're
    seeing real eigenvalues or the old proxy.

    the label is a compact numerical fingerprint. with G available:
        λ{|λ_max|}ω{arg(λ_max)/π}π
    without:
        K{|K|}ω{arg(K)/π}π
    both read directly — |λ|≈1 is near the stability edge, ω/π≈0.5
    is quarter-Nyquist. no hand-chosen binning.
    """
    mags_K = np.abs(centroid_K)
    phases_K = np.abs(np.angle(centroid_K))  # [0, π]
    K_mag_med = float(np.median(mags_K))
    K_mag_std = float(mags_K.std())
    K_freq_med = float(np.median(phases_K))
    K_freq_std = float(np.std(phases_K))

    # if G is present and _eigenvalues_all is importable, compute the
    # actual eigenvalues. pick λ_max (the one with larger |λ|) per
    # pair — that's the dominant mode.
    use_eigen = (centroid_G is not None
                 and _eigenvalues_all is not None
                 and np.any(np.abs(centroid_G) > 1e-12))

    if use_eigen:
        lam1, lam2 = _eigenvalues_all(centroid_K, centroid_G)
        # elementwise pick the root with larger magnitude
        abs1 = np.abs(lam1)
        abs2 = np.abs(lam2)
        lam_max = np.where(abs1 >= abs2, lam1, lam2)
        mags_lam = np.abs(lam_max)
        phases_lam = np.abs(np.angle(lam_max))
        lam_mag_med = float(np.median(mags_lam))
        lam_mag_std = float(mags_lam.std())
        lam_freq_med = float(np.median(phases_lam))
        lam_freq_std = float(np.std(phases_lam))
        # stability indicator: max |λ| across all pairs
        lam_mag_max = float(mags_lam.max())
        # label precision is 1 decimal in the compact tag — AR(2)
        # eigenvalues drift at the 1-2% level under refinement even
        # when the underlying physics converges; 1-digit rounding
        # makes tag strings stable at the level humans care about.
        # precise numeric values are still in |λ|_med, arg(λ)_med
        # for anyone reading the full record.
        tag = f"λ{lam_mag_med:.1f}ω{lam_freq_med / np.pi:.1f}π"
        return {
            "mode": "eigen",
            "|λ|_med": round(lam_mag_med, 4),
            "|λ|_std": round(lam_mag_std, 4),
            "|λ|_max": round(lam_mag_max, 4),
            "arg(λ)_med_rad": round(lam_freq_med, 4),
            "arg(λ)_std_rad": round(lam_freq_std, 4),
            "ω_over_π": round(lam_freq_med / np.pi, 4),
            "stable": bool(lam_mag_max < 1.0),
            # keep K stats for reference / backward compat
            "|K|_med": round(K_mag_med, 4),
            "|K|_std": round(K_mag_std, 4),
            "tag": tag,
        }

    # fallback: K-only label (v4 brains or legacy paths that didn't
    # populate G). 1-decimal precision matches the eigen-label format
    # so mixed-version brains don't show artificial churn.
    tag = f"K{K_mag_med:.1f}ω{K_freq_med / np.pi:.1f}π"
    return {
        "mode": "K-only",
        "|K|_med": round(K_mag_med, 4),
        "|K|_std": round(K_mag_std, 4),
        "arg(K)_med_rad": round(K_freq_med, 4),
        "arg(K)_std_rad": round(K_freq_std, 4),
        "ω_over_π": round(K_freq_med / np.pi, 4),
        "tag": tag,
    }


def _composition_tag(
    well_hist: np.ndarray,
    global_hist: np.ndarray,
    well_count: int,
    global_count: int,
    *,
    top_k: int = 6,
) -> dict:
    """rank bytes by KL-contribution, not raw log-lift.

    naive log-lift (log P_well / P_global) surfaces bytes that are
    OVER-represented in the well but may be rare in absolute terms —
    a single UTF-8 smart quote in one source file beats a thousand
    `(` brackets because the bracket is common globally. that's
    technically discriminative but not *characterizing*.

    the principled choice is KL-contribution per byte:
        c(b) = P_well(b) * log(P_well(b) / P_global(b))
    bytes that are BOTH frequent in the well AND over-represented
    relative to the brain dominate. summing c(b) over all b gives the
    total KL divergence of this well from the brain (also returned
    as kl_bits). this is the principle-preserving choice: the tokens
    with highest contribution ARE the reason the well diverges from
    the average.
    """
    if well_count <= 0 or global_count <= 0:
        return {"top": "", "tokens": [], "kl_bits": 0.0}
    local_p = well_hist / max(float(well_count), 1.0)
    global_p = global_hist / max(float(global_count), 1.0)
    eps = 1e-8
    # per-byte KL contribution (in nats). only positive contributions
    # matter — a byte under-represented in the well pulls KL down but
    # isn't "characterizing" the well.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log((local_p + eps) / (global_p + eps))
    contrib = local_p * log_ratio
    contrib = np.where(np.isfinite(contrib), contrib, 0.0)
    pos = contrib > 0.0
    if not pos.any():
        return {"top": "", "tokens": [], "kl_bits": 0.0}

    idxs = np.where(pos)[0]
    order = idxs[np.argsort(-contrib[idxs])][:top_k]

    tokens = []
    for b in order:
        bi = int(b)
        ch = chr(bi) if 32 <= bi < 127 else f"\\x{bi:02x}"
        tokens.append(ch)

    kl_nats = float(contrib[pos].sum())
    kl_bits = round(kl_nats / np.log(2.0), 3)

    return {
        "top": "".join(tokens),
        "tokens": tokens,
        "kl_bits": kl_bits,
    }


def _topology_tag(name: str, centroids: dict, top_k: int = 2) -> dict:
    """two nearest K-space neighbors among the other wells.

    metric is the same RMS-over-complex-pairs distance the brain uses
    for routing, so neighbors here mean neighbors in the attractor
    basin arrangement that governs ingest.
    """
    others = [(n, c) for n, c in centroids.items() if n != name]
    if not others:
        return {"neighbors": []}
    me = centroids[name]
    dists = []
    for n, c in others:
        d = float(np.sqrt(np.mean(np.abs(me - c) ** 2).real))
        dists.append((d, n))
    dists.sort()
    nearest = [n for _d, n in dists[:top_k]]
    return {
        "neighbors": nearest,
        "nearest_dist": round(float(dists[0][0]), 4) if dists else 0.0,
    }


def name_wells(
    ef,
    *,
    top_bytes: int = 6,
    n_neighbors: int = 2,
    write_annotations: bool = True,
) -> dict:
    """derive a descriptive label per well from its own structure.

    returns dict[well_name -> {physics, composition, topology, name}]

    if write_annotations=True (default), the result is stored under
    ef.annotations['well_labels'] so callers can persist it by just
    save()-ing the ef — annotations round-trips through the existing
    .engram format.

    composition tags are empty for wells whose Source isn't byte-level
    (ef._well_sum_raw not populated) — that's expected; the physics
    and topology tags still carry useful structure.

    names are deterministic functions of the ef state, so the same
    brain produces the same labels every call.
    """
    labels: dict[str, dict] = {}
    if not ef._well_sum_K:
        return labels

    # per-well complex P-dim centroids for physics + topology. both
    # K-centroid (for routing-distance and the legacy K-proxy label)
    # and G-centroid (for actual eigenvalue computation) are tracked.
    # v4 brains loaded before the format bump have sum_G absent; we
    # fall back to K-only labels for those wells gracefully.
    P = ef.dim // 2
    centroids: dict[str, np.ndarray] = {}
    centroids_G: dict[str, Optional[np.ndarray]] = {}
    for name, K_sum in ef._well_sum_K.items():
        c = ef._well_count.get(name, 0)
        if c <= 0:
            continue
        centroids[name] = (K_sum / float(c)).astype(np.complex128)
        G_sum = ef._well_sum_G.get(name) if hasattr(ef, "_well_sum_G") else None
        if G_sum is not None and np.any(np.abs(G_sum) > 1e-12):
            centroids_G[name] = (G_sum / float(c)).astype(np.complex128)
        else:
            centroids_G[name] = None

    # global byte histogram for compositional tag
    have_bytes = bool(ef._well_sum_raw)
    if have_bytes:
        global_hist = np.zeros(256, dtype=np.float64)
        global_count = 0.0
        for n, h in ef._well_sum_raw.items():
            global_hist += h
            global_count += float(ef._well_count.get(n, 0))
    else:
        global_hist = None
        global_count = 0.0

    for name, cen in centroids.items():
        phys = _physics_tag(cen, centroids_G.get(name))
        topo = _topology_tag(name, centroids, top_k=n_neighbors)

        if have_bytes and name in ef._well_sum_raw:
            comp = _composition_tag(
                ef._well_sum_raw[name],
                global_hist,
                int(ef._well_count.get(name, 0)),
                int(global_count),
                top_k=top_bytes,
            )
        else:
            comp = {"top": "", "tokens": [], "kl_bits": 0.0}

        # human-readable composite. physics always present; the
        # compositional and neighbor parts join only when non-empty.
        parts = [phys["tag"]]
        if comp["top"]:
            parts.append(comp["top"])
        if topo["neighbors"]:
            parts.append(f"~{','.join(topo['neighbors'])}")
        composite = ":".join(parts)

        labels[name] = {
            "physics": phys,
            "composition": comp,
            "topology": topo,
            "count": int(ef._well_count.get(name, 0)),
            "name": composite,
        }

    if write_annotations:
        if ef.annotations is None:
            ef.annotations = {}
        ef.annotations["well_labels"] = labels

    return labels


def wells_by_label(ef) -> dict:
    """convenience: return {composite_label: well_name} for the labels
    currently in ef.annotations. inverse of name_wells() output.
    """
    if ef.annotations is None or "well_labels" not in ef.annotations:
        return {}
    out = {}
    for well, rec in ef.annotations["well_labels"].items():
        out[rec.get("name", well)] = well
    return out
